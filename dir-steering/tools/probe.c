#include "../../q36.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float attn, ffn; } setting;

static void fail(const char *why) {
    fprintf(stderr, "steering-probe: %s\n", why);
    exit(1);
}

static double logsumexp(const float *x, int n) {
    double max = -INFINITY, sum = 0;
    for (int i = 0; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) sum += exp((double)x[i] - max);
    return max + log(sum);
}

static void json_bytes(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') fputc('\\', fp);
        if (c < 32) fprintf(fp, "\\u%04x", c);
        else fputc(c, fp);
    }
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s MODEL BANK PROMPTS OUT [ATTN:FFN,... [TOKENS]]\n", argv[0]);
        return 2;
    }
    int max_tokens = 0;
    if (argc == 7) {
        char *end;
        long n = strtol(argv[6], &end, 10);
        if (!argv[6][0] || *end || n < 0 || n > 256) fail("TOKENS must be 0..256");
        max_tokens = (int)n;
    }
    setting settings[17] = {{0, 0}};
    int n_settings = 1;
    char *spec = strdup(argc >= 6 ? argv[5] : "0:0.25,0:0.5,0:1,0.5:0.5,1:1");
    if (!spec) fail("out of memory");
    char *save = NULL;
    for (char *part = strtok_r(spec, ",", &save); part; part = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        errno = 0;
        float attn = strtof(part, &end);
        if (end == part || *end != ':' || errno) fail("invalid attention:FFN settings");
        char *next = end + 1;
        float ffn = strtof(next, &end);
        if (end == next || *end || errno || !isfinite(attn) || !isfinite(ffn) ||
            fabsf(attn) > 100 || fabsf(ffn) > 100 || n_settings == 17)
            fail("invalid attention:FFN settings");
        if (attn == 0 && ffn == 0) continue;
        settings[n_settings++] = (setting){attn, ffn};
    }
    free(spec);
    FILE *input = fopen(argv[3], "r");
    FILE *output = fopen(argv[4], "w");
    if (!input || !output) fail("cannot open prompt or output file");
    q36_engine_options options = {
        .model_path = argv[1], .directional_steering_file = argv[2],
        .context_size = 512, .prefill_chunk = 64,
#ifdef __APPLE__
        .backend = Q36_BACKEND_METAL,
#else
        .backend = Q36_BACKEND_VULKAN,
#endif
    };
    q36_engine *engine = NULL;
    q36_session *session = NULL;
    if (q36_engine_open(&engine, &options) || q36_session_create(&session, engine, 512))
        fail("cannot open model or session");
    const char *prefixes[] = {"I", "I'm", "Sorry", "Unfortunately"};
    int refusal_ids[4], n_ids = 0;
    for (int i = 0; i < 4; i++) {
        q36_tokens tokens = {0};
        q36_tokenize_text(engine, prefixes[i], &tokens);
        if (!tokens.len) fail("cannot tokenize refusal prefix");
        int duplicate = 0;
        for (int j = 0; j < n_ids; j++) if (refusal_ids[j] == tokens.v[0]) duplicate = 1;
        if (!duplicate) refusal_ids[n_ids++] = tokens.v[0];
        q36_tokens_free(&tokens);
    }
    fprintf(output, "{\"type\":\"config\",\"metric\":\"first-token refusal-prefix probability\",\"refusal_token_ids\":[");
    for (int i = 0; i < n_ids; i++) fprintf(output, "%s%d", i ? "," : "", refusal_ids[i]);
    fprintf(output, "],\"max_tokens\":%d}\n", max_tokens);
    int vocab = q36_engine_vocab_size(engine);
    float *baseline = malloc((size_t)vocab * sizeof(float));
    if (!baseline) fail("out of memory");
    char *line = NULL;
    size_t cap = 0;
    int index = 0;
    char error[192] = {0};
    while (getline(&line, &cap, input) >= 0) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;
        int expected_id = -1;
        char *answer = strchr(line, '\t');
        if (answer) {
            *answer++ = '\0';
            q36_tokens expected = {0};
            q36_tokenize_text(engine, answer, &expected);
            if (expected.len) expected_id = expected.v[0];
            q36_tokens_free(&expected);
        }
        q36_tokens prompt = {0};
        q36_encode_chat_prompt(engine, "You are a helpful assistant.", line, Q36_THINK_NONE, &prompt);
        if (prompt.len + max_tokens > 512) fail("prompt plus completion exceeds context");
        double baseline_lse = 0;
        for (int k = 0; k < n_settings; k++) {
            q36_session_invalidate(session);
            if (q36_session_set_directional_steering(session, settings[k].attn, settings[k].ffn) ||
                q36_session_sync(session, &prompt, error, sizeof(error))) fail(error);
            const float *logits = q36_session_logits(session, &vocab);
            if (!logits) fail("cannot read logits");
            double lse = logsumexp(logits, vocab);
            if (!isfinite(lse)) fail("non-finite logits");
            if (!k) {
                memcpy(baseline, logits, (size_t)vocab * sizeof(float));
                baseline_lse = lse;
            }
            double refusal = 0, kl = 0;
            for (int j = 0; j < n_ids; j++) refusal += exp((double)logits[refusal_ids[j]] - lse);
            for (int j = 0; j < vocab; j++) {
                double logp = (double)baseline[j] - baseline_lse;
                kl += exp(logp) * (logp - ((double)logits[j] - lse));
            }
            int top = q36_session_argmax(session);
            fprintf(output,
                    "{\"type\":\"measurement\",\"index\":%d,\"prompt_tokens\":%d,"
                    "\"attn\":%.9g,\"ffn\":%.9g,\"refusal_probability\":%.12g,"
                    "\"refusal_logodds\":%.12g,\"kl_from_baseline\":%.12g,"
                    "\"top_token\":%d,\"top_probability\":%.12g,\"expected_token\":%d",
                    index, prompt.len, settings[k].attn, settings[k].ffn, refusal,
                    log(refusal + 1e-8) - log(1.0 - refusal + 1e-8), fmax(kl, 0),
                    top, exp((double)logits[top] - lse), expected_id);
            if (max_tokens) {
                int generated = 0, ended = 0;
                fputs(",\"completion\":\"", output);
                while (generated < max_tokens) {
                    int token = q36_session_argmax(session);
                    if (token == q36_token_eos(engine)) {
                        ended = 1;
                        break;
                    }
                    size_t len;
                    char *piece = q36_token_text(engine, token, &len);
                    json_bytes(output, piece, len);
                    free(piece);
                    generated++;
                    if (q36_session_eval(session, token, error, sizeof(error))) fail(error);
                }
                fprintf(output, "\",\"generated_tokens\":%d,\"ended\":%s",
                        generated, ended ? "true" : "false");
            }
            fputs("}\n", output);
            fflush(output);
        }
        q36_tokens_free(&prompt);
        fprintf(stderr, "steering-probe: prompt %d\n", ++index);
    }
    if (ferror(input) || fclose(output)) fail("prompt or result I/O failed");
    fclose(input);
    free(line);
    free(baseline);
    q36_session_free(session);
    q36_engine_close(engine);
    return 0;
}
