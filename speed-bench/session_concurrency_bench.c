#include "q36.h"
#include "q36_gpu.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "concurrency-bench: %s\n", message);
    exit(1);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int integer(const char *s) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || !*s || *end || n < 1 || n > INT_MAX) fail("invalid positive integer");
    return (int)n;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp || fseek(fp, 0, SEEK_END)) fail("cannot open prompt");
    long size = ftell(fp);
    if (size < 0 || fseek(fp, 0, SEEK_SET)) fail("cannot seek prompt");
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, fp) != (size_t)size) fail("cannot read prompt");
    fclose(fp);
    text[size] = 0;
    return text;
}

int main(int argc, char **argv) {
    const char *model = Q36_DEFAULT_MODEL_PATH, *path = NULL, *mode = "plain";
    int count = 2, context = 128, tokens = 64, append = 0, chunk = 0;
    bool verify = false, mtp = false, cold = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            puts("Usage: speed-bench/session_concurrency_bench --prompt-file FILE [options]\n"
                 "  -m FILE              Model GGUF\n"
                 "  --ctx N              Prompt tokens per stream (default 128)\n"
                 "  --concurrency N      Simultaneous streams, 1..8 (default 2)\n"
                 "  --gen-tokens N       Exact output tokens per stream (default 64)\n"
                 "  --append-tokens N    Resume another session during plain decode\n"
                 "  --prefill-chunk N    Maximum prefill rows per dispatch\n"
                 "  --mode MODE          plain or serial-spec\n"
                 "  --mtp                Load the embedded MTP head\n"
                 "  --cold               Include the first decode's pipeline setup\n"
                 "  --verify             Compare every output token with sequential greedy decode");
            return 0;
        }
        if (!strcmp(arg, "--verify")) { verify = true; continue; }
        if (!strcmp(arg, "--cold")) { cold = true; continue; }
        if (!strcmp(arg, "--mtp")) { mtp = true; continue; }
        if (i + 1 == argc) fail("option needs a value");
        const char *value = argv[++i];
        if (!strcmp(arg, "-m")) model = value;
        else if (!strcmp(arg, "--prompt-file")) path = value;
        else if (!strcmp(arg, "--mode")) mode = value;
        else if (!strcmp(arg, "--concurrency")) count = integer(value);
        else if (!strcmp(arg, "--ctx")) context = integer(value);
        else if (!strcmp(arg, "--gen-tokens")) tokens = integer(value);
        else if (!strcmp(arg, "--append-tokens")) append = integer(value);
        else if (!strcmp(arg, "--prefill-chunk")) chunk = integer(value);
        else fail("unknown option");
    }
    int room = tokens > append ? tokens : append;
    if (!path || count > 8 || room >= Q36_CONTEXT_MAX ||
        context >= Q36_CONTEXT_MAX - room) fail("invalid prompt, concurrency or context");
    bool spec = !strcmp(mode, "serial-spec");
    if (strcmp(mode, "plain") && !spec) fail("mode must be plain or serial-spec");
    if (spec && !mtp) fail("speculative modes need --mtp");
    if (append && (spec || tokens < 8)) fail("continued prefill needs plain mode and at least eight output tokens");
    q36_engine_options opt = {
        .model_path = model, .backend = Q36_BACKEND_VULKAN,
        .mtp_path = mtp ? model : NULL, .mtp_draft_tokens = mtp ? 2 : 1,
        .context_size = context + room + 1, .session_count = count + !!append,
        .prefill_chunk = (uint32_t)chunk,
        .cache_type_k = Q36_KV_CACHE_Q8_0, .cache_type_v = Q36_KV_CACHE_Q4_0,
    };
    q36_engine *engine = NULL;
    if (q36_engine_open(&engine, &opt)) fail("engine open failed");
    if (mtp && !q36_engine_has_mtp(engine)) fail("model has no MTP head");
    struct stat st;
    if (stat(model, &st)) fail("cannot stat model");
    q36_context_memory memory = q36_context_memory_estimate_configured(opt.backend,
        opt.context_size, opt.prefill_chunk, opt.cache_type_k, opt.cache_type_v);
    uint64_t estimate = (uint64_t)st.st_size + (uint64_t)opt.session_count * memory.total_bytes;
    /* Reserve the verifier and predictor state before allocating any sessions.
     * Leave ten percent of the device working set outside the benchmark. */
    if (mtp) estimate += (uint64_t)count * (256u << 20);
    uint64_t budget = q36_gpu_recommended_working_set_size();
    if (budget && estimate > budget / 10u * 9u) {
        fprintf(stderr, "concurrency-bench: estimated %.2f GiB exceeds %.2f GiB budget\n",
                estimate / 1073741824.0, budget * 0.9 / 1073741824.0);
        q36_engine_close(engine);
        return 77;
    }
    char *text = read_file(path);
    q36_tokens corpus = {0};
    q36_tokenize_text(engine, text, &corpus);
    free(text);
    /* Stagger the streams so identical routing cannot flatter the batch. */
    if (corpus.len < context + 17 * (count - 1) ||
        (append && corpus.len < context + append + 17 * count)) fail("prompt corpus is too short");
    q36_session *sessions[8] = {0};
    q36_tokens prompts[8] = {0};
    int *generated[8] = {0}, done[8] = {0};
    char err[256] = {0};
    double prefill_start = now();
    for (int i = 0; i < count; i++) {
        prompts[i] = (q36_tokens){.v = corpus.v + 17 * i, .len = context, .cap = context};
        generated[i] = malloc((size_t)tokens * sizeof(int));
        if (!generated[i] || q36_session_create(&sessions[i], engine, opt.context_size) ||
            q36_session_sync(sessions[i], &prompts[i], err, sizeof(err))) fail(err);
    }
    q36_session *prefill = NULL;
    q36_tokens prefill_prompt = {.v = append ? corpus.v + 17 * count : NULL,
        .len = context, .cap = context};
    if (append && (q36_session_create(&prefill, engine, opt.context_size) ||
        q36_session_sync(prefill, &prefill_prompt, err, sizeof(err)))) fail(err);
    double prefill_seconds = now() - prefill_start;
    q36_gpu_print_memory_report("concurrency-prefill");
    double *steps = malloc((size_t)tokens * sizeof(double));
    if (!steps) fail("cannot allocate step timings");
    const int eos = q36_token_eos(engine);
    double start = 0;
    int total = 0, cycles = 0;
    double finished[8] = {0};
    const int requested_tokens = tokens;
    double mixed_seconds = 0;
    /* Warm the decode pipelines before measuring steady throughput. */
    for (int phase = 0; phase < (cold ? 1 : 2); phase++) {
        bool warmup = !cold && phase == 0;
        tokens = warmup ? (requested_tokens < 16 ? requested_tokens : 16) : requested_tokens;
        if (phase) for (int i = 0; i < count; i++) {
            q36_session_invalidate(sessions[i]);
            if (q36_session_sync(sessions[i], &prompts[i], err, sizeof(err))) fail(err);
        }
        if (phase && append) {
            prefill_prompt.len = context;
            q36_session_invalidate(prefill);
            if (q36_session_sync(prefill, &prefill_prompt, err, sizeof(err))) fail(err);
        }
        memset(done, 0, sizeof(done));
        memset(finished, 0, sizeof(finished));
        total = cycles = 0;
        mixed_seconds = 0;
        start = now();
        while (total < tokens * count) {
            double step_start = now();
            q36_decode_item items[8];
            int members[8], active = 0;
            for (int i = 0; i < count; i++) {
                if (done[i] == tokens) continue;
                int token = q36_session_argmax_excluding(sessions[i], eos);
                if (token < 0) fail("cannot choose token");
                members[active] = i;
                items[active++] = (q36_decode_item){.session = sessions[i], .token = token};
            }
            if (spec) {
                for (int j = 0; j < active; j++) {
                    int i = members[j], accepted[2];
                    int n = q36_session_eval_speculative_argmax_excluding(sessions[i],
                        items[j].token, tokens - done[i], eos, accepted, 2, err, sizeof(err));
                    if (n < 1) fail(err);
                    memcpy(generated[i] + done[i], accepted, (size_t)n * sizeof(int));
                    done[i] += n;
                    total += n;
                }
            } else {
                if (append && cycles == 4) {
                    prefill_prompt.len = context + append;
                    double t0 = now();
                    if (q36_sessions_eval_batch_with_prefill(items, active, prefill,
                            &prefill_prompt, err, sizeof(err))) fail(err);
                    mixed_seconds = now() - t0;
                } else if (q36_sessions_eval_batch(items, active, err, sizeof(err))) fail(err);
                for (int j = 0; j < active; j++) {
                    int i = members[j];
                    generated[i][done[i]++] = items[j].token;
                    total++;
                }
            }
            steps[cycles++] = now() - step_start;
            for (int i = 0; i < count; i++)
                if (!finished[i] && done[i] == tokens) finished[i] = now() - start;
        }
    }
    double seconds = now() - start;
    q36_gpu_print_memory_report("concurrency-decode");
    qsort(steps, (size_t)cycles, sizeof(double), compare_double);
    double fastest = finished[0], slowest = finished[0];
    for (int i = 0; i < count; i++) {
        if (finished[i] < fastest) fastest = finished[i];
        if (finished[i] > slowest) slowest = finished[i];
    }
    /* Verification is outside the timing window and needs no second set of
     * model sessions. Compare with the ordinary single-stream arithmetic. */
    if (verify) for (int i = 0; i < count; i++) {
        q36_session_invalidate(sessions[i]);
        if (q36_session_sync(sessions[i], &prompts[i], err, sizeof(err))) fail(err);
    }
    if (verify) for (int i = 0; i < count; i++) {
        for (int t = 0; t < tokens; t++) {
            int token = q36_session_argmax_excluding(sessions[i], eos);
            if (token != generated[i][t]) {
                fprintf(stderr, "stream=%d token=%d batch=%d sequential=%d\n",
                        i, t, generated[i][t], token);
                fail("greedy output mismatch");
            }
            if (q36_session_eval(sessions[i], token, err, sizeof(err))) fail(err);
        }
    }
    puts("concurrency,context,mode,mtp,generated,cycles,prefill_seconds,decode_seconds,aggregate_tps,slowest_stream_tps,fastest_stream_tps,step_p50_ms,step_p95_ms,step_max_ms,mixed_step_ms,verified");
    printf("%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
           count, context, mode, mtp, total, cycles, prefill_seconds, seconds,
           total / seconds, tokens / slowest, tokens / fastest,
           1000 * steps[(cycles - 1) / 2], 1000 * steps[(cycles * 95 + 99) / 100 - 1],
           1000 * steps[cycles - 1], 1000 * mixed_seconds, verify);
    free(steps);
    for (int i = 0; i < count; i++) {
        q36_session_free(sessions[i]);
        free(generated[i]);
    }
    q36_session_free(prefill);
    q36_tokens_free(&corpus);
    q36_engine_close(engine);
    return 0;
}
