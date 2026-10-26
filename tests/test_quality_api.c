#define main quality_scorer_main
#define q36_token_text quality_test_token_text
#define q36_tokenize_text quality_test_tokenize_text
#define q36_tokens_free quality_test_tokens_free
#include "../gguf-tools/quality-testing/score_openrouter.c"
#undef main
#undef q36_token_text
#undef q36_tokenize_text
#undef q36_tokens_free
#include <assert.h>

char *quality_test_token_text(q36_engine *engine, int token, size_t *len) {
    (void)engine;
    const char *text[] = {"A", "B", "AB", "", "\xf0\x9f"};
    assert(token >= 0 && token < 5);
    *len = strlen(text[token]);
    char *copy = malloc(*len + 1);
    assert(copy);
    memcpy(copy, text[token], *len + 1);
    return copy;
}

void quality_test_tokenize_text(q36_engine *engine, const char *text, q36_tokens *out) {
    (void)engine;
    assert(!out->v && out->len == 0);
    out->v = malloc(2 * sizeof(*out->v));
    assert(out->v);
    if (!strcmp(text, "A")) out->v[out->len++] = 0;
    else if (!strcmp(text, "B")) out->v[out->len++] = 1;
    else if (!strcmp(text, "AB")) {
        out->v[out->len++] = 0;
        out->v[out->len++] = 1;
    }
}

void quality_test_tokens_free(q36_tokens *tokens) {
    free(tokens->v);
    memset(tokens, 0, sizeof(*tokens));
}

static void check_top1(void) {
    const struct {
        const char *alts;
        int count;
        unsigned matches;
    } fixtures[] = {
        {"[]", 0, 0},
        {"[{\"bytes\":[65],\"logprob\":-1},"
          "{\"bytes\":[66],\"logprob\":-2}]", 1, 1u << 0},
        {"[{\"bytes\":[65],\"logprob\":-2},"
          "{\"bytes\":[66],\"logprob\":-1}]", 1, 1u << 1},
        {"[{\"bytes\":null,\"logprob\":-1},"
          "{\"bytes\":[65],\"logprob\":-2}]", 0, 0},
        {"[{\"bytes\":[239,191,189],\"logprob\":-1},"
          "{\"bytes\":[65],\"logprob\":-2}]", 0, 0},
        {"[{\"bytes\":[65,66],\"logprob\":-1},"
          "{\"bytes\":[65],\"logprob\":-2}]", 0, 0},
        {"[{\"bytes\":[65],\"logprob\":-1},"
          "{\"bytes\":[66],\"logprob\":-1},"
          "{\"bytes\":[65],\"logprob\":-1}]", 1, (1u << 0) | (1u << 1)},
        {"[{\"bytes\":null,\"logprob\":-1},"
          "{\"bytes\":[65],\"logprob\":-2},"
          "{\"bytes\":[66],\"logprob\":-1}]", 1, 1u << 1},
        {"[{\"bytes\":null,\"logprob\":-1},"
          "{\"bytes\":[239,191,189],\"logprob\":-1},"
          "{\"bytes\":[65],\"logprob\":-2}]", 0, 0},
        {"[{\"bytes\":[65],\"logprob\":-1.0000000000001},"
          "{\"bytes\":[66],\"logprob\":-1}]", 1, 1u << 1},
    };
    for (size_t f = 0; f < sizeof(fixtures) / sizeof(*fixtures); f++) {
        char json[1024];
        snprintf(json, sizeof(json), "{\"choices\":[{\"logprobs\":{\"content\":["
                 "{\"bytes\":[65],\"logprob\":-2,\"top_logprobs\":%s}]}}]}", fixtures[f].alts);
        api_ref ref;
        assert(api_ref_parse(json, &ref));
        assert(ref.n_pos == 1);
        api_pos *pos = &ref.pos[0];
        const int rotations = pos->n_alts ? pos->n_alts : 1;
        for (int rotation = 0; rotation < rotations; rotation++) {
            for (int greedy = 0; greedy < 3; greedy++) {
                api_metrics metrics = {.top1_count = 7, .top1_match = 3};
                const int token = api_top1_update(NULL, pos, greedy, &metrics);
                const int match = !!(fixtures[f].matches & (1u << greedy));
                assert(metrics.top1_count == 7 + fixtures[f].count);
                assert(metrics.top1_match == 3 + match);
                assert((token >= 0) == fixtures[f].count);
                assert((token == greedy) == match);
            }
            if (pos->n_alts) {
                const api_alt first = pos->alts[0];
                memmove(pos->alts, pos->alts + 1, (size_t)(pos->n_alts - 1) * sizeof(*pos->alts));
                pos->alts[pos->n_alts - 1] = first;
            }
        }
        api_ref_free(&ref);
    }
}

static void check_token_alignment(void) {
    api_ref ref;
    assert(api_ref_parse("{\"choices\":[{\"logprobs\":{\"content\":["
                         "{\"logprob\":-1,\"bytes\":[65]},"
                         "{\"logprob\":-2,\"bytes\":[66]}]}}]}", &ref));
    int ids[] = {0, 1};
    q36_tokens target = {.v = ids, .len = 2};
    assert(api_ref_matches_target(NULL, &ref, &target));
    ids[0] = 2;
    ids[1] = 3;
    assert(!api_ref_matches_target(NULL, &ref, &target));
    target.len = 1;
    assert(!api_ref_matches_target(NULL, &ref, &target));
    api_ref_free(&ref);
    const char *bytes[] = {"null", "[]", "[239,191,189]", "[240,159]"};
    ids[0] = 4;
    for (int i = 0; i < 4; i++) {
        char json[256];
        snprintf(json, sizeof(json), "{\"choices\":[{\"logprobs\":{\"content\":["
                 "{\"logprob\":-1,\"bytes\":%s}]}}]}", bytes[i]);
        assert(api_ref_parse(json, &ref));
        assert(api_ref_matches_target(NULL, &ref, &target) == (i == 3));
        api_ref_free(&ref);
    }
}

int main(void) {
    check_token_alignment();
    check_top1();
    const char *invalid[] = {
        "{}",
        "{\"choices\":[]}",
        ("{\"choices\":[{\"logprobs\":null,\"message\":{\"content\":\"[text]\","
         "\"reasoning_details\":[{\"text\":\"reasoning\"}]}}]}"),
        "{\"choices\":[{\"logprobs\":{\"content\":[]}}]}",
        "{\"choices\":[{\"logprobs\":{\"content\":[{}]}}]}",
        "{\"choices\":[{\"logprobs\":{\"content\":[{\"bytes\":[65],\"logprob\":null}]}}]}",
        "{\"choices\":[{\"message\":{\"logprobs\":{\"content\":[{\"logprob\":-1}]}}}]}",
        "{\"choices\":[{\"logprobs\":null},{\"logprobs\":{\"content\":[{\"logprob\":-1}]}}]}",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        api_ref ref;
        assert(!api_ref_parse(invalid[i], &ref));
        assert(ref.n_pos == 0);
        api_ref_free(&ref);
    }
    api_ref ref;
    assert(api_ref_parse(
        "{\"ignored\":{\"logprobs\":null},\"choices\":[{\"message\":{\"content\":\"[x]\"},"
        "\"logprobs\":{\"content\":[{\"logprob\":-0.5,\"top_logprobs\":["
        "{\"bytes\":[65],\"logprob\":-0.25}]},{\"logprob\":-1.5}]}}]}", &ref));
    assert(ref.n_pos == 2 && ref.pos[0].logprob == -0.5);
    assert(ref.pos[1].logprob == -1.5);
    assert(ref.pos[0].n_alts == 1);
    assert(ref.pos[0].alts[0].len == 1 && ref.pos[0].alts[0].bytes[0] == 'A');
    assert(ref.pos[0].alts[0].logprob == -0.25);
    api_ref_free(&ref);
    puts("quality API parser: PASS");
    return 0;
}
