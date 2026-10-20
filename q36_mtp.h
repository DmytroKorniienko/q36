#ifndef Q36_MTP_H
#define Q36_MTP_H

#include <string.h>

typedef struct {
    double seconds[3];
    unsigned samples[3];
    double accepted[3];
    unsigned attempts[3];
    unsigned calls;
} q36_mtp_policy;

/* Compare time per committed token, with an occasional probe so a change
 * in the continuation can make a different depth profitable. */
static inline int q36_mtp_policy_depth(q36_mtp_policy *p, int cap) {
    if (cap < 2) return cap;
    if (cap > 4) cap = 4;
    unsigned call = p->calls++;
    int arms = cap - 1;
    for (int i = 0; i < arms; i++)
        if (!p->samples[i]) return i + 2;
    if ((call & 15u) == 15u) return 2 + (int)((call / 16u) % (unsigned)arms);
    double expected[3], product = 1.0, total = 1.0;
    for (int i = 0; i < arms; i++) {
        product *= p->attempts[i] ? p->accepted[i] : 1.0;
        expected[i] = total += product;
    }
    int best = 0;
    for (int i = 1; i < arms; i++)
        if (p->seconds[i] * expected[best] < p->seconds[best] * expected[i]) best = i;
    return best + 2;
}

static inline void q36_mtp_policy_observe(q36_mtp_policy *p, int depth,
                                           int tokens, double seconds) {
    if (depth < 2 || depth > 4 || tokens < 1 || tokens > depth || !(seconds > 0)) return;
    int i = depth - 2;
    if (p->samples[i] < 8u) p->samples[i]++;
    double rate = 1.0 / p->samples[i];
    p->seconds[i] += rate * (seconds - p->seconds[i]);
    /* A longer verification also observes every accepted shorter prefix.
     * Condition each probability on its predecessor having matched. */
    for (int j = 0; j < depth - 1 && j < tokens; j++) {
        if (p->attempts[j] < 8u) p->attempts[j]++;
        p->accepted[j] += ((tokens > j + 1 ? 1.0 : 0.0) - p->accepted[j]) / p->attempts[j];
    }
}

/* Reuse a continuation only when the longest matching suffix is unambiguous.
 * The known first token participates in the match but is not in history yet.
 * Bound the scan so CPU work stays small as the context grows. */
static inline int q36_mtp_lookup(const int *history, int len, int first,
                                  int *draft, int cap) {
    if (!history || !draft || cap < 2 || len < 13 || cap > len - 11) return 0;
    int best = -1, longest = 11, ambiguous = 0;
    int start = len > 65536 ? len - 65536 : 11;
    for (int pos = start; pos <= len - cap; pos++) {
        if (history[pos] != first) continue;
        int n = 1;
        while (n < 16 && n <= pos && n <= len &&
               history[pos - n] == history[len - n]) n++;
        if (n < 12 || n < longest) continue;
        if (n > longest) {
            best = pos;
            longest = n;
            ambiguous = 0;
        } else if (best >= 0 && memcmp(history + best + 1, history + pos + 1,
                                       (size_t)(cap - 1) * sizeof(int))) {
            ambiguous = 1;
        }
    }
    if (best < 0 || ambiguous) return 0;
    draft[0] = first;
    memcpy(draft + 1, history + best + 1, (size_t)(cap - 1) * sizeof(int));
    return cap;
}

#endif
