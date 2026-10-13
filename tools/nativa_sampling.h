#ifndef NATIVA_SAMPLING_H
#define NATIVA_SAMPLING_H

#include <stdint.h>
#include <math.h>

typedef struct {
    int id;
    float logit;
} TokenScore;

static inline float sample_random_float(void) {
    static uint32_t rng_state = 2463534242u;
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & 0x00FFFFFF) / (float)0x01000000;
}

static inline int sample_argmax(const float* logits, int size) {
    if (size <= 0) return 0;
    int best = 0;
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best = i;
        }
    }
    return best;
}

static inline void topk_find_min(const TokenScore* best, int count, float* out_min_val, int* out_min_idx) {
    float min_val = best[0].logit;
    int min_idx = 0;
    for (int j = 1; j < count; j++) {
        if (best[j].logit < min_val) {
            min_val = best[j].logit;
            min_idx = j;
        }
    }
    *out_min_val = min_val;
    *out_min_idx = min_idx;
}

static inline int topk_sample_from_candidates(const TokenScore* best, int count, float temp) {
    if (count <= 0) return 0;
    float max_l = best[0].logit;
    for (int j = 1; j < count; j++) {
        if (best[j].logit > max_l) max_l = best[j].logit;
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < count; j++) {
        sum_exp += expf((best[j].logit - max_l) / temp);
    }
    float r = sample_random_float() * sum_exp;
    float acc = 0.0f;
    for (int j = 0; j < count; j++) {
        acc += expf((best[j].logit - max_l) / temp);
        if (acc >= r) return best[j].id;
    }
    return best[0].id;
}

static inline int sample_topk(const float* logits, int size, float temp, int top_k) {
    if (size <= 0) return 0;
    if (temp <= 0.05f || top_k <= 1) return sample_argmax(logits, size);
    if (top_k > 50) top_k = 50;

    TokenScore best[50] = {0};
    int count = 0;
    float min_best = 1e30f;
    int min_idx = 0;

    for (int i = 0; i < size; i++) {
        float l = logits[i];
        if (count < top_k) {
            best[count].id = i;
            best[count].logit = l;
            count++;
            if (count == top_k) {
                topk_find_min(best, count, &min_best, &min_idx);
            }
        } else if (l > min_best) {
            best[min_idx].id = i;
            best[min_idx].logit = l;
            topk_find_min(best, count, &min_best, &min_idx);
        }
    }

    return topk_sample_from_candidates(best, count, temp);
}

static inline int sample_temperature(const float* logits, int size, float temp, float top_p) {
    (void)top_p;
    if (size <= 0) return 0;
    if (temp <= 0.05f) return sample_argmax(logits, size);

    float max_logit = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float threshold = max_logit - 10.0f * temp;
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        if (logits[i] >= threshold) {
            sum_exp += expf((logits[i] - max_logit) / temp);
        }
    }

    float r = sample_random_float() * sum_exp;
    float acc = 0.0f;
    for (int i = 0; i < size; i++) {
        if (logits[i] >= threshold) {
            acc += expf((logits[i] - max_logit) / temp);
            if (acc >= r) return i;
        }
    }
    return sample_argmax(logits, size);
}

#endif /* NATIVA_SAMPLING_H */
