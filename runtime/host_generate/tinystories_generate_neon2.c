#define _POSIX_C_SOURCE 200809L
#define LLM_INT8_ACT 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>

#include "../llm.h"
#include "../../firmware/esp32_tinystories/generated/vocab.h"

static const int PROMPT_IDS[] = {433, 447, 259, 405};
static const int N_GENERATE = 200;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc failed: %zu bytes\n", n);
        exit(1);
    }
    return p;
}

static void *stage_alloc(size_t n) {
    return xmalloc(n);
}

static uint8_t *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    if (end < 0) {
        fprintf(stderr, "ftell failed\n");
        exit(1);
    }
    *size = (size_t)end;
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = xmalloc(*size);

    if (fread(buf, 1, *size, f) != *size) {
        fprintf(stderr, "short read\n");
        exit(1);
    }

    fclose(f);
    return buf;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void emit_token(int tok) {
    if (tok < 0 || tok >= VOCAB_N)
        return;

    int begin = VOCAB_OFF[tok];
    int end = VOCAB_OFF[tok + 1];

    fwrite(VOCAB_BLOB + begin, 1, (size_t)(end - begin), stdout);
}


static inline int32_t dot_i8_neon(const int8_t *a, const int8_t *b, int n) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int i = 0;

    for (; i + 32 <= n; i += 32) {
        int8x16_t a0 = vld1q_s8(a + i);
        int8x16_t b0 = vld1q_s8(b + i);
        int8x16_t a1 = vld1q_s8(a + i + 16);
        int8x16_t b1 = vld1q_s8(b + i + 16);

        int16x8_t p00 = vmull_s8(vget_low_s8(a0),  vget_low_s8(b0));
        int16x8_t p01 = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));
        int16x8_t p10 = vmull_s8(vget_low_s8(a1),  vget_low_s8(b1));
        int16x8_t p11 = vmull_s8(vget_high_s8(a1), vget_high_s8(b1));

        acc0 = vaddq_s32(acc0, vpaddlq_s16(p00));
        acc0 = vaddq_s32(acc0, vpaddlq_s16(p01));
        acc1 = vaddq_s32(acc1, vpaddlq_s16(p10));
        acc1 = vaddq_s32(acc1, vpaddlq_s16(p11));
    }

    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        acc0 = vaddq_s32(acc0, vpaddlq_s16(lo));
        acc0 = vaddq_s32(acc0, vpaddlq_s16(hi));
    }

    int32x4_t acc = vaddq_s32(acc0, acc1);

    int32_t sum =
        vgetq_lane_s32(acc, 0) +
        vgetq_lane_s32(acc, 1) +
        vgetq_lane_s32(acc, 2) +
        vgetq_lane_s32(acc, 3);

    for (; i < n; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];

    return sum;
}

static void matvec_i8_range_neon(
    const QT *t,
    const int8_t *xq,
    float x_scale,
    float *y,
    int row_begin,
    int row_end
) {
    int g = t->group;
    int ng = t->n_groups;
    int cols = t->cols;

    for (int r = row_begin; r < row_end; r++) {
        const int8_t *w = t->w8 + (size_t)r * cols;
        const float *sc = t->scale8 + (size_t)r * ng;

        float acc = 0.0f;

        for (int gi = 0; gi < ng; gi++) {
            int begin = gi * g;
            int end = begin + g;
            if (end > cols)
                end = cols;

            int32_t dot = dot_i8_neon(
                w + begin,
                xq + begin,
                end - begin
            );

            acc += (float)dot * sc[gi];
        }

        y[r] = acc * x_scale;
    }
}

static void matvec_neon(const QT *t, const float *x, float *y) {
    static int8_t xq[LLM_Q8_MAX_INPUT];
    float xs;

    if (t->w8 == NULL) {
        MATVEC(t, x, y);
        return;
    }

    quantize_act(x, t->cols, xq, &xs);
    matvec_i8_range_neon(t, xq, xs, y, 0, t->rows);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 2;
    }

    size_t image_size;
    uint8_t *image = read_file(argv[1], &image_size);

    Model model;
    if (llm_load(image, &model)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    if (model.image_bytes != image_size) {
        fprintf(stderr, "model image size mismatch: %zu vs %zu\n",
                model.image_bytes, image_size);
        return 1;
    }

    if (model.out_vocab != VOCAB_N) {
        fprintf(stderr, "vocab mismatch: model=%d vocab=%d\n",
                model.out_vocab, VOCAB_N);
        return 1;
    }

    int want = llm_core_stage_count(&model);
    int staged = llm_stage_core_int8_alloc(&model, stage_alloc);

    if (staged != want) {
        fprintf(stderr, "core staging failed: %d/%d\n", staged, want);
        return 1;
    }

    void *head_buf = stage_alloc(llm_stage_int8_bytes(&model.out_head));
    llm_stage_int8(&model.out_head, head_buf);
    staged++;

    model.layer_matvec = matvec_neon;
    model.head_matvec = matvec_neon;

    int D = model.c.dim;
    int L = model.c.n_layers;
    int P = model.c.ple_dim;
    int F = model.c.ffn;
    int S = model.c.seq_len;
    int V = model.out_vocab;

    Scratch s = {0};

    s.x      = xmalloc((size_t)D * sizeof(float));
    s.h      = xmalloc((size_t)(F > D ? F : D) * sizeof(float));
    s.qkv    = xmalloc((size_t)3 * D * sizeof(float));
    s.att    = xmalloc((size_t)D * sizeof(float));
    s.g1     = xmalloc((size_t)F * sizeof(float));
    s.g2     = xmalloc((size_t)(P > F ? P : F) * sizeof(float));
    s.ple    = xmalloc((size_t)L * P * sizeof(float));
    s.tmpP   = xmalloc((size_t)L * P * sizeof(float));
    s.trow   = xmalloc((size_t)L * P * sizeof(float));
    s.logits = xmalloc((size_t)V * sizeof(float));
    s.scores = xmalloc((size_t)S * sizeof(float));
    s.kcache = xmalloc((size_t)L * S * D * sizeof(float));
    s.vcache = xmalloc((size_t)L * S * D * sizeof(float));

    printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d\n",
           model.c.vocab, model.out_vocab,
           D, L, model.c.n_heads, F, P);
    printf("int8 staged tensors: %d\n", staged);

    printf("prompt: ");

    int pos = 0;

    for (size_t i = 0;
         i < sizeof(PROMPT_IDS) / sizeof(PROMPT_IDS[0]);
         i++) {
        int tok = PROMPT_IDS[i];
        emit_token(tok);
        llm_forward(&model, tok, pos++, &s);
    }

    printf("\n\ngenerated:\n");

    double start = now_sec();
    double forward_time = 0.0;
    int generated = 0;

    for (int step = 0;
         step < N_GENERATE && pos < model.c.seq_len;
         step++) {

        int best = 0;
        float best_value = s.logits[0];

        for (int v = 1; v < V; v++) {
            if (s.logits[v] > best_value) {
                best_value = s.logits[v];
                best = v;
            }
        }

        emit_token(best);

        double t0 = now_sec();
        llm_forward(&model, best, pos++, &s);
        forward_time += now_sec() - t0;

        generated++;
    }

    double elapsed = now_sec() - start;

    printf("\n\n--- benchmark ---\n");
    printf("tokens: %d\n", generated);
    printf("total: %.3f s\n", elapsed);
    printf("throughput: %.2f tok/s\n", generated / elapsed);
    printf("forward: %.2f ms/token\n",
           forward_time * 1000.0 / generated);

    free(image);
    return 0;
}
