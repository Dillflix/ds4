#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void ds4_gpu_test_reset_attention_indexed_streaming_exact_launches(void);
extern void ds4_gpu_test_set_attention_indexed_streaming_exact_heads(uint32_t);
extern void ds4_gpu_test_set_attention_indexed_streaming_exact_audit_mode(uint32_t);
extern uint64_t ds4_gpu_test_get_attention_indexed_streaming_exact_launches(uint32_t);
extern int ds4_gpu_test_get_attention_indexed_streaming_exact_resources(
    uint32_t, uint32_t *, uint32_t *, uint32_t *, uint32_t *, uint32_t *);

enum {
    HEAD_DIM = 512u, MAX_HEADS = 64u, SHARD_HEADS = 32u,
    RAW_CAP = 2304u, N_COMP = 8192u, TOP_K = 512u, RATIO = 4u,
    FIRST_DECODE_POS = 32768u, LIVE_RAW = 128u, LIVE_WINDOW = 128u,
    FIRST_DECODE_RAW_START = 385u, STRESS_WINDOW = 2048u,
    GUARD_FLOATS = 64u,
};
static const float SENTINEL = -12345.5f;

typedef enum {
    TOPK_PRODUCTION, TOPK_SELECTED_TIE, TOPK_INVALID_DUPLICATE,
    TOPK_VISIBLE_CUTOFF, TOPK_ALL_INVALID,
} topk_pattern;

typedef struct {
    const char *name, *fixture_class;
    uint32_t n_head, pos, n_raw, raw_start, window;
    topk_pattern pattern;
} exact_case;

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u;
    x *= 0x846ca68bu; x ^= x >> 16u; return x;
}

static float fixture_value(uint64_t i, uint32_t salt) {
    uint32_t bits = mix32((uint32_t)i ^
        ((uint32_t)(i >> 32u) * 0x9e3779b9u) ^ salt);
    return (float)((int32_t)(bits % 2001u) - 1000) / 2048.0f;
}

static void fill_topk(int32_t *topk, topk_pattern pattern, uint32_t visible) {
    for (uint32_t k = 0; k < TOP_K; k++) {
        int32_t v;
        switch (pattern) {
        case TOPK_PRODUCTION:
            v = (int32_t)((k * 1543u + 37u) % N_COMP); break;
        case TOPK_SELECTED_TIE:
            v = k == 0u ? 23 : k == 1u ? 24 :
                (int32_t)((k * 1543u + 37u) % N_COMP); break;
        case TOPK_INVALID_DUPLICATE:
            if (k % 17u == 0u) v = -1;
            else if (k % 19u == 0u) v = (int32_t)(N_COMP + 11u);
            else if (k % 7u == 0u) v = 23;
            else v = (int32_t)((k * 1103u + 71u) % N_COMP);
            break;
        case TOPK_VISIBLE_CUTOFF:
            if (k % 9u == 0u) v = -7;
            else if (k & 1u) v = (int32_t)(visible + 3u + k);
            else v = (int32_t)((k * 73u + 5u) % visible);
            break;
        default:
            v = (k & 1u) ? -1 : (int32_t)(N_COMP + k); break;
        }
        topk[k] = v;
    }
}

static uint32_t effective_raw(const exact_case *t) {
    uint32_t first = t->pos + 1u - t->n_raw;
    uint32_t last = first + t->n_raw - 1u, count = 0u;
    if (t->pos >= first) {
        uint32_t lo = first;
        if (t->window && t->pos + 1u > t->window) {
            uint32_t wlo = t->pos + 1u - t->window;
            if (wlo > lo) lo = wlo;
        }
        uint32_t hi = t->pos < last ? t->pos : last;
        if (hi >= lo) { count = hi - lo + 1u; if (count > 256u) count = 256u; }
    }
    return count;
}

static int compare_exact(const exact_case *t, uint32_t group,
                         const float *ref, const float *got, uint64_t n) {
    if (!memcmp(ref, got, (size_t)n * sizeof(float))) return 1;
    uint64_t i = 0; while (i < n && !memcmp(ref + i, got + i, sizeof(float))) i++;
    uint32_t rb = 0, gb = 0; memcpy(&rb, ref + i, 4); memcpy(&gb, got + i, 4);
    fprintf(stderr, "mismatch case=%s group=%u index=%llu ref=%a/0x%08x got=%a/0x%08x\n",
            t->name, group, (unsigned long long)i,
            (double)ref[i], rb, (double)got[i], gb);
    return 0;
}

static int compare_stage(const exact_case *t, uint32_t group, uint32_t mode,
                         const float *ref, const float *got, uint64_t n) {
    static const char *names[] = {
        "output", "max-score", "denominator", "numerator"
    };
    if (!memcmp(ref, got, (size_t)n * sizeof(float))) {
        printf("stage-exact,case=%s,group=%u,stage=%s\n",
               t->name, group, names[mode]);
        return 1;
    }
    uint64_t i = 0u;
    while (i < n && !memcmp(ref + i, got + i, sizeof(float))) i++;
    uint32_t rb = 0u, gb = 0u;
    memcpy(&rb, ref + i, sizeof(rb));
    memcpy(&gb, got + i, sizeof(gb));
    fprintf(stderr,
            "stage-mismatch case=%s group=%u stage=%s index=%llu "
            "ref=%a/0x%08x got=%a/0x%08x\n",
            t->name, group, names[mode], (unsigned long long)i,
            (double)ref[i], rb, (double)got[i], gb);
    return 0;
}

static int check_guards(const exact_case *t, uint32_t group,
                        const float *got, const float *initial) {
    const uint64_t payload = (uint64_t)MAX_HEADS * HEAD_DIM;
    const uint64_t active = (uint64_t)t->n_head * HEAD_DIM;
    if (memcmp(got, initial, GUARD_FLOATS * sizeof(float))) {
        fprintf(stderr, "prefix guard changed case=%s group=%u\n", t->name, group);
        return 0;
    }
    for (uint64_t i = active; i < payload; i++) {
        if (memcmp(got + GUARD_FLOATS + i,
                   initial + GUARD_FLOATS + i, sizeof(float))) {
            fprintf(stderr, "inactive tail changed case=%s group=%u index=%llu\n",
                    t->name, group, (unsigned long long)i); return 0;
        }
    }
    if (memcmp(got + GUARD_FLOATS + payload,
               initial + GUARD_FLOATS + payload,
               GUARD_FLOATS * sizeof(float))) {
        fprintf(stderr, "suffix guard changed case=%s group=%u\n", t->name, group);
        return 0;
    }
    return 1;
}

static int launch(ds4_gpu_tensor *heads, const float *sinks,
                  ds4_gpu_tensor *q, ds4_gpu_tensor *raw,
                  ds4_gpu_tensor *comp, ds4_gpu_tensor *topk,
                  const exact_case *t) {
    return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        heads, sinks, (uint64_t)MAX_HEADS * sizeof(float), 0u,
        q, raw, comp, 0u, topk, 1u, t->pos, t->n_raw, RAW_CAP,
        t->raw_start, N_COMP, TOP_K, t->window, RATIO, t->n_head, HEAD_DIM);
}

static int run_exact(ds4_gpu_tensor *base, ds4_gpu_tensor *heads,
                     const float *initial, float *observed,
                     const float *sinks, ds4_gpu_tensor *q,
                     ds4_gpu_tensor *raw, ds4_gpu_tensor *comp,
                     ds4_gpu_tensor *topk, const exact_case *t,
                     uint32_t group, float *out) {
    const uint64_t payload = (uint64_t)MAX_HEADS * HEAD_DIM;
    const uint64_t storage = payload + 2u * GUARD_FLOATS;
    ds4_gpu_test_set_attention_indexed_streaming_exact_heads(group);
    ds4_gpu_test_reset_attention_indexed_streaming_exact_launches();
    if (!ds4_gpu_tensor_write(base, 0u, initial, storage * sizeof(float)) ||
        !launch(heads, sinks, q, raw, comp, topk, t) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(base, 0u, observed, storage * sizeof(float))) {
        fprintf(stderr, "launch failed case=%s group=%u\n", t->name, group);
        return 0;
    }
    uint64_t calls = ds4_gpu_test_get_attention_indexed_streaming_exact_launches(group);
    if ((group == 0u && calls) || (group && calls != 1u)) {
        fprintf(stderr, "selector failed case=%s group=%u calls=%llu\n",
                t->name, group, (unsigned long long)calls); return 0;
    }
    if (!check_guards(t, group, observed, initial)) return 0;
    memcpy(out, observed + GUARD_FLOATS,
           (size_t)t->n_head * HEAD_DIM * sizeof(float));
    return 1;
}

static uint32_t env_u32(const char *name, uint32_t max) {
    const char *s = getenv(name); if (!s || !*s) return 0u;
    char *end = NULL; unsigned long v = strtoul(s, &end, 10);
    return end == s || *end || v > max ? 0u : (uint32_t)v;
}

static int time_one(ds4_gpu_tensor *heads, const float *sinks,
                    ds4_gpu_tensor *q, ds4_gpu_tensor *raw,
                    ds4_gpu_tensor *comp, ds4_gpu_tensor *topk,
                    const exact_case *t, uint32_t group, uint32_t repeats,
                    ds4_gpu_timer *timer, float *per_call_ms) {
    ds4_gpu_test_set_attention_indexed_streaming_exact_heads(group);
    if (!ds4_gpu_timer_record_start(timer)) return 0;
    for (uint32_t i = 0; i < repeats; i++)
        if (!launch(heads, sinks, q, raw, comp, topk, t)) return 0;
    float elapsed = 0.0f;
    if (!ds4_gpu_timer_record_end(timer) ||
        !ds4_gpu_timer_elapsed_ms(timer, &elapsed)) return 0;
    *per_call_ms = elapsed / (float)repeats; return 1;
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y);
}
static float median(const float *x, uint32_t n) {
    return n & 1u ? x[n / 2u] : 0.5f * (x[n / 2u - 1u] + x[n / 2u]);
}

static int paired_timing(ds4_gpu_tensor *heads, const float *sinks,
                         ds4_gpu_tensor *q, ds4_gpu_tensor *raw,
                         ds4_gpu_tensor *comp, ds4_gpu_tensor *topk,
                         const exact_case *t, uint32_t group,
                         uint32_t rounds, uint32_t repeats) {
    float *ctl = malloc(rounds * sizeof(float));
    float *can = malloc(rounds * sizeof(float));
    float *ratio = malloc(rounds * sizeof(float));
    ds4_gpu_timer *timer = ds4_gpu_timer_create(); int ok = !!(ctl && can && ratio && timer);
    if (!ok) goto done;
    for (uint32_t g = 0; g <= group; g += group) {
        ds4_gpu_test_set_attention_indexed_streaming_exact_heads(g);
        if (!launch(heads, sinks, q, raw, comp, topk, t) || !ds4_gpu_synchronize()) {
            ok = 0; goto done;
        }
        if (!group) break;
    }
    for (uint32_t r = 0; r < rounds; r++) {
        uint32_t first = (r & 1u) ? group : 0u, second = first ? 0u : group;
        float a, b;
        if (!time_one(heads, sinks, q, raw, comp, topk, t, first, repeats, timer, &a) ||
            !time_one(heads, sinks, q, raw, comp, topk, t, second, repeats, timer, &b)) {
            ok = 0; goto done;
        }
        ctl[r] = first ? b : a; can[r] = first ? a : b; ratio[r] = ctl[r] / can[r];
    }
    double mean = 0.0, sumsq = 0.0; float min = ratio[0], max = ratio[0];
    for (uint32_t r = 0; r < rounds; r++) {
        mean += ratio[r]; if (ratio[r] < min) min = ratio[r]; if (ratio[r] > max) max = ratio[r];
    }
    mean /= rounds;
    for (uint32_t r = 0; r < rounds; r++) { double d = ratio[r] - mean; sumsq += d * d; }
    qsort(ctl, rounds, sizeof(float), cmp_float);
    qsort(can, rounds, sizeof(float), cmp_float);
    qsort(ratio, rounds, sizeof(float), cmp_float);
    printf("timing,case=%s,control=shipping,candidate=H%u,rounds=%u,repeats=%u,"
           "control_median_ms=%.9f,candidate_median_ms=%.9f,"
           "paired_speedup_median=%.9f,paired_speedup_sd=%.9f,"
           "paired_speedup_min=%.9f,paired_speedup_max=%.9f\n",
           t->name, group, rounds, repeats, median(ctl, rounds), median(can, rounds),
           median(ratio, rounds), rounds > 1u ? sqrt(sumsq / (rounds - 1u)) : 0.0,
           min, max);
done:
    ds4_gpu_timer_free(timer); free(ratio); free(can); free(ctl); return ok;
}

static int tensor_matches(const char *label, const ds4_gpu_tensor *t,
                          const void *expected, uint64_t bytes) {
    void *got = malloc((size_t)bytes); int ok = got && ds4_gpu_tensor_read(t, 0u, got, bytes);
    if (ok) ok = !memcmp(got, expected, (size_t)bytes);
    if (!ok) fprintf(stderr, "input mutation/readback failure label=%s\n", label);
    free(got); return ok;
}

int main(void) {
    static const exact_case cases[] = {
        {"production-four-gpu-shard-after-32k", "production", SHARD_HEADS,
         FIRST_DECODE_POS, LIVE_RAW, FIRST_DECODE_RAW_START, LIVE_WINDOW, TOPK_PRODUCTION},
        {"selected-score-tie-after-32k-h32", "adversarial", SHARD_HEADS,
         FIRST_DECODE_POS, LIVE_RAW, FIRST_DECODE_RAW_START, LIVE_WINDOW, TOPK_SELECTED_TIE},
        {"stress-max-768-rows", "stress-only", 64u, 32767u, RAW_CAP, 512u,
         STRESS_WINDOW, TOPK_PRODUCTION},
        {"wrapped-invalid-duplicate-32k-h32", "adversarial", 32u, 32767u,
         RAW_CAP, RAW_CAP - 300u, STRESS_WINDOW, TOPK_INVALID_DUPLICATE},
        {"visible-cutoff-4k-h17", "adversarial", 17u, 4095u, RAW_CAP, 911u,
         STRESS_WINDOW, TOPK_VISIBLE_CUTOFF},
        {"raw-only-partial-group-32k-h13", "adversarial", 13u, 32767u,
         RAW_CAP, 1733u, STRESS_WINDOW, TOPK_ALL_INVALID},
    };
    const uint64_t qn = (uint64_t)MAX_HEADS * HEAD_DIM;
    const uint64_t rn = (uint64_t)RAW_CAP * HEAD_DIM;
    const uint64_t cn = (uint64_t)N_COMP * HEAD_DIM;
    const uint64_t hn = qn + 2u * GUARD_FLOATS;
    float *sinks = malloc(MAX_HEADS * sizeof(float)), *sinks_copy = malloc(MAX_HEADS * sizeof(float));
    float *qh = malloc((size_t)qn * sizeof(float)), *rh = malloc((size_t)rn * sizeof(float));
    float *ch = malloc((size_t)cn * sizeof(float)); int32_t *kh = malloc(TOP_K * sizeof(int32_t));
    float *ref = malloc((size_t)qn * sizeof(float)), *got = malloc((size_t)qn * sizeof(float));
    float *hi = malloc((size_t)hn * sizeof(float)), *ho = malloc((size_t)hn * sizeof(float));
    ds4_gpu_tensor *q = NULL, *raw = NULL, *comp = NULL, *topk = NULL, *base = NULL, *heads = NULL;
    int rc = 1, initialized = 0;
    if (!sinks || !sinks_copy || !qh || !rh || !ch || !kh || !ref || !got || !hi || !ho) goto cleanup;
    if (FIRST_DECODE_RAW_START != (FIRST_DECODE_POS + 1u - LIVE_RAW) % RAW_CAP) {
        fprintf(stderr, "first-decode ring identity failed\n"); goto cleanup;
    }
    static const float sink_pattern[8] = {0, 6, -6, .25f, -.25f, 1.5f, -2, 0};
    for (uint32_t h = 0; h < MAX_HEADS; h++) sinks[h] = sink_pattern[h & 7u];
    memcpy(sinks_copy, sinks, MAX_HEADS * sizeof(float));
    for (uint64_t i = 0; i < qn; i++) qh[i] = fixture_value(i, 0x13579bdfu);
    for (uint64_t i = 0; i < rn; i++) rh[i] = fixture_value(i, 0x2468ace0u);
    for (uint64_t i = 0; i < cn; i++) ch[i] = fixture_value(i, 0xdeadbeefu);
    memcpy(ch + 24u * HEAD_DIM, ch + 23u * HEAD_DIM, HEAD_DIM * sizeof(float));
    for (uint32_t d = 0; d < HEAD_DIM; d += 37u) rh[d] = (d & 1u) ? -0.0f : 0.0f;
    for (uint64_t i = 0; i < hn; i++) hi[i] =
        i < GUARD_FLOATS || i >= GUARD_FLOATS + qn ? fixture_value(i, 0xa55aa55au) : SENTINEL;
    if (!ds4_gpu_init()) goto cleanup; initialized = 1;
    for (uint32_t group = 4u; group <= 8u; group += 4u) {
        uint32_t regs=0, shared=0, local=0, threads=0, blocks=0;
        if (!ds4_gpu_test_get_attention_indexed_streaming_exact_resources(
                group, &regs, &shared, &local, &threads, &blocks)) goto cleanup;
        printf("resources,group=%u,registers=%u,static_shared_bytes=%u,local_bytes=%u,"
               "max_threads_per_block=%u,active_blocks_per_sm=%u\n",
               group, regs, shared, local, threads, blocks);
        if (regs > 128u || local || threads < 256u || blocks < 2u) goto cleanup;
    }
    q=ds4_gpu_tensor_alloc(qn*4u); raw=ds4_gpu_tensor_alloc(rn*4u);
    comp=ds4_gpu_tensor_alloc(cn*4u); topk=ds4_gpu_tensor_alloc(TOP_K*4u);
    base=ds4_gpu_tensor_alloc(hn*4u); heads=base ? ds4_gpu_tensor_view(base,GUARD_FLOATS*4u,qn*4u):NULL;
    if (!q||!raw||!comp||!topk||!base||!heads ||
        !ds4_gpu_tensor_write(q,0,qh,qn*4u) || !ds4_gpu_tensor_write(raw,0,rh,rn*4u) ||
        !ds4_gpu_tensor_write(comp,0,ch,cn*4u) || !ds4_gpu_set_model_map(sinks,MAX_HEADS*4u)) goto cleanup;
    for (uint32_t ci=0; ci<sizeof(cases)/sizeof(cases[0]); ci++) {
        const exact_case *t=&cases[ci]; uint32_t visible=(t->pos+1u)/RATIO;
        if (visible>N_COMP) visible=N_COMP; fill_topk(kh,t->pattern,visible);
        printf("fixture,case=%s,class=%s,pos=%u,allocated_n_raw=%u,raw_cap=%u,raw_start=%u,"
               "effective_raw_count=%u,n_comp=%u,top_k=%u,window=%u,ratio=%u,n_head=%u,head_dim=%u\n",
               t->name,t->fixture_class,t->pos,t->n_raw,RAW_CAP,t->raw_start,effective_raw(t),
               N_COMP,TOP_K,t->window,RATIO,t->n_head,HEAD_DIM);
        if ((t->pattern==TOPK_SELECTED_TIE && (kh[0]!=23||kh[1]!=24)) ||
            !ds4_gpu_tensor_write(topk,0,kh,TOP_K*4u) ||
            !run_exact(base,heads,hi,ho,sinks,q,raw,comp,topk,t,0,ref)) goto cleanup;
        uint64_t nonzero=0; for(uint64_t i=0;i<(uint64_t)t->n_head*HEAD_DIM;i++) {
            if(!isfinite(ref[i])) goto cleanup;
            nonzero += ref[i]!=0.0f;
        }
        if(!nonzero) goto cleanup;
        for(uint32_t group=4;group<=8;group+=4) {
            if (ci == 0u) {
                for (uint32_t mode = 1u; mode <= 3u; mode++) {
                    ds4_gpu_test_set_attention_indexed_streaming_exact_audit_mode(mode);
                    if (!run_exact(base,heads,hi,ho,sinks,q,raw,comp,topk,t,0,ref) ||
                        !run_exact(base,heads,hi,ho,sinks,q,raw,comp,topk,t,group,got))
                        goto cleanup;
                    if (!compare_stage(t,group,mode,ref,got,
                                       (uint64_t)t->n_head*HEAD_DIM)) {
                        ds4_gpu_test_set_attention_indexed_streaming_exact_audit_mode(0u);
                        goto cleanup;
                    }
                }
                ds4_gpu_test_set_attention_indexed_streaming_exact_audit_mode(0u);
                if (!run_exact(base,heads,hi,ho,sinks,q,raw,comp,topk,t,0,ref))
                    goto cleanup;
            }
            if(!run_exact(base,heads,hi,ho,sinks,q,raw,comp,topk,t,group,got) ||
               !compare_exact(t,group,ref,got,(uint64_t)t->n_head*HEAD_DIM)) goto cleanup;
            printf("exact,case=%s,group=%u,values=%llu\n",t->name,group,
                   (unsigned long long)t->n_head*HEAD_DIM);
        }
        if(!tensor_matches("topk-per-case",topk,kh,TOP_K*4u)) goto cleanup;
    }
    {
        uint32_t rounds=env_u32("DS4_STREAMING_EXACT_TIMING_ROUNDS",101u);
        uint32_t repeats=env_u32("DS4_STREAMING_EXACT_TIMING_REPEATS",10000u);
        if ((rounds==0)!=(repeats==0)) goto cleanup;
        if(rounds) {
            const exact_case *t=&cases[0]; fill_topk(kh,t->pattern,N_COMP);
            if(!ds4_gpu_tensor_write(topk,0,kh,TOP_K*4u)) goto cleanup;
            for(uint32_t group=4;group<=8;group+=4)
                if(!paired_timing(heads,sinks,q,raw,comp,topk,t,group,rounds,repeats)) goto cleanup;
        }
    }
    if(!tensor_matches("q",q,qh,qn*4u)||!tensor_matches("raw",raw,rh,rn*4u)||
       !tensor_matches("comp",comp,ch,cn*4u)||!tensor_matches("topk",topk,kh,TOP_K*4u)||
       memcmp(sinks,sinks_copy,MAX_HEADS*4u)) goto cleanup;
    puts("immutability,q=pass,raw=pass,comp=pass,topk=pass,sinks=pass"); rc=0;
cleanup:
    ds4_gpu_test_set_attention_indexed_streaming_exact_audit_mode(0u);
    ds4_gpu_test_set_attention_indexed_streaming_exact_heads(0u);
    ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(base); ds4_gpu_tensor_free(topk);
    ds4_gpu_tensor_free(comp); ds4_gpu_tensor_free(raw); ds4_gpu_tensor_free(q);
    if(initialized) ds4_gpu_cleanup(); free(ho);free(hi);free(got);free(ref);free(kh);
    free(ch);free(rh);free(qh);free(sinks_copy);free(sinks);
    if(!rc) puts("exact streaming indexed-decode experiment: OK"); return rc;
}
