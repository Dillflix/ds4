#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
#include "ds4_gpu.h"
#endif

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory when the payload is small enough,
 * performs a fixed greedy decode run without allowing EOS, restores the
 * snapshot or replays the prefix, and continues to the next frontier.  Snapshot
 * save/restore time is intentionally outside both timing windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#if !defined(_WIN32)
#include <execinfo.h>
#include <signal.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#define DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES (UINT64_C(1) << 30)

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    const char *dump_frontier_logits_dir;
    const char *dump_decode_logits_dir;
    ds4_dist_options dist;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool cuda_tensor_parallel;
    bool show_output;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

#if !defined(_WIN32)
/* Opt-in postmortem aid for production-shaped diagnostics.  The handler is
 * deliberately absent from ordinary runs so it cannot affect signal handling
 * or the benchmark's hot path.  backtrace_symbols_fd() is the GNU/Linux
 * best-effort primitive available before a debugger is attached; the signal
 * is then re-raised with its default disposition so the usual core dump is
 * still produced when the host is configured for one. */
static void bench_crash_trace_handler(int signo) {
    static const char message[] =
        "ds4-bench: fatal host signal; native backtrace follows\n";
    void *frames[64];
    (void)write(STDERR_FILENO, message, sizeof(message) - 1u);
    const int count = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
    if (count > 0) backtrace_symbols_fd(frames, count, STDERR_FILENO);
    (void)signal(signo, SIG_DFL);
    (void)raise(signo);
}

static void bench_install_crash_trace(void) {
    if (getenv("DS4_BENCH_CRASH_TRACE") == NULL) return;
    (void)signal(SIGSEGV, bench_crash_trace_handler);
    (void)signal(SIGBUS, bench_crash_trace_handler);
    (void)signal(SIGABRT, bench_crash_trace_handler);
}
#else
static void bench_install_crash_trace(void) {}
#endif

typedef struct {
    const char *path;
    const char *phase;
    bool warned;
} bench_progress_journal;

static bool bench_progress_journal_flush(FILE *fp) {
    if (fflush(fp) != 0) return false;
#if !defined(_WIN32)
    if (fsync(fileno(fp)) != 0) return false;
#endif
    return true;
}

static bool bench_progress_journal_init(bench_progress_journal *journal) {
    if (!journal || !journal->path || journal->path[0] == '\0') return true;
    FILE *fp = fopen(journal->path, "wb");
    if (!fp) return false;
    const bool ok =
        fprintf(fp, "realtime_sec,realtime_nsec,phase,event,current,total\n") > 0 &&
        bench_progress_journal_flush(fp);
    const int close_rc = fclose(fp);
    return ok && close_rc == 0;
}

static void bench_progress_journal_note(void *ud,
                                        const char *event,
                                        int current,
                                        int total) {
    bench_progress_journal *journal = ud;
    if (!journal || !journal->path || journal->path[0] == '\0') return;
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        ts.tv_sec = 0;
        ts.tv_nsec = 0;
    }
    FILE *fp = fopen(journal->path, "ab");
    bool ok = fp != NULL;
    if (ok) {
        ok = fprintf(fp, "%lld,%ld,%s,%s,%d,%d\n",
                     (long long)ts.tv_sec,
                     ts.tv_nsec,
                     journal->phase ? journal->phase : "unknown",
                     event ? event : "unknown",
                     current,
                     total) > 0 &&
             bench_progress_journal_flush(fp);
        if (fclose(fp) != 0) ok = false;
    }
    if (!ok && !journal->warned) {
        journal->warned = true;
        fprintf(stderr,
                "ds4-bench: failed to update progress journal %s: %s\n",
                journal->path,
                strerror(errno));
    }
}

static void bench_progress_journal_mark(
        bench_progress_journal *journal,
        const char             *phase,
        const char             *event,
        int                     current,
        int                     total) {
    if (!journal) return;
    journal->phase = phase;
    bench_progress_journal_note(journal, event, current, total);
}

static uint64_t bench_snapshot_max_bytes(void) {
    const char *env = getenv("DS4_BENCH_SNAPSHOT_MAX_BYTES");
    if (!env || env[0] == '\0') return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    if (!strcmp(env, "unlimited") || !strcmp(env, "UNLIMITED") ||
        !strcmp(env, "inf") || !strcmp(env, "INF")) {
        return UINT64_MAX;
    }
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (env[0] == '\0' || !end || *end != '\0') {
        fprintf(stderr,
                "ds4-bench: invalid DS4_BENCH_SNAPSHOT_MAX_BYTES=%s; using default %llu\n",
                env,
                (unsigned long long)DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES);
        return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    }
    return (uint64_t)v;
}

static uint32_t bench_tile_audit_capacity(void) {
    const char *env = getenv("DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY");
    if (!env || env[0] == '\0') return 4096u;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || !end || *end != '\0' || v == 0ul || v > UINT32_MAX) {
        fprintf(stderr,
                "ds4-bench: invalid DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY=%s\n",
                env);
        return 0u;
    }
    return (uint32_t)v;
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-decode-logits-dir")) {
            c.dump_decode_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--show-output")) {
            c.show_output = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) {
            fprintf(stderr,
                    "ds4-bench: non-finite frontier logit at %d (vocab index %d)\n",
                    frontier,
                    i);
            free(logits);
            return 1;
        }
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }

    char raw_path[PATH_MAX];
    const int raw_n = snprintf(raw_path,
                               sizeof(raw_path),
                               "%s/frontier_%06d.logits.f32",
                               cfg->dump_frontier_logits_dir,
                               frontier);
    if (raw_n <= 0 || (size_t)raw_n >= sizeof(raw_path)) {
        fprintf(stderr, "ds4-bench: raw frontier logits path is too long\n");
        free(logits);
        return 1;
    }
    fp = fopen(raw_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n",
                raw_path, strerror(errno));
        free(logits);
        return 1;
    }
    const size_t raw_written =
        fwrite(logits, sizeof(logits[0]), (size_t)vocab, fp);
    const int raw_close = fclose(fp);
    if (raw_written != (size_t)vocab || raw_close != 0) {
        fprintf(stderr, "ds4-bench: failed to write %s\n", raw_path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

/* Correctness-only decode evidence.  The caller records token timing before
 * entering this function, so filesystem work cannot inflate that token's
 * latency sample.  It can still perturb later samples and is therefore kept
 * out of every throughput run by the evidence driver. */
static int write_decode_logits_raw(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 decode_step) {
    if (!cfg->dump_decode_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying decode logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr,
                "ds4-bench: failed to copy decode logits at frontier %d step %d\n",
                frontier,
                decode_step);
        free(logits);
        return 1;
    }
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) {
            fprintf(stderr,
                    "ds4-bench: non-finite decode logit at frontier %d "
                    "step %d (vocab index %d)\n",
                    frontier,
                    decode_step,
                    i);
            free(logits);
            return 1;
        }
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.decode_%06d.logits.f32",
                           cfg->dump_decode_logits_dir,
                           frontier,
                           decode_step);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: decode logits path is too long\n");
        free(logits);
        return 1;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n",
                path, strerror(errno));
        free(logits);
        return 1;
    }
    const size_t written =
        fwrite(logits, sizeof(logits[0]), (size_t)vocab, fp);
    const int close_rc = fclose(fp);
    free(logits);
    if (written != (size_t)vocab || close_rc != 0) {
        fprintf(stderr, "ds4-bench: failed to write %s\n", path);
        return 1;
    }
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

int main(int argc, char **argv) {
    bench_install_crash_trace();
    bench_config cfg = parse_options(argc, argv);
    bench_progress_journal progress_journal = {
        .path = getenv("DS4_BENCH_PROGRESS_JOURNAL"),
        .phase = "initialization",
    };
    if (!bench_progress_journal_init(&progress_journal)) {
        fprintf(stderr,
                "ds4-bench: failed to initialize progress journal %s: %s\n",
                progress_journal.path,
                strerror(errno));
        return 2;
    }

    /* Hint the packer at the largest ctx this bench run will exercise
     * so per-layer KV bytes are priced for the real session size, not
     * a stale 4096 default. Single-tier and CPU paths ignore this. */
    int placement_ctx_hint = cfg.ctx_max;
    if (cfg.ctx_alloc > placement_ctx_hint) placement_ctx_hint = cfg.ctx_alloc;

    ds4_gpu_config gpu_cfg = {0};
    bool skip_cuda = false;
    int untimed_warmup_tokens = 0;
    const bool have_gpu_config = cfg.gpu_vram_arg || cfg.gpu_devices_arg;
    if (have_gpu_config) {
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-bench: %s\n", gpu_err);
            return 2;
        }
        cfg.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
    }

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    int nsys_decode_skip = -1;
    int nsys_decode_tokens = 0;
    const char *q8_cache_pretiming_state_csv =
        getenv("DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV");
    const char *q8_binding_state_csv =
        getenv("DS4_CUDA_Q8_BINDING_STATE_CSV");
    const char *q8_allocation_state_csv =
        getenv("DS4_CUDA_Q8_ALLOCATION_STATE_CSV");
    const char *cuda_memory_state_csv =
        getenv("DS4_CUDA_MEMORY_STATE_CSV");
    const char *q8_cache_audit_csv = cfg.backend == DS4_BACKEND_CUDA
        ? getenv("DS4_CUDA_Q8_CACHE_AUDIT_CSV") : NULL;
    bool q8_cache_audit_done = false;
    const char *warmup_env = getenv("DS4_BENCH_UNTIMED_WARMUP_TOKENS");
    if (warmup_env && warmup_env[0]) {
        errno = 0;
        char *end = NULL;
        const long parsed = strtol(warmup_env, &end, 10);
        if (errno != 0 || end == warmup_env || *end != '\0' ||
            parsed <= 0 || parsed > cfg.ctx_max || parsed >= cfg.ctx_alloc) {
            fprintf(stderr,
                    "ds4-bench: invalid DS4_BENCH_UNTIMED_WARMUP_TOKENS=%s\n",
                    warmup_env);
            return 2;
        }
        if (cfg.backend != DS4_BACKEND_CUDA ||
            cfg.dist.role != DS4_DISTRIBUTED_NONE) {
            fprintf(stderr,
                    "ds4-bench: untimed prefill warm-up requires local CUDA\n");
            return 2;
        }
        untimed_warmup_tokens = (int)parsed;
    }
    const char *decode_skip_env = getenv("DS4_NSYS_CAPTURE_DECODE_SKIP");
    const char *decode_tokens_env = getenv("DS4_NSYS_CAPTURE_DECODE_TOKENS");
    if ((decode_skip_env && decode_skip_env[0]) ||
        (decode_tokens_env && decode_tokens_env[0])) {
        if (!decode_skip_env || !decode_skip_env[0] ||
            !decode_tokens_env || !decode_tokens_env[0]) {
            fprintf(stderr,
                    "ds4-bench: DS4_NSYS_CAPTURE_DECODE_SKIP and "
                    "DS4_NSYS_CAPTURE_DECODE_TOKENS must be set together\n");
            return 2;
        }
        errno = 0;
        char *skip_end = NULL;
        const long parsed_skip = strtol(decode_skip_env, &skip_end, 10);
        const int skip_errno = errno;
        errno = 0;
        char *tokens_end = NULL;
        const long parsed_tokens = strtol(decode_tokens_env, &tokens_end, 10);
        const int tokens_errno = errno;
        if (skip_errno != 0 || skip_end == decode_skip_env ||
            !skip_end || *skip_end != '\0' || parsed_skip < 0 ||
            parsed_skip > INT_MAX || tokens_errno != 0 ||
            tokens_end == decode_tokens_env || !tokens_end ||
            *tokens_end != '\0' || parsed_tokens <= 0 ||
            parsed_tokens > INT_MAX ||
            parsed_skip > INT_MAX - parsed_tokens ||
            parsed_skip + parsed_tokens > cfg.gen_tokens) {
            fprintf(stderr,
                    "ds4-bench: invalid bounded decode capture skip=%s "
                    "tokens=%s for --gen-tokens=%d\n",
                    decode_skip_env,
                    decode_tokens_env,
                    cfg.gen_tokens);
            return 2;
        }
        if (cfg.backend != DS4_BACKEND_CUDA ||
            cfg.dist.role != DS4_DISTRIBUTED_NONE ||
            cfg.ctx_start != cfg.ctx_max) {
            fprintf(stderr,
                    "ds4-bench: bounded decode capture requires local CUDA "
                    "and one context frontier\n");
            return 2;
        }
        if (getenv("DS4_NSYS_CAPTURE_PREFILL") != NULL) {
            fprintf(stderr,
                    "ds4-bench: prefill and decode profiler captures are "
                    "mutually exclusive\n");
            return 2;
        }
        nsys_decode_skip = (int)parsed_skip;
        nsys_decode_tokens = (int)parsed_tokens;
    }
    if (((q8_cache_pretiming_state_csv && q8_cache_pretiming_state_csv[0]) ||
         (q8_binding_state_csv && q8_binding_state_csv[0]) ||
         (q8_allocation_state_csv && q8_allocation_state_csv[0]) ||
         (cuda_memory_state_csv && cuda_memory_state_csv[0])) &&
        untimed_warmup_tokens == 0) {
        fprintf(stderr,
                "ds4-bench: CUDA Q8 pre-timing state export "
                "requires DS4_BENCH_UNTIMED_WARMUP_TOKENS\n");
        return 2;
    }
#endif

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .cuda_tensor_parallel = cfg.cuda_tensor_parallel,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    bench_progress_journal_mark(
        &progress_journal, "engine-create", "start", 0, 1);
    if (have_gpu_config && !skip_cuda) {
        const bool was_auto =
            (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
            (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
        char layout[256];
        if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                   layout, sizeof(layout)) > 0) {
            fprintf(stdout, "%s\n", layout);
            fflush(stdout);
        }
        if (ds4_engine_create_with_gpu_config(
                &engine, &opt, &gpu_cfg) != 0) return 1;
    } else if (ds4_engine_open(&engine, &opt) != 0) {
        return 1;
    }
    bench_progress_journal_mark(
        &progress_journal, "engine-create", "complete", 1, 1);
    if (getenv("DS4_BENCH_ROUTED_QUANT_AUDIT") != NULL) {
        ds4_engine_log_routed_quant_audit(engine);
    }
    log_context_memory(opt.backend,
                       cfg.ctx_alloc,
                       ds4_engine_prefill_chunk(engine),
                       cfg.ssd_streaming);

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    bench_progress_journal_mark(
        &progress_journal, "session-create", "start", 0, 1);
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    bench_progress_journal_mark(
        &progress_journal, "session-create", "complete", 1, 1);
    progress_journal.phase = untimed_warmup_tokens > 0 ?
        "untimed-warmup" : "measured-prefill";
    if (progress_journal.path && progress_journal.path[0]) {
        ds4_session_set_progress(session,
                                 bench_progress_journal_note,
                                 &progress_journal);
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (untimed_warmup_tokens > 0) {
        ds4_tokens warmup_prefix = {
            .v = prompt.v,
            .len = untimed_warmup_tokens,
            .cap = untimed_warmup_tokens,
        };
        char warmup_err[256];
        fprintf(stderr,
                "ds4-bench: starting untimed CUDA warm-up frontier %d\n",
                untimed_warmup_tokens);
        bench_progress_journal_mark(
            &progress_journal, "untimed-warmup", "frontier-start",
            0, untimed_warmup_tokens);
        if (ds4_session_sync(session, &warmup_prefix,
                             warmup_err, sizeof(warmup_err)) != 0) {
            fprintf(stderr,
                    "ds4-bench: untimed CUDA warm-up failed: %s\n",
                    warmup_err);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        fprintf(stderr,
                "ds4-bench: completed untimed CUDA warm-up frontier %d\n",
                untimed_warmup_tokens);
        bench_progress_journal_mark(
            &progress_journal, "untimed-warmup", "frontier-complete",
            untimed_warmup_tokens, untimed_warmup_tokens);

        if (q8_cache_audit_csv && q8_cache_audit_csv[0]) {
            if (!ds4_gpu_q8_audit_write_csv(q8_cache_audit_csv)) {
                fprintf(stderr,
                        "ds4-bench: failed to write untimed CUDA Q8 cache audit %s\n",
                        q8_cache_audit_csv);
                ds4_session_free(session);
                ds4_tokens_free(&prompt);
                ds4_engine_close(engine);
                return 1;
            }
            ds4_gpu_q8_audit_end();
            q8_cache_audit_done = true;
            fprintf(stderr,
                    "ds4-bench: wrote untimed CUDA Q8 cache audit %s\n",
                    q8_cache_audit_csv);
        }

        /* Recreate the session so every reported frontier starts from the
         * same empty sequence state while retaining the engine-owned Q8
         * cache populated by the untimed pass. */
        ds4_session_free(session);
        session = NULL;
        bench_progress_journal_mark(
            &progress_journal, "session-recreate", "start", 0, 1);
        if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
            fprintf(stderr,
                    "ds4-bench: failed to recreate session after warm-up\n");
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        bench_progress_journal_mark(
            &progress_journal, "session-recreate", "complete", 1, 1);
        progress_journal.phase = "measured-prefill";
        if (progress_journal.path && progress_journal.path[0]) {
            ds4_session_set_progress(session,
                                     bench_progress_journal_note,
                                     &progress_journal);
        }
        if (q8_cache_pretiming_state_csv &&
            q8_cache_pretiming_state_csv[0] &&
            !ds4_gpu_q8_cache_state_write_csv(
                q8_cache_pretiming_state_csv)) {
            fprintf(stderr,
                    "ds4-bench: failed to write pre-timing CUDA Q8 cache state %s\n",
                    q8_cache_pretiming_state_csv);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        if (q8_binding_state_csv && q8_binding_state_csv[0] &&
            !ds4_gpu_q8_binding_state_write_csv(q8_binding_state_csv)) {
            fprintf(stderr,
                    "ds4-bench: failed to write CUDA Q8 binding state %s\n",
                    q8_binding_state_csv);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        if (q8_allocation_state_csv && q8_allocation_state_csv[0] &&
            !ds4_gpu_q8_allocation_state_write_csv(
                q8_allocation_state_csv)) {
            fprintf(stderr,
                    "ds4-bench: failed to write CUDA Q8 allocation state %s\n",
                    q8_allocation_state_csv);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        if (cuda_memory_state_csv && cuda_memory_state_csv[0] &&
            !ds4_gpu_memory_state_write_csv(cuda_memory_state_csv)) {
            fprintf(stderr,
                    "ds4-bench: failed to write CUDA memory state %s\n",
                    cuda_memory_state_csv);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        if ((q8_binding_state_csv && q8_binding_state_csv[0]) ||
            (q8_allocation_state_csv && q8_allocation_state_csv[0])) {
            /* Liveness is sampled only by the untimed warm-up.  Timed
             * dispatches retain only the disabled fast-path check; they do
             * not take the usage mutex or update a counter. */
            ds4_gpu_q8_binding_usage_end();
        }
    }
#endif
    maybe_warn_distributed_step_shape(&cfg, session);

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed = cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR;
    ds4_session_snapshot snap = {0};
    const uint64_t snapshot_max_bytes = bench_snapshot_max_bytes();
    bool warned_large_snapshot = false;
    char err[256];
    int previous = 0;
    int rc = 0;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    const bool nsys_capture_prefill =
        cfg.backend == DS4_BACKEND_CUDA &&
        getenv("DS4_NSYS_CAPTURE_PREFILL") != NULL;
    bool nsys_capture_done = false;
    bool nsys_decode_capture_active = false;
    bool nsys_decode_capture_done = false;
    const char *tile_audit_csv = cfg.backend == DS4_BACKEND_CUDA
        ? getenv("DS4_CUDA_PREFILL_TILE_AUDIT_CSV") : NULL;
    bool tile_audit_done = false;
    const char *q8_cache_state_csv = cfg.backend == DS4_BACKEND_CUDA
        ? getenv("DS4_CUDA_Q8_CACHE_STATE_CSV") : NULL;
#endif

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        const bool capture_this_prefill =
            nsys_capture_prefill && !nsys_capture_done;
        const bool audit_this_prefill =
            tile_audit_csv && tile_audit_csv[0] && !tile_audit_done;
        if (audit_this_prefill) {
            const uint32_t capacity = bench_tile_audit_capacity();
            if (capacity == 0u ||
                !ds4_gpu_prefill_tile_audit_begin(capacity)) {
                fprintf(stderr,
                        "ds4-bench: failed to initialize CUDA tile audit\n");
                rc = 1;
                break;
            }
        }
        if (capture_this_prefill) {
            fprintf(stderr,
                    "ds4-bench: starting Nsight CUDA capture for prefill frontier %d\n",
                    frontier);
            if (!ds4_gpu_profiler_start()) {
                fprintf(stderr, "ds4-bench: failed to start Nsight CUDA capture\n");
                rc = 1;
                break;
            }
        }
#endif
        bench_progress_journal_mark(
            &progress_journal, "measured-prefill", "frontier-start",
            previous, frontier);
        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
            if (audit_this_prefill) ds4_gpu_prefill_tile_audit_end();
#endif
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        bench_progress_journal_mark(
            &progress_journal, "measured-prefill", "frontier-complete",
            frontier, frontier);
        if (getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
            fprintf(stderr,
                    "ds4-bench: completed measured prefill frontier %d\n",
                    frontier);
            fflush(stderr);
        }
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        if (capture_this_prefill) {
            if (!ds4_gpu_profiler_stop()) {
                fprintf(stderr, "ds4-bench: failed to stop Nsight CUDA capture\n");
                rc = 1;
                break;
            }
            nsys_capture_done = true;
            fprintf(stderr,
                    "ds4-bench: stopped Nsight CUDA capture for prefill frontier %d\n",
                    frontier);
        }
        if (audit_this_prefill) {
            if (!ds4_gpu_prefill_tile_audit_write_csv(tile_audit_csv)) {
                fprintf(stderr,
                        "ds4-bench: failed to write CUDA tile audit %s\n",
                        tile_audit_csv);
                ds4_gpu_prefill_tile_audit_end();
                rc = 1;
                break;
            }
            ds4_gpu_prefill_tile_audit_end();
            tile_audit_done = true;
            fprintf(stderr, "ds4-bench: wrote CUDA tile audit %s\n",
                    tile_audit_csv);
        }
        if (q8_cache_audit_csv && q8_cache_audit_csv[0] &&
            !q8_cache_audit_done) {
            if (!ds4_gpu_q8_audit_write_csv(q8_cache_audit_csv)) {
                fprintf(stderr,
                        "ds4-bench: failed to write CUDA Q8 cache audit %s\n",
                        q8_cache_audit_csv);
                rc = 1;
                break;
            }
            ds4_gpu_q8_audit_end();
            q8_cache_audit_done = true;
            fprintf(stderr, "ds4-bench: wrote CUDA Q8 cache audit %s\n",
                    q8_cache_audit_csv);
        }
#endif
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        const bool need_restore_after_generation =
            cfg.gen_tokens > 0 && frontier < cfg.ctx_max;
        const bool force_frontier_snapshot =
            getenv("DS4_BENCH_FORCE_FRONTIER_SNAPSHOT") != NULL;
        bool have_snapshot = false;
        if ((need_restore_after_generation || force_frontier_snapshot) &&
            !distributed &&
            getenv("DS4_BENCH_DISABLE_SNAPSHOT") == NULL) {
            const uint64_t payload_bytes = ds4_session_payload_bytes(session);
            const bool large_snapshot_forced =
                getenv("DS4_BENCH_FORCE_SNAPSHOT") != NULL;
            if (payload_bytes > snapshot_max_bytes && !large_snapshot_forced) {
                if (!warned_large_snapshot) {
                    fprintf(stderr,
                            "ds4-bench: session payload snapshot is %.2f GiB, above the %.2f GiB benchmark limit; "
                            "replaying prefixes instead (set DS4_BENCH_FORCE_SNAPSHOT=1 to force snapshots)\n",
                            bytes_to_gib(payload_bytes),
                            bytes_to_gib(snapshot_max_bytes));
                    warned_large_snapshot = true;
                }
            } else if (payload_bytes > 0) {
                if (getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                    fprintf(stderr,
                            "ds4-bench: starting snapshot at frontier %d "
                            "bytes=%llu\n",
                            frontier, (unsigned long long)payload_bytes);
                    fflush(stderr);
                }
                if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                have_snapshot = true;
                if (getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                    fprintf(stderr,
                            "ds4-bench: completed snapshot at frontier %d\n",
                            frontier);
                    fflush(stderr);
                }
            }
        }

        const double gen_t0 = bench_now_sec();
        double gen_first_sec = 0.0;
        double gen_steady_sec = 0.0;
        int gen_done = 0;
        int *gen_token_buf = cfg.show_output && cfg.gen_tokens > 0
            ? malloc((size_t)cfg.gen_tokens * sizeof(gen_token_buf[0]))
            : NULL;
        int gen_token_count = 0;
        if (cfg.gen_tokens > 0) {
            if (getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                fprintf(stderr,
                        "ds4-bench: starting decode frontier %d tokens=%d "
                        "session=%p\n",
                        frontier, cfg.gen_tokens, (void *)session);
                fflush(stderr);
            }
            bench_progress_journal_mark(
                &progress_journal, "decode", "frontier-start",
                0, cfg.gen_tokens);
        }
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (i == 0 && getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                fprintf(stderr,
                        "ds4-bench: selecting first decode token at frontier %d\n",
                        frontier);
                fflush(stderr);
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (i == 0 && getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                fprintf(stderr,
                        "ds4-bench: selected first decode token at frontier %d "
                        "token=%d\n",
                        frontier, token);
                fflush(stderr);
            }
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
            if (nsys_decode_tokens > 0 && !nsys_decode_capture_done &&
                i == nsys_decode_skip) {
                fprintf(stderr,
                        "ds4-bench: starting Nsight CUDA capture for decode "
                        "frontier %d steps %d..%d\n",
                        frontier,
                        i + 1,
                        i + nsys_decode_tokens);
                if (!ds4_gpu_profiler_start()) {
                    fprintf(stderr,
                            "ds4-bench: failed to start bounded decode capture\n");
                    rc = 1;
                    break;
                }
                nsys_decode_capture_active = true;
            }
#endif
            if (i == 0 || ((i + 1) % 16) == 0) {
                bench_progress_journal_mark(
                    &progress_journal, "decode", "token-start",
                    i + 1, cfg.gen_tokens);
            }
            const double token_t0 = bench_now_sec();
            if (i == 0 && getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                fprintf(stderr,
                        "ds4-bench: entering first decode eval at frontier %d\n",
                        frontier);
                fflush(stderr);
            }
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
                if (nsys_decode_capture_active) {
                    (void)ds4_gpu_profiler_stop();
                    nsys_decode_capture_active = false;
                }
#endif
                rc = 1;
                break;
            }
            if (i == 0 && getenv("DS4_BENCH_PHASE_TRACE") != NULL) {
                fprintf(stderr,
                        "ds4-bench: completed first decode eval at frontier %d\n",
                        frontier);
                fflush(stderr);
            }
            const double token_t1 = bench_now_sec();
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
            if (nsys_decode_capture_active &&
                i + 1 == nsys_decode_skip + nsys_decode_tokens) {
                if (!ds4_gpu_profiler_stop()) {
                    fprintf(stderr,
                            "ds4-bench: failed to stop bounded decode capture\n");
                    nsys_decode_capture_active = false;
                    rc = 1;
                    break;
                }
                nsys_decode_capture_active = false;
                nsys_decode_capture_done = true;
                fprintf(stderr,
                        "ds4-bench: stopped Nsight CUDA capture for decode "
                        "frontier %d after %d tokens\n",
                        frontier,
                        nsys_decode_tokens);
            }
#endif
            if (i == 0) gen_first_sec = token_t1 - token_t0;
            else gen_steady_sec += token_t1 - token_t0;
            if (gen_token_buf) gen_token_buf[gen_token_count++] = token;
            gen_done++;
            if (i == 0 || ((i + 1) % 16) == 0 ||
                i + 1 == cfg.gen_tokens) {
                bench_progress_journal_mark(
                    &progress_journal, "decode", "token-complete",
                    i + 1, cfg.gen_tokens);
            }
            if (write_decode_logits_raw(&cfg,
                                        engine,
                                        session,
                                        frontier,
                                        i + 1) != 0) {
                rc = 1;
                break;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (rc == 0 && cfg.gen_tokens > 0) {
            bench_progress_journal_mark(
                &progress_journal, "decode", "frontier-complete",
                gen_done, cfg.gen_tokens);
        }
        if (cfg.show_output && gen_token_buf && gen_token_count > 0) {
            fprintf(stderr, "ds4-bench: gen[ctx=%d] decoded text: \"", frontier);
            for (int i = 0; i < gen_token_count; i++) {
                size_t tlen = 0;
                char *txt = ds4_token_text(engine, gen_token_buf[i], &tlen);
                if (txt) {
                    fwrite(txt, 1, tlen, stderr);
                    free(txt);
                }
            }
            fprintf(stderr, "\"\n");
            fflush(stderr);
        }
        free(gen_token_buf);
        if (rc != 0) break;
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        if (nsys_decode_tokens > 0 && !nsys_decode_capture_done) {
            fprintf(stderr,
                    "ds4-bench: bounded decode capture did not reach its "
                    "requested token window\n");
            rc = 1;
            break;
        }
#endif

        if (!need_restore_after_generation) {
            /* Nothing later depends on the frontier state. */
        } else if (distributed || !have_snapshot) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        const int gen_steady_tokens = gen_done > 1 ? gen_done - 1 : 0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                gen_first_sec * 1000.0,
                gen_steady_tokens,
                gen_steady_sec > 0.0 ? (double)gen_steady_tokens / gen_steady_sec : 0.0,
                (unsigned long long)(have_snapshot ? snap.len : 0));
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (rc == 0 && q8_cache_state_csv && q8_cache_state_csv[0]) {
        if (!ds4_gpu_q8_cache_state_write_csv(q8_cache_state_csv)) {
            fprintf(stderr,
                    "ds4-bench: failed to write CUDA Q8 cache state %s\n",
                    q8_cache_state_csv);
            rc = 1;
        } else {
            fprintf(stderr, "ds4-bench: wrote CUDA Q8 cache state %s\n",
                    q8_cache_state_csv);
        }
    }
#endif

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
