#define _POSIX_C_SOURCE 200809L

#include "cosine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#ifndef BUILD_FLAGS
#define BUILD_FLAGS ""
#endif

#ifndef BUILD_COMPILER
#define BUILD_COMPILER "cc"
#endif

static long long now_wall_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static long long now_cpu_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static int proc_threads(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (f == NULL) {
        return -1;
    }
    char line[256];
    int threads = -1;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "Threads:", 8) == 0) {
            threads = atoi(line + 8);
            break;
        }
    }
    fclose(f);
    return threads;
}

static int parse_int_arg(const char *name, const char *value, int *out) {
    char *end = NULL;
    long v = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        fprintf(stderr, "invalid value for %s: %s\n", name, value);
        return 0;
    }
    *out = (int)v;
    return 1;
}

int main(int argc, char **argv) {
    const char *metadata = "data/metadata.json";
    const char *expected_dataset_sha = "";
    int warmup = 5;
    int runs = 30;
    int repeat = 50;
    int self_check = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--metadata") == 0 && i + 1 < argc) {
            metadata = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (!parse_int_arg("--warmup", argv[++i], &warmup)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            if (!parse_int_arg("--runs", argv[++i], &runs)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            if (!parse_int_arg("--repeat", argv[++i], &repeat)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--expected-dataset-sha") == 0 && i + 1 < argc) {
            expected_dataset_sha = argv[++i];
        } else if (strcmp(argv[i], "--self-check") == 0) {
            self_check = 1;
        } else {
            fprintf(stderr, "Usage: %s [--metadata PATH] [--warmup N] [--runs N] [--repeat N] [--expected-dataset-sha SHA] [--self-check]\n", argv[0]);
            return 1;
        }
    }

    if (warmup < 0 || runs <= 0 || repeat <= 0) {
        fprintf(stderr, "warmup must be >= 0, runs and repeat must be > 0\n");
        return 1;
    }

    Dataset ds;
    char err[256] = {0};
    if (!load_dataset(metadata, &ds, err, sizeof(err))) {
        fprintf(stderr, "failed to load dataset: %s\n", err);
        return 1;
    }

    if (expected_dataset_sha[0] != '\0' && strcmp(expected_dataset_sha, ds.dataset_sha256) != 0) {
        fprintf(stderr, "dataset sha mismatch: expected %s got %s\n", expected_dataset_sha, ds.dataset_sha256);
        free_dataset(&ds);
        return 1;
    }

    if (self_check) {
        double checksum = cosine_all_pairs_checksum(&ds);
        printf("{\"type\":\"self_check\",\"impl\":\"c-naive\",\"ok\":true,\"N\":%zu,\"M\":%zu,\"D\":%zu,\"dataset_sha256\":\"%s\",\"checksum\":%.17g}\n",
               ds.N,
               ds.M,
               ds.D,
               ds.dataset_sha256,
               checksum);
        free_dataset(&ds);
        return 0;
    }

    printf("{\"type\":\"meta\",\"impl\":\"c-naive\",\"N\":%zu,\"M\":%zu,\"D\":%zu,\"repeat\":%d,\"warmup\":%d,\"runs\":%d,\"dataset_sha256\":\"%s\",\"build_flags\":\"%s\",\"runtime\":{\"compiler\":\"%s\",\"compiler_version\":\"%s\",\"warmup_executed\":%d}}\n",
           ds.N,
           ds.M,
           ds.D,
           repeat,
           warmup,
           runs,
           ds.dataset_sha256,
           BUILD_FLAGS,
           BUILD_COMPILER,
           __VERSION__,
           warmup);
    fflush(stdout);

    for (int i = 0; i < warmup; i++) {
        volatile double sink = cosine_all_pairs_checksum(&ds);
        (void)sink;
    }

    for (int run_id = 0; run_id < runs; run_id++) {
        struct rusage ru0;
        struct rusage ru1;
        getrusage(RUSAGE_SELF, &ru0);

        int threads0 = proc_threads();
        long long t0 = now_wall_ns();
        long long c0 = now_cpu_ns();

        double checksum_acc = 0.0;
        for (int r = 0; r < repeat; r++) {
            checksum_acc += cosine_all_pairs_checksum(&ds);
        }

        long long c1 = now_cpu_ns();
        long long t1 = now_wall_ns();
        int threads1 = proc_threads();
        getrusage(RUSAGE_SELF, &ru1);

        long long wall_ns = (t1 - t0) / repeat;
        long long cpu_ns = (c1 - c0) / repeat;

        long long ctx_vol = (long long)(ru1.ru_nvcsw - ru0.ru_nvcsw);
        long long ctx_invol = (long long)(ru1.ru_nivcsw - ru0.ru_nivcsw);
        long long minflt = (long long)(ru1.ru_minflt - ru0.ru_minflt);
        long long majflt = (long long)(ru1.ru_majflt - ru0.ru_majflt);
        int max_threads = threads0 > threads1 ? threads0 : threads1;

        printf("{\"type\":\"run\",\"impl\":\"c-naive\",\"run_id\":%d,\"wall_ns\":%lld,\"cpu_ns\":%lld,\"checksum\":%.17g,\"max_rss_kb\":%ld,\"ctx_voluntary\":%lld,\"ctx_involuntary\":%lld,\"minor_faults\":%lld,\"major_faults\":%lld,\"max_threads\":%d}\n",
               run_id,
               wall_ns,
               cpu_ns,
               checksum_acc,
               ru1.ru_maxrss,
               ctx_vol,
               ctx_invol,
               minflt,
               majflt,
               max_threads);
        fflush(stdout);
    }

    free_dataset(&ds);
    return 0;
}
