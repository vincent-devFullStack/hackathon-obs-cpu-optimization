#include "cosine.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <vector>

#ifndef BUILD_FLAGS
#define BUILD_FLAGS ""
#endif

#ifndef BUILD_COMPILER
#define BUILD_COMPILER "c++"
#endif

#ifndef BUILD_PROFILE
#define BUILD_PROFILE "unknown"
#endif

static long long now_wall_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + static_cast<long long>(ts.tv_nsec);
}

static long long now_cpu_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + static_cast<long long>(ts.tv_nsec);
}

static bool parse_int(const std::string &s, int &out) {
    try {
        std::size_t idx = 0;
        long v = std::stol(s, &idx, 10);
        if (idx != s.size()) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static int proc_threads() {
    std::ifstream in("/proc/self/status");
    if (!in) {
        return -1;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("Threads:", 0) == 0) {
            try {
                return std::stoi(line.substr(8));
            } catch (...) {
                return -1;
            }
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    std::string metadata = "data/metadata.json";
    std::string expected_dataset_sha;
    int warmup = 5;
    int runs = 30;
    int repeat = 50;
    bool self_check = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--metadata" && i + 1 < argc) {
            metadata = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!parse_int(argv[++i], warmup)) {
                std::cerr << "invalid --warmup\n";
                return 1;
            }
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!parse_int(argv[++i], runs)) {
                std::cerr << "invalid --runs\n";
                return 1;
            }
        } else if (arg == "--repeat" && i + 1 < argc) {
            if (!parse_int(argv[++i], repeat)) {
                std::cerr << "invalid --repeat\n";
                return 1;
            }
        } else if (arg == "--expected-dataset-sha" && i + 1 < argc) {
            expected_dataset_sha = argv[++i];
        } else if (arg == "--self-check") {
            self_check = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " [--metadata PATH] [--warmup N] [--runs N] [--repeat N] [--expected-dataset-sha SHA] [--self-check]\n";
            return 1;
        }
    }

    if (warmup < 0 || runs <= 0 || repeat <= 0) {
        std::cerr << "warmup must be >= 0, runs and repeat must be > 0\n";
        return 1;
    }

    Dataset ds{};
    std::string err;
    if (!load_dataset(metadata, ds, err)) {
        std::cerr << "failed to load dataset: " << err << "\n";
        return 1;
    }

    if (!expected_dataset_sha.empty() && ds.dataset_sha256 != expected_dataset_sha) {
        std::cerr << "dataset sha mismatch: expected " << expected_dataset_sha << " got " << ds.dataset_sha256 << "\n";
        return 1;
    }

    std::cout << std::setprecision(17);

    if (self_check) {
        double checksum = cosine_all_pairs_checksum(ds);
        std::cout << "{\"type\":\"self_check\",\"impl\":\"cpp-naive\",\"ok\":true,\"N\":" << ds.N
                  << ",\"M\":" << ds.M
                  << ",\"D\":" << ds.D
                  << ",\"dataset_sha256\":\"" << ds.dataset_sha256
                  << "\",\"checksum\":" << checksum
                  << "}" << std::endl;
        return 0;
    }

    std::cout << "{\"type\":\"meta\",\"impl\":\"cpp-naive\",\"N\":" << ds.N
              << ",\"M\":" << ds.M
              << ",\"D\":" << ds.D
              << ",\"repeat\":" << repeat
              << ",\"warmup\":" << warmup
              << ",\"runs\":" << runs
              << ",\"dataset_sha256\":\"" << ds.dataset_sha256
              << "\",\"build_flags\":\"" << BUILD_FLAGS
              << "\",\"runtime\":{\"compiler\":\"" << BUILD_COMPILER
              << "\",\"compiler_version\":\"" << __VERSION__
              << "\",\"build_profile\":\"" << BUILD_PROFILE
              << "\",\"warmup_executed\":" << warmup
              << "}}" << std::endl;

    for (int i = 0; i < warmup; i++) {
        volatile double sink = cosine_all_pairs_checksum(ds);
        (void)sink;
    }

    for (int run_id = 0; run_id < runs; run_id++) {
        struct rusage ru0 {};
        struct rusage ru1 {};
        getrusage(RUSAGE_SELF, &ru0);
        int threads0 = proc_threads();

        long long t0 = now_wall_ns();
        long long c0 = now_cpu_ns();

        double checksum_acc = 0.0;
        for (int r = 0; r < repeat; r++) {
            checksum_acc += cosine_all_pairs_checksum(ds);
        }

        long long c1 = now_cpu_ns();
        long long t1 = now_wall_ns();
        int threads1 = proc_threads();
        getrusage(RUSAGE_SELF, &ru1);

        long long wall_ns = (t1 - t0) / repeat;
        long long cpu_ns = (c1 - c0) / repeat;

        long long ctx_vol = static_cast<long long>(ru1.ru_nvcsw - ru0.ru_nvcsw);
        long long ctx_invol = static_cast<long long>(ru1.ru_nivcsw - ru0.ru_nivcsw);
        long long minflt = static_cast<long long>(ru1.ru_minflt - ru0.ru_minflt);
        long long majflt = static_cast<long long>(ru1.ru_majflt - ru0.ru_majflt);
        int max_threads = threads0 > threads1 ? threads0 : threads1;

        std::cout << "{\"type\":\"run\",\"impl\":\"cpp-naive\",\"run_id\":" << run_id
                  << ",\"wall_ns\":" << wall_ns
                  << ",\"cpu_ns\":" << cpu_ns
                  << ",\"checksum\":" << checksum_acc
                  << ",\"max_rss_kb\":" << ru1.ru_maxrss
                  << ",\"ctx_voluntary\":" << ctx_vol
                  << ",\"ctx_involuntary\":" << ctx_invol
                  << ",\"minor_faults\":" << minflt
                  << ",\"major_faults\":" << majflt
                  << ",\"max_threads\":" << max_threads
                  << "}" << std::endl;
    }

    return 0;
}
