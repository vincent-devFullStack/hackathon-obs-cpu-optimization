#include "cosine.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace {

bool is_little_endian() {
    std::uint16_t x = 1;
    return *reinterpret_cast<std::uint8_t *>(&x) == 1;
}

void byteswap_f64(std::vector<double> &vals) {
    for (double &v : vals) {
        auto *p = reinterpret_cast<std::uint8_t *>(&v);
        std::swap(p[0], p[7]);
        std::swap(p[1], p[6]);
        std::swap(p[2], p[5]);
        std::swap(p[3], p[4]);
    }
}

bool read_text_file(const std::string &path, std::string &out, std::string &err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open metadata: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::size_t find_key(const std::string &json, const std::string &key) {
    return json.find("\"" + key + "\"");
}

bool extract_string(const std::string &json, const std::string &key, std::string &out) {
    std::size_t pos = find_key(json, key);
    if (pos == std::string::npos) {
        return false;
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return false;
    }
    pos++;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        pos++;
    }
    if (pos >= json.size() || json[pos] != '"') {
        return false;
    }
    pos++;
    std::size_t end = json.find('"', pos);
    if (end == std::string::npos) {
        return false;
    }
    out.assign(json.data() + pos, end - pos);
    return true;
}

bool extract_size_t(const std::string &json, const std::string &key, std::size_t &out) {
    std::size_t pos = find_key(json, key);
    if (pos == std::string::npos) {
        return false;
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return false;
    }
    pos++;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        pos++;
    }
    std::size_t end = pos;
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        end++;
    }
    if (end == pos) {
        return false;
    }
    out = static_cast<std::size_t>(std::stoull(json.substr(pos, end - pos)));
    return true;
}

std::string dirname_of(const std::string &path) {
    std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return ".";
    }
    if (slash == 0) {
        return "/";
    }
    return path.substr(0, slash);
}

std::string join_path(const std::string &dir, const std::string &file) {
    if (!file.empty() && file[0] == '/') {
        return file;
    }
    return dir + "/" + file;
}

bool read_f64_file(const std::string &path, std::size_t expected_count, std::vector<double> &out, std::string &err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open data file: " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    std::streamoff sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::size_t expected_bytes = expected_count * sizeof(double);
    if (sz < 0 || static_cast<std::size_t>(sz) != expected_bytes) {
        err = "bad data file size: " + path;
        return false;
    }

    out.resize(expected_count);
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(expected_bytes));
    if (!in) {
        err = "cannot read data file: " + path;
        return false;
    }

    if (!is_little_endian()) {
        byteswap_f64(out);
    }

    return true;
}

std::string shell_escape_single_quoted(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

bool dataset_sha256_concat(const std::string &e_path, const std::string &a_path, std::string &out_sha, std::string &err) {
    std::string cmd = "cat " + shell_escape_single_quoted(e_path) + " " + shell_escape_single_quoted(a_path) + " | sha256sum";
    std::array<char, 256> buf{};
    FILE *pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        err = "cannot run sha256sum";
        return false;
    }
    if (fgets(buf.data(), static_cast<int>(buf.size()), pipe) == nullptr) {
        pclose(pipe);
        err = "cannot read sha256sum output";
        return false;
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        err = "sha256sum command failed";
        return false;
    }
    std::string line(buf.data());
    std::size_t sep = line.find_first_of(" \t\r\n");
    if (sep == std::string::npos || sep != 64) {
        err = "invalid sha256 output";
        return false;
    }
    out_sha = line.substr(0, 64);
    for (char c : out_sha) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) {
            err = "invalid sha256 hex";
            return false;
        }
    }
    return true;
}

}  // namespace

bool load_dataset(const std::string &metadata_path, Dataset &out, std::string &err) {
    std::string json;
    if (!read_text_file(metadata_path, json, err)) {
        return false;
    }

    std::string format;
    std::string dtype;
    std::string e_file;
    std::string a_file;

    if (!extract_string(json, "format", format) || format != "cosine-benchmark-v1") {
        err = "unsupported metadata format";
        return false;
    }
    if (!extract_string(json, "dtype", dtype) || dtype != "float64-le") {
        err = "unsupported metadata dtype";
        return false;
    }
    if (!extract_size_t(json, "N", out.N) || !extract_size_t(json, "M", out.M) || !extract_size_t(json, "D", out.D)) {
        err = "missing N/M/D in metadata";
        return false;
    }
    if (!extract_string(json, "E_file", e_file) || !extract_string(json, "A_file", a_file)) {
        err = "missing E_file/A_file in metadata";
        return false;
    }

    std::string base = dirname_of(metadata_path);
    out.e_path = join_path(base, e_file);
    out.a_path = join_path(base, a_file);

    if (!read_f64_file(out.e_path, out.N * out.D, out.E, err)) {
        return false;
    }
    if (!read_f64_file(out.a_path, out.M * out.D, out.A, err)) {
        return false;
    }
    if (!dataset_sha256_concat(out.e_path, out.a_path, out.dataset_sha256, err)) {
        return false;
    }

    return true;
}

double cosine_all_pairs_checksum(const Dataset &ds) {
    std::vector<double> axis_norms(ds.M, 0.0);
    for (std::size_t j = 0; j < ds.M; j++) {
        std::size_t a_base = j * ds.D;
        double s = 0.0;
        for (std::size_t k = 0; k < ds.D; k++) {
            double x = ds.A[a_base + k];
            s += x * x;
        }
        axis_norms[j] = std::sqrt(s);
    }

    double checksum = 0.0;
    for (std::size_t i = 0; i < ds.N; i++) {
        std::size_t e_base = i * ds.D;
        double emb_norm_sq = 0.0;
        for (std::size_t k = 0; k < ds.D; k++) {
            double x = ds.E[e_base + k];
            emb_norm_sq += x * x;
        }
        double emb_norm = std::sqrt(emb_norm_sq);
        if (emb_norm == 0.0) {
            continue;
        }

        for (std::size_t j = 0; j < ds.M; j++) {
            std::size_t a_base = j * ds.D;
            double dot = 0.0;
            for (std::size_t k = 0; k < ds.D; k++) {
                dot += ds.E[e_base + k] * ds.A[a_base + k];
            }
            double denom = emb_norm * axis_norms[j];
            if (denom != 0.0) {
                checksum += dot / denom;
            }
        }
    }

    return checksum;
}
