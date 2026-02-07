#define _POSIX_C_SOURCE 200809L

#include "cosine.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static void set_error(char *err, size_t err_sz, const char *fmt, ...) {
    if (err == NULL || err_sz == 0) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_sz, fmt, ap);
    va_end(ap);
}

static int is_little_endian(void) {
    uint16_t x = 1;
    return *((uint8_t *)&x) == 1;
}

static void byteswap_f64(double *vals, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint8_t *p = (uint8_t *)&vals[i];
        uint8_t t;
        t = p[0];
        p[0] = p[7];
        p[7] = t;
        t = p[1];
        p[1] = p[6];
        p[6] = t;
        t = p[2];
        p[2] = p[5];
        p[5] = t;
        t = p[3];
        p[3] = p[4];
        p[4] = t;
    }
}

static int read_text_file(const char *path, char **out_buf, size_t *out_len, char *err, size_t err_sz) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        set_error(err, err_sz, "cannot open metadata: %s", path);
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        set_error(err, err_sz, "cannot seek metadata: %s", path);
        return 0;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        set_error(err, err_sz, "cannot get metadata size: %s", path);
        return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        set_error(err, err_sz, "cannot rewind metadata: %s", path);
        return 0;
    }

    char *buf = (char *)malloc((size_t)sz + 1);
    if (buf == NULL) {
        fclose(f);
        set_error(err, err_sz, "out of memory");
        return 0;
    }

    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) {
        free(buf);
        set_error(err, err_sz, "cannot read metadata: %s", path);
        return 0;
    }
    buf[got] = '\0';
    *out_buf = buf;
    if (out_len != NULL) {
        *out_len = got;
    }
    return 1;
}

static char *find_key(const char *json, const char *key) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr((char *)json, needle);
}

static int extract_size_t(const char *json, const char *key, size_t *out) {
    char *pos = find_key(json, key);
    if (pos == NULL) {
        return 0;
    }
    pos = strchr(pos, ':');
    if (pos == NULL) {
        return 0;
    }
    pos++;
    while (*pos != '\0' && isspace((unsigned char)*pos)) {
        pos++;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(pos, &end, 10);
    if (end == pos || errno != 0) {
        return 0;
    }
    *out = (size_t)v;
    return 1;
}

static int extract_string(const char *json, const char *key, char *out, size_t out_sz) {
    char *pos = find_key(json, key);
    if (pos == NULL) {
        return 0;
    }
    pos = strchr(pos, ':');
    if (pos == NULL) {
        return 0;
    }
    pos++;
    while (*pos != '\0' && isspace((unsigned char)*pos)) {
        pos++;
    }
    if (*pos != '"') {
        return 0;
    }
    pos++;
    char *end = strchr(pos, '"');
    if (end == NULL) {
        return 0;
    }
    size_t len = (size_t)(end - pos);
    if (len + 1 > out_sz) {
        return 0;
    }
    memcpy(out, pos, len);
    out[len] = '\0';
    return 1;
}

static int dirname_of(const char *path, char *out, size_t out_sz) {
    size_t len = strlen(path);
    if (len + 1 > out_sz) {
        return 0;
    }
    memcpy(out, path, len + 1);
    char *slash = strrchr(out, '/');
    if (slash == NULL) {
        if (out_sz < 2) {
            return 0;
        }
        out[0] = '.';
        out[1] = '\0';
        return 1;
    }
    *slash = '\0';
    if (out[0] == '\0') {
        if (out_sz < 2) {
            return 0;
        }
        out[0] = '/';
        out[1] = '\0';
    }
    return 1;
}

static int join_path(const char *dir, const char *file, char *out, size_t out_sz) {
    if (file[0] == '/') {
        size_t len = strlen(file);
        if (len + 1 > out_sz) {
            return 0;
        }
        memcpy(out, file, len + 1);
        return 1;
    }
    size_t dlen = strlen(dir);
    size_t flen = strlen(file);
    size_t need = dlen + 1 + flen + 1;
    if (need > out_sz) {
        return 0;
    }
    memcpy(out, dir, dlen);
    out[dlen] = '/';
    memcpy(out + dlen + 1, file, flen);
    out[dlen + 1 + flen] = '\0';
    return 1;
}

static int shell_escape_single_quoted(const char *in, char *out, size_t out_sz) {
    size_t j = 0;
    if (out_sz < 3) {
        return 0;
    }
    out[j++] = '\'';
    for (size_t i = 0; in[i] != '\0'; i++) {
        if (in[i] == '\'') {
            const char *esc = "'\\''";
            for (size_t k = 0; esc[k] != '\0'; k++) {
                if (j + 1 >= out_sz) {
                    return 0;
                }
                out[j++] = esc[k];
            }
        } else {
            if (j + 1 >= out_sz) {
                return 0;
            }
            out[j++] = in[i];
        }
    }
    if (j + 2 > out_sz) {
        return 0;
    }
    out[j++] = '\'';
    out[j] = '\0';
    return 1;
}

static int dataset_sha256_concat(const char *e_path, const char *a_path, char out_sha[65], char *err, size_t err_sz) {
    char esc_e[PATH_MAX * 2];
    char esc_a[PATH_MAX * 2];
    if (!shell_escape_single_quoted(e_path, esc_e, sizeof(esc_e)) ||
        !shell_escape_single_quoted(a_path, esc_a, sizeof(esc_a))) {
        set_error(err, err_sz, "dataset path too long for sha256 command");
        return 0;
    }

    char cmd[PATH_MAX * 5];
    int n = snprintf(cmd, sizeof(cmd), "cat %s %s | sha256sum", esc_e, esc_a);
    if (n <= 0 || (size_t)n >= sizeof(cmd)) {
        set_error(err, err_sz, "sha256 command too long");
        return 0;
    }

    FILE *pipe = popen(cmd, "r");
    if (pipe == NULL) {
        set_error(err, err_sz, "cannot run sha256sum");
        return 0;
    }

    char line[256];
    if (fgets(line, sizeof(line), pipe) == NULL) {
        pclose(pipe);
        set_error(err, err_sz, "cannot read sha256sum output");
        return 0;
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        set_error(err, err_sz, "sha256sum command failed");
        return 0;
    }

    size_t len = strcspn(line, " \t\r\n");
    if (len != 64) {
        set_error(err, err_sz, "invalid sha256 output");
        return 0;
    }
    for (size_t i = 0; i < len; i++) {
        if (!isxdigit((unsigned char)line[i])) {
            set_error(err, err_sz, "invalid sha256 hex output");
            return 0;
        }
    }

    memcpy(out_sha, line, 64);
    out_sha[64] = '\0';
    return 1;
}

static int read_f64_file(const char *path, size_t expected_count, double **out, char *err, size_t err_sz) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        set_error(err, err_sz, "cannot open data file: %s", path);
        return 0;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        set_error(err, err_sz, "cannot seek data file: %s", path);
        return 0;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        set_error(err, err_sz, "cannot stat data file: %s", path);
        return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        set_error(err, err_sz, "cannot rewind data file: %s", path);
        return 0;
    }

    size_t expected_bytes = expected_count * sizeof(double);
    if ((size_t)sz != expected_bytes) {
        fclose(f);
        set_error(err, err_sz, "bad file size for %s: got %ld, expected %zu", path, sz, expected_bytes);
        return 0;
    }

    double *buf = (double *)malloc(expected_bytes);
    if (buf == NULL) {
        fclose(f);
        set_error(err, err_sz, "out of memory");
        return 0;
    }
    size_t got = fread(buf, sizeof(double), expected_count, f);
    fclose(f);
    if (got != expected_count) {
        free(buf);
        set_error(err, err_sz, "cannot read data file: %s", path);
        return 0;
    }

    if (!is_little_endian()) {
        byteswap_f64(buf, expected_count);
    }

    *out = buf;
    return 1;
}

int load_dataset(const char *metadata_path, Dataset *out, char *err, size_t err_sz) {
    memset(out, 0, sizeof(*out));

    char *json = NULL;
    if (!read_text_file(metadata_path, &json, NULL, err, err_sz)) {
        return 0;
    }

    char format[64];
    char dtype[64];
    char e_file[PATH_MAX];
    char a_file[PATH_MAX];
    if (!extract_string(json, "format", format, sizeof(format)) || strcmp(format, "cosine-benchmark-v1") != 0) {
        free(json);
        set_error(err, err_sz, "unsupported metadata format");
        return 0;
    }
    if (!extract_string(json, "dtype", dtype, sizeof(dtype)) || strcmp(dtype, "float64-le") != 0) {
        free(json);
        set_error(err, err_sz, "unsupported dtype");
        return 0;
    }
    if (!extract_size_t(json, "N", &out->N) || !extract_size_t(json, "M", &out->M) || !extract_size_t(json, "D", &out->D)) {
        free(json);
        set_error(err, err_sz, "missing N/M/D in metadata");
        return 0;
    }
    if (!extract_string(json, "E_file", e_file, sizeof(e_file)) || !extract_string(json, "A_file", a_file, sizeof(a_file))) {
        free(json);
        set_error(err, err_sz, "missing E_file/A_file in metadata");
        return 0;
    }
    free(json);

    char meta_dir[PATH_MAX];
    char e_path[PATH_MAX];
    char a_path[PATH_MAX];
    if (!dirname_of(metadata_path, meta_dir, sizeof(meta_dir)) ||
        !join_path(meta_dir, e_file, e_path, sizeof(e_path)) ||
        !join_path(meta_dir, a_file, a_path, sizeof(a_path))) {
        set_error(err, err_sz, "path too long");
        return 0;
    }

    if (!read_f64_file(e_path, out->N * out->D, &out->E, err, err_sz)) {
        return 0;
    }
    if (!read_f64_file(a_path, out->M * out->D, &out->A, err, err_sz)) {
        free(out->E);
        out->E = NULL;
        return 0;
    }

    if (!dataset_sha256_concat(e_path, a_path, out->dataset_sha256, err, err_sz)) {
        free(out->E);
        free(out->A);
        out->E = NULL;
        out->A = NULL;
        return 0;
    }

    snprintf(out->e_path, sizeof(out->e_path), "%s", e_path);
    snprintf(out->a_path, sizeof(out->a_path), "%s", a_path);

    return 1;
}

void free_dataset(Dataset *ds) {
    if (ds == NULL) {
        return;
    }
    free(ds->E);
    free(ds->A);
    ds->E = NULL;
    ds->A = NULL;
    ds->N = ds->M = ds->D = 0;
}

double cosine_all_pairs_checksum(const Dataset *ds) {
    size_t N = ds->N;
    size_t M = ds->M;
    size_t D = ds->D;

    double *axis_norms = (double *)malloc(M * sizeof(double));
    if (axis_norms == NULL) {
        return 0.0;
    }

    for (size_t j = 0; j < M; j++) {
        size_t a_base = j * D;
        double s = 0.0;
        for (size_t k = 0; k < D; k++) {
            double x = ds->A[a_base + k];
            s += x * x;
        }
        axis_norms[j] = sqrt(s);
    }

    double checksum = 0.0;
    for (size_t i = 0; i < N; i++) {
        size_t e_base = i * D;

        double emb_norm_sq = 0.0;
        for (size_t k = 0; k < D; k++) {
            double x = ds->E[e_base + k];
            emb_norm_sq += x * x;
        }
        double emb_norm = sqrt(emb_norm_sq);
        if (emb_norm == 0.0) {
            continue;
        }

        for (size_t j = 0; j < M; j++) {
            size_t a_base = j * D;
            double dot = 0.0;
            for (size_t k = 0; k < D; k++) {
                dot += ds->E[e_base + k] * ds->A[a_base + k];
            }
            double denom = emb_norm * axis_norms[j];
            if (denom != 0.0) {
                checksum += dot / denom;
            }
        }
    }

    free(axis_norms);
    return checksum;
}
