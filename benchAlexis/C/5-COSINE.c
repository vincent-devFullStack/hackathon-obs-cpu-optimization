#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 5000000
#endif

typedef struct {
    char *name;
    double *vector;
    size_t dim;
} LabelVector;

typedef struct {
    const char *data;
    size_t len;
    size_t pos;
    char error[256];
} Parser;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

static void set_error(Parser *p, const char *fmt, ...) {
    if (p->error[0] != '\0') {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(p->error, sizeof(p->error), fmt, args);
    va_end(args);
}

static void skip_ws(Parser *p) {
    while (p->pos < p->len) {
        unsigned char c = (unsigned char)p->data[p->pos];
        if (!isspace(c)) {
            break;
        }
        p->pos++;
    }
}

static bool expect_char(Parser *p, char expected) {
    skip_ws(p);
    if (p->pos >= p->len || p->data[p->pos] != expected) {
        set_error(p, "expected '%c' at byte %zu", expected, p->pos);
        return false;
    }
    p->pos++;
    return true;
}

static bool sb_reserve(StrBuf *sb, size_t need) {
    if (need <= sb->cap) {
        return true;
    }
    size_t new_cap = sb->cap == 0 ? 64 : sb->cap;
    while (new_cap < need) {
        new_cap *= 2;
    }
    char *new_data = (char *)realloc(sb->data, new_cap);
    if (new_data == NULL) {
        return false;
    }
    sb->data = new_data;
    sb->cap = new_cap;
    return true;
}

static bool sb_push_char(StrBuf *sb, char c) {
    if (!sb_reserve(sb, sb->len + 2)) {
        return false;
    }
    sb->data[sb->len++] = c;
    sb->data[sb->len] = '\0';
    return true;
}

static bool sb_push_utf8(StrBuf *sb, unsigned int cp) {
    if (cp <= 0x7F) {
        return sb_push_char(sb, (char)cp);
    }
    if (cp <= 0x7FF) {
        if (!sb_reserve(sb, sb->len + 3)) {
            return false;
        }
        sb->data[sb->len++] = (char)(0xC0 | (cp >> 6));
        sb->data[sb->len++] = (char)(0x80 | (cp & 0x3F));
        sb->data[sb->len] = '\0';
        return true;
    }
    if (cp <= 0xFFFF) {
        if (!sb_reserve(sb, sb->len + 4)) {
            return false;
        }
        sb->data[sb->len++] = (char)(0xE0 | (cp >> 12));
        sb->data[sb->len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        sb->data[sb->len++] = (char)(0x80 | (cp & 0x3F));
        sb->data[sb->len] = '\0';
        return true;
    }
    if (cp <= 0x10FFFF) {
        if (!sb_reserve(sb, sb->len + 5)) {
            return false;
        }
        sb->data[sb->len++] = (char)(0xF0 | (cp >> 18));
        sb->data[sb->len++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        sb->data[sb->len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        sb->data[sb->len++] = (char)(0x80 | (cp & 0x3F));
        sb->data[sb->len] = '\0';
        return true;
    }
    return false;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    return -1;
}

static bool parse_u4(Parser *p, unsigned int *out_cp) {
    if (p->pos + 4 > p->len) {
        set_error(p, "incomplete unicode escape at byte %zu", p->pos);
        return false;
    }
    unsigned int cp = 0;
    for (int i = 0; i < 4; i++) {
        int hv = hex_value(p->data[p->pos + (size_t)i]);
        if (hv < 0) {
            set_error(p, "invalid unicode escape at byte %zu", p->pos + (size_t)i);
            return false;
        }
        cp = (cp << 4) | (unsigned int)hv;
    }
    p->pos += 4;
    *out_cp = cp;
    return true;
}

static bool parse_string(Parser *p, char **out_str) {
    skip_ws(p);
    if (p->pos >= p->len || p->data[p->pos] != '"') {
        set_error(p, "expected string at byte %zu", p->pos);
        return false;
    }
    p->pos++;

    StrBuf sb = {0};
    while (p->pos < p->len) {
        char c = p->data[p->pos++];
        if (c == '"') {
            if (!sb_reserve(&sb, sb.len + 1)) {
                free(sb.data);
                set_error(p, "out of memory");
                return false;
            }
            sb.data[sb.len] = '\0';
            *out_str = sb.data;
            return true;
        }
        if (c == '\\') {
            if (p->pos >= p->len) {
                free(sb.data);
                set_error(p, "incomplete escape sequence at byte %zu", p->pos);
                return false;
            }
            char esc = p->data[p->pos++];
            switch (esc) {
                case '"':
                case '\\':
                case '/':
                    if (!sb_push_char(&sb, esc)) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                case 'b':
                    if (!sb_push_char(&sb, '\b')) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                case 'f':
                    if (!sb_push_char(&sb, '\f')) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                case 'n':
                    if (!sb_push_char(&sb, '\n')) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                case 'r':
                    if (!sb_push_char(&sb, '\r')) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                case 't':
                    if (!sb_push_char(&sb, '\t')) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                case 'u': {
                    unsigned int cp1 = 0;
                    if (!parse_u4(p, &cp1)) {
                        free(sb.data);
                        return false;
                    }

                    if (cp1 >= 0xD800 && cp1 <= 0xDBFF) {
                        if (p->pos + 6 <= p->len && p->data[p->pos] == '\\' && p->data[p->pos + 1] == 'u') {
                            p->pos += 2;
                            unsigned int cp2 = 0;
                            if (!parse_u4(p, &cp2)) {
                                free(sb.data);
                                return false;
                            }
                            if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                cp1 = 0x10000 + (((cp1 - 0xD800) << 10) | (cp2 - 0xDC00));
                            } else {
                                free(sb.data);
                                set_error(p, "invalid surrogate pair at byte %zu", p->pos);
                                return false;
                            }
                        } else {
                            free(sb.data);
                            set_error(p, "missing low surrogate at byte %zu", p->pos);
                            return false;
                        }
                    }

                    if (!sb_push_utf8(&sb, cp1)) {
                        free(sb.data);
                        set_error(p, "out of memory");
                        return false;
                    }
                    break;
                }
                default:
                    free(sb.data);
                    set_error(p, "invalid escape sequence at byte %zu", p->pos - 1);
                    return false;
            }
            continue;
        }
        if ((unsigned char)c < 0x20) {
            free(sb.data);
            set_error(p, "control character in string at byte %zu", p->pos - 1);
            return false;
        }
        if (!sb_push_char(&sb, c)) {
            free(sb.data);
            set_error(p, "out of memory");
            return false;
        }
    }

    free(sb.data);
    set_error(p, "unterminated string");
    return false;
}

static bool parse_number(Parser *p, double *out_value) {
    skip_ws(p);
    if (p->pos >= p->len) {
        set_error(p, "unexpected end while parsing number");
        return false;
    }
    const char *start = p->data + p->pos;
    char *end = NULL;
    errno = 0;
    double v = strtod(start, &end);
    if (end == start) {
        set_error(p, "expected number at byte %zu", p->pos);
        return false;
    }
    if (errno == ERANGE) {
        set_error(p, "number out of range near byte %zu", p->pos);
        return false;
    }
    p->pos = (size_t)(end - p->data);
    if (out_value != NULL) {
        *out_value = v;
    }
    return true;
}

static bool match_literal(Parser *p, const char *lit) {
    size_t n = strlen(lit);
    if (p->pos + n > p->len) {
        return false;
    }
    if (strncmp(p->data + p->pos, lit, n) != 0) {
        return false;
    }
    p->pos += n;
    return true;
}

static bool skip_value(Parser *p);

static bool skip_array(Parser *p) {
    if (!expect_char(p, '[')) {
        return false;
    }
    skip_ws(p);
    if (p->pos < p->len && p->data[p->pos] == ']') {
        p->pos++;
        return true;
    }

    while (true) {
        if (!skip_value(p)) {
            return false;
        }
        skip_ws(p);
        if (p->pos >= p->len) {
            set_error(p, "unterminated array");
            return false;
        }
        if (p->data[p->pos] == ',') {
            p->pos++;
            continue;
        }
        if (p->data[p->pos] == ']') {
            p->pos++;
            return true;
        }
        set_error(p, "expected ',' or ']' at byte %zu", p->pos);
        return false;
    }
}

static bool skip_object(Parser *p) {
    if (!expect_char(p, '{')) {
        return false;
    }
    skip_ws(p);
    if (p->pos < p->len && p->data[p->pos] == '}') {
        p->pos++;
        return true;
    }

    while (true) {
        char *key = NULL;
        if (!parse_string(p, &key)) {
            return false;
        }
        free(key);
        if (!expect_char(p, ':')) {
            return false;
        }
        if (!skip_value(p)) {
            return false;
        }
        skip_ws(p);
        if (p->pos >= p->len) {
            set_error(p, "unterminated object");
            return false;
        }
        if (p->data[p->pos] == ',') {
            p->pos++;
            continue;
        }
        if (p->data[p->pos] == '}') {
            p->pos++;
            return true;
        }
        set_error(p, "expected ',' or '}' at byte %zu", p->pos);
        return false;
    }
}

static bool skip_value(Parser *p) {
    skip_ws(p);
    if (p->pos >= p->len) {
        set_error(p, "unexpected end while skipping value");
        return false;
    }
    char c = p->data[p->pos];
    if (c == '{') {
        return skip_object(p);
    }
    if (c == '[') {
        return skip_array(p);
    }
    if (c == '"') {
        char *tmp = NULL;
        if (!parse_string(p, &tmp)) {
            return false;
        }
        free(tmp);
        return true;
    }
    if (c == 't') {
        if (match_literal(p, "true")) {
            return true;
        }
    }
    if (c == 'f') {
        if (match_literal(p, "false")) {
            return true;
        }
    }
    if (c == 'n') {
        if (match_literal(p, "null")) {
            return true;
        }
    }
    return parse_number(p, NULL);
}

static bool parse_number_array(Parser *p, double **out_values, size_t *out_count) {
    if (!expect_char(p, '[')) {
        return false;
    }

    size_t cap = 0;
    size_t count = 0;
    double *vals = NULL;

    skip_ws(p);
    if (p->pos < p->len && p->data[p->pos] == ']') {
        p->pos++;
        *out_values = vals;
        *out_count = 0;
        return true;
    }

    while (true) {
        double v = 0.0;
        if (!parse_number(p, &v)) {
            free(vals);
            return false;
        }
        if (count == cap) {
            size_t new_cap = cap == 0 ? 64 : cap * 2;
            double *new_vals = (double *)realloc(vals, new_cap * sizeof(double));
            if (new_vals == NULL) {
                free(vals);
                set_error(p, "out of memory");
                return false;
            }
            vals = new_vals;
            cap = new_cap;
        }
        vals[count++] = v;

        skip_ws(p);
        if (p->pos >= p->len) {
            free(vals);
            set_error(p, "unterminated number array");
            return false;
        }
        if (p->data[p->pos] == ',') {
            p->pos++;
            continue;
        }
        if (p->data[p->pos] == ']') {
            p->pos++;
            *out_values = vals;
            *out_count = count;
            return true;
        }
        free(vals);
        set_error(p, "expected ',' or ']' at byte %zu", p->pos);
        return false;
    }
}

static bool parse_label_object(Parser *p, LabelVector *out_label) {
    if (!expect_char(p, '{')) {
        return false;
    }

    out_label->name = NULL;
    out_label->vector = NULL;
    out_label->dim = 0;

    skip_ws(p);
    if (p->pos < p->len && p->data[p->pos] == '}') {
        p->pos++;
        set_error(p, "label object missing required fields");
        return false;
    }

    while (true) {
        char *key = NULL;
        if (!parse_string(p, &key)) {
            return false;
        }
        if (!expect_char(p, ':')) {
            free(key);
            return false;
        }

        if (strcmp(key, "name") == 0) {
            free(out_label->name);
            out_label->name = NULL;
            if (!parse_string(p, &out_label->name)) {
                free(key);
                return false;
            }
        } else if (strcmp(key, "vector") == 0) {
            free(out_label->vector);
            out_label->vector = NULL;
            out_label->dim = 0;
            if (!parse_number_array(p, &out_label->vector, &out_label->dim)) {
                free(key);
                return false;
            }
        } else {
            if (!skip_value(p)) {
                free(key);
                return false;
            }
        }
        free(key);

        skip_ws(p);
        if (p->pos >= p->len) {
            set_error(p, "unterminated label object");
            return false;
        }
        if (p->data[p->pos] == ',') {
            p->pos++;
            continue;
        }
        if (p->data[p->pos] == '}') {
            p->pos++;
            break;
        }
        set_error(p, "expected ',' or '}' at byte %zu", p->pos);
        return false;
    }

    if (out_label->name == NULL || out_label->vector == NULL) {
        set_error(p, "label object missing 'name' or 'vector'");
        return false;
    }
    return true;
}

static void free_label_vectors(LabelVector *labels, size_t count) {
    if (labels == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        free(labels[i].name);
        free(labels[i].vector);
    }
    free(labels);
}

static bool parse_labels_array(Parser *p, LabelVector **out_labels, size_t *out_count, size_t *out_dim) {
    if (!expect_char(p, '[')) {
        return false;
    }

    size_t cap = 0;
    size_t count = 0;
    size_t expected_dim = 0;
    LabelVector *labels = NULL;

    skip_ws(p);
    if (p->pos < p->len && p->data[p->pos] == ']') {
        p->pos++;
        *out_labels = labels;
        *out_count = 0;
        *out_dim = 0;
        return true;
    }

    while (true) {
        LabelVector lv = {0};
        if (!parse_label_object(p, &lv)) {
            free_label_vectors(labels, count);
            return false;
        }

        if (count == 0) {
            expected_dim = lv.dim;
        } else if (lv.dim != expected_dim) {
            free(lv.name);
            free(lv.vector);
            free_label_vectors(labels, count);
            set_error(p, "inconsistent vector size in labels array");
            return false;
        }

        if (count == cap) {
            size_t new_cap = cap == 0 ? 32 : cap * 2;
            LabelVector *new_labels = (LabelVector *)realloc(labels, new_cap * sizeof(LabelVector));
            if (new_labels == NULL) {
                free(lv.name);
                free(lv.vector);
                free_label_vectors(labels, count);
                set_error(p, "out of memory");
                return false;
            }
            labels = new_labels;
            cap = new_cap;
        }
        labels[count++] = lv;

        skip_ws(p);
        if (p->pos >= p->len) {
            free_label_vectors(labels, count);
            set_error(p, "unterminated labels array");
            return false;
        }
        if (p->data[p->pos] == ',') {
            p->pos++;
            continue;
        }
        if (p->data[p->pos] == ']') {
            p->pos++;
            *out_labels = labels;
            *out_count = count;
            *out_dim = expected_dim;
            return true;
        }
        free_label_vectors(labels, count);
        set_error(p, "expected ',' or ']' at byte %zu", p->pos);
        return false;
    }
}

static bool parse_root_labels(Parser *p, LabelVector **out_labels, size_t *out_count, size_t *out_dim) {
    if (!expect_char(p, '{')) {
        return false;
    }
    bool found_labels = false;

    while (true) {
        skip_ws(p);
        if (p->pos >= p->len) {
            set_error(p, "unterminated root object");
            return false;
        }
        if (p->data[p->pos] == '}') {
            p->pos++;
            break;
        }

        char *key = NULL;
        if (!parse_string(p, &key)) {
            return false;
        }
        if (!expect_char(p, ':')) {
            free(key);
            return false;
        }

        if (strcmp(key, "labels") == 0) {
            if (found_labels) {
                free(key);
                set_error(p, "duplicate 'labels' field");
                return false;
            }
            if (!parse_labels_array(p, out_labels, out_count, out_dim)) {
                free(key);
                return false;
            }
            found_labels = true;
        } else {
            if (!skip_value(p)) {
                free(key);
                return false;
            }
        }
        free(key);

        skip_ws(p);
        if (p->pos >= p->len) {
            set_error(p, "unterminated root object");
            return false;
        }
        if (p->data[p->pos] == ',') {
            p->pos++;
            continue;
        }
        if (p->data[p->pos] == '}') {
            p->pos++;
            break;
        }
        set_error(p, "expected ',' or '}' at byte %zu", p->pos);
        return false;
    }

    if (!found_labels) {
        set_error(p, "missing 'labels' field");
        return false;
    }
    return true;
}

static bool read_file(const char *path, char **out_data, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        fprintf(stderr, "Cannot seek file: %s\n", path);
        return false;
    }
    long fsize = ftell(f);
    if (fsize < 0) {
        fclose(f);
        fprintf(stderr, "Cannot tell file size: %s\n", path);
        return false;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "Cannot seek file: %s\n", path);
        return false;
    }

    size_t size = (size_t)fsize;
    char *buf = (char *)malloc(size + 1);
    if (buf == NULL) {
        fclose(f);
        fprintf(stderr, "Out of memory while reading: %s\n", path);
        return false;
    }

    size_t got = fread(buf, 1, size, f);
    fclose(f);
    if (got != size) {
        free(buf);
        fprintf(stderr, "Cannot read file: %s\n", path);
        return false;
    }
    buf[size] = '\0';
    *out_data = buf;
    *out_len = size;
    return true;
}

static bool load_labels_from_json(const char *path, LabelVector **out_labels, size_t *out_count, size_t *out_dim) {
    char *json = NULL;
    size_t len = 0;
    if (!read_file(path, &json, &len)) {
        return false;
    }

    Parser p = {
        .data = json,
        .len = len,
        .pos = 0,
        .error = {0},
    };

    bool ok = parse_root_labels(&p, out_labels, out_count, out_dim);
    if (ok) {
        skip_ws(&p);
        if (p.pos != p.len) {
            ok = false;
            set_error(&p, "unexpected trailing characters at byte %zu", p.pos);
        }
    }
    if (!ok) {
        fprintf(stderr, "JSON parse error in %s: %s\n", path, p.error[0] ? p.error : "unknown error");
    }
    free(json);
    return ok;
}

static void write_csv_field(FILE *f, const char *s) {
    bool needs_quote = false;
    for (const char *p = s; *p != '\0'; p++) {
        if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        fputs(s, f);
        return;
    }
    fputc('"', f);
    for (const char *p = s; *p != '\0'; p++) {
        if (*p == '"') {
            fputc('"', f);
        }
        fputc(*p, f);
    }
    fputc('"', f);
}

static double vector_norm(const double *v, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

static double vector_dot(const double *a, const double *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static bool compute_and_write_csv(const char *out_path,
                                  const LabelVector *embeddings,
                                  size_t n_embeddings,
                                  const LabelVector *axes,
                                  size_t n_axes,
                                  size_t dim) {
    FILE *f = fopen(out_path, "w");
    if (f == NULL) {
        fprintf(stderr, "Cannot open output CSV: %s\n", out_path);
        return false;
    }

    double *emb_norms = (double *)malloc(n_embeddings * sizeof(double));
    double *axis_norms = (double *)malloc(n_axes * sizeof(double));
    if (emb_norms == NULL || axis_norms == NULL) {
        fclose(f);
        free(emb_norms);
        free(axis_norms);
        fprintf(stderr, "Out of memory while computing norms\n");
        return false;
    }

    for (size_t i = 0; i < n_embeddings; i++) {
        emb_norms[i] = vector_norm(embeddings[i].vector, dim);
    }
    for (size_t j = 0; j < n_axes; j++) {
        axis_norms[j] = vector_norm(axes[j].vector, dim);
    }

    fprintf(f, "label_name_embeddings,label_name_axis,cosine_value,sqrt(1-cosine^2)\n");
    for (size_t i = 0; i < n_embeddings; i++) {
        for (size_t j = 0; j < n_axes; j++) {
            double cos = 0.0;
            if (emb_norms[i] != 0.0 && axis_norms[j] != 0.0) {
                double dot = vector_dot(embeddings[i].vector, axes[j].vector, dim);
                cos = dot / (emb_norms[i] * axis_norms[j]);
            }
            double sqrt_val = 0.0;
            if (fabs(cos) <= 1.0) {
                double inside = 1.0 - (cos * cos);
                if (inside < 0.0) {
                    inside = 0.0;
                }
                sqrt_val = sqrt(inside);
            }

            write_csv_field(f, embeddings[i].name);
            fputc(',', f);
            write_csv_field(f, axes[j].name);
            fprintf(f, ",%.17g,%.17g\n", cos, sqrt_val);
        }
    }

    fclose(f);
    free(emb_norms);
    free(axis_norms);
    return true;
}

static bool get_executable_dir(char *out_dir, size_t out_size) {
    if (out_size == 0) {
        return false;
    }
    ssize_t n = readlink("/proc/self/exe", out_dir, out_size - 1);
    if (n <= 0 || (size_t)n >= out_size) {
        return false;
    }
    out_dir[n] = '\0';
    char *slash = strrchr(out_dir, '/');
    if (slash == NULL) {
        return false;
    }
    *slash = '\0';
    return true;
}

static bool build_path(char *dst, size_t dst_size, const char *dir, const char *file_name) {
    size_t dir_len = strlen(dir);
    size_t file_len = strlen(file_name);
    size_t needed = dir_len + 1 + file_len + 1;
    if (needed > dst_size) {
        return false;
    }
    memcpy(dst, dir, dir_len);
    dst[dir_len] = '/';
    memcpy(dst + dir_len + 1, file_name, file_len);
    dst[dir_len + 1 + file_len] = '\0';
    return true;
}

int main(int argc, char **argv) {
    char emb_path[PATH_MAX];
    char axis_path[PATH_MAX];
    char out_path[PATH_MAX];

    if (argc == 4) {
        snprintf(emb_path, sizeof(emb_path), "%s", argv[1]);
        snprintf(axis_path, sizeof(axis_path), "%s", argv[2]);
        snprintf(out_path, sizeof(out_path), "%s", argv[3]);
    } else if (argc == 1) {
        char base_dir[PATH_MAX];
        if (!get_executable_dir(base_dir, sizeof(base_dir))) {
            fprintf(stderr, "Cannot resolve executable directory. Provide explicit paths:\n");
            fprintf(stderr, "Usage: %s <embeddings.json> <axis.json> <output.csv>\n", argv[0]);
            return 1;
        }
        if (!build_path(emb_path, sizeof(emb_path), base_dir, "vectors-OrangeIA.json") ||
            !build_path(axis_path, sizeof(axis_path), base_dir, "axis-vectors-OrangeIA.json") ||
            !build_path(out_path, sizeof(out_path), base_dir, "CosineCompute-OrangeIA.csv")) {
            fprintf(stderr, "Resolved path is too long\n");
            return 1;
        }
    } else {
        fprintf(stderr, "Usage: %s [embeddings.json axis.json output.csv]\n", argv[0]);
        return 1;
    }

    LabelVector *embeddings = NULL;
    LabelVector *axes = NULL;
    size_t n_embeddings = 0;
    size_t n_axes = 0;
    size_t emb_dim = 0;
    size_t axis_dim = 0;

    if (!load_labels_from_json(emb_path, &embeddings, &n_embeddings, &emb_dim)) {
        return 1;
    }
    if (!load_labels_from_json(axis_path, &axes, &n_axes, &axis_dim)) {
        free_label_vectors(embeddings, n_embeddings);
        return 1;
    }
    if (n_embeddings == 0 || n_axes == 0) {
        fprintf(stderr, "Input files contain no labels\n");
        free_label_vectors(embeddings, n_embeddings);
        free_label_vectors(axes, n_axes);
        return 1;
    }
    if (emb_dim != axis_dim) {
        fprintf(stderr, "Vector dimension mismatch: embeddings=%zu axis=%zu\n", emb_dim, axis_dim);
        free_label_vectors(embeddings, n_embeddings);
        free_label_vectors(axes, n_axes);
        return 1;
    }

    bool ok = compute_and_write_csv(out_path, embeddings, n_embeddings, axes, n_axes, emb_dim);
    free_label_vectors(embeddings, n_embeddings);
    free_label_vectors(axes, n_axes);

    if (!ok) {
        return 1;
    }

    printf("Cosines generees et enregistrees dans : %s\n", out_path);
    return 0;
}
