#ifndef COSINE_H
#define COSINE_H

#include <stddef.h>

#ifndef DATASET_PATH_MAX
#define DATASET_PATH_MAX 4096
#endif

typedef struct {
    size_t N;
    size_t M;
    size_t D;
    char e_path[DATASET_PATH_MAX];
    char a_path[DATASET_PATH_MAX];
    char dataset_sha256[65];
    double *E;
    double *A;
} Dataset;

int load_dataset(const char *metadata_path, Dataset *out, char *err, size_t err_sz);
void free_dataset(Dataset *ds);
double cosine_all_pairs_checksum(const Dataset *ds);

#endif
