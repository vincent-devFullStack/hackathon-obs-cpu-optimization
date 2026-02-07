#ifndef COSINE_HPP
#define COSINE_HPP

#include <cstddef>
#include <string>
#include <vector>

struct Dataset {
    std::size_t N;
    std::size_t M;
    std::size_t D;
    std::string e_path;
    std::string a_path;
    std::string dataset_sha256;
    std::vector<double> E;
    std::vector<double> A;
};

bool load_dataset(const std::string &metadata_path, Dataset &out, std::string &err);
double cosine_all_pairs_checksum(const Dataset &ds);

#endif
