#include <random>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include "CSRMatrix.hpp"

CSRMatrix generate_random_csr(int N, double density, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> idx_dist(0, N - 1);

    long long target_nnz = static_cast<long long>(density * N * N);

    // Generate unique (row, col) pairs -> use a set keyed by flattened index
    std::unordered_set<long long> seen;
    seen.reserve(target_nnz * 2);

    std::vector<std::pair<int,int>> coords;
    coords.reserve(target_nnz);

    while (static_cast<long long>(coords.size()) < target_nnz) {
        int r = idx_dist(gen);
        int c = idx_dist(gen);
        long long key = static_cast<long long>(r) * N + c;
        if (seen.insert(key).second) {          // true if newly inserted
            coords.emplace_back(r, c);
        }
    }

    // Sort by row, then col; CSR requirment
    std::sort(coords.begin(), coords.end());

    CSRMatrix mat;
    mat.rows = N; mat.cols = N; mat.nnz = static_cast<int>(coords.size());
    mat.data.resize(mat.nnz);
    mat.indices.resize(mat.nnz);
    mat.indptr.assign(N + 1, 0);

    for (int i = 0; i < mat.nnz; ++i) {
        mat.data[i]    = val_dist(gen);
        mat.indices[i] = coords[i].second;
        mat.indptr[coords[i].first + 1]++;
    }
    // Convert counts to offsets
    for (int r = 0; r < N; ++r) mat.indptr[r + 1] += mat.indptr[r];

    return mat;
}