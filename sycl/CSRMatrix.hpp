#ifndef CSRMAT_H
#define CSRMAT_H

#include <vector>

struct CSRMatrix {
    std::vector<float> data;
    std::vector<int> indices;
    std::vector<int> indptr;
    
    int rows, cols, nnz;
};

CSRMatrix generate_random_csr(int N, double density, unsigned seed) {}

#endif
