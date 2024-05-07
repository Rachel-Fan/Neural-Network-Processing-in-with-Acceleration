#ifndef MATRIX_VECTOR_MUL_H
#define MATRIX_VECTOR_MUL_H

#include <vector>

void matrixVectorMultiplyCUDA(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec, std::vector<double> &res);

#endif
