#include <vector>
#include <iostream>
#include <mkl.h>

// Function to perform matrix-vector multiplication using Intel MKL
void matrixVectorMultiplyMKL(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec, std::vector<double> &res)
{
    if (mat.empty() || mat[0].empty())
        throw std::runtime_error("Matrix is empty.");
    int rows = mat.size();
    int cols = mat[0].size();
    for (const auto &row : mat)
    {
        if (row.size() != cols)
            throw std::runtime_error("Matrix is not rectangular.");
    }
    if (vec.size() != cols)
        throw std::runtime_error("Vector size does not match the number of matrix columns.");

    res.resize(rows); // Ensure result vector is properly sized.

    std::vector<double> flat_matrix(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat_matrix[j * rows + i] = mat[i][j];

    // Call the MKL matrix-vector multiplication function
    cblas_dgemv(CblasColMajor, CblasNoTrans, rows, cols, 1.0, flat_matrix.data(), rows, vec.data(), 1, 0.0, res.data(), 1);
    std::cout << "Intel MKL matrix-vector multiplication completed." << std::endl;
}

// Function to perform vector addition using Intel MKL
void vectorAdditionMKL(const std::vector<double> &vecA, const std::vector<double> &vecB, std::vector<double> &result)
{
    if (vecA.size() != vecB.size())
        throw std::invalid_argument("Vectors must be of the same size.");

    int size = vecA.size();
    result.resize(size); // Resize result to match the input size

    // Copy vecA into result since cblas_daxpy computes y := alpha * x + y
    std::copy(vecA.begin(), vecA.end(), result.begin());

    // Perform the addition result = 1.0 * vecB + result
    cblas_daxpy(size, 1.0, vecB.data(), 1, result.data(), 1);

    std::cout << "Intel MKL vector addition completed." << std::endl;
}