#include <vector>
#include <cuda_runtime.h>
#include "matrixVectorMul.h"

__global__ void matVecMultiplyKernel(double *mat, double *vec, double *res, int rows, int cols)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows)
    {
        double temp = 0;
        for (int j = 0; j < cols; j++)
        {
            temp += mat[index * cols + j] * vec[j];
        }
        res[index] = temp;
    }
}

void matrixVectorMultiplyCUDA(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec, std::vector<double> &res)
{
    // Flatten the matrix and prepare device memory, etc.
    // Kernel launch and memory transfer to/from the device
}
