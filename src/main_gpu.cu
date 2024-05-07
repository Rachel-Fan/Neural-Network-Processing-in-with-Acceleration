#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMultiplyKernel(double *d_mat, double *d_vec, double *d_res, int rows, int cols)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < rows)
    {
        double sum = 0;
        for (int i = 0; i < cols; i++)
        {
            sum += d_mat[idx * cols + i] * d_vec[i];
        }
        d_res[idx] = sum;
    }
}

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const double *d_a, const double *d_b, double *d_result, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        d_result[idx] = d_a[idx] + d_b[idx];
    }
}

void readMatrixAndVector(const std::string &matrixFile, const std::string &vectorFile, std::vector<double> &flat_matrix, std::vector<double> &vector, int size)
{
    std::ifstream matFile(matrixFile);
    std::ifstream vecFile(vectorFile);
    if (!matFile.is_open() || !vecFile.is_open())
    {
        std::cerr << "Failed to open matrix or vector file.\n";
        return;
    }

    double value;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (matFile >> value)
            {
                flat_matrix[i * size + j] = value;
            }
        }
    }

    for (int i = 0; i < size; i++)
    {
        if (vecFile >> value)
        {
            vector[i] = value;
        }
    }

    matFile.close();
    vecFile.close();
}

// Function to save results to a file
void saveResults(const std::string &filename, const std::vector<double> &res)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    for (double value : res)
    {
        file << value << std::endl;
    }
    file.close();
}

void writePerformanceData(const std::string &operation, int size, double elapsed_ms)
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);

    std::stringstream ss;
    ss << std::put_time(&now_tm, "%Y%m%d");
    std::string filename = "../data/performance_data_gpu_" + ss.str() + ".csv";

    // Read current file content to determine if it's empty
    std::ifstream read_file(filename);
    bool is_empty = read_file.peek() == std::ifstream::traits_type::eof();
    read_file.close();

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // If file is empty, write header
    if (is_empty)
    {
        file << "Acceleration,Operation,MatrixSize,Time_ms\n";
    }

    file << "CUDA" << "," << operation << "," << size << "," << elapsed_ms << std::endl;
    file.close();
}

// Function to perform matrix-vector multiplication and vector addition
void processOperations(const std::string &matrixPath, const std::string &vectorPath, int size)
{
    std::vector<double> flat_matrix(size * size), vector(size); // Vectors to hold the flat matrix and the vector

    // Read the matrix and vector from the file
    readMatrixAndVector(matrixPath, vectorPath, flat_matrix, vector, size);

    double *d_mat, *d_vec, *d_res;
    cudaMalloc(&d_mat, size * size * sizeof(double));
    cudaMalloc(&d_vec, size * sizeof(double));
    cudaMalloc(&d_res, size * sizeof(double));

    cudaMemcpy(d_mat, flat_matrix.data(), size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vector.data(), size * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Matrix-vector multiplication
    auto startMul = std::chrono::high_resolution_clock::now();
    matrixVectorMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_vec, d_res, size, size);
    cudaDeviceSynchronize();
    auto endMul = std::chrono::high_resolution_clock::now();

    std::vector<double> result(size);
    cudaMemcpy(result.data(), d_res, size * sizeof(double), cudaMemcpyDeviceToHost);
    saveResults("../data/multiplication_GPU_" + std::to_string(size) + ".txt", result);
    writePerformanceData("Multiplication", size, std::chrono::duration<double, std::milli>(endMul - startMul).count());

    // Vector addition (example with the same vector for simplicity)
    auto startAdd = std::chrono::high_resolution_clock::now();
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_vec, d_res, size); // Adding vector to itself
    cudaDeviceSynchronize();
    auto endAdd = std::chrono::high_resolution_clock::now();

    cudaMemcpy(result.data(), d_res, size * sizeof(double), cudaMemcpyDeviceToHost);
    saveResults("../data/addition_GPU_" + std::to_string(size) + ".txt", result);
    writePerformanceData("Addition", size, std::chrono::duration<double, std::milli>(endAdd - startAdd).count());

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);
}

int main()
{
    std::vector<int> sizes = {1000, 5000, 10000};
    for (int size : sizes)
    {
        std::string matrixPath = "../data/matrix_" + std::to_string(size) + ".txt";
        std::string vectorPath = "../data/vector_" + std::to_string(size) + ".txt";
        processOperations(matrixPath, vectorPath, size);
    }
    return 0;
}
