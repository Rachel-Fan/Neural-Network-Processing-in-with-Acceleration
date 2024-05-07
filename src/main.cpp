#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip> // For std::put_time
#include <sstream> // For std::stringstream
#include <ctime>   // For std::time_t and std::tm

// Forward declarations
void matrixVectorMultiplyOpenBLAS(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec, std::vector<double> &res);
void matrixVectorMultiplyMKL(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec, std::vector<double> &res);

void vectorAdditionOpenBLAS(const std::vector<double> &vecA, const std::vector<double> &vecB, std::vector<double> &result);
void vectorAdditionMKL(const std::vector<double> &vecA, const std::vector<double> &vecB, std::vector<double> &result);

// Function Definition for reading matrix and vector
void readMatrixAndVector(const std::string &matrixFile, const std::string &vectorFile, std::vector<std::vector<double>> &matrix, std::vector<double> &vector, int size)
{
    std::ifstream matFile(matrixFile);
    std::ifstream vecFile(vectorFile);

    if (!matFile || !vecFile)
    {
        std::cerr << "Cannot open the files: " << matrixFile << " or " << vectorFile << std::endl;
        return;
    }

    matrix.resize(size, std::vector<double>(size));
    vector.resize(size);
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            matFile >> matrix[i][j];
        }
        vecFile >> vector[i];
    }
    matFile.close();
    vecFile.close();
}

void writePerformanceData(const std::string &library, const std::string &operation, int size, double elapsed_ms, const std::string &suffix)
{
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm *ptm = std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(ptm, "%Y%m%d");
    std::string filename = "../data/performance_data_cpu_" + ss.str() + suffix + ".csv";

    std::ofstream file(filename, std::ios::app);
    file << library << "," << operation << "," << size << "," << elapsed_ms << std::endl;
    file.close();
}

void saveResultsToFile(const std::string &filename, const std::vector<double> &results)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    for (const auto &value : results)
    {
        file << value << std::endl;
    }
    file.close();
}

void runPerformanceTest(const std::string &library, void (*multiplyFunc)(const std::vector<std::vector<double>> &, const std::vector<double> &, std::vector<double> &), void (*addFunc)(const std::vector<double> &, const std::vector<double> &, std::vector<double> &), const std::vector<int> &sizes)
{
    for (int size : sizes)
    {
        std::string basePath = "../data/";
        std::string matrixPath = basePath + "matrix_" + std::to_string(size) + ".txt";
        std::string vectorPath = basePath + "vector_" + std::to_string(size) + ".txt";

        std::vector<std::vector<double>> matrix;
        std::vector<double> vector, vectorAddend(size, 1.0);
        readMatrixAndVector(matrixPath, vectorPath, matrix, vector, size);

        std::vector<double> resultMultiplication(size, 0.0);
        auto startMul = std::chrono::high_resolution_clock::now();
        multiplyFunc(matrix, vector, resultMultiplication);
        auto endMul = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsedMul = endMul - startMul;
        writePerformanceData(library, "Multiplication", size, elapsedMul.count(), "_BLAS_MKL");
        saveResultsToFile("../data/multiplications_CPU_" + library + "_" + std::to_string(size) + ".txt", resultMultiplication);

        std::vector<double> resultAddition(size, 0.0); // Correct initialization
        auto startAdd = std::chrono::high_resolution_clock::now();
        addFunc(vector, vectorAddend, resultAddition);
        auto endAdd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsedAdd = endAdd - startAdd;
        writePerformanceData(library, "Addition", size, elapsedAdd.count(), "_BLAS_MKL");
        saveResultsToFile("../data/additions_CPU_" + library + "_" + std::to_string(size) + ".txt", resultAddition);
    }
}

int main()
{
    std::vector<int> sizes = {1000, 5000, 10000}; // Sizes to test

    // Initialize the CSV file with headers for multiplication and addition
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm *ptm = std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(ptm, "%Y%m%d");
    std::string filenameMul = "../data/performance_data_cpu_" + ss.str() + "_BLAS_MKL.csv";
    std::ofstream fileMul(filenameMul, std::ofstream::out);
    fileMul << "Library,Operation,MatrixSize,Time_ms\n";
    fileMul.close();

    // Run performance tests for each library and operation
    runPerformanceTest("OpenBLAS", matrixVectorMultiplyOpenBLAS, vectorAdditionOpenBLAS, sizes);
    runPerformanceTest("MKL", matrixVectorMultiplyMKL, vectorAdditionMKL, sizes);

    return 0;
}
