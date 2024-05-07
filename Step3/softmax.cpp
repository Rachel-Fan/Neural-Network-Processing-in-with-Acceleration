#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip> // For std::setprecision

// Softmax function for a vector
std::vector<double> softmax(const std::vector<double> &input)
{
    std::vector<double> result(input.size());
    double sum = 0.0;

    // Calculate exponentials of input elements and sum
    for (size_t i = 0; i < input.size(); ++i)
    {
        result[i] = exp(input[i]);
        sum += result[i];
    }

    // Normalize by the sum
    for (double &val : result)
    {
        val /= sum;
    }

    return result;
}

// Softmax function for a matrix, processing row-wise
std::vector<std::vector<double>> softmax_matrix(const std::vector<std::vector<double>> &matrix)
{
    std::vector<std::vector<double>> result(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        result[i] = softmax(matrix[i]);
    }
    return result;
}

// Function to read input from a text file into a matrix
std::vector<std::vector<double>> read_matrix_from_file(const std::string &filename)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val)
        {
            row.push_back(val);
        }
        matrix.push_back(row);
    }

    infile.close();
    return matrix;
}

int main()
{
    // File containing matrix data
    std::string filename = "../data/matrix_5000.txt";

    // Read matrix from file
    std::vector<std::vector<double>> matrix = read_matrix_from_file(filename);

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate softmax for each row of the matrix
    std::vector<std::vector<double>> output = softmax_matrix(matrix);

    // Stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate processing time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Write processing time to CSV file
    std::ofstream csv_file("../data/softmax_cpu_processing_time.csv");
    csv_file << "Processing Time (ms)\n";
    // Set precision to output more digits
    csv_file << std::setprecision(10) << duration << std::endl;
    csv_file.close();

    // Print result
    std::cout << "Processing time: " << duration << " ms" << std::endl;

    return 0;
}
