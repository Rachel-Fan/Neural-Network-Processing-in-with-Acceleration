#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip> // For std::setprecision

// Function to read a matrix from a text file
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

// Function to apply 2D convolution on a matrix
std::vector<std::vector<double>> apply_convolution(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &filter)
{
    int filterSize = filter.size();
    int size = input.size() - filterSize + 1;
    std::vector<std::vector<double>> output(size, std::vector<double>(size));

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            double sum = 0.0;
            for (int fi = 0; fi < filterSize; ++fi)
            {
                for (int fj = 0; fj < filterSize; ++fj)
                {
                    sum += input[i + fi][j + fj] * filter[fi][fj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

int main()
{
    // Define a simple 3x3 filter (e.g., an edge detection filter)
    std::vector<std::vector<double>> filter = {
        {1, 0, -1},
        {0, 0, 0},
        {-1, 0, 1}};

    // File containing matrix data
    std::string filename = "../data/matrix_5000.txt";

    // Read matrix from file
    std::vector<std::vector<double>> matrix = read_matrix_from_file(filename);

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Apply convolution
    std::vector<std::vector<double>> output = apply_convolution(matrix, filter);

    // Stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate processing time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Write processing time to CSV file
    std::ofstream csv_file("../data/cnn_cpu_processing_time.csv");
    csv_file << "Processing Time (ms)\n";
    csv_file << std::setprecision(10) << duration << std::endl;
    csv_file.close();

    // Print result (optional, could be large)
    // for (const auto& row : output) {
    //     for (double val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
