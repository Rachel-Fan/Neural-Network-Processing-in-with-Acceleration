import numpy as np
import time

# Function to read a matrix from a text file
def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        matrix = [list(map(float, line.strip().split())) for line in file]
    return np.array(matrix)

# Function to apply 2D convolution on a matrix
def apply_convolution(input_matrix, filter_matrix):
    filter_size = filter_matrix.shape[0]
    output_size = input_matrix.shape[0] - filter_size + 1
    output_matrix = np.zeros((output_size, output_size))

    # Apply the filter to each position
    for i in range(output_size):
        for j in range(output_size):
            output_matrix[i, j] = np.sum(input_matrix[i:i+filter_size, j:j+filter_size] * filter_matrix)

    return output_matrix

def main():
    filename = "data/matrix_5000.txt"
    filter_matrix = np.array([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1]
    ])

    # Read the input matrix from file
    matrix = read_matrix_from_file(filename)

    # Start the timer
    start_time = time.time()

    # Apply convolution
    output = apply_convolution(matrix, filter_matrix)

    # Stop the timer
    end_time = time.time()

    # Calculate processing time in milliseconds
    duration = (end_time - start_time) * 1000  # convert to milliseconds

    # Output processing time
    print(f"Processing time: {duration:.2f} ms")

    # Save processing time to a CSV file
    with open("data/cnn_python_processing_time.csv", "w") as file:
        file.write("Processing Time (ms)\n")
        file.write(f"{duration:.10f}\n")

if __name__ == "__main__":
    main()
