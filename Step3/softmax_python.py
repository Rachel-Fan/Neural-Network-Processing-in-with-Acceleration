import numpy as np
import time

# Softmax function for a vector
def softmax(vector):
    e_vector = np.exp(vector - np.max(vector))  # subtract max for numerical stability
    return e_vector / e_vector.sum()

# Function to read a matrix from a text file
def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        matrix = []
        for line in file:
            row = list(map(float, line.strip().split()))
            matrix.append(row)
    return np.array(matrix)

# Main function to process the matrix
def main():
    filename = "data/matrix_5000.txt"  # Adjust the path as needed

    # Read matrix from file
    matrix = read_matrix_from_file(filename)

    # Start the timer
    start_time = time.time()

    # Apply softmax to each row of the matrix
    softmax_matrix = np.apply_along_axis(softmax, 1, matrix)

    # Stop the timer
    end_time = time.time()

    # Calculate processing time in milliseconds
    duration = (end_time - start_time) * 1000  # Convert seconds to milliseconds

    # Output results and processing time
    print("Processing time: {:.2f} ms".format(duration))

    # Save processing time to a CSV file
    with open("data/softmax_python_processing_time.csv", "w") as file:
        file.write("Processing Time (ms)\n")
        file.write(f"{duration:.10f}\n")


if __name__ == "__main__":
    main()
