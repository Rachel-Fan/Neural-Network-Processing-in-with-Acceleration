import pandas as pd
import matplotlib.pyplot as plt

def read_and_prepare_data():
    # Read CPU and GPU performance data
    df_cpu = pd.read_csv("data/performance_data_cpu_20240506_BLAS_MKL.csv")
    
    # Read GPU performance data, skipping the first row (header)
    df_gpu = pd.read_csv("data/performance_data_gpu_20240506.csv", header=0)

    # Combine the two DataFrames
    df = pd.concat([df_cpu, df_gpu], ignore_index=True)
    
    return df


def create_comparison_graph(df):
    operations = ['Multiplication', 'Addition']
    matrix_sizes = [1000, 5000, 10000]

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    # Loop over each subplot to create a plot
    for idx, (op, size) in enumerate([(op, size) for op in operations for size in matrix_sizes]):
        ax = axes[idx]
        df_op_size = df[(df['Operation'] == op) & (df['MatrixSize'] == size)]

        # Collecting unique libraries to set x positions
        libraries = df_op_size['Library'].unique()
        x_pos = range(len(libraries))

        # Bar width configuration
        bar_width = 0.35
        
        # Plotting each library's performance
        for pos, lib in enumerate(libraries):
            df_lib = df_op_size[df_op_size['Library'] == lib]
            ax.bar(pos, df_lib['Time_ms'], width=bar_width, label=lib)
        
        ax.set_title(f'Performance Comparison - {op}, {size}')
        ax.set_xlabel('Library')
        ax.set_ylabel('Time (ms)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(libraries)  # Setting x labels as library names
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show plot before saving for debugging
    plt.show()
    # Save the figure to a file
    plt.savefig('data/Step1_2_performance_comparison.png')

def main():
    df = read_and_prepare_data()
    create_comparison_graph(df)

if __name__ == "__main__":
    main()
