import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_graphs(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Filter the DataFrame for Multiplication and Addition operations
    df_mul = df[df['Operation'] == 'Multiplication']
    df_add = df[df['Operation'] == 'Addition']

    # Set up the plot - create two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(len(df_mul['MatrixSize'].unique()))

    # Plot for Multiplication
    for i, acc in enumerate(df_mul['Library'].unique()):
        df_acc_mul = df_mul[df_mul['Library'] == acc]
        bars = axes[0].bar(index + i * bar_width, df_acc_mul['Time_ms'], width=bar_width, label=acc)
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate('{}'.format(height),
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

    axes[0].set_title('Performance Comparison - Multiplication')
    axes[0].set_xlabel('Matrix Size')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_xticks(index + bar_width / len(df_mul['Library'].unique()) / 2)
    axes[0].set_xticklabels(df_mul['MatrixSize'].unique())
    axes[0].legend()

    # Plot for Addition
    for i, acc in enumerate(df_add['Library'].unique()):
        df_acc_add = df_add[df_add['Library'] == acc]
        bars = axes[1].bar(index + i * bar_width, df_acc_add['Time_ms'], width=bar_width, label=acc)
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate('{}'.format(height),
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')


    axes[1].set_title('Performance Comparison - Addition')
    axes[1].set_xlabel('Matrix Size')
    axes[1].set_ylabel('Time (ms)')
    axes[1].set_xticks(index + bar_width / len(df_add['Library'].unique()))
    axes[1].set_xticklabels(df_add['MatrixSize'].unique())
    axes[1].legend()

    # Adjust layout and show/save the plots
    plt.tight_layout()
    plt.show()
    # Optionally save the figure
    plt.savefig('data/Step1_comparison_graphs.png')

# Path to the CSV file
csv_file_path = 'data/performance_data_cpu_20240506_BLAS_MKL.csv'  # Modify this path to your actual CSV file path
plot_comparison_graphs(csv_file_path)
