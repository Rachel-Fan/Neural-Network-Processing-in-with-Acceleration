import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('data')

def read_data_and_plot(ax, file_cpu, file_python, operation_name):
    # Read data, skipping the first row (header assumed)
    df_cpu = pd.read_csv(file_cpu, header=0)
    df_python = pd.read_csv(file_python, header=0)

    # Adjust the column name if needed after verifying CSV headers
    time_column = 'Processing Time (ms)'  # Replace 'Time_ms' with the actual column name if different

    # Create DataFrame for easier plotting
    data = {
        'Library': ['C++ CPU', 'Python'],
        'Time_ms': [df_cpu[time_column].mean(), df_python[time_column].mean()]
    }
    df = pd.DataFrame(data)

    # Create bar chart
    bars = ax.bar(df['Library'], df['Time_ms'], color=['blue', 'green'])

    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

    ax.set_ylabel('Processing Time (ms)')
    ax.set_title(f'Performance Comparison - {operation_name}')

def main():
    # Create a figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Softmax comparison
    read_data_and_plot(ax1, 'softmax_cpu_processing_time.csv', 'softmax_python_processing_time.csv', 'Softmax Operation')

    # CNN comparison
    read_data_and_plot(ax2, 'cnn_cpu_processing_time.csv', 'cnn_python_processing_time.csv', 'CNN Operation')

    plt.tight_layout()
    plt.show()

    # Optionally save the figure
    plt.savefig('Step3_comparison_graphs.png')

if __name__ == "__main__":
    main()
