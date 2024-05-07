import numpy as np
import matplotlib.pyplot as plt

def generate_fft_input(size, frequencies, amplitudes, sample_rate, noise_level=0.0):
    """
    Generates a dataset for FFT with multiple frequency components and optional noise.
    """
    t = np.arange(size) / sample_rate  # Time vector
    signal = np.zeros(size)
    for frequency, amplitude in zip(frequencies, amplitudes):
        signal += amplitude * np.cos(2 * np.pi * frequency * t)  # Sum of cosine waves

    # Add Gaussian noise
    if noise_level > 0:
        signal += np.random.normal(scale=noise_level, size=size)

    complex_signal = signal + 0j  # Convert to complex numbers by adding imaginary part as 0
    return complex_signal

def save_data_to_file(data, filename):
    """
    Saves complex data to a text file.
    """
    np.savetxt(filename, np.column_stack((data.real, data.imag)), fmt=['%f', '%f'])

def plot_signal(data):
    """
    Plots the real part of the complex signal.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(data.real, label='Real part')
    plt.title('Input Signal for FFT')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Adjusted Parameters for a Larger Dataset
SIZE = 1024  # 1,048,576 samples, significantly larger than before
FREQUENCIES = [50, 150, 300, 600, 1200]  # More varied frequencies
AMPLITUDES = [1.0, 0.8, 0.6, 0.4, 0.2]  # Corresponding amplitudes
SAMPLE_RATE = 48000  # Higher sample rate
NOISE_LEVEL = 0.1  # Slightly higher noise level

# Generate the FFT input data
fft_input = generate_fft_input(SIZE, FREQUENCIES, AMPLITUDES, SAMPLE_RATE, NOISE_LEVEL)

# Save the data to a text file
save_data_to_file(fft_input, 'fft_small_input_data.txt')

# Optionally, plot a small part of the signal to keep it manageable
plot_signal(fft_input[:1024])  # Only plot the first 1024 samples
