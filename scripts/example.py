import numpy as np
import matplotlib.pyplot as plt

# Generate a signal with multiple frequencies
fs = 1000  # Sampling frequency (samples per second)
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
frequencies = [50, 150, 300]  # Frequencies in Hz
signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

# Compute FFT
fft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(fft_values), 1/fs)

# Plot the magnitude of the FFT
plt.figure(figsize=(12, 6))
plt.stem(frequencies[:len(frequencies)//2], np.abs(fft_values)[:len(frequencies)//2], 'b', markerfmt=" ", basefmt="-b")
plt.title('FFT of the signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()
