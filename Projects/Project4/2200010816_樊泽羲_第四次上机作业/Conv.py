import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
M = 500
Q = 200

# Generate sequences x and h
x = np.array([np.sin(n/2) for n in range(1, M)], dtype=float)
h = np.array([np.exp(1/n) for n in range(1, Q)], dtype=float)

# Direct convolution
N = M + Q - 2
y_direct = np.zeros(N, dtype=float)

start_direct = time.time()
for n in range(N):
    # convolution sum for each output index
    for q in range(Q-1):
        idx = n - q
        if 0 <= idx < M-1:
            y_direct[n] += h[q] * x[idx]
time_direct = time.time() - start_direct

# Prepare zero-padding for FFT convolution
def next_pow_two(n):
    return 1 << (n - 1).bit_length()

padded_len = next_pow_two(N)
x_padded = np.pad(x, (0, padded_len - len(x)))
h_padded = np.pad(h, (0, padded_len - len(h)))

# FFT-based convolution
start_fft = time.time()
X = np.fft.fft(x_padded)
H = np.fft.fft(h_padded)
Y = X * H
y_fft_full = np.fft.ifft(Y)
y_fft = y_fft_full[:N].real  # truncate and take real part
time_fft = time.time() - start_fft

# Compute maximum absolute error
max_error = np.max(np.abs(y_direct - y_fft))

# Print performance and error
print(f"Maximum absolute error: {max_error:.3e}")
print(f"Time for direct convolution: {time_direct:.4f} s")
print(f"Time for FFT-based convolution: {time_fft:.4f} s")

# Visualization

# 1. Compare convolution outputs
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(y_direct, label='Direct', color='blue')
plt.title('Direct Convolution Result')
plt.ylabel('y[n]')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(y_fft, label='FFT-based', color='red')
plt.title('FFT-based Convolution Result')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Error between methods
plt.figure(figsize=(10, 3))
plt.plot(np.abs(y_direct - y_fft), color='black')
plt.title('Absolute Error |y_direct - y_fft|')
plt.xlabel('n')
plt.ylabel('Error')
plt.grid(True)
plt.show()
