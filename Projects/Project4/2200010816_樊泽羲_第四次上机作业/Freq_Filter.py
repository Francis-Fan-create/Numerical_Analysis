import numpy as np
import matplotlib.pyplot as plt

# Define the signal function f(t)
def f(t):
    return np.exp(-t**2 / 10) * (np.sin(2*t) + 2*np.cos(4*t) + 0.4*np.sin(t)*np.sin(50*t))

# Parameters
N = 256
k_indices = np.arange(0, N+1) # Renamed from k to k_indices for clarity in plots
t_k = 2 * np.pi * k_indices / N

# Sample the original signal
y_k_original = f(t_k) # Renamed from y_k to y_k_original

# Compute next power of two for zero-padding
def next_pow_two(x):
    return 1 << (x - 1).bit_length()

pad_length = next_pow_two(len(y_k_original))
y_k_padded_original = np.concatenate([y_k_original, np.zeros(pad_length - len(y_k_original))])

# Compute FFT of the original signal
Y_k_original_fft = np.fft.fft(y_k_padded_original)

# Define m values to test
m_values = [5, 10, 20, 40, 80, 100]

all_y_k_prime_filtered = []
all_Y_k_fft_filtered = []
errors_l2_norm = []

# Loop over different m values
for m_val in m_values:
    # Filter: zero out high-frequency components
    Y_k_fft_current_filtered = Y_k_original_fft.copy()
    
    # Apply the original filtering algorithm
    # Y_k_filtered[m+1:-m] = 0
    # Ensure m_val is not so large that -m_val becomes non-negative or wraps around incorrectly.
    # The slice Y_k_fft_current_filtered[m_val+1 : -m_val] handles this.
    # If m_val is 0, -m_val is 0, so Y_k_fft_current_filtered[1:0] is an empty slice (no filtering).
    # If m_val is large, e.g., m_val >= pad_length/2, then -m_val will select a small portion from the end.
    if m_val > 0: # Only apply if m_val is positive
        if m_val < pad_length // 2: # Standard case
             Y_k_fft_current_filtered[m_val+1 : pad_length-m_val] = 0
        else: # m_val is large, potentially filtering nothing or very little.
              # This case means we keep all or almost all frequencies.
              # For m_val >= pad_length/2, m_val+1 could be >= pad_length-m_val.
              # The original Y_k_filtered[m+1:-m] = 0 would mean:
              # e.g. m=128, pad_length=257. Y[129 : -128] which is Y[129 : 257-128=129]. Empty slice.
              # Let's ensure the slice indices are valid and reflect the intent.
              # If m_val is very large, it means we are keeping more high frequencies.
              # The original Y_k_filtered[m+1:-m] = 0 is equivalent to:
              # Y_k_filtered[m+1 : len(Y_k_filtered)-m] = 0
              # This is the most direct interpretation of the original algorithm.
              idx_start = m_val + 1
              idx_end = len(Y_k_fft_current_filtered) - m_val
              if idx_start < idx_end : # Ensure the slice is valid
                  Y_k_fft_current_filtered[idx_start:idx_end] = 0
    # If m_val is 0, no filtering is done based on Y_k_filtered[m+1:-m] = 0 logic, as Y_k_filtered[1:0] is empty.

    all_Y_k_fft_filtered.append(Y_k_fft_current_filtered)

    # Inverse FFT to obtain the filtered signal
    y_k_prime_padded_current = np.fft.ifft(Y_k_fft_current_filtered)
    # Truncate to original signal length (N+1 points) and take real part
    y_k_prime_current = y_k_prime_padded_current[:len(y_k_original)].real
    all_y_k_prime_filtered.append(y_k_prime_current)

    # Calculate L2 norm of the difference
    error = np.linalg.norm(y_k_original - y_k_prime_current)
    errors_l2_norm.append(error)

# Plotting
plt.style.use('seaborn-v0_8-paper') # Using a style often seen in papers for aesthetics

# Plot 1: Original and Filtered Signals for different m
plt.figure(figsize=(12, 7))
plt.plot(k_indices, y_k_original, label='Original $y_k$', color='black', linewidth=2, alpha=0.8)
colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
for i, m_val in enumerate(m_values):
    plt.plot(k_indices, all_y_k_prime_filtered[i], label=f'Filtered $y_k\'$ (m={m_val})', linestyle='--', color=colors[i], alpha=0.85)
plt.title('Original Signal vs. Filtered Signals for Varying Cutoff $m$', fontsize=16)
plt.xlabel('Sample Index $k$', fontsize=14)
plt.ylabel('Signal Amplitude $y_k$', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot 2: Absolute Difference between original and filtered signals for different m
plt.figure(figsize=(12, 7))
for i, m_val in enumerate(m_values):
    plt.plot(k_indices, np.abs(y_k_original - all_y_k_prime_filtered[i]), label=f'$|y_k - y_k\'|$ (m={m_val})', color=colors[i], alpha=0.85)
plt.title('Absolute Difference $|y_k - y_k\'|$ for Varying Cutoff $m$', fontsize=16)
plt.xlabel('Sample Index $k$', fontsize=14)
plt.ylabel('Absolute Difference', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Calculate frequencies for spectrum plot
# Using FFT indices for the x-axis as it's common for discrete spectra.
# Only positive frequencies are typically shown (up to Nyquist).
freq_indices_plot = np.arange(pad_length // 2)


# Plot 3: L2 Norm of Error vs. m
plt.figure(figsize=(10, 6))
plt.plot(m_values, errors_l2_norm, marker='o', linestyle='-', color='crimson', markersize=8)
plt.title('L2 Norm of Difference $||y_k - y_k\'||_2$ vs. Cutoff $m$', fontsize=16)
plt.xlabel('Cutoff Parameter $m$', fontsize=14)
plt.ylabel('L2 Norm of Difference', fontsize=14)
plt.xticks(m_values) # Ensure all m_values are shown as ticks
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()