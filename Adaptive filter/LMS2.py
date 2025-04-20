import numpy as np
import matplotlib.pyplot as plt

# System parameters
np.random.seed(0)
filter_order = 4  # Match this with the true system order
mu = 0.01
num_samples = 500

# True system (length = filter_order)
h_true = np.array([0.1, 0.3, -0.4, 0.2])

# Generate input signal and desired output
x = np.random.randn(num_samples)
d = np.convolve(x, h_true, mode='full')[:num_samples] + 0.01 * np.random.randn(num_samples)

# Initialize LMS
w = np.zeros(filter_order)
W_history = np.zeros((num_samples, filter_order))  # For weight evolution
error = np.zeros(num_samples)

# LMS Algorithm
for n in range(filter_order, num_samples):
    x_vec = x[n-filter_order:n][::-1]
    y = np.dot(w, x_vec)
    e = d[n] - y
    w = w + mu * e * x_vec
    error[n] = e ** 2
    W_history[n] = w

# Smooth MSE using moving average
window = 20
mse_smooth = np.convolve(error, np.ones(window)/window, mode='valid')

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 7))

# 1. Smoothed MSE
axs[0].plot(mse_smooth)
axs[0].set_title('Smoothed LMS MSE Convergence')
axs[0].set_ylabel('Mean Squared Error')
axs[0].set_xlabel('Iteration')

# 2. Weight evolution
for i in range(filter_order):
    axs[1].plot(W_history[:, i], label=f'w[{i}]')
axs[1].set_title('LMS Weight Evolution Over Time')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Weight Value')
axs[1].legend()

plt.tight_layout()
plt.show()
