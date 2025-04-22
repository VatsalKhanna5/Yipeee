import numpy as np
import matplotlib.pyplot as plt

# LMS Parameters
N = 100
mu = 0.1
filter_order = 5

# Signals
x = np.random.randn(N)
unknown_system = np.array([0.1, 0.3, -0.4, 0.2, 0.1])
d = np.convolve(x, unknown_system, mode='full')[:N]
d += 0.05 * np.random.randn(N)

# LMS Variables
w = np.zeros(filter_order)
y = np.zeros(N)
e = np.zeros(N)

for n in range(filter_order, N):
    x_n = x[n-filter_order:n][::-1]
    y[n] = np.dot(w, x_n)
    e[n] = d[n] - y[n]
    w = w + 2 * mu * e[n] * x_n

# Plot Results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(e**2)
plt.title('LMS Error Convergence (dB)')
plt.xlabel('Iteration')
plt.ylabel('Error (dB)')
plt.grid()

plt.subplot(2, 1, 2)
plt.stem(unknown_system, linefmt='r-', markerfmt='ro', basefmt='r-', label='True System')
plt.stem(w, linefmt='g--', markerfmt='go', basefmt='g--', label='Estimated Weights')
plt.title('True vs LMS Estimated Weights')
plt.xlabel('Weight Index')
plt.ylabel('Value')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
