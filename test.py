import numpy as np
import matplotlib.pyplot as plt

# Define Gaussian membership function
def gauss_mf(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma)**2)

# Range for input variable
x = np.linspace(0, 1, 500)

# Centers for VL, L, M, H, VH
centers = {
    "VL": 0.0,
    "L": 0.25,
    "M": 0.5,
    "H": 0.75,
    "VH": 1.0
}

sigma = 0.12  # width of Gaussian

plt.figure(figsize=(10, 5))

for label, c in centers.items():
    plt.plot(x, gauss_mf(x, c, sigma), label=label)

plt.xlabel("Input Variable")
plt.ylabel("Membership Degree")
plt.grid(True)

plt.show()
