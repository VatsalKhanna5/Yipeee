import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian membership function
def gaussian_mf(x, center, sigma=0.5):
    """
    Gaussian membership function
    μᵢ(x) = exp(-(x-cᵢ)²/σ²)
    """
    return np.exp(-((x - center)**2) / sigma)

# Define input range and centers
x = np.linspace(-3, 3, 1000)
centers = [-2, -1, 0, 1, 2]
labels = ['VL', 'L', 'M', 'H', 'VH']
colors = ['blue', 'green', 'red', 'orange', 'purple']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Gaussian Membership Functions for Fuzzy-Driven HNANC System', fontsize=16, fontweight='bold')

# Plot 1: ΔEerr (Change in Error Energy)
ax1 = axes[0, 0]
for i, (center, label, color) in enumerate(zip(centers, labels, colors)):
    y = gaussian_mf(x, center)
    ax1.plot(x, y, color=color, linewidth=2.5, label=f'{label} (c={center})')

ax1.set_title('Membership Functions for ΔEₑᵣᵣ\n(Change in Error Energy)', fontsize=14, fontweight='bold')
ax1.set_xlabel('ΔEₑᵣᵣ', fontsize=12)
ax1.set_ylabel('Membership Degree μ(x)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_xlim(-3, 3)
ax1.set_ylim(0, 1.1)

# Plot 2: ΔEy (Change in Control Signal Energy)
ax2 = axes[0, 1]
for i, (center, label, color) in enumerate(zip(centers, labels, colors)):
    y = gaussian_mf(x, center)
    ax2.plot(x, y, color=color, linewidth=2.5, label=f'{label} (c={center})')

ax2.set_title('Membership Functions for ΔEᵧ\n(Change in Control Signal Energy)', fontsize=14, fontweight='bold')
ax2.set_xlabel('ΔEᵧ', fontsize=12)
ax2.set_ylabel('Membership Degree μ(x)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_xlim(-3, 3)
ax2.set_ylim(0, 1.1)

# Plot 3: Combined view showing overlap
ax3 = axes[1, 0]
for i, (center, label, color) in enumerate(zip(centers, labels, colors)):
    y = gaussian_mf(x, center)
    ax3.plot(x, y, color=color, linewidth=2.5, label=f'{label}', alpha=0.8)
    ax3.fill_between(x, 0, y, color=color, alpha=0.2)

ax3.set_title('Combined Membership Functions\n(Showing Overlap and Coverage)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Input Variable Value', fontsize=12)
ax3.set_ylabel('Membership Degree μ(x)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')
ax3.set_xlim(-3, 3)
ax3.set_ylim(0, 1.1)

# Plot 4: Individual membership functions separately
ax4 = axes[1, 1]
x_individual = np.linspace(-3, 3, 200)
for i, (center, label, color) in enumerate(zip(centers, labels, colors)):
    y = gaussian_mf(x_individual, center)
    ax4.subplot = plt.subplot(2, 2, 4)
    plt.plot(x_individual, y, color=color, linewidth=3, label=f'{label}: μ(x) = exp(-((x-{center})²/0.5))')

ax4.set_title('Mathematical Representation\nμᵢ(x) = exp(-(x-cᵢ)²/0.5)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Input Value', fontsize=12)
ax4.set_ylabel('Membership Degree', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
ax4.set_xlim(-3, 3)
ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

# Print mathematical details
print("="*60)
print("GAUSSIAN MEMBERSHIP FUNCTION PARAMETERS")
print("="*60)
print(f"Function: μᵢ(x) = exp(-(x-cᵢ)²/σ²)")
print(f"Sigma (σ²): 0.5")
print(f"Input Range: [-3, 3]")
print(f"Centers (cᵢ): {centers}")
print(f"Labels: {labels}")
print("\nMembership Function Equations:")
for center, label in zip(centers, labels):
    print(f"{label}: μ(x) = exp(-((x-({center}))²/0.5))")

# Create a separate detailed plot for paper inclusion
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot with better styling for paper
for i, (center, label, color) in enumerate(zip(centers, labels, colors)):
    y = gaussian_mf(x, center)
    ax.plot(x, y, color=color, linewidth=2.5, label=f'{label}', marker='', markersize=3)

ax.set_title('Gaussian Membership Functions for TSK Fuzzy Inference System', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Input Variable (ΔEₑᵣᵣ, ΔEᵧ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Membership Degree μ(x)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.05)

# Add annotations for key points
ax.annotate('μᵢ(x) = exp(-(x-cᵢ)²/0.5)', xy=(1.5, 0.8), xytext=(1.5, 0.9),
            fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# Add vertical lines at centers
for center in centers:
    ax.axvline(x=center, color='gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("NOTES FOR PAPER:")
print("="*60)
print("1. All membership functions use Gaussian shape with σ² = 0.5")
print("2. Centers are evenly distributed: c ∈ {-2, -1, 0, 1, 2}")
print("3. Input range: [-3, 3] covers 99.7% of normal distribution")
print("4. Linguistic variables: Very Low (VL), Low (L), Medium (M), High (H), Very High (VH)")
print("5. These functions provide smooth transitions and good overlap for fuzzy inference")