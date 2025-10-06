import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Load both datasets
# ---------------------
# File paths (edit these to your actual file names)
ref_file = "refData"
my_file  = "temporals.txt"

# Load reference (skip comment lines)
ref = np.loadtxt(ref_file, comments="#")
my  = np.loadtxt(my_file,  comments="#")

# Unpack columns
t_ref, E_ref, eps_ref, Z_ref = ref.T[:4]
t_my,  E_my,  eps_my,  Z_my  = my.T[:4]

# ---------------------
# Plot setup
# ---------------------
plt.rcParams.update({
    "font.size": 12,
    "lines.linewidth": 1.8,
    "figure.figsize": (8, 10)
})

fig, axes = plt.subplots(3, 1, sharex=True)

# ---- Energy ----
axes[0].plot(t_ref, E_ref, '-', color='C0', label='Ref DNS (512^3)')
axes[0].plot(t_my,  E_my,  'o', mfc='none', color='C0', label='spectralHAT')
axes[0].set_ylabel('Kinetic energy')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='best', frameon=False)

# ---- Dissipation ----
axes[1].plot(t_ref, eps_ref, '-', color='C1', label='Ref DNS (512³)')
axes[1].plot(t_my,  eps_my,  'o', mfc='none', color='C1', label='spectralHAT')
axes[1].set_ylabel('Dissipation $\\varepsilon$')
axes[1].grid(True, alpha=0.3)

# ---- Enstrophy ----
axes[2].plot(t_ref, Z_ref, '-', color='C2', label='Ref DNS (512³)')
axes[2].plot(t_my,  Z_my,  'o', mfc='none', color='C2', label='spectralHAT')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Enstrophy $\\mathcal{Z}$')
axes[2].grid(True, alpha=0.3)

# ---- Titles & layout ----
fig.suptitle('Taylor-Green Vortex: Comparison with Reference DNS', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig('output.pdf')
