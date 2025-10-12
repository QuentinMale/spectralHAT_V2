import numpy as np
import matplotlib.pyplot as plt
import glob, re, os

# ---- gather files and their times ----
spec_dir = "SPECTRA"
files = sorted(glob.glob(os.path.join(spec_dir, "spectrum_*.txt")))

times = []
for fname in files:
    with open(fname, "r") as f:
        header = f.readline()  # first line contains "# Spectrum at t=..."
    m = re.search(r"t\s*=\s*([0-9.Ee+-]+)", header)
    t = float(m.group(1)) if m else np.nan
    times.append(t)
times = np.array(times)

# ---- plot selection + KCM theory overlays ----
plt.figure(figsize=(9,6))
Emax = 0.0
knyq = 0.0

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, fname in enumerate(files):
    # fname = files[i]
    t_sel = times[i]
    data = np.loadtxt(fname)
    k, Ek = data[:,0], data[:,1]

    # Update axes ranges
    Emax = max(Emax, float(Ek.max()))
    knyq = max(knyq, float(k.max()))

    # plot measured spectrum (skip k=0)
    plt.loglog(k[1:], Ek[1:], label=f"DNS t≈{t_sel:.3f}", lw=2, color='k', alpha=0.1)

knyq /= 2
# ---- cosmetics & limits ----
plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy spectrum $E(k)$")
if Emax > 0:
    plt.ylim(1e-4*Emax, 2.0*Emax)
if knyq > 1:
    plt.xlim(1.0, knyq)

plt.title("Energy spectra")
plt.grid(True, which="both", ls=":", alpha=0.6)
# plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("spectra.pdf", dpi=200)
# plt.show()
