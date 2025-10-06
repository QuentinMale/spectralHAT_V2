import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# Directory containing the spectra
spec_dir = "SPECTRA"
files = sorted(glob.glob(os.path.join(spec_dir, "spectrum_*.txt")))

plt.figure(figsize=(7,5))

Emax = 0.0
kmax = 0.0

for fname in files[::10]:
    # --- Extract time from the header line ---
    with open(fname, "r") as f:
        header = f.readline()
    m = re.search(r"t\s*=\s*([0-9.Ee+-]+)", header)
    t = float(m.group(1)) if m else 0.0

    # --- Load k and E(k) data ---
    data = np.loadtxt(fname)
    k, Ek = data[:,0], data[:,1]

    Emax = max(Emax, np.max(Ek))
    kmax = max(kmax, np.max(k))

    # --- Plot (log–log for usual turbulence spectra) ---
    plt.loglog(k[1:], Ek[1:], label=f"t={t:.3f}")  # skip k=0

plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy spectrum $E(k)$")
plt.ylim(1e-3*Emax, Emax*2.)
plt.xlim(None, kmax/2.)
plt.title("Energy spectra evolution")
# plt.legend(fontsize=8, loc="best")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig('spectra.pdf')