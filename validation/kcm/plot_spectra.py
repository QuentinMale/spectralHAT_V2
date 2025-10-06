import numpy as np
import matplotlib.pyplot as plt
import glob, re, os

class kcm_spectrum:
  def __init__(self,station=0):
    self.station_ = station
    self.epst_ = [22.8, 9.13, 4.72, 3.41]
    self.lt_  = [0.25, 0.288, 0.321, 0.332]
    self.etat_= [0.11e-3, 0.14e-3, 0.16e-3, 0.18e-3]
    self.ck_ = 1.613;
    self.alfa_ = [0.39, 1.2, 4.0, 2.1, 0.522, 10.0, 12.58]
    self.eps_ = self.epst_[station]
    self.ckEpsTwo3rd_ = self.ck_*pow(self.eps_, 0.666666666667)
    self.l_ = self.lt_[station]
    self.eta_ = self.etat_[station]
  def evaluate(self, k):
    kl = k*self.l_
    keta = k*self.eta_
    term1 = kl / pow( (pow(kl, self.alfa_[1]) + self.alfa_[0]), 0.83333333333333)
    term1 = pow(term1,5.66666666666667)
    term2 = 0.5 + 0.31830988618379*np.arctan(self.alfa_[5]*np.log10(keta) + self.alfa_[6])
    espec = self.ckEpsTwo3rd_*pow(k,-1.66666666666667)*term1 * np.exp(-self.alfa_[3]*keta) * (1.0 + self.alfa_[4]*term2)
    return espec

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

# ---- target times and nearest files ----
target_times = np.array([0.0, 0.1116071429, 0.2232142857, 0.3125])
nearest_idx = []
for tt in target_times:
    i = int(np.nanargmin(np.abs(times - tt)))
    nearest_idx.append(i)

# ensure unique ordering by time
nearest_idx = sorted(set(nearest_idx), key=lambda i: times[i])

# ---- plot selection + KCM theory overlays ----
plt.figure(figsize=(9,6))
Emax = 0.0
knyq = 0.0

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for j, i in enumerate(nearest_idx):
    fname = files[i]
    t_sel = times[i]
    data = np.loadtxt(fname)
    k, Ek = data[:,0], data[:,1]

    # Update axes ranges
    Emax = max(Emax, float(Ek.max()))
    knyq = max(knyq, float(k.max()))

    # plot measured spectrum (skip k=0)
    plt.loglog(k[1:], Ek[1:], label=f"DNS t≈{t_sel:.3f}", lw=2, color=colors[j % len(colors)])

    # theory for corresponding station j (0..3), evaluated on same k
    spec = kcm_spectrum(station=j)
    Ek_th = spec.evaluate(k)
    plt.loglog(k[1:], Ek_th[1:], ls="--", lw=1.8, color=colors[j % len(colors)],
               label=f"Theory (station {j})")

# ---- cosmetics & limits ----
plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy spectrum $E(k)$")
if Emax > 0:
    plt.ylim(1e-3*Emax, 2.0*Emax)
if knyq > 1:
    plt.xlim(1.0, knyq)

plt.title("Energy spectra vs KCM theory (nearest output times)")
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("spectra_vs_theory.pdf", dpi=200)
plt.show()