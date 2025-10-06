import cupy as cp
import numpy as np
from spectralHAT.hat import SpectralDNS
import os, io

class RefTimeseriesLogger:
    """
    Callback to record: Time, Energy, Dissipation rate (-dE/dt), Enstrophy
    - 'Dissipation' is computed as ε = 2ν Z (periodic, incompressible).
    - Also computes a numerical -dE/dt (from finite difference).
    """
    def __init__(self, path: str = None, also_print: bool = True,
                 do_spectrum: bool = True, spectrum_dir: str = "SPECTRA"):
        self.path = path
        self.also_print = also_print
        self.prev_t = None
        self.prev_E = None
        self.fh = None
        self.spectrum_dir = spectrum_dir
        self.do_spectrum = do_spectrum
        if path is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.fh = open(path, "w", buffering=1)
            self.fh.write("# Time  Energy  Dissipation(-dE/dt)  Enstrophy  [-dE/dt_num]\n")
        if do_spectrum:
            os.makedirs(self.spectrum_dir, exist_ok=True)

    def __call__(self, t, stats, solver):
        nstep = getattr(solver, "nstep", 0)
        # energy from stats (already computed by solver)
        E = float(stats["kinetic_energy"])
        # enstrophy & dissipation from solver
        Z = solver.enstrophy()
        eps = 2.0 * solver.nu * Z  # = dissipation rate = -dE/dt (theoretical)
        div = solver.divergence_max()

        print(f"[TG] t={t:.3f}, E={E:.6f}, Z={Z:.6f}, div={div:.6e}")

        # optional finite-difference estimate of -dE/dt for diagnostics
        dE_num = 0.0
        if self.prev_t is not None and t > self.prev_t:
            dE_num = -(E - self.prev_E) / (t - self.prev_t)

        self.prev_t, self.prev_E = t, E

        # format line
        if dE_num is None:
            line = f"{t:.8f} {E:.12f} {eps:.12f} {Z:.12f}\n"
        else:
            line = f"{t:.8f} {E:.12f} {eps:.12f} {Z:.12f} {dE_num:.12f}\n"

        if self.fh is not None:
            self.fh.write(line)
        if self.also_print:
            print(line.strip())
        
        # --- optional spectrum dump ---
        if self.do_spectrum:
            k, E_k, _ = solver.energy_spectrum()
            fname = os.path.join(self.spectrum_dir, f"spectrum_{nstep:06d}.txt")
            np.savetxt(fname, np.column_stack([k, E_k]),
                       fmt="%.8e", header=f"# Spectrum at t={t:.8f}")
            if self.also_print:
                print(f"  -> saved spectrum to {fname}")

    def close(self):
        if self.fh is not None:
            self.fh.close()

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

# ----- Utilities for HIT IC construction -----
def make_wavenumbers(N, L):
    """Return kx, ky, kz (CuPy) and |k|, with 2π periodicity for box length L."""
    d = L / N
    base = cp.fft.fftfreq(N, d=d) * 2*cp.pi  # 2π n / L
    kx = base[:, None, None]
    ky = base[None, :, None]
    kz = base[None, None, :]
    K2 = kx**2 + ky**2 + kz**2
    K = cp.sqrt(K2)
    return kx, ky, kz, K, K2

def shell_partition(K, k0, dk):
    """Integer shell index per mode, with shell width dk and offset k0."""
    return cp.asarray(cp.floor((K - k0) / dk), dtype=cp.int32)

def project_div_free(uhat, kx, ky, kz, K2):
    """Apply Helmholtz projection in Fourier space to make the field solenoidal."""
    ux, uy, uz = uhat
    # Avoid divide-by-zero at k=0: set projector = I (no change) then zero out later
    invK2 = cp.where(K2==0, 0.0, 1.0/K2)
    k_dot_u = kx*ux + ky*uy + kz*uz
    ux = ux - kx * (k_dot_u * invK2)
    uy = uy - ky * (k_dot_u * invK2)
    uz = uz - kz * (k_dot_u * invK2)
    # Zero the mean mode (k=0) to avoid a bulk flow
    mask0 = (K2 == 0)
    ux = cp.where(mask0, 0.0, ux)
    uy = cp.where(mask0, 0.0, uy)
    uz = cp.where(mask0, 0.0, uz)
    return cp.stack([ux, uy, uz], axis=0)

def compute_shell_energy(uhat, shell_ids):
    """Unnormalized shell energy: (1/2) sum_shell |û|^2 over components."""
    e_per_mode = 0.5 * cp.sum(cp.abs(uhat)**2, axis=0)  # shape (N,N,N)
    # Group by shell index with bincount
    smax = int(cp.max(shell_ids).get()) if shell_ids.size else -1
    flat_ids = shell_ids.ravel()
    flat_e = e_per_mode.ravel()
    shell_e = cp.bincount(flat_ids, weights=flat_e, minlength=smax+1)
    return shell_e  # length = n_shells

def build_hit_initial_field(N, L, target_E_of_k, seed=1234):
    """
    Generate a real, divergence-free U(x) with target 1D energy spectrum E(k).
    target_E_of_k: callable taking numpy array of k (float) -> E(k)
    """
    cp.random.seed(seed)

    # 1) start from 3 real white-noise fields
    U0 = cp.random.standard_normal((3, N, N, N), dtype=cp.float32)

    # 2) FFT to k-space and project to solenoidal
    uhat = cp.fft.fftn(U0, axes=(1,2,3))
    kx, ky, kz, K, K2 = make_wavenumbers(N, L)
    uhat = project_div_free(uhat, kx, ky, kz, K2)

    # 3) define shells and desired shell energies from E(k)
    dk = 2.0*cp.pi / L             # fundamental spacing in |k|
    k0 = 0.0                       # start shells at 0, [0,dk), [dk,2dk), ...
    shell_ids = shell_partition(K, k0, dk)
    sidx = cp.unique(shell_ids)    # valid shells that actually appear
    k_centers = (cp.asarray(sidx, dtype=cp.float32) + 0.5) * dk  # center of each shell
    k_centers_np = cp.asnumpy(k_centers)                          # to numpy
    E_target_np = target_E_of_k(k_centers_np)                     # numpy
    E_target = cp.asarray(E_target_np, dtype=cp.float32)

    # Convert 1D spectrum (per dk) to discrete shell energy target:
    # E_shell_target ≈ E(k_center) * dk
    E_shell_target = E_target * float(dk.get() if hasattr(dk,'get') else dk) * (N**6)

    # 4) measure current shell energy and rescale per shell
    E_shell_meas = compute_shell_energy(uhat, shell_ids)
    eps = 1e-30
    scale = cp.sqrt(cp.maximum(E_shell_target, 0.0) / cp.maximum(E_shell_meas, eps))

    # apply shell scale to all modes in each shell
    scale_field = scale[shell_ids]  # shape (N,N,N)
    uhat = uhat * scale_field[None, :, :, :]

    def two_thirds_mask(N, L):
        kx, ky, kz, _, _ = make_wavenumbers(N, L)
        kmax = (N//2)*(2*cp.pi/L)
        kc = (2.0/3.0)*kmax
        return (cp.abs(kx) <= kc) & (cp.abs(ky) <= kc) & (cp.abs(kz) <= kc)

    mask = two_thirds_mask(N, L)
    uhat = uhat * mask[None, ...]

    # 5) IFFT back to real space
    U = cp.fft.ifftn(uhat, axes=(1,2,3)).real.astype(cp.float32)
    return U  # shape (3,N,N,N)

if __name__ == "__main__":
    N = 128
    L = 5.12
    spec = kcm_spectrum(station=0)
    nu = 15.69e-6

    solver = SpectralDNS(N=N, L=L, nu=nu, les_model="smagorinsky", precision="float32", dealias_mode="two_thirds")

    # Build an initial field matching E_kcm(k)
    U0 = build_hit_initial_field(N, float(L.get() if hasattr(L,'get') else L),
                                 target_E_of_k=spec.evaluate,
                                 seed=42)

    # Prepare IC: ignore X, just return our precomputed field
    def hit_ic(_X):
        return U0

    solver.prepare_ic(hit_ic)

    logger = RefTimeseriesLogger(path="temporals.txt", also_print=False)
    os.makedirs('SOLUT', exist_ok=True)

    solver.run(T=0.38, cfl=0.8, fourier=0.3, log_every=1, sol_every=10,
               callback=logger)
    logger.close()
