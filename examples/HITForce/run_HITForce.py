#!/usr/bin/env python3
import argparse, sys, time
import numpy as np
import scipy.interpolate as spi
import os
import cupy as cp
from cupyx.scipy.fft import irfftn as cu_irfftn

from spectralHAT.hat import SpectralDNS

# -------------------- helpers (from your snippet) --------------------
def div0(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
    return c

def abs2(x):
    return x.real**2 + x.imag**2

# -------------------- IC generator --------------------
def make_ic_random_HIT(Nk, N_out, k0, seed=0, L=2.0*np.pi, urms=1.0):
    """
    Create a divergence-free random velocity field on a Nk^3 spectral grid,
    then iFFT to real-space and (optionally) resample to an N_out^3 grid.
    Returns (ur, vr, wr) on the N_out grid.
    """
    start = time.time()

    # require even sizes (as in your code)
    if not ((Nk % 2 == 0) and (N_out % 2 == 0)):
        raise SystemExit("Nk and N_out must be even.")

    halfN = Nk // 2
    xs, xe = 0.0, L
    dx = L / Nk

    # cell-centers for Nk grid
    x = np.linspace(xs, xe, Nk + 1)
    xc = 0.5 * (x[1:] + x[:-1])                     # Nk centers
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")

    # wave numbers (rFFT along k3)
    k = np.concatenate((np.arange(halfN), np.arange(-halfN, 0, 1)), axis=0)
    khalf = np.arange(halfN + 1)
    k1, k2, k3 = np.meshgrid(k, k, khalf, indexing="ij")
    kmag  = np.sqrt(k1**2 + k2**2 + k3**2)
    k12   = np.sqrt(k1**2 + k2**2)
    k1k12 = div0(k1, k12)
    k2k12 = div0(k2, k12)
    k3kmag = div0(k3, kmag)
    k12kmag = div0(k12, kmag)

    # energy spectrum
    Ek = 16.0*np.sqrt(2.0/np.pi) * (kmag**4) / (k0**5) * np.exp(-2.0*(kmag**2)/(k0**2))

    # random phases
    rng = np.random.default_rng(seed)
    phi1 = rng.uniform(0, 2*np.pi, kmag.shape)
    phi2 = rng.uniform(0, 2*np.pi, kmag.shape)
    phi3 = rng.uniform(0, 2*np.pi, kmag.shape)

    prefix = np.sqrt(2.0 * div0(Ek, 4.0*np.pi*(kmag**2)))
    a = prefix * np.exp(1j*phi1) * np.cos(phi3)
    b = prefix * np.exp(1j*phi2) * np.sin(phi3)

    # divergence-free Fourier coefficients (rFFT in k3)
    uf = k2k12 * a + k1k12 * k3kmag * b
    vf = k2k12 * k3kmag * b - k1k12 * a
    wf = -k12kmag * b

    # enforce Hermitian symmetry on kz=0 plane (to yield real signal)
    N = Nk
    uf[N:halfN:-1, N:halfN:-1, 0] = np.conj(uf[1:halfN, 1:halfN, 0])
    uf[N:halfN:-1, 0, 0]          = np.conj(uf[1:halfN, 0, 0])
    uf[0, N:halfN:-1, 0]          = np.conj(uf[0, 1:halfN, 0])
    uf[halfN-1:0:-1, N:halfN-1:-1, 0] = np.conj(uf[halfN+1:N, 1:halfN+1, 0])

    vf[N:halfN:-1, N:halfN:-1, 0] = np.conj(vf[1:halfN, 1:halfN, 0])
    vf[halfN-1:0:-1, N:halfN-1:-1, 0] = np.conj(vf[halfN+1:N, 1:halfN+1, 0])
    vf[N:halfN:-1, 0, 0]          = np.conj(vf[1:halfN, 0, 0])
    vf[0, N:halfN:-1, 0]          = np.conj(vf[0, 1:halfN, 0])

    wf[N:halfN:-1, N:halfN:-1, 0] = np.conj(wf[1:halfN, 1:halfN, 0])
    wf[halfN-1:0:-1, N:halfN-1:-1, 0] = np.conj(wf[halfN+1:N, 1:halfN+1, 0])
    wf[N:halfN:-1, 0, 0]          = np.conj(wf[1:halfN, 0, 0])
    wf[0, N:halfN:-1, 0]          = np.conj(wf[0, 1:halfN, 0])

    # normalize for unnormalized numpy irfftn
    uf *= Nk**3
    vf *= Nk**3
    wf *= Nk**3

    # sharp cutoff to target maximum resolvable k on Nk grid
    kmagc = 0.5 * Nk
    uf[kmag > kmagc] = 0.0
    vf[kmag > kmagc] = 0.0
    wf[kmag > kmagc] = 0.0

    # back to real space on Nk grid
    u = np.fft.irfftn(uf, s=(Nk, Nk, Nk), axes=(0, 1, 2))
    v = np.fft.irfftn(vf, s=(Nk, Nk, Nk), axes=(0, 1, 2))
    w = np.fft.irfftn(wf, s=(Nk, Nk, Nk), axes=(0, 1, 2))

    # resample to solver grid N_out^3 (cell centers), periodic domain
    xr = np.linspace(xs, xe, N_out + 1)
    xrc = 0.5*(xr[1:] + xr[:-1])
    Xr, Yr, Zr = np.meshgrid(xrc, xrc, xrc, indexing="ij")
    pts = (xc, xc, xc)
    Ur = spi.interpn(pts, u, (Xr, Yr, Zr), method="linear", bounds_error=False, fill_value=None)
    Vr = spi.interpn(pts, v, (Xr, Yr, Zr), method="linear", bounds_error=False, fill_value=None)
    Wr = spi.interpn(pts, w, (Xr, Yr, Zr), method="linear", bounds_error=False, fill_value=None)

    Ur *= urms
    Vr *= urms
    Wr *= urms

    # quick energy print
    En = 0.5*np.mean(Ur**2 + Vr**2 + Wr**2)
    urms = np.sqrt(2/3*En)
    print(f"[IC] Nk={Nk}, N={N_out}, k0={k0:.3g}, seed={seed}  ->  <E>= {En:.6e} <urms>= {urms:.6e} (built in {time.time()-start:.2f}s)")

    return Ur.astype(np.float64), Vr.astype(np.float64), Wr.astype(np.float64)

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

Re_lambda = 82.0
L = 2 * cp.pi
nu = 15.69e-6
N = 64

l_t = L * 0.19
forcing_eps = Re_lambda**6 * nu**3 / 15.0**3 / l_t**4
u_prim = Re_lambda**2 * nu / 15.0 / l_t
tau_t = l_t / u_prim
print(Re_lambda, forcing_eps, u_prim, tau_t)

os.makedirs('SOLUT', exist_ok=True)

# 1) Build initial condition on CPU
ur, vr, wr = make_ic_random_HIT(Nk=256, N_out=N, k0=4, seed=42, L=L, urms=u_prim)

# 2) Instantiate solver (on GPU)
solver = SpectralDNS(
    N=N, L=L, nu=nu, precision="float64",
    dealias_mode="three_halves", les_model="smagorinsky", Cs=0.17,
    forcing_eps=forcing_eps
)

# 3) Load IC into solver (CuPy arrays) and project to div-free
solver.set_velocity_real(cp.asarray(ur, dtype=solver.dtype),
                            cp.asarray(vr, dtype=solver.dtype),
                            cp.asarray(wr, dtype=solver.dtype))

# 4) Run
logger = RefTimeseriesLogger(path="temporals.txt", also_print=False)
solver.run(T=50 * tau_t, cfl=0.8, fourier=0.3, log_every=10, sol_every=400,
            callback=logger)
logger.close()
