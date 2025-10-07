import cupy as cp
from spectralHAT.hat import SpectralDNS
import os, io

class RefTimeseriesLogger:
    """
    Callback to record: Time, Energy, Dissipation rate (-dE/dt), Enstrophy
    - 'Dissipation' is computed as ε = 2ν Z (periodic, incompressible).
    - Also computes a numerical -dE/dt (from finite difference).
    """
    def __init__(self, path: str = None, also_print: bool = True):
        self.path = path
        self.also_print = also_print
        self.prev_t = None
        self.prev_E = None
        self.fh = None
        if path is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.fh = open(path, "w", buffering=1)
            self.fh.write("# Time  Energy  Dissipation(-dE/dt)  Enstrophy  [-dE/dt_num]\n")

    def __call__(self, t, stats, solver):
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

    def close(self):
        if self.fh is not None:
            self.fh.close()

def taylor_green_ic(X, Lc=1.0, V0=1.0):
    x, y, z = X
    U = cp.empty((3,)+x.shape, dtype=x.dtype)
    U[0] = V0 * cp.sin(x/Lc) * cp.cos(y/Lc) * cp.cos(z/Lc)
    U[1] = -V0 * cp.cos(x)    * cp.sin(y)    * cp.cos(z)
    U[2] = 0.0
    return U

if __name__ == "__main__":
    N = 64
    Lc = 1.0
    V0 = 1.0
    Re = 1600.
    Lbox = 2 * cp.pi * Lc
    nu = V0 * Lc / Re
    tc = Lc / V0

    solver = SpectralDNS(N=N, L=Lbox, nu=nu, precision="float32", dealias_mode="phase_shift")
    solver.prepare_ic(lambda X: taylor_green_ic(X, Lc=Lc, V0=V0))

    logger = RefTimeseriesLogger(path="temporals.txt", also_print=False)
    os.makedirs('SOLUT', exist_ok=True)

    solver.run(T=20.0*tc, cfl=0.8, fourier=0.3, log_every=5, sol_every=10,
            callback=logger)
    logger.close()
