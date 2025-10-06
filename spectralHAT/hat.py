# spectral_dns_cupy.py
import cupy as cp
import numpy as np
from cupyx.scipy.fft import rfftn, irfftn, rfftfreq, fftfreq
import vtk
from vtk.util import numpy_support as vtknp

class SpectralDNS:
    def __init__(self, N=128, L=2*cp.pi, nu=1/1600, precision="float32",
                 dealias_mode: str = "two_thirds"):
        self.Nx = self.Ny = self.Nz = int(N)
        self.L = L
        self.nu = float(nu)
        self.dtype = cp.float32 if precision == "float32" else cp.float64
        self.cdtype = cp.complex64 if self.dtype == cp.float32 else cp.complex128
        self.zf = self.Nz//2 + 1
        self.dealias_mode = dealias_mode

        # Real-space mesh (cached for ICs)
        grid = cp.mgrid[0:self.Nx, 0:self.Ny, 0:self.Nz].astype(self.dtype)
        self.X = grid * (self.L/self.Nx)  # [x,y,z] on GPU

        # Spectral wavenumbers
        kx = (2*cp.pi) * fftfreq(self.Nx, d=self.L/self.Nx).astype(self.dtype)
        ky = (2*cp.pi) * fftfreq(self.Ny, d=self.L/self.Ny).astype(self.dtype)
        kz = (2*cp.pi) * rfftfreq(self.Nz, d=self.L/self.Nz).astype(self.dtype)
        self.KX, self.KY, self.KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.invK2 = cp.where(self.K2 == 0, 0, 1.0/self.K2).astype(self.dtype)

        # --- build dealiasing mask based on mode ---
        self.dealias = self._build_dealias_mask(self.dealias_mode)

        self.__post_init_precompute()

        # Work arrays allocated once
        self.U_hat = cp.zeros((3, self.Nx, self.Ny, self.zf), dtype=self.cdtype)


    def _build_dealias_mask(self, mode: str):
        """
        Returns a boolean mask in spectral space applied to the nonlinear term.
        - "two_thirds": classic 2/3 truncation in each direction
        - "nyquist": keep all resolvable modes (no truncation)
        """
        if mode == "nyquist":
            # Keep everything up to the FFT's resolvable limit.
            return cp.ones((self.Nx, self.Ny, self.zf), dtype=cp.bool_)

        if mode == "two_thirds":
            ix = cp.abs(cp.fft.fftfreq(self.Nx))*self.Nx
            iy = cp.abs(cp.fft.fftfreq(self.Ny))*self.Ny
            iz = cp.abs(cp.fft.rfftfreq(self.Nz))*self.Nz
            IX, IY, IZ = cp.meshgrid(ix, iy, iz, indexing='ij')
            cutx = (2.0/3.0)*(self.Nx//2)
            cuty = (2.0/3.0)*(self.Ny//2)
            cutz = (2.0/3.0)*(self.Nz//2)
            return (IX < cutx) & (IY < cuty) & (IZ < cutz)

        raise ValueError(f"Unknown dealias_mode: {mode!r}. Use 'two_thirds' or 'nyquist'.")

    # ---------------- utilities ----------------
    @staticmethod
    def _fft3r(u):   # real -> complex (rfftn across all 3 axes)
        return rfftn(u, axes=(-3, -2, -1))

    @staticmethod
    def _ifft3c(uh): # complex -> real
        return irfftn(uh, s=None, axes=(-3, -2, -1))

    def _curl_hat(self, U_hat):
        i = 1j
        wx = i*(self.KY*U_hat[2] - self.KZ*U_hat[1])
        wy = i*(self.KZ*U_hat[0] - self.KX*U_hat[2])
        wz = i*(self.KX*U_hat[1] - self.KY*U_hat[0])
        return cp.stack([wx, wy, wz], axis=0)

    def _project_divfree(self, A_hat):
        # P = I - kk^T/|k|^2
        KdotA = self.KX*A_hat[0] + self.KY*A_hat[1] + self.KZ*A_hat[2]
        Px = A_hat[0] - self.KX*(KdotA*self.invK2)
        Py = A_hat[1] - self.KY*(KdotA*self.invK2)
        Pz = A_hat[2] - self.KZ*(KdotA*self.invK2)
        return cp.stack([Px, Py, Pz], axis=0)

    def _rhs(self, U_hat):
        # vorticity in spectral -> real
        W_hat = self._curl_hat(U_hat)
        U = self._ifft3c(U_hat)
        W = self._ifft3c(W_hat)

        # nonlinearity N = u x w (real) -> spectral
        Nx_hat = self._fft3r(U[1]*W[2] - U[2]*W[1])
        Ny_hat = self._fft3r(U[2]*W[0] - U[0]*W[2])
        Nz_hat = self._fft3r(U[0]*W[1] - U[1]*W[0])
        N_hat = cp.stack([Nx_hat, Ny_hat, Nz_hat], axis=0)

        # dealias and project (pressure elimination)
        N_hat *= self.dealias
        N_hat = self._project_divfree(N_hat)

        # viscous term
        return N_hat - (self.nu * self.K2)[None, ...] * U_hat

    def _rk4(self, U_hat, dt):
        k1 = self._rhs(U_hat)
        k2 = self._rhs(U_hat + 0.5*dt*k1)
        k3 = self._rhs(U_hat + 0.5*dt*k2)
        k4 = self._rhs(U_hat + dt*k3)
        return U_hat + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # ---------------- public API ----------------
    def prepare_ic(self, ic_func):
        """
        ic_func must take self.X (a 3xNxNyNz GPU array) and return U(x) as shape (3,Nx,Ny,Nz) on GPU.
        We project to divergence-free just in case.
        """
        U0 = ic_func(self.X).astype(self.dtype, copy=False)
        for c in range(3):
            self.U_hat[c] = self._fft3r(U0[c]).astype(self.cdtype, copy=False)
        self.U_hat = self._project_divfree(self.U_hat)

    def __post_init_precompute(self):
        # call from __init__ after constructing KX/KY/KZ, dealias, etc.
        self.dx = self.L / self.Nx
        self.dy = self.L / self.Ny
        self.dz = self.L / self.Nz
        self.dx_min = float(min(self.dx, self.dy, self.dz))
        # max resolved k^2 after dealiasing (safest for diffusion limit)
        k2_masked = cp.where(self.dealias, self.K2, 0.0)
        self.k2_max = float(k2_masked.max().get())  # scalar on host

    def _umax(self):
        U = self._ifft3c(self.U_hat)
        umax = float(cp.max(cp.abs(U)).get())
        return umax

    def run(self, T, cfl=0.5, fourier=0.8, max_dt=None, min_dt=None,
            callback=None, log_every=10, sol_every=None, emit_first=True):
        """
        Advance until physical time T with adaptive dt = min(dt_CFL, dt_Fourier).
        """
        assert T >= 0.0
        if log_every is not None and log_every <= 0:
            log_every = None

        t = 0.0
        n = 0
        alpha_rk4 = 2.785  # RK4 stability bound

        if not hasattr(self, "dx_min"):
            self.__post_init_precompute()
        k2max = self.k2_max
        viscous_active = (self.nu > 0.0 and k2max > 0.0)

        if emit_first:
            if log_every > 0 and callback is not None:
                self._maybe_callback(t, callback)
            if sol_every > 0:
                fname = f"SOLUT/velocity_{n:06d}.vti"
                self.save_velocity_vti(fname)

        while t < T:
            # --- compute adaptive timestep ---
            umax = self._umax()
            dt_cfl  = cfl * self.dx_min / umax if umax > 0.0 else float("inf")
            dt_four = fourier * alpha_rk4 / (self.nu * k2max) if viscous_active else float("inf")
            dt = min(dt_cfl, dt_four, T - t)

            if max_dt is not None: dt = min(dt, max_dt)
            if min_dt is not None: dt = max(dt, min_dt)
            if not (dt > 0.0 and cp.isfinite(dt)):
                raise RuntimeError("Adaptive dt became non-positive or non-finite.")

            # --- iteration-based callback output ---
            self.U_hat = self._rk4(self.U_hat, dt)
            t += dt
            n += 1

            # --- iteration-based output ---
            if log_every is not None and callback is not None:
                if n % int(log_every) == 0:
                    self._maybe_callback(t, callback)

            # --- write velocity field every N iterations ---
            if sol_every is not None and (n % int(sol_every) == 0):
                fname = f"SOLUT/velocity_{n:06d}.vti"
                self.save_velocity_vti(fname)

    def kinetic_energy(self):
        U = self._ifft3c(self.U_hat)
        return float(0.5*cp.mean(U[0]**2 + U[1]**2 + U[2]**2).get())

    def _maybe_callback(self, t, callback):
        if callback is None:
            return
        stats = {"t": float(t), "kinetic_energy": self.kinetic_energy()}
        callback(t, stats, self)

    def energy_spectrum(self):
        """
        Spherically averaged energy spectrum E(k) using rFFT data in self.U_hat.
        - Uses existing KX, KY, KZ (rFFT grid along z).
        - Reconstructs Hermitian pair via weights along kz.
        - Bins by m = round(|k|/dk). Excludes DC, caps at Nyquist.
        Returns: k (nbins,), E_k (nbins,), (TKE_real, TKE_spec)
        """
        Nx = self.Nx; Ny = self.Ny; Nz = self.Nz
        Lx = self.L
        Nxyz = Nx * Ny * Nz

        # radial grid + Nyquist cap (cubic box)
        dk  = 2.0 * cp.pi / Lx
        kc  = cp.pi * Nx / Lx
        eps = kc / 1_000_000.0

        nbins  = Nx + 1
        E_k    = cp.zeros(nbins, dtype=cp.float64)
        k_grid = dk * cp.arange(nbins, dtype=cp.float64)

        # |k| from existing rFFT grids
        kk = cp.sqrt(self.KX*self.KX + self.KY*self.KY + self.KZ*self.KZ)

        # bin index m ~ round(kk/dk); cap at Nyquist bin
        m = cp.floor(kk/dk + 0.5).astype(cp.int64)
        mcap = int(cp.floor((kc/dk) + 0.5).get())  # ~ Nx//2

        # Hermitian weights along rFFT axis (z):
        # - interior kz planes count twice
        # - kz=0 plane counts once
        # - if Nz even: kz=Nyquist plane (index zf-1) counts once
        zf = self.zf
        wz = cp.ones((zf,), dtype=self.dtype)
        if Nz % 2 == 0:
            # even Nz: [0, 1..zf-2, zf-1(Nyquist)]
            if zf > 2:
                wz[1:zf-1] = 2.0
        else:
            # odd Nz: [0, 1..zf-1], no Nyquist plane
            if zf > 1:
                wz[1:] = 2.0
        # broadcast weights to 3D half-spectrum
        WZ = wz[None, None, :]

        # mask: exclude near-DC, cap at kc, keep valid bins
        valid = (kk > eps) & (kk <= kc) & (m >= 0) & (m <= min(Nx, mcap))

        # Accumulate per component using self.U_hat (forward rFFT, unnormalized)
        for c in range(3):
            # Normalize like the reference: divide by total grid points
            F = self.U_hat[c] / Nxyz
            mag2_half = 0.5 * (cp.abs(F)**2) * WZ  # 0.5*|F|^2 with Hermitian weight

            # Add shell power into bins: E_k[m] += (0.5|F|^2)/dk
            cp.add.at(E_k, m[valid], (mag2_half[valid] / dk))

        return np.asarray(k_grid.get()), np.asarray(E_k.get()), dk
    
    def save_velocity_vti(self, filename="velocity.vti"):
        """
        Save the current velocity field to a VTK .vti file (ImageData format)
        readable by ParaView/VisIt.

        Writes point-centered velocity vector 'Velocity' with components (u,v,w).
        """
        # Inverse transform to real space
        U = self._ifft3c(self.U_hat)
        # Move to host
        U = [u.get() for u in U]

        u, v, w = U
        Nx, Ny, Nz = u.shape
        dx = self.L / Nx
        dy = self.L / Ny
        dz = self.L / Nz

        # Create VTK image data object
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(Nx, Ny, Nz)
        imageData.SetSpacing(dx, dy, dz)
        imageData.SetOrigin(0.0, 0.0, 0.0)

        # Stack velocity components and flatten (VTK wants Fortran order)
        V = np.stack([u, v, w], axis=-1).astype(np.float32)
        V_flat = np.ascontiguousarray(V.reshape(-1, 3), dtype=np.float32)

        # Convert to VTK array and attach as point data
        vtk_array = vtknp.numpy_to_vtk(num_array=V_flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName("Velocity")
        imageData.GetPointData().AddArray(vtk_array)
        imageData.GetPointData().SetActiveVectors("Velocity")

        # Write .vti file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(imageData)
        writer.Write()

        print(f"Velocity field saved to {filename} (VTK ImageData)")

    def enstrophy(self):
        """
        Enstrophy Z = 0.5 * <|ω|^2>, with ω = curl(u).
        Uses current spectral field self.U_hat.
        """
        W_hat = self._curl_hat(self.U_hat)     # ω̂ = i k × û
        W = self._ifft3c(W_hat)                # ω in real space
        Z = 0.5 * (W[0]**2 + W[1]**2 + W[2]**2).mean()
        return float(Z.get())

    def dissipation(self):
        """
        Dissipation ε = 2 * ν * Z for incompressible, periodic flow.
        """
        Z = self.enstrophy()
        return 2.0 * self.nu * Z

# ----------- example ICs and usage -----------
def taylor_green_ic(X):
    x, y, z = X
    U = cp.empty((3,)+x.shape, dtype=x.dtype)
    U[0] = cp.sin(x)*cp.cos(y)*cp.cos(z)
    U[1] = -cp.cos(x)*cp.sin(y)*cp.cos(z)
    U[2] = 0.0
    return U

if __name__ == "__main__":
    solver = SpectralDNS(N=128, nu=1.0/1600, precision="float32", dealias_mode="two_thirds")

    # Run 1: Taylor–Green for T=0.05
    solver.prepare_ic(taylor_green_ic)
    solver.run(T=10.0, cfl=0.8, fourier=0.3, log_every=1,
               callback=lambda t, s, _: print(f"[TG] t={s['t']:.3f}, E={s['kinetic_energy']:.6f}"))
    
    k, E_k, dk = solver.energy_spectrum()
    TKE_spec = cp.sum(E_k * dk)
    TKE_real = solver.kinetic_energy()
    print(f"TKE(real)={TKE_real:.6e}  TKE(spec)={TKE_spec:.6e}  rel.err={(abs(TKE_real-TKE_spec)/TKE_real):.3e}")
