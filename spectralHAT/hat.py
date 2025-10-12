# spectral_dns_cupy.py
import cupy as cp
import numpy as np
from cupyx.scipy.fft import rfftn, irfftn, rfftfreq, fftfreq
import vtk
from vtk.util import numpy_support as vtknp

class SpectralDNS:
    def __init__(self, N=128, L=2*cp.pi, nu=1/1600, precision="float64",
                 dealias_mode: str = "two_thirds",
                 les_model=None, Cs=0.17, delta=None,
                 forcing_eps=None):
        self.Nx = self.Ny = self.Nz = int(N)
        self.L = L
        self.nu = float(nu)
        self.dtype = cp.float32 if precision == "float32" else cp.float64
        self.cdtype = cp.complex64 if self.dtype == cp.float32 else cp.complex128
        self.zf = self.Nz//2 + 1
        self.dealias_mode = dealias_mode
        self.les_model = les_model  # None or "smagorinsky"
        self.Cs = float(Cs)
        # filter width: default cube root of cell volume
        if delta is None:
            self.DELTA = float((self.L/self.Nx * self.L/self.Ny * self.L/self.Nz) ** (1.0/3.0))
        else:
            self.DELTA = float(delta)
        self.forcing_eps = float(forcing_eps) if forcing_eps is not None else None
        self.nstep = 0

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

        self.Kmag = cp.sqrt(self.K2).astype(self.dtype)
        # low-pass filter on the rFFT grid for low-k forcing
        self._G_lowk = self._build_lowpass(4.0*cp.pi/self.L, "sharp").astype(self.dtype)

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
        - "phase_shift": handled in _rhs via grid-shift
        - "three_halves": 3/2 padding
        """
        ix = cp.abs(cp.fft.fftfreq(self.Nx))*self.Nx
        iy = cp.abs(cp.fft.fftfreq(self.Ny))*self.Ny
        iz = cp.abs(cp.fft.rfftfreq(self.Nz))*self.Nz
        IX, IY, IZ = cp.meshgrid(ix, iy, iz, indexing='ij')
        if (mode == "nyquist") or (mode == "phase_shift") or (mode == "three_halves"):
            cutx = (1.0)*(self.Nx//2)
            cuty = (1.0)*(self.Ny//2)
            cutz = (1.0)*(self.Nz//2)
        elif mode == "two_thirds":
            cutx = (2.0/3.0)*(self.Nx//2)
            cuty = (2.0/3.0)*(self.Ny//2)
            cutz = (2.0/3.0)*(self.Nz//2)
        else:
            raise ValueError(f"Unknown dealias_mode: {mode!r}."
                              "Use 'two_thirds', 'nyquist', 'phase_shift' or '3/2 padding'.")
        return (IX < cutx) & (IY < cuty) & (IZ < cutz)

    def _build_lowpass(self, kf: float, kind: str = "sharp"):
        """Return low-pass G(k,kf) on the rFFT grid."""
        if kind == "sharp":
            return (self.Kmag <= kf).astype(self.dtype)
        elif kind == "gaussian":
            # smooth low-pass: exp(-(k/kf)^4) (steeper than ^2, common in HIT)
            r = self.Kmag / float(kf)
            return cp.exp(-(r**4)).astype(self.dtype)
        else:
            raise ValueError(f"Unknown forcing_filter '{kind}', use 'sharp' or 'gaussian'.")

    def _pad3_halfrfft(self, A_hat, Np):
        """
        Zero-pad a 3D rFFT half-spectrum A_hat (Nx,Ny,Nz//2+1) to (Np,Np,Np//2+1).
        Assumes even Nx,Ny,Nz and Np = 3*Nx//2.
        Returns complex array on GPU.
        """
        Nx, Ny, zf = A_hat.shape
        assert Nx == self.Nx and Ny == self.Ny and zf == self.zf
        assert Np % 2 == 0 and Nx % 2 == 0 and Ny % 2 == 0

        zfp = Np//2 + 1
        out = cp.zeros((Np, Np, zfp), dtype=A_hat.dtype)

        nxh = Nx//2; nyh = Ny//2  # Nyquist plane indices in x,y

        # Low-k blocks (remember: last axis is half-spectrum and copies directly)
        out[0:nxh+1, 0:nyh+1, :zf] = A_hat[0:nxh+1, 0:nyh+1, :]
        out[0:nxh+1, Np-(Ny-nyh-1):Np, :zf] = A_hat[0:nxh+1, nyh+1:Ny, :]
        out[Np-(Nx-nxh-1):Np, 0:nyh+1, :zf] = A_hat[nxh+1:Nx, 0:nyh+1, :]
        out[Np-(Nx-nxh-1):Np, Np-(Ny-nyh-1):Np, :zf] = A_hat[nxh+1:Nx, nyh+1:Ny, :]

        # z half-spectrum is already correct size in 'out'; higher kz remain zero
        return out

    def _truncate_from3_halfrfft(self, A_pad_hat):
        """
        Truncate a padded rFFT half-spectrum of shape (Np,Np,Np//2+1) down to base
        (Nx,Ny,Nz//2+1).
        """
        Np, _, zfp = A_pad_hat.shape
        Nx, Ny, zf = self.Nx, self.Ny, self.zf
        nxh = Nx//2; nyh = Ny//2
        out = cp.zeros((Nx, Ny, zf), dtype=A_pad_hat.dtype)

        out[0:nxh+1, 0:nyh+1, :zf] = A_pad_hat[0:nxh+1, 0:nyh+1, :zf]
        out[0:nxh+1, nyh+1:Ny, :zf] = A_pad_hat[0:nxh+1, Np-(Ny-nyh-1):Np, :zf]
        out[nxh+1:Nx, 0:nyh+1, :zf] = A_pad_hat[Np-(Nx-nxh-1):Np, 0:nyh+1, :zf]
        out[nxh+1:Nx, nyh+1:Ny, :zf] = A_pad_hat[Np-(Nx-nxh-1):Np, Np-(Ny-nyh-1):Np, :zf]
        return out

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

    def _gradients_real(self, U_hat):
        """du_i/dx_j in real space, shape (3,3,Nx,Ny,Nz)."""
        i = 1j
        # spectral derivatives (3,3,Nx,Ny,zf)
        dU_hat = cp.stack([
            cp.stack([i*self.KX*U_hat[0], i*self.KY*U_hat[0], i*self.KZ*U_hat[0]], axis=0),
            cp.stack([i*self.KX*U_hat[1], i*self.KY*U_hat[1], i*self.KZ*U_hat[1]], axis=0),
            cp.stack([i*self.KX*U_hat[2], i*self.KY*U_hat[2], i*self.KZ*U_hat[2]], axis=0),
        ], axis=0)

        # inverse FFT all derivatives at once -> real space (3,3,Nx,Ny,Nz)
        G = irfftn(dU_hat, s=(self.Nx, self.Ny, self.Nz), axes=(-3, -2, -1))
        return G

    def _sgs_smagorinsky(self, U_hat):
        G = self._gradients_real(U_hat)               # (3,3,Nx,Ny,Nz)
        S = 0.5 * (G + cp.swapaxes(G, 0, 1))          # symmetric strain
        Ssq = cp.sum(S*S, axis=(0,1))                 # sum_{i,j} S_ij^2
        Smag = cp.sqrt(2.0*Ssq) + 1e-30
        nu_t = (self.Cs*self.DELTA)**2 * Smag         # (Nx,Ny,Nz)

        # τ_ij = 2 ν_t S_ij  (real space)
        tau = 2.0 * S * nu_t[None, None, :, :, :]

        # F_i = ∂_j τ_ij  (compute in spectral space)
        t_x = rfftn(tau[:,0], axes=(-3,-2,-1))        # shape (3,Nx,Ny,zf) for j=x
        t_y = rfftn(tau[:,1], axes=(-3,-2,-1))        # j=y
        t_z = rfftn(tau[:,2], axes=(-3,-2,-1))        # j=z
        i = 1j
        F_hat = i*self.KX[None]*t_x + i*self.KY[None]*t_y + i*self.KZ[None]*t_z

        return F_hat

    def _forcing_lowk_hat(self, U_hat):
        """
        Low-wavenumber forcing: ^f = alpha G Û, with alpha chosen so that
        <f.u> = epsilon each step.
        """
        if self.forcing_eps is None:
            return cp.zeros_like(U_hat)

        # filtered velocity in spectral space
        Uhat_f = cp.stack([self._G_lowk*U_hat[0],
                           self._G_lowk*U_hat[1],
                           self._G_lowk*U_hat[2]], axis=0)

        # --- compute K_< directly in spectral space via Parseval (no IFFT) ---
        Nxyz = self.Nx * self.Ny * self.Nz

        # Hermitian weights for the rFFT axis (z):
        zf = self.zf
        wz = cp.ones((zf,), dtype=self.dtype)
        if self.Nz % 2 == 0 and zf > 2:
            wz[1:zf-1] = 2.0  # Nyquist plane counted once
        elif self.Nz % 2 == 1 and zf > 1:
            wz[1:] = 2.0

        # broadcast weights
        WZ = wz[None, None, :]

        # apply normalization: divide by Nxyz^2 because rfftn/irfftn are unnormalized
        mag2 = (cp.abs(Uhat_f[0])**2 + cp.abs(Uhat_f[1])**2 + cp.abs(Uhat_f[2])**2) * WZ
        Klt = 0.5 * cp.sum(mag2) / (Nxyz**2)

        alpha = self.forcing_eps / (2.0*Klt + 1e-30)  # scalar (CuPy)

        # ^f = alpha * ^U_<   (already solenoidal)
        F_hat = cp.stack([alpha*Uhat_f[0],
                          alpha*Uhat_f[1],
                          alpha*Uhat_f[2]], axis=0).astype(self.cdtype, copy=False)

        return F_hat

    def _project_divfree(self, A_hat):
        # P = I - kk^T/|k|^2
        KdotA = self.KX*A_hat[0] + self.KY*A_hat[1] + self.KZ*A_hat[2]
        Px = A_hat[0] - self.KX*(KdotA*self.invK2)
        Py = A_hat[1] - self.KY*(KdotA*self.invK2)
        Pz = A_hat[2] - self.KZ*(KdotA*self.invK2)
        return cp.stack([Px, Py, Pz], axis=0)

    def _phase_factor(self, dx_shift, dy_shift, dz_shift):
        """Return exp(i k·Δ) on the rFFT grid."""
        phase = cp.exp(1j * (self.KX*dx_shift + self.KY*dy_shift + self.KZ*dz_shift)).astype(self.cdtype)
        return phase  # shape (Nx,Ny,zf)

    def _nonlinear_hat_from_Uhat(self, U_hat, phase=None):
        """
        Compute N_hat = FFT(u x w) with optional grid shift via spectral phase.
        If 'phase' is not None, it must be shape (Nx,Ny,zf) with exp(i k·Δ).
        """
        if phase is None:
            U_hat_s = U_hat
        else:
            U_hat_s = cp.stack([U_hat[0]*phase, U_hat[1]*phase, U_hat[2]*phase], axis=0)

        # vorticity from the shifted field
        W_hat_s = self._curl_hat(U_hat_s)

        # go to real space
        U_s = self._ifft3c(U_hat_s)
        W_s = self._ifft3c(W_hat_s)

        # compute nonlinearity in real space
        Nx_hat = self._fft3r(U_s[1]*W_s[2] - U_s[2]*W_s[1])
        Ny_hat = self._fft3r(U_s[2]*W_s[0] - U_s[0]*W_s[2])
        Nz_hat = self._fft3r(U_s[0]*W_s[1] - U_s[1]*W_s[0])
        N_hat_s = cp.stack([Nx_hat, Ny_hat, Nz_hat], axis=0)

        # unshift back if needed (multiply by conj phase)
        if phase is not None:
            phase_c = cp.conj(phase)
            N_hat = cp.stack([N_hat_s[0]*phase_c, N_hat_s[1]*phase_c, N_hat_s[2]*phase_c], axis=0)
            return N_hat
        else:
            return N_hat_s
    
    def _nonlinear_hat_three_halves(self, U_hat):
        """
        Exact quadratic de-aliasing via 3/2 padding.
        Steps:
        1) pad Û and Ŵ to Np = 3N/2
        2) irfftn on padded grids -> u_pad, w_pad
        3) compute u×w on padded grid
        4) rfftn back, then truncate to base grid
        5) final projection to divergence-free happens in caller
        """
        Nx = self.Nx
        Np = (3 * Nx) // 2
        assert 2*Np % 3 == 0, "Np must be 3/2*N (even)"

        # vorticity in spectral (base)
        W_hat = self._curl_hat(U_hat)

        # pad spectra to Np
        U0p = self._pad3_halfrfft(U_hat[0], Np)
        U1p = self._pad3_halfrfft(U_hat[1], Np)
        U2p = self._pad3_halfrfft(U_hat[2], Np)
        W0p = self._pad3_halfrfft(W_hat[0], Np)
        W1p = self._pad3_halfrfft(W_hat[1], Np)
        W2p = self._pad3_halfrfft(W_hat[2], Np)

        # inverse FFT on padded grid (real)
        u0 = irfftn(U0p, s=(Np, Np, Np), axes=(-3, -2, -1))
        u1 = irfftn(U1p, s=(Np, Np, Np), axes=(-3, -2, -1))
        u2 = irfftn(U2p, s=(Np, Np, Np), axes=(-3, -2, -1))
        w0 = irfftn(W0p, s=(Np, Np, Np), axes=(-3, -2, -1))
        w1 = irfftn(W1p, s=(Np, Np, Np), axes=(-3, -2, -1))
        w2 = irfftn(W2p, s=(Np, Np, Np), axes=(-3, -2, -1))

        # nonlinear on padded grid
        Nx_hat_p = rfftn(u1*w2 - u2*w1, axes=(-3, -2, -1))
        Ny_hat_p = rfftn(u2*w0 - u0*w2, axes=(-3, -2, -1))
        Nz_hat_p = rfftn(u0*w1 - u1*w0, axes=(-3, -2, -1))

        # truncate to base grid
        Nx_hat = self._truncate_from3_halfrfft(Nx_hat_p)
        Ny_hat = self._truncate_from3_halfrfft(Ny_hat_p)
        Nz_hat = self._truncate_from3_halfrfft(Nz_hat_p)

        return cp.stack([Nx_hat, Ny_hat, Nz_hat], axis=0)

    def _rhs(self, U_hat):
        # viscosity term
        visc_hat = -(self.nu * self.K2)[None, ...] * U_hat
        visc_hat *= self.dealias

        # Smagorinsky SGS
        if self.les_model == "smagorinsky":
            SGS_hat = self._sgs_smagorinsky(U_hat)
            SGS_hat *= self.dealias
        else:
            SGS_hat = 0.0

        # Forcing
        if (self.forcing_eps is not None):
            Fforce_hat = self._forcing_lowk_hat(U_hat)
        else:
            Fforce_hat = 0.0

        if self.dealias_mode == "three_halves":
            N_hat = self._nonlinear_hat_three_halves(U_hat)
            N_hat *= self.dealias
            return self._project_divfree(N_hat + visc_hat + SGS_hat + Fforce_hat)

        elif self.dealias_mode == "phase_shift":
            # compute N_hat for each shift on the shifted grid, then unshift back and average
            N_acc = 0.0
            for phase in self._phases8:                     # includes (0,0,0)
                N_shift = self._nonlinear_hat_from_Uhat(U_hat, phase=phase)
                N_acc += N_shift
            N_hat = (N_acc / len(self._phases8))
            return self._project_divfree(N_hat + visc_hat + SGS_hat + Fforce_hat)

        else:
            # --- default path (your existing dealiasing) ---
            # compute curl, go to real space
            W_hat = self._curl_hat(U_hat)
            U = self._ifft3c(U_hat)
            W = self._ifft3c(W_hat)

            Nx_hat = self._fft3r(U[1]*W[2] - U[2]*W[1])
            Ny_hat = self._fft3r(U[2]*W[0] - U[0]*W[2])
            Nz_hat = self._fft3r(U[0]*W[1] - U[1]*W[0])
            N_hat = cp.stack([Nx_hat, Ny_hat, Nz_hat], axis=0)

            N_hat *= self.dealias

            return self._project_divfree(N_hat + visc_hat + SGS_hat + Fforce_hat)

    def _rk4(self, U_hat, dt):
        k1 = self._rhs(U_hat)
        k2 = self._rhs(U_hat + 0.5*dt*k1)
        k3 = self._rhs(U_hat + 0.5*dt*k2)
        k4 = self._rhs(U_hat + dt*k3)
        return U_hat + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def divergence_max(self):
        i = 1j
        div_hat = i * (self.KX*self.U_hat[0] + self.KY*self.U_hat[1] + self.KZ*self.U_hat[2])
        div = self._ifft3c(div_hat)
        return float(cp.max(cp.abs(div)).get())

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

        if self.dealias_mode == "phase_shift":
            dxh, dyh, dzh = 0.5*self.dx, 0.5*self.dy, 0.5*self.dz
            # 8 combinations of (0 or 0.5*h) per axis
            self._phases8 = []
            for ax in (0.0, dxh):
                for ay in (0.0, dyh):
                    for az in (0.0, dzh):
                        self._phases8.append(self._phase_factor(ax, ay, az))  # shape (Nx,Ny,zf)

    def _umax(self):
        U = self._ifft3c(self.U_hat)
        umax = float(cp.max(cp.abs(U)).get())
        return umax

    def _nu_eff_max(self):
        if self.les_model != "smagorinsky":
            return self.nu
        # cheap estimate using current field (call when computing dt_four)
        G = self._gradients_real(self.U_hat)
        S = 0.5*(G + cp.swapaxes(G,0,1))
        Ssq = cp.zeros_like(S[0,0])
        for i in range(3):
            for j in range(3):
                Ssq = Ssq + S[i,j]*S[i,j]
        Smag = cp.sqrt(2.0*Ssq) + 1e-30
        nu_t_max = float(cp.max((self.Cs*self.DELTA)**2 * Smag).get())
        return self.nu + nu_t_max

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
            if sol_every is not None and sol_every > 0:
                fname = f"SOLUT/velocity_{n:06d}.vti"
                self.save_velocity_vti(fname)

        while t < T:
            # --- compute adaptive timestep ---
            umax = self._umax()
            dt_cfl  = cfl * self.dx_min / umax if umax > 0.0 else float("inf")
            nu_for_dt = self._nu_eff_max() if viscous_active else 0.0
            dt_four = fourier * alpha_rk4 / (nu_for_dt * k2max) if viscous_active else float("inf")
            dt = min(dt_cfl, dt_four, T - t)

            if max_dt is not None: dt = min(dt, max_dt)
            if min_dt is not None: dt = max(dt, min_dt)
            if not (dt > 0.0 and cp.isfinite(dt)):
                raise RuntimeError("Adaptive dt became non-positive or non-finite.")

            # --- iteration-based callback output ---
            self.U_hat = self._rk4(self.U_hat, dt)
            t += dt
            n += 1
            self.nstep = n

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

    def get_velocity_real(self):
        """Return (u,v,w) as CuPy arrays in real space."""
        U = self._ifft3c(self.U_hat)      # shape (3,Nx,Ny,Nz), CuPy
        return U[0], U[1], U[2]

    def set_velocity_real(self, u, v, w):
        """Set (u,v,w) in real space; updates U_hat with projection to div-free."""
        U = cp.stack([u, v, w], axis=0).astype(self.dtype, copy=False)
        for c in range(3):
            self.U_hat[c] = self._fft3r(U[c]).astype(self.cdtype, copy=False)
        # ensure ∇·u = 0 after any filtering/rescaling
        self.U_hat = self._project_divfree(self.U_hat)

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

        # --- vorticity in real space (via spectral curl) ---
        W_hat = self._curl_hat(self.U_hat)     # (3, Nx, Ny, zf), complex on GPU
        W = self._ifft3c(W_hat)                # (3, Nx, Ny, Nz), real on GPU
        wx, wy, wz = [comp.get() for comp in W]  # CuPy -> NumPy

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

        Wv = np.stack([wx, wy, wz], axis=-1).astype(np.float32)
        Wv_flat = np.ascontiguousarray(Wv.reshape(-1, 3), dtype=np.float32)
        vtk_array = vtknp.numpy_to_vtk(num_array=Wv_flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName("Vorticity")
        imageData.GetPointData().AddArray(vtk_array)
        imageData.GetPointData().SetActiveVectors("Vorticity")

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

    # 2/3 dealiasing
    solver = SpectralDNS(N=128, nu=1.0/1600, precision="float64", dealias_mode="two_thirds")

    solver.prepare_ic(taylor_green_ic)
    solver.run(T=10.0, cfl=0.8, fourier=0.3, log_every=1,
               callback=lambda t, s, _: print(f"[TG] t={s['t']:.3f}, E={s['kinetic_energy']:.6f}"))

    k, E_k, dk = solver.energy_spectrum()
    TKE_spec = cp.sum(E_k * dk)
    TKE_real = solver.kinetic_energy()
    print(f"TKE(real)={TKE_real:.6e}  TKE(spec)={TKE_spec:.6e}  rel.err={(abs(TKE_real-TKE_spec)/TKE_real):.3e}")

    # phase shift dealiasing
    solver = SpectralDNS(N=128, nu=1.0/1600, precision="float64", dealias_mode="phase_shift")

    # Run 1: Taylor–Green for T=0.05
    solver.prepare_ic(taylor_green_ic)
    solver.run(T=10.0, cfl=0.8, fourier=0.3, log_every=1,
               callback=lambda t, s, _: print(f"[TG] t={s['t']:.3f}, E={s['kinetic_energy']:.6f}"))

    k, E_k, dk = solver.energy_spectrum()
    TKE_spec = cp.sum(E_k * dk)
    TKE_real = solver.kinetic_energy()
    print(f"TKE(real)={TKE_real:.6e}  TKE(spec)={TKE_spec:.6e}  rel.err={(abs(TKE_real-TKE_spec)/TKE_real):.3e}")

    # 3/2 padding dealiasing
    solver = SpectralDNS(N=128, nu=1.0/1600, precision="float64", dealias_mode="three_halves")

    # Run 1: Taylor–Green for T=0.05
    solver.prepare_ic(taylor_green_ic)
    solver.run(T=10.0, cfl=0.8, fourier=0.3, log_every=1,
               callback=lambda t, s, _: print(f"[TG] t={s['t']:.3f}, E={s['kinetic_energy']:.6f}"))

    k, E_k, dk = solver.energy_spectrum()
    TKE_spec = cp.sum(E_k * dk)
    TKE_real = solver.kinetic_energy()
    print(f"TKE(real)={TKE_real:.6e}  TKE(spec)={TKE_spec:.6e}  rel.err={(abs(TKE_real-TKE_spec)/TKE_real):.3e}")
