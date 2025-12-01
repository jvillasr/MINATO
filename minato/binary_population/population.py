import numpy as np
import pandas as pd
import kepler


class BinaryPopulation:
    """
    Intrinsic binary population model: draws masses, periods, eccentricities, and
    orbital parameters, with optional fixed-value overrides and Roche/smear guards.
    """

    def __init__(self):
        # Primary mass
        self.M1_min = 8
        self.M1_max = 20
        self.gamma = -2.35
        # Mass ratio
        self.q_min = 0.01
        self.q_max = 1.0
        self.kappa = 0.0
        # Log period
        self.logP_min = -0.15
        self.logP_max = 3.5
        self.pi = 0.0
        # Systemic velocity
        self.gamma_range = (-1, 1)
        # Eccentricity
        self._e_max = 0.95
        self.use_period_ecc_cap = True
        self.eta = -0.5

        self.M1_values = None        # e.g., [8, 15, 20]
        self.logP_values = None      # e.g., [np.log10(2), np.log10(5)]
        self.q_values = None         # e.g., [0.2, 0.4, 0.6]
        self.e_values = None         # e.g., [0.0, 0.2, 0.5]

        # For fixed lists: "random" (default) or "cycle"
        self.fixed_values_mode = "random"
        # Optional weights per parameter for "random" mode
        self.fixed_values_weights = {}   # e.g., {"M1": [0.2, 0.5, 0.3], "q": [0.3, 0.3, 0.4]}

        # Internal pointer for cycle mode (don’t touch)
        self._cycle_idx = {}
        # If fixed e potentially violates caps, what to do? "clip" | "error"
        self.fixed_e_enforcement = "clip"

        # --- period-safety and smear controls ---
        self.use_roche_guard = True      # enforce Roche-safe minimum period
        self.roche_margin_frac = 0.10    # 10% headroom below RL at periastron
        self.use_smear_flag = True       # compute exposure-smear flag (15 min)
        self.t_exp_sec = 900.0           # BOSS exposure ≈ 15 min
        self.dv_smear_limit = 20.0       # km/s allowed intra-exposure Δv

        # Radii source: "proxy" (mass–radius power law) or "iso" (isochrones)
        self.radius_source = "proxy"
        self.R_proxy_alpha = 0.64        # R/Rsun ≈ (M/Msun)^alpha (MS hot stars)
        self.R_proxy_norm = 1.00         # R at 1 Msun

        # to use with isochrones (IsoBank): set these to callable funcs
        self._iso_lookup = None   # function M -> (Teff, logg, R)
        self._iso_age = None      # scalar logAge or callable M -> logAge

    @property
    def e_max(self):
        return self._e_max

    @e_max.setter
    def e_max(self, val):
        val = float(val)
        if not (0.0 <= val < 1.0):
            raise ValueError(f"e_max must be in [0,1), got {val}")
        self._e_max = val

    def summary(self):
        return {
            "fixed_values_mode": self.fixed_values_mode,
            "M1_values": self.M1_values,
            "logP_values": self.logP_values,
            "q_values": self.q_values,
            "e_values": self.e_values,
            "fixed_values_weights": self.fixed_values_weights,
            "fixed_e_enforcement": getattr(self, "fixed_e_enforcement", "clip"),
            "e_max": getattr(self, "e_max", None),
            "use_period_ecc_cap": getattr(self, "use_period_ecc_cap", True),
        }

    # -------------------- Radius support --------------------
    def attach_iso_radius(self, iso_lookup_func, age_source):
        """
        Optional: provide an isochrone-based radius accessor.
        iso_lookup_func: callable (M, logAge) -> (Teff, logg, R_sun)
        age_source:     either a scalar logAge or callable M -> logAge
        """
        self._iso_lookup = iso_lookup_func
        self._iso_age = age_source
        self.radius_source = "iso"

    def _estimate_radius(self, M):
        """Return radius in R_sun for array-like M using current source."""
        M = np.asarray(M, dtype=float)
        if self.radius_source == "iso" and (self._iso_lookup is not None) and (self._iso_age is not None):
            if np.isscalar(self._iso_age):
                logAge = float(self._iso_age)
                R = np.array([self._iso_lookup(m, logAge)[2] for m in M], float)
            else:
                # age_source is callable M -> logAge
                R = np.array([self._iso_lookup(m, float(self._iso_age(m)))[2] for m in M], float)
            return R
        # fallback: simple MS proxy
        return self.R_proxy_norm * np.power(M, self.R_proxy_alpha)

    # -------------------- Roche limit guard --------------------
    @staticmethod
    def _eggleton_rl_over_a(q):
        """Roche-lobe radius over separation for the primary (M2/M1=q)."""
        q23 = np.power(q, 2.0 / 3.0)
        return 0.49 * q23 / (0.6 * q23 + np.log(1.0 + np.power(q, 1.0 / 3.0)))

    def _roche_safe_Pmin_days(self, M1, M2, R1, R2, e):
        """
        Vectorized: minimum orbital period (days) so both stars underfill RLs
        at periastron by (1+roche_margin_frac).
        M in Msun, R in Rsun, e array-like.
        """
        G = 6.67430e-8
        Ms = 1.989e33
        Rs = 6.957e10
        DAY = 86400.0

        M1 = np.asarray(M1, float)
        M2 = np.asarray(M2, float)
        R1 = np.asarray(R1, float)
        R2 = np.asarray(R2, float)
        e = np.asarray(e, float)
        q = np.clip(M2 / M1, 1e-6, 1e6)

        rl1 = self._eggleton_rl_over_a(q)        # RL1/a
        rl2 = self._eggleton_rl_over_a(1.0 / q)  # RL2/a

        a_peri_req = np.maximum(R1 / rl1, R2 / rl2) * (1.0 + self.roche_margin_frac)  # in Rsun
        a = a_peri_req / np.maximum(1.0 - e, 1e-6)                                    # semi-major axis (Rsun)

        a_cm = a * Rs
        Mtot = (M1 + M2) * Ms
        P_sec = 2 * np.pi * np.sqrt(a_cm**3 / (G * Mtot))
        return P_sec / DAY

    # -------------------- Exposure smearing flag --------------------
    def _smear_ok(self, K_kms, P_days):
        """Return boolean array: intra-exposure Δv <= dv_smear_limit."""
        DAY = 86400.0
        K_kms = np.asarray(K_kms, float)
        P_days = np.asarray(P_days, float)
        dv = (2 * np.pi * K_kms / np.maximum(P_days, 1e-9)) * (self.t_exp_sec / DAY)
        return dv <= self.dv_smear_limit

    # -------------------- Period safety clamp --------------------
    def _enforce_roche_guard_on_P(self, M1, q, e, P_days):
        """
        Clamp P to be >= Roche-safe minimum (vectorized).
        Returns (P_safe, n_clamped)
        """
        M1 = np.asarray(M1, float)
        q = np.asarray(q, float)
        e = np.asarray(e, float)
        P = np.asarray(P_days, float)

        M2 = M1 * q
        R1 = self._estimate_radius(M1)
        R2 = self._estimate_radius(M2)

        Pmin = self._roche_safe_Pmin_days(M1, M2, R1, R2, e)
        P_safe = np.maximum(P, Pmin)
        n_clamped = int(np.count_nonzero(P_safe > P + 1e-12))
        return P_safe, n_clamped

    def generate_intrinsic_sample_vectorized(self, N=100, f_bin=0.5, save_sample=False):
        # Number of binaries
        n_bin = int(np.round(f_bin * N))

        # 1) For the binary stars, sample all parameters in one go:
        M1_array = self.draw_M1(n_bin)         # shape (n_bin,)
        logP_array = self.draw_logP(n_bin)     # shape (n_bin,)
        q_array = self.draw_q(n_bin)           # shape (n_bin,)

        # Eccentricities might depend on period, so:
        P_array = 10 ** logP_array
        e_array = self.draw_e_vectorized(P_array)

        # --- remember the drawn P before any safety enforcement
        P_drawn = P_array.copy()
        # --- Roche guard (vectorized clamp) ---
        if self.use_roche_guard:
            P_array, n_clamped = self._enforce_roche_guard_on_P(M1_array, q_array, e_array, P_array)
            if n_clamped > 0:
                print(f"[roche] clamped {n_clamped}/{n_bin} periods to avoid RL overflow.")
            logP_array = np.log10(P_array)
        P_clamped_flag = (P_array > P_drawn + 1e-12)

        # 2) Sample additional orbital parameters for all n_bin at once
        orb_data = self.sample_orbital_extras_vectorized(
            M1_array,
            logP_array,
            q_array,
            e_array,
            gamma_range=self.gamma_range,
        )

        # 3) Build a DataFrame for binary stars:
        df_binaries = pd.DataFrame({
            "M1": M1_array,
            "M2": orb_data["M2"],
            "q": q_array,
            "i": np.degrees(orb_data["i_rad"]),
            "P": orb_data["P"],                     # final P
            "P_drawn": P_drawn,                     # original draw
            "P_clamped": P_clamped_flag,            # boolean
            "e": e_array,
            "Tp": orb_data["Tp"],
            "omega_deg": orb_data["omega_deg"],
            "gamma": orb_data["gamma"],
            "K1": orb_data["K1"],
            "K2": orb_data["K2"],
            "is_binary": True,
            "synthetic_ID": [f"SYN_{i:04d}" for i in range(n_bin)],
        })

        # Exposure smearing diagnostics
        if self.use_smear_flag:
            smear_ok1 = self._smear_ok(df_binaries["K1"].values, df_binaries["P"].values)
            smear_ok2 = self._smear_ok(df_binaries["K2"].values, df_binaries["P"].values)
            dv1 = (2 * np.pi * df_binaries["K1"].values / np.maximum(df_binaries["P"].values, 1e-9)) * (self.t_exp_sec / 86400.0)
            dv2 = (2 * np.pi * df_binaries["K2"].values / np.maximum(df_binaries["P"].values, 1e-9)) * (self.t_exp_sec / 86400.0)
            df_binaries["smear_ok"] = smear_ok1 & smear_ok2
            df_binaries["dv_exp1_kms"] = dv1.astype(np.float32)
            df_binaries["dv_exp2_kms"] = dv2.astype(np.float32)
        else:
            df_binaries["smear_ok"] = np.nan
            df_binaries["dv_exp1_kms"] = np.nan
            df_binaries["dv_exp2_kms"] = np.nan

        # 4) For single stars:
        n_singles = N - n_bin
        df_singles = pd.DataFrame({
            "M1": self.sample_primary_mass(n_singles),
            "M2": np.nan,
            "q": np.nan,
            "i": np.nan,
            "P": np.nan,
            "P_drawn": np.nan,
            "P_clamped": False,
            "e": np.nan,
            "gamma": np.random.uniform(self.gamma_range[0], self.gamma_range[1], size=n_singles),
            "is_binary": False,
            "synthetic_ID": [f"SYN_{(n_bin + i):04d}" for i in range(n_singles)],
            "smear_ok": np.nan,
            "dv_exp1_kms": np.nan,
            "dv_exp2_kms": np.nan,
        })

        intrinsic_df = pd.concat([df_binaries, df_singles], ignore_index=True)

        if save_sample:
            intrinsic_df.to_pickle(f"mock_sample_N{N}_fbin{int(f_bin*100)}_pi{int(self.pi)}.pkl")

        return intrinsic_df

    def add_semi_major_axis(self, df):
        """
        Add the semi-major axis 'a' to the DataFrame in AU units.
        """
        required_columns = {"P", "M1", "q"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns for computing semi-major axis: {missing}")

        # Gravitational constant in AU^3 / (day^2 * solar mass)
        G = 2.959122082855911e-4

        df = df.copy()
        df["M2"] = df["M1"] * df["q"]
        df["a"] = ((G * (df["M1"] + df["M2"]) * (df["P"] ** 2)) / (4 * np.pi ** 2)) ** (1 / 3)
        return df

    def rvcurve(self, t, P, Tp, e, omega_deg, gamma, K1, K2, SB2=False):
        """
        Compute radial velocities for the primary (and optionally secondary).
        """
        omega = omega_deg * np.pi / 180.0
        M = (2 * np.pi / P) * (t - Tp)
        E = kepler.solve(M, e)
        theta = 2.0 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0))

        v1 = gamma + K1 * (np.cos(theta + omega) + e * np.cos(omega))
        if SB2:
            v2 = gamma - K2 * (np.cos(theta + omega) + e * np.cos(omega))
            return v1, v2
        return v1

    ######################################################################
    # Sampling functions
    ######################################################################
    def sample_primary_mass(self, N):
        """
        Sample primary masses from a Salpeter-like IMF dN/dM ~ M^gamma, with M in [self.M1_min, self.M1_max].
        """
        Mmin = self.M1_min
        Mmax = self.M1_max
        gamma = self.gamma

        alpha = gamma + 1.0
        A = (Mmax**alpha - Mmin**alpha)
        u = np.random.random(N)
        masses = ((u * A) + Mmin**alpha) ** (1.0 / alpha)
        return masses

    def sample_logP(self, N):
        """
        Sample x = log10(P) from a PDF f(y) ~ y^pi, y=x-logP_min in [0, y_max].
        """
        pi = self.pi
        logP_min = self.logP_min
        logP_max = self.logP_max

        if pi <= -1.0:
            raise ValueError(
                f"pi={pi} <= -1 is not integrable with logP_min={logP_min} at y=0. "
                "Use pi>-1 or shift logP_min > 0 (excluding sub-day orbits)."
            )
        if logP_min >= logP_max:
            raise ValueError("logP_min must be < logP_max.")

        y_max = logP_max - logP_min
        u = np.random.rand(N)

        if abs(pi) < 1e-9:
            y_samples = np.random.uniform(0.0, y_max, N)
        else:
            y_samples = y_max * (u ** (1.0 / (pi + 1)))

        x_samples = y_samples + logP_min
        return x_samples

    def sample_eccentricity(self, N, e_max=None):
        """
        Sample eccentricities in the interval [0, e_max], following a power-law f(e) ~ e^eta.
        """
        eta = self.eta
        if e_max is None:
            e_max = self.e_max
        if not (0.0 <= e_max < 1.0):
            raise ValueError(f"e_max must be between 0 and 0.99; got {e_max}")
        if abs(eta + 1) < 1e-9:
            raise ValueError("eta = -1 is not supported by this simple approach (logarithmic).")

        u = np.random.rand(N)
        e_samples = e_max * (u ** (1.0 / (eta + 1.0)))
        return e_samples

    def period_ecc_cap(self, P):
        """
        Maximum eccentricity for a given orbital period P.
        """
        return 1.0 - (P / 2) ** (-2.0 / 3.0)

    def sample_ecc(self, P):
        """
        Draw eccentricity based on the orbital period P.
        """
        if P < 2:
            return 0.0
        cap = self.e_max
        if getattr(self, "use_period_ecc_cap", True):
            cap = min(cap, float(self.period_ecc_cap(P)))
        cap = min(max(cap, 0.0), 0.999999)
        return self.sample_eccentricity(1, e_max=cap)[0]

    def sample_ecc_vectorized(self, P_array, tol=1e-5):
        """
        Vectorized draw of eccentricities for an array of periods P_array.
        """
        P = np.asarray(P_array, dtype=float)
        e_cap = np.full(P.shape, float(self.e_max), dtype=float)

        if getattr(self, "use_period_ecc_cap", True):
            e_cap = np.minimum(e_cap, self.period_ecc_cap(P))

        e_cap = np.clip(e_cap, 0.0, 0.999999)

        if abs(self.eta + 1.0) < 1e-9:
            raise ValueError("eta = -1 is not supported by this sampler.")
        u = np.random.rand(P.size)
        e = e_cap * (u ** (1.0 / (self.eta + 1.0)))

        e[P < 2.0] = 0.0
        if tol is not None:
            e[np.abs(e) < tol] = 0.0
        return e

    def sample_q(self, N):
        """
        Sample mass ratio q in [q_min, q_max], possibly with a mild power law q^kappa.
        """
        kappa = self.kappa
        q_min = self.q_min
        q_max = self.q_max

        if abs(kappa) < 1e-3:
            return np.random.uniform(q_min, q_max, N)
        alpha = kappa + 1
        Qmin_alpha = q_min**alpha
        Qmax_alpha = q_max**alpha
        A = Qmax_alpha - Qmin_alpha
        u = np.random.rand(N)
        qvals = ((u * A) + Qmin_alpha) ** (1.0 / alpha)
        return qvals

    def sample_orbital_extras_vectorized(self, M1_array, logP_array, q_array, e_array,
                                         gamma_range=None, inc_mode="random"):
        """
        Vectorized sampling of inclination, omega, gamma, Tp, and K1/K2.
        """
        if gamma_range is None:
            gamma_range = self.gamma_range

        P_array = 10.0 ** logP_array
        M2_array = q_array * M1_array

        if inc_mode == "random":
            cos_i = np.random.uniform(-1, 1, size=M1_array.shape)
            i_array = np.arccos(cos_i)
        else:
            i_array = np.full(M1_array.shape, np.radians(90.0))

        omega_deg_array = np.random.uniform(0, 360, size=M1_array.shape)
        zero_e_mask = (e_array == 0)
        omega_deg_array[zero_e_mask] = 90.0

        gamma_array = np.random.uniform(gamma_range[0], gamma_range[1], size=M1_array.shape)

        Tp_array = np.random.random(size=M1_array.shape) * P_array

        G = 4.309e-3 * 3.0857e13  # combined factor
        P_sec_array = P_array * 86400.0

        denom = (M1_array + M2_array) ** (2 / 3)
        sin_i = np.sin(i_array)
        factor = (2 * np.pi * G) ** (1 / 3) * (P_sec_array ** (-1 / 3))

        K1_array = factor * (M2_array * sin_i) / denom
        K2_array = factor * (M1_array * sin_i) / denom

        return {
            "P": P_array,
            "M2": M2_array,
            "i_rad": i_array,
            "omega_deg": omega_deg_array,
            "gamma": gamma_array,
            "Tp": Tp_array,
            "K1": K1_array,
            "K2": K2_array,
        }

    def _draw_from_values(self, name, values, n, rng=None, mode="random", weights=None):
        """Return length-n array drawn from 'values' by mode; updates a cycle index per name."""
        if rng is None:
            rng = np.random

        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name}_values must be a 1D non-empty list/array")

        if mode == "random":
            p = None
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                if w.shape != values.shape:
                    raise ValueError(f"{name} weights must match number of values")
                p = w / w.sum()
            return rng.choice(values, size=n, replace=True, p=p)

        if mode == "cycle":
            k = values.size
            start = self._cycle_idx.get(name, 0)
            idx = (np.arange(start, start + n) % k)
            self._cycle_idx[name] = (start + n) % k
            return values[idx]

        raise ValueError(f"Unknown fixed_values_mode: {mode}")

    def draw_M1(self, n, rng=None):
        if self.M1_values is not None:
            return self._draw_from_values(
                "M1",
                self.M1_values,
                n,
                rng,
                mode=self.fixed_values_mode,
                weights=self.fixed_values_weights.get("M1"),
            )
        return self.sample_primary_mass(n)

    def draw_logP(self, n, rng=None):
        if self.logP_values is not None:
            return self._draw_from_values(
                "logP",
                self.logP_values,
                n,
                rng,
                mode=self.fixed_values_mode,
                weights=self.fixed_values_weights.get("logP"),
            )
        return self.sample_logP(n)

    def draw_q(self, n, rng=None):
        if self.q_values is not None:
            return self._draw_from_values(
                "q",
                self.q_values,
                n,
                rng,
                mode=self.fixed_values_mode,
                weights=self.fixed_values_weights.get("q"),
            )
        return self.sample_q(n)

    def draw_e_vectorized(self, P_array, rng=None):
        """
        If e_values is provided, draw from it; otherwise use the vectorized sampler.
        Applies P<2 circularization and caps (global e_max + period cap) if needed.
        """
        P = np.asarray(P_array, dtype=float)
        n = P.size

        if self.e_values is None:
            return self.sample_ecc_vectorized(P)

        e = self._draw_from_values(
            "e",
            self.e_values,
            n,
            rng,
            mode=self.fixed_values_mode,
            weights=self.fixed_values_weights.get("e"),
        ).astype(float)

        e[P < 2.0] = 0.0

        cap = float(self.e_max)
        if getattr(self, "use_period_ecc_cap", True):
            cap = np.minimum(cap, self.period_ecc_cap(P))
        cap = np.clip(cap, 0.0, 0.999999)

        if self.fixed_e_enforcement == "clip":
            e = np.minimum(e, cap)
        elif self.fixed_e_enforcement == "error":
            if np.any(e > cap + 1e-12):
                raise ValueError("Fixed e values exceed allowed cap for some periods.")
        else:
            raise ValueError(f"Unknown fixed_e_enforcement: {self.fixed_e_enforcement}")

        return e
