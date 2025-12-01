import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kepler
import sys
# sys.path.append('/Users/villasenor/science/github/jvillasr/MINATO')
# from minato import myRC
# import emcee
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocess import Pool as MP_Pool

class BinarySimulations:
    def __init__(self):
        """
        Fixed-value options (cheat sheet)

        What these do

        They let you override the usual random draws and force certain parameters to take values from lists you provide.
            •	M1_values: Optional[list[float]] — allowed primary masses (applies to binaries and singles when you draw masses for all stars).
            •	logP_values: Optional[list[float]] — allowed log10(P/days) values for binaries.
            •	q_values: Optional[list[float]] — allowed mass-ratio values for binaries.
            •	e_values: Optional[list[float]] — allowed eccentricities for binaries. These are still constrained by physics (see “Eccentricity rules”).

        How they are used:
            •	fixed_values_mode: "random" | "cycle"
            •	"random" (default): pick with replacement from the list (optionally weighted).
            •	"cycle": walk through your list in order and wrap around as needed (deterministic given a seed).
            •	fixed_values_weights: dict[str, list[float]] (only if mode=“random”)
        Per-parameter weights, e.g. {"M1": [0.2, 0.5, 0.3]} must match the length of M1_values.

        Eccentricity rules (important)
            •	Global maximum: e_max (float in [0,1)) — user knob, e.g. simulator.e_max = 0.9.
            •	Period cap: period_ecc_cap(P) (method). Enabled by use_period_ecc_cap = True.
            •	For P < 2 d, we set e = 0 (circularized).
            •	When e_values is used, we still enforce caps:
            •	fixed_e_enforcement: "clip" | "error"
        "clip" lowers any too-large e to the allowed cap; "error" raises an exception if a provided e exceeds the cap for some period.

        The simplest mental model
            •	If you set a *_values list → that parameter will be drawn from your list (by random or cycle).
            •	If you don’t set it (leave as None) → the usual distribution is used.
            •	You can mix: e.g., fix M1 and q, but sample logP and e from distributions.
        """
        # Default parameters
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

        self.M1_values   = None       # e.g., [8, 15, 20]
        self.logP_values = None       # e.g., [np.log10(2), np.log10(5)]
        self.q_values    = None       # e.g., [0.2, 0.4, 0.6]
        self.e_values    = None       # e.g., [0.0, 0.2, 0.5]  (see note on caps below)

        # For fixed lists: "random" (default) or "cycle"
        self.fixed_values_mode = "random"
        # Optional weights per parameter for "random" mode
        self.fixed_values_weights = {}   # e.g., {"M1": [0.2, 0.5, 0.3], "q": [0.3, 0.3, 0.4]}

        # Internal pointer for cycle mode (don’t touch)
        self._cycle_idx = {}
        # If fixed e potentially violates caps, what to do? "clip" | "error"
        self.fixed_e_enforcement = "clip"

        # Storage for data and results
        self.coverage_dict = None
        self.real_dRV = None
        self.mock_data = None
        self.sampler = None
        self.obs_results_df = None

        # --- NEW: period-safety controls ---
        self.use_roche_guard   = True      # enforce Roche-safe minimum period
        self.roche_margin_frac = 0.10      # 10% headroom below RL at periastron
        self.use_smear_flag    = True      # compute exposure-smear flag (15 min)
        self.t_exp_sec         = 900.0     # BOSS exposure ≈ 15 min
        self.dv_smear_limit    = 20.0      # km/s allowed intra-exposure Δv (≈0.2 FWHM)

        # Radii source: "proxy" (mass–radius power law) or "iso" (isochrones)
        self.radius_source     = "proxy"
        self.R_proxy_alpha     = 0.64      # R/Rsun ≈ (M/Msun)^alpha (MS hot stars)
        self.R_proxy_norm      = 1.00      # R at 1 Msun

        # to use with isochrones (IsoBank): set these to callable funcs
        self._iso_lookup = None   # function M -> (Teff, logg, R)
        self._iso_age    = None   # scalar logAge or callable M -> logAge

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

    def load_data(self, df: pd.DataFrame) -> dict:
        """
        Reads a pandas DaraFrame with columns: ID, MJD, mean_rv_er.
        Returns a dictionary keyed by star_id, 
        each item is a tuple of arrays (mjds, rv_errors).
        """
        # Define the required columns.
        required_columns = {"ID", "MJD", "mean_rv_er"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")
        # Build the coverage dictionary.
        self.coverage_dict = {}
        for star_id, group in df.groupby('ID'):
            group_sorted = group.sort_values(by='MJD')
            mjds = group_sorted['MJD'].values
            rv_errs = group_sorted['mean_rv_er'].values
            self.coverage_dict[star_id] = (mjds, rv_errs)
        return self.coverage_dict

# -------------------- Radius support --------------------
    def attach_iso_radius(self, iso_lookup_func, age_source):
        """
        Optional: provide an isochrone-based radius accessor.
        iso_lookup_func: callable (M, logAge) -> (Teff, logg, R_sun)
        age_source:     either a scalar logAge or callable M -> logAge
        """
        self._iso_lookup = iso_lookup_func
        self._iso_age    = age_source
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
        q23 = np.power(q, 2.0/3.0)
        return 0.49*q23 / (0.6*q23 + np.log(1.0 + np.power(q, 1.0/3.0)))

    def _roche_safe_Pmin_days(self, M1, M2, R1, R2, e):
        """
        Vectorized: minimum orbital period (days) so both stars underfill RLs
        at periastron by (1+roche_margin_frac).
        M in Msun, R in Rsun, e array-like.
        """
        G   = 6.67430e-8
        Ms  = 1.989e33
        Rs  = 6.957e10
        DAY = 86400.0

        M1 = np.asarray(M1, float); M2 = np.asarray(M2, float)
        R1 = np.asarray(R1, float); R2 = np.asarray(R2, float)
        e  = np.asarray(e,  float)
        q = np.clip(M2/M1, 1e-6, 1e6)

        rl1 = self._eggleton_rl_over_a(q)        # RL1/a
        rl2 = self._eggleton_rl_over_a(1.0/q)    # RL2/a

        a_peri_req = np.maximum(R1/rl1, R2/rl2) * (1.0 + self.roche_margin_frac)   # in Rsun
        a = a_peri_req / np.maximum(1.0 - e, 1e-6)                                 # semi-major axis (Rsun)

        a_cm = a * Rs
        Mtot = (M1 + M2) * Ms
        P_sec = 2*np.pi*np.sqrt(a_cm**3 / (G*Mtot))
        return P_sec / DAY

    # -------------------- Exposure smearing flag --------------------
    def _smear_ok(self, K_kms, P_days):
        """Return boolean array: intra-exposure Δv <= dv_smear_limit."""
        DAY = 86400.0
        K_kms = np.asarray(K_kms, float)
        P_days = np.asarray(P_days, float)
        dv = (2*np.pi * K_kms / np.maximum(P_days, 1e-9)) * (self.t_exp_sec / DAY)
        return dv <= self.dv_smear_limit

    # -------------------- Period safety clamp --------------------
    def _enforce_roche_guard_on_P(self, M1, q, e, P_days):
        """
        Clamp P to be >= Roche-safe minimum (vectorized).
        Returns (P_safe, n_clamped)
        """
        M1 = np.asarray(M1, float)
        q  = np.asarray(q,  float)
        e  = np.asarray(e,  float)
        P  = np.asarray(P_days, float)

        M2 = M1 * q
        R1 = self._estimate_radius(M1)
        R2 = self._estimate_radius(M2)

        Pmin = self._roche_safe_Pmin_days(M1, M2, R1, R2, e)
        P_safe = np.maximum(P, Pmin)
        n_clamped = int(np.count_nonzero(P_safe > P + 1e-12))
        return P_safe, n_clamped

    def simulate_mock_observations(self, pi=0.0, kappa=0.0, eta=-0.5, gamma=-2.35, 
                                N=100, f_bin=0.5, save_sample=False, intrinsic_sample=None, ideal_sampling=False,
                                n_epochs=None, rv_error_common=None, seed=None):
        """
        Parent function to generate (or load) the intrinsic mock sample and simulate observations.
        
        Parameters:
        pi, kappa, eta, gamma: Parameters for the binary distributions.
        N: Total number of stars.
        f_bin: Binary fraction.
        # observation_strategy: Dictionary specifying "t_array" and "rv_errors". For example:
        #                         {"t_array": np.array([0, 30, 300]),
        #                         "rv_errors": np.array([0.5, 0.5, 0.5])}
        save_sample: If True, save the generated intrinsic sample to disk.
        intrinsic_sample: Optionally, an already-generated intrinsic sample (DataFrame).
        ideal_sampling:
            False -> use real survey cadence from self.coverage_dict (requires load_data()).
            True  -> use two quadratures (existing behavior).
            "phase_uniform" -> use n_epochs points uniformly spaced in phase (NEW).
        n_epochs: int, required when ideal_sampling == "phase_uniform".
        rv_error_common: float [km/s], per-epoch RV uncertainty for synthetic cadences.
        seed: int or None. If provided, sets NumPy RNG for reproducible simulations.
        
        Returns:
        obs_results_df: A DataFrame with the simulated observation results.
        """
        if seed is not None:
            np.random.seed(int(seed))
        self._last_seed = int(seed) if seed is not None else None

        if self.coverage_dict is None and ideal_sampling is False:
            raise ValueError("No coverage data loaded. Please load observational data using load_data before simulating observations.")

        # If no intrinsic sample is provided, try loading from file; otherwise, generate it.
        if intrinsic_sample is None:
            # print("Generating intrinsic sample ...")
            intrinsic_sample = self.generate_intrinsic_sample_vectorized(N=N, f_bin=f_bin, 
                                                            save_sample=save_sample)
            # intrinsic_sample_nonvec = generate_intrinsic_sample(N=N, f_bin=f_bin, 
            #                                                 pi=pi, kappa=kappa, eta=eta, gamma=gamma,
            #                                                 save_file=save_file)
        else:
            print("Using provided intrinsic sample ...")
            # intrinsic_sample = pd.read_pickle(intrinsic_sample)

        # plot_sampled_params(intrinsic_sample)

        # Now simulate observations using the chosen observing strategy.
        self.obs_results_df = self.compute_rvs(intrinsic_sample, ideal_sampling=ideal_sampling,
                                            n_epochs=n_epochs, rv_error_common=rv_error_common)
        return self.obs_results_df

    def generate_intrinsic_sample_vectorized(self, N=100, f_bin=0.5, save_sample=False):
        # Number of binaries
        n_bin = int(np.round(f_bin * N))
        
        # 1) For the binary stars, sample all parameters in one go:
        M1_array = self.draw_M1(n_bin)      # shape (n_bin,)
        logP_array = self.draw_logP(n_bin)     # shape (n_bin,)
        q_array = self.draw_q(n_bin)           # shape (n_bin,)

        # Eccentricities might depend on period, so:
        P_array = 10 ** logP_array
        e_array = self.draw_e_vectorized(P_array)

        # --- NEW: remember the drawn P before any safety enforcement
        P_drawn = P_array.copy()
        # --- Roche guard (vectorized clamp) ---
        if self.use_roche_guard:
            P_array, n_clamped = self._enforce_roche_guard_on_P(M1_array, q_array, e_array, P_array)
            if n_clamped > 0:
                print(f"[roche] clamped {n_clamped}/{n_bin} periods to avoid RL overflow.")
            logP_array = np.log10(P_array)
        P_clamped_flag = (P_array > P_drawn + 1e-12)

        # 2) Sample additional orbital parameters for all n_bin at once
        orb_data = self.sample_orbital_extras_vectorized(M1_array, logP_array, q_array, e_array, 
                                                    gamma_range=self.gamma_range) # for BLOeM: (100,240)
        
        # 3) Build a DataFrame for binary stars:
        df_binaries = pd.DataFrame({
            'M1': M1_array,
            'M2': orb_data['M2'],
            'q': q_array,
            'i': np.degrees(orb_data['i_rad']),
            'P': orb_data['P'],                     # final P
            'P_drawn': P_drawn,                     # NEW: original draw
            'P_clamped': P_clamped_flag,            # NEW: boolean
            'e': e_array,
            'Tp': orb_data['Tp'],
            'omega_deg': orb_data['omega_deg'],
            'gamma': orb_data['gamma'],
            'K1': orb_data['K1'],
            'K2': orb_data['K2'],
            'is_binary': True,
            'synthetic_ID': [f"SYN_{i:04d}" for i in range(n_bin)]
        })

        # --- NEW: 15-min exposure smearing diagnostics ---
        if self.use_smear_flag:
            smear_ok1 = self._smear_ok(df_binaries['K1'].values, df_binaries['P'].values)
            smear_ok2 = self._smear_ok(df_binaries['K2'].values, df_binaries['P'].values)
            dv1 = (2*np.pi * df_binaries['K1'].values / np.maximum(df_binaries['P'].values, 1e-9)) * (self.t_exp_sec/86400.0)
            dv2 = (2*np.pi * df_binaries['K2'].values / np.maximum(df_binaries['P'].values, 1e-9)) * (self.t_exp_sec/86400.0)
            df_binaries['smear_ok'] = smear_ok1 & smear_ok2
            df_binaries['dv_exp1_kms'] = dv1.astype(np.float32)
            df_binaries['dv_exp2_kms'] = dv2.astype(np.float32)

        # 4) For single stars:
        n_singles = N - n_bin
        # You can sample single-star masses if needed, or just fill placeholders:
        df_singles = pd.DataFrame({
            'M1'       : self.sample_primary_mass(n_singles),
            'M2'       : np.nan,  # or zero, if you prefer
            'q'        : np.nan,
            'i'        : np.nan,
            'P'        : np.nan,
            'P_drawn'  : np.nan,
            'P_clamped': False,
            'e'        : np.nan,
            'gamma': np.random.uniform(self.gamma_range[0], self.gamma_range[1], size=n_singles),
            'is_binary': False,
            'synthetic_ID': [f"SYN_{(n_bin + i):04d}" for i in range(n_singles)]
        })

        # 5) Combine them:
        intrinsic_df = pd.concat([df_binaries, df_singles], ignore_index=True)
        
        if save_sample:
            intrinsic_df.to_pickle(f'mock_sample_N{N}_fbin{int(f_bin*100)}_pi{int(self.pi)}.pkl')

        return intrinsic_df

    def _compute_sigmad_vectorized(self, rv_obs, rv_errors):
        rv_obs = np.asarray(rv_obs)
        rv_errors = np.asarray(rv_errors)
        diff = np.abs(rv_obs[:, None] - rv_obs[None, :])
        err = np.sqrt(rv_errors[:, None]**2 + rv_errors[None, :]**2)
        sigma = diff / err
        return np.max(sigma)

    def two_quadratures(self, P, Tp, e, omega_deg, n_fallback=8000):
        """
        Return the times of maximum and minimum RV over one orbit.
        Analytic extrema; falls back to a dense grid with rvcurve if anything looks odd.
        """
        omega = np.deg2rad(omega_deg)

        # RV extrema at theta = -omega and theta = pi - omega
        thetas = np.array([-omega, np.pi - omega], dtype=float)

        # True anomaly -> Eccentric anomaly (robust formulas)
        cosE = (e + np.cos(thetas)) / (1.0 + e * np.cos(thetas))
        sinE = (np.sqrt(1.0 - e**2) * np.sin(thetas)) / (1.0 + e * np.cos(thetas))
        E = np.arctan2(sinE, cosE)

        # Mean anomaly -> time since periastron
        M = (E - e * np.sin(E)) % (2.0 * np.pi)
        t_ext = Tp + (P / (2.0 * np.pi)) * M
        t_ext.sort()

        # Very defensive fallback (NaNs, degeneracy): dense grid using rvcurve
        if not np.all(np.isfinite(t_ext)) or np.isclose(t_ext[0], t_ext[1]):
            t_grid = Tp + np.linspace(0.0, P, int(n_fallback), endpoint=False)
            v1 = self.rvcurve(t_grid, P, Tp, e, omega_deg, gamma=0.0, K1=1.0, K2=0.0, SB2=False)
            i_max, i_min = np.argmax(v1), np.argmin(v1)
            t_ext = np.array([t_grid[i_max], t_grid[i_min]], dtype=float)

        return t_ext
    
    def phase_grid(self, P, Tp, n_epochs):
        """Return n_epochs times uniformly spaced in orbital phase over one period."""
        phases = np.linspace(0.0, 1.0, int(n_epochs), endpoint=False)
        return Tp + phases * P

    def compute_rvs(self, intrinsic_df, ideal_sampling=False, n_epochs=None, rv_error_common=None):
        """
        Simulate RV observations for each star in the intrinsic sample based on the specified observing strategy.
        
        Parameters:
        intrinsic_df: DataFrame containing the intrinsic parameters of the simulated stars.
        # observation_strategy: Dictionary with keys:
        #                         "t_array": 1D array of observation times (e.g., [0, 30, 300])
        #                         "rv_errors": 1D array of RV uncertainties for each epoch.
        ideal_sampling:
            False           -> real cadence & errors from coverage_dict (requires load_data)
            True            -> 2 quadratures; errors = fixed if rv_error_common else sampled from coverage
            "phase_uniform" -> n_epochs uniform in orbital phase; requires rv_error_common (singles not allowed)

        Returns:
        obs_results_df: A DataFrame with the simulated observation results.
        """
        results_obs = []

        intrinsic_df = self.add_semi_major_axis(intrinsic_df)
        has_coverage = self.coverage_dict is not None
        real_star_ids = list(self.coverage_dict.keys()) if has_coverage else []

        # default n_epochs if user forgets (only matters for synthetic cadences)
        if ideal_sampling == "phase_uniform" and (n_epochs is None or int(n_epochs) < 1):
            n_epochs = 8

        if ideal_sampling is True and (n_epochs is not None) and int(n_epochs) != 2:
            raise ValueError(
                "ideal_sampling=True uses two_quadratures (2 epochs). "
                "To request N>2 epochs, set ideal_sampling='phase_uniform'."
            )

        for idx, row in intrinsic_df.iterrows():
            chosen_id = "synthetic"  # <-- default, so later 'real_ID_used' is always defined

            if row['is_binary'] and ideal_sampling is True:
                # two quadratures
                t_array = self.two_quadratures(row['P'], row['Tp'], row['e'], row['omega_deg'])

                if rv_error_common is not None:
                    rv_errors = np.full(len(t_array), float(rv_error_common), dtype=float)
                else:
                    if not has_coverage:
                        raise ValueError("For two-quadrature sampling without coverage, rv_error_common must be provided.")
                    # we *do* need coverage for errors → pick a real star now
                    chosen_id = np.random.choice(real_star_ids)
                    _, rv_errors_full = self.coverage_dict[chosen_id]
                    rv_errors = np.random.choice(np.asarray(rv_errors_full).ravel(), size=len(t_array), replace=True)

            elif row['is_binary'] and ideal_sampling == "phase_uniform":
                t_array = self.phase_grid(row['P'], row['Tp'], int(n_epochs))

                if rv_error_common is not None:
                    rv_errors = np.full(len(t_array), float(rv_error_common), dtype=float)
                else:
                    raise ValueError("For 'phase_uniform' sampling, rv_error_common must be provided.")

            elif (row['is_binary'] is False) and (ideal_sampling is not False):
                # your explicit rule for singles
                raise ValueError("Ideal sampling modes are only applicable to binary stars; single stars require a coverage dictionary (ideal_sampling=False).")

            elif ideal_sampling is False:
                if not has_coverage:
                    raise ValueError("Real cadence requires coverage_dict; call load_data().")
                # real cadence → we need coverage for times and errors
                chosen_id = np.random.choice(real_star_ids)
                t_array, rv_errors = self.coverage_dict[chosen_id]

            else:
                raise ValueError(f"Unsupported ideal_sampling value: {ideal_sampling!r}")

            # For binary stars, compute the RV curve; for single stars, use gamma.
            if row['is_binary']:
                # Calculate the true RV curve. We pass SB2=True so that for binaries rvcurve returns a tuple.
                v_orbit = self.rvcurve(t_array, row['P'], row['Tp'], row['e'],
                                        row['omega_deg'], row['gamma'],
                                        row['K1'], row['K2'], SB2=True)
            else:
                # For single stars, there are no orbital variations.
                v_orbit = np.full(t_array.shape, row['gamma'], dtype=float)

            # Generate Gaussian noise once per epoch set
            noise   = np.random.normal(0, rv_errors, size=len(t_array))

            # If rvcurve returned two components -> SB2
            if isinstance(v_orbit, tuple):
                v1_true, v2_true = v_orbit
                rv_obs1 = v1_true + noise
                rv_obs2 = v2_true + noise
            else:                        # SB1 or single star
                v1_true = v_orbit
                v2_true = np.full_like(v1_true, np.nan)  # No secondary RV for single stars
                rv_obs1 = v1_true + noise
                rv_obs2 = np.full_like(rv_obs1, np.nan)

            dRV_true = np.ptp(v1_true)

            n_eps = len(t_array)
            rv_mean = np.mean(rv_obs1)
            # Compute the maximum delta RV and the detection significance.
            dRV = rv_obs1.max() - rv_obs1.min()
            sigma_detect = self._compute_sigmad_vectorized(rv_obs1, rv_errors)
            
            results_obs.append({
                'synthetic_ID': row['synthetic_ID'],
                'real_ID_used': chosen_id,
                'is_binary': row['is_binary'],
                'M1': row['M1'], 'q': row['q'],
                'P': row['P'], 'P_drawn': row['P_drawn'], 'P_clamped': row['P_clamped'],
                'Tp': row['Tp'], 'e': row['e'], 'omega_deg': row['omega_deg'], 
                'gamma': row['gamma'], 'K1': row['K1'], 'K2': row['K2'], 
                'i_deg': row['i'], 'a': row['a'],
                'mjd_array': t_array,
                'dRV_true': dRV_true,
                'dRV_max': dRV,
                'sigma_d': sigma_detect,
                'rv_mean': rv_mean,
                'n_eps': n_eps,
                'rv_true': v1_true,
                'rv_true2': v2_true,
                'rv_array': rv_obs1,
                'rv_array2': rv_obs2,  # Store secondary RVs
                'rv_errors': rv_errors,
                'smear_ok': row['smear_ok'],
                'dv_exp1_kms': row['dv_exp1_kms'],
                'dv_exp2_kms': row['dv_exp2_kms'],
                'rng_seed': getattr(self, '_last_seed', None)
                # additional parameters if needed
            })

        return pd.DataFrame(results_obs)

    def add_semi_major_axis(self, df):
        """
        Add the semi-major axis 'a' to the DataFrame in AU units.

        Parameters:
        df (pd.DataFrame): DataFrame containing the columns 'P', 'M1', and 'q'.

        Returns:
        pd.DataFrame: DataFrame with the added column 'a'.
        """
        # Check for required columns.
        required_columns = {'P', 'M1', 'q'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f'DataFrame is missing required columns for computing semi-major axis: {missing}')
        
        # Gravitational constant in AU^3 / (day^2 * solar mass)
        G = 2.959122082855911e-4

        # Compute M2 from M1 and q
        df['M2'] = df['M1'] * df['q']

        # Compute the semi-major axis 'a' in AU
        df['a'] = ((G * (df['M1'] + df['M2']) * (df['P']**2)) / (4 * np.pi**2))**(1/3)

        return df

    def rvcurve(self, t, P, Tp, e, omega_deg, gamma, K1, K2, SB2=False):
        """
        Compute radial velocities for the primary (and optionally secondary)
        given:
        t: times (array)
        P: orbital period
        Tp: reference time of periastron (same units as t)
        e: eccentricity
        omega_deg: argument of periastron (degrees)
        gamma: systemic velocity
        K1, K2: RV semi-amplitudes of primary, secondary
        SB2: whether to return both v1 and v2
        """
        # Convert omega to radians
        omega = omega_deg * np.pi / 180.0
        
        # Compute mean anomaly (can be negative or > 2π, that's okay)
        M = (2 * np.pi / P) * (t - Tp)
        
        # If your solver needs M mod 2π, do it once:
        # M = M % (2*np.pi)   # optional, depending on the kepler.solve function
        
        # Solve Kepler's Equation for E (eccentric anomaly)
        E = kepler.solve(M, e)
        
        # True anomaly
        theta = 2.0 * np.arctan(
            np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2.0)
        )
        
        # Primary radial velocity:
        v1 = gamma + K1 * (np.cos(theta + omega) + e * np.cos(omega))
        
        if SB2:
            # If you want secondary velocities, similarly:
            v2 = gamma - K2 * (np.cos(theta + omega) + e * np.cos(omega))
            return v1, v2
        else:
            return v1


    def plot_sampled_params(self):
        df = self.obs_results_df

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 3, figsize=(20, 12))
        fig.subplots_adjust(hspace=0.35)
        # Plot the distribution of P
        axs[0, 0].hist(df['P'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        axs[0, 0].set_xlabel(r'$P_{\rm orb}$ [d]')
        axs[0, 0].set_ylabel('Number')
        # axs[0, 0].set_title('Distribution of Orbital Period (P)')
        axs[0, 0].grid(True)

        # Plot the distribution of M1
        axs[0, 1].hist(df['M1'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        axs[0, 1].set_xlabel('$M_1$ [M$_{\odot}$]')
        # axs[0, 1].set_ylabel('Number')
        # axs[0, 1].set_title('Distribution of Primary Mass (M1)')
        axs[0, 1].grid(True)

        # Plot the distribution of Tp
        axs[0, 2].hist(df['Tp'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[0, 2].hist(intrinsic_sample_nonvec['Tp'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[0, 2].set_xlabel('$T_p$ [d]')
        # axs[0, 2].set_ylabel('Number')
        # axs[0, 2].set_title('Distribution of Time of Periastron Passage (Tp)')
        axs[0, 2].grid(True)

        # Plot the distribution of q
        axs[1, 0].hist(df['q'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[1, 0].hist(intrinsic_sample_nonvec['q'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[1, 0].set_xlabel('Mass Ratio ($q$)')
        axs[1, 0].set_ylabel('Number')
        # axs[1, 0].set_title('Distribution of Mass Ratio (q)')
        axs[1, 0].grid(True)

        # Plot the distribution of i
        axs[1, 1].hist(df['i_deg'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[1, 1].hist(intrinsic_sample_nonvec['i'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[1, 1].set_xlabel('Inclination ($i$) [deg]')
        # axs[1, 1].set_ylabel('Number')
        # axs[1, 1].set_title('Distribution of Inclination (i)')
        axs[1, 1].grid(True)

        # Plot the distribution of omega_deg
        axs[1, 2].hist(df['omega_deg'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[1, 2].hist(intrinsic_sample_nonvec['omega_deg'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[1, 2].set_xlabel('Argument of Periastron ($\omega$) [deg]')
        # axs[1, 2].set_ylabel('Number')
        # axs[1, 2].set_title('Distribution of Argument of Periastron (omega)')
        axs[1, 2].grid(True)

        # Plot the distribution of K1
        axs[2, 0].hist(df['K1'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[2, 0].hist(intrinsic_sample_nonvec['K1'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[2, 0].set_xlabel('$K_1$ [km\,s$^{-1}$]')
        axs[2, 0].set_ylabel('Number')
        # axs[2, 0].set_title('Distribution of Radial Velocity Semi-Amplitude (K1)')
        axs[2, 0].grid(True)

        # Plot the distribution of e
        axs[2, 1].hist(df['e'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[2, 1].hist(intrinsic_sample_nonvec['e'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[2, 1].set_xlabel('Eccentricity ($e$)')
        # axs[2, 1].set_ylabel('Number of Binaries')
        # axs[2, 1].set_title('Distribution of Eccentricity (e)')
        axs[2, 1].grid(True)

        # Plot the distribution of gamma
        axs[2, 2].hist(df['gamma'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[2, 2].hist(intrinsic_sample_nonvec['gamma'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[2, 2].set_xlabel('Systemic Velocity ($\gamma$) [km\,s$^{-1}$]')
        # axs[2, 2].set_ylabel('Number of Binaries')
        # axs[2, 2].set_title('Distribution of Systemic Velocity (gamma)')
        axs[2, 2].grid(True)

        # Adjust layout
        # plt.tight_layout()
        plt.savefig('parameter_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    ######################################################################
    # Sampling functions
    ######################################################################

    def sample_primary_mass(self, N):
        """
        Sample primary masses from a Salpeter-like IMF 
        dN/dM ~ M^gamma, with M in [self.M1_min, self.M1_max].
        """
        Mmin = self.M1_min
        Mmax = self.M1_max
        gamma = self.gamma
        # Inverse-transform sampling for a power law:
        # Cumulative distribution for M^(gamma+1).
        # If gamma != -1, the formula is:
        #   F(M) = [M^(gamma+1) - Mmin^(gamma+1)] / [Mmax^(gamma+1) - Mmin^(gamma+1)]
        
        alpha = gamma + 1.0
        A = (Mmax**alpha - Mmin**alpha)
        # Uniform random [0,1]
        u = np.random.random(N)
        
        # invert:
        masses = ((u * A) + Mmin**alpha)**(1./alpha)
        return masses

    def sample_logP(self, N):
        """
        Sample x = log10(P) from a PDF:
            f(y) ~ y^pi,  where y = x - logP_min in [0, y_max],
        and x in [logP_min, logP_max], subject to pi > -1.

        Parameters
        ----------
        N : int
            Number of samples to draw.
        pi : float
            Power-law exponent in the shifted variable y. Must satisfy pi > -1.
            - pi = 0 -> uniform in [logP_min, logP_max].
            - -0.5, -0.9, etc. -> mild favoring of short periods if logP_min < 0.
        logP_min : float
            Lower bound for x = log10(P). 
            If logP_min = -0.3 => P_min ~ 0.5 days.
        logP_max : float
            Upper bound for x = log10(P).
            If logP_max = 3.5 => P_max ~ 3162 days (~8.7 years).

        Returns
        -------
        x_samples : np.ndarray
            Array of length N with log10(P) values in [logP_min, logP_max].

        Notes
        -----
        1) We shift the domain: y = x - logP_min  in [0, y_max], where y_max = logP_max - logP_min.
        2) PDF in y-space:  f(y) ~ y^pi, integrable near y=0 iff pi > -1.
        3) Inverse CDF approach:
            if pi=0 => uniform in [0, y_max],
            else => y = y_max * u^(1/(pi+1)),  u in [0,1].
        4) Then shift back x = y + logP_min.

        Examples
        --------
        # 1) Generate 1000 logP for short periods down to 0.5 d, up to 3162 d,
        #    with a mild short-period bias pi = -0.5.
        #>>> logP_samples = sample_logP(1000, pi=-0.5, logP_min=-0.3, logP_max=3.5)
        #>>> P_days = 10**logP_samples
        
        # 2) Uniform in logP (Öpik's law), from 0.5 d to 3162 d:
        #>>> logP_samples = sample_logP(1000, pi=0.0, logP_min=-0.3, logP_max=3.5)
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
        
        # Domain shift: y in [0, y_max], where y = x - logP_min
        y_min = 0.0
        y_max = logP_max - logP_min  # must be > 0
        
        # Sample y
        u = np.random.rand(N)  # uniform deviate in [0, 1]
        
        if abs(pi) < 1e-9:
            # pi ~ 0 => uniform in [0, y_max]
            y_samples = np.random.uniform(y_min, y_max, N)
        else:
            # f(y) ~ y^pi => CDF: F(y) = y^(pi+1) / y_max^(pi+1)
            # => y = y_max * (u)^(1/(pi+1)), valid only if pi>-1
            y_samples = y_max * (u ** (1.0 / (pi + 1)))
        
        # Shift back
        x_samples = y_samples + logP_min  # x in [logP_min, logP_max]
        
        return x_samples

    def sample_eccentricity(self, N, e_max=None):
        """
        Sample eccentricities in the interval [0, e_max], following a power-law:
            f(e) ~ e^eta   for 0 <= e <= e_max
        using inverse-transform sampling.

        Parameters
        ----------
        N : int
            Number of eccentricities to draw.
        eta : float, optional
            Power-law exponent for e^eta. Default = -0.5.
        e_max : float, optional
            Upper bound for eccentricities. Default = 0.95.

        Returns
        -------
        e_samples : np.ndarray
            Array of shape (N,) with eccentricities in [0, e_max].

        Notes
        -----
        - If eta = -1, the distribution becomes log(e). That requires a separate approach.
        - For 0 <= e < 2 days orbits (very short period) you often set e=0 manually in code elsewhere.
        - This function assumes 0 <= e_max <= 1, typical for eccentricities.
        """
        eta = self.eta
        if e_max is None:
            e_max = self.e_max
        if not (0.0 <= e_max < 1.0):
            raise ValueError(f"e_max must be between 0 and 0.99; got {e_max}")
        if abs(eta + 1) < 1e-9:
            raise ValueError("eta = -1 is not supported by this simple approach (logarithmic).")

        # Draw uniform random deviates u in [0,1].
        u = np.random.rand(N)

        # Inverse CDF:  u = e^(eta+1) / e_max^(eta+1)
        # => e = e_max * (u)^(1/(eta+1))
        e_samples = e_max * (u ** (1.0/(eta+1.0)))

        return e_samples

    def period_ecc_cap(self, P):
        """
        Maximum eccentricity for a given orbital period P.
        """
        return 1.0 - (P / 2)**(-2.0/3.0)

    def sample_ecc(self, P):
        """
        Draw eccentricity based on the orbital period P.
        - For P < 2 days, assume circular (e = 0).
        - For P >= 2 days, use sample_eccentricity() with rejection sampling to ensure e < e_max(P).
        """
        if P < 2:
            return 0.0  # Circularized for short-period binaries
        # Rejection sampling for P >= 2 days
        cap = self.e_max
        if getattr(self, "use_period_ecc_cap", True):
            cap = min(cap, float(self.period_ecc_cap(P)))
        cap = min(max(cap, 0.0), 0.999999)  # safety clip
        return self.sample_eccentricity(1, e_max=cap)[0]

    def sample_ecc_vectorized(self, P_array, tol=1e-5):
        """
        Vectorized draw of eccentricities for an array of periods P_array.
        Uses f(e) ~ e^eta truncated at cap = min(self.e_max, period_ecc_cap(P)) if enabled.
        Sets e=0 for P<2 d. Optionally zeroes tiny values below `tol`.
        """
        P = np.asarray(P_array, dtype=float)

        # global cap
        e_cap = np.full(P.shape, float(self.e_max), dtype=float)

        # optional period-dependent cap
        if getattr(self, "use_period_ecc_cap", True):
            # period_ecc_cap must be vectorized (works fine with NumPy arrays)
            e_cap = np.minimum(e_cap, self.period_ecc_cap(P))

        # safety
        e_cap = np.clip(e_cap, 0.0, 0.999999)

        # draw (inverse CDF for e^eta on [0, e_cap])
        if abs(self.eta + 1.0) < 1e-9:
            raise ValueError("eta = -1 is not supported by this sampler.")
        u = np.random.rand(P.size)
        e = e_cap * (u ** (1.0 / (self.eta + 1.0)))

        # short-period circularization
        e[P < 2.0] = 0.0

        # tiny values to exact zero (optional)
        if tol is not None:
            e[np.abs(e) < tol] = 0.0

        return e

    def sample_q(self, N):
        """
        Sample mass ratio q in [q_min, q_max], 
        possibly with a mild power law q^kappa.
        """
        kappa = self.kappa
        q_min = self.q_min
        q_max = self.q_max
        # If kappa=0 => uniform in [q_min, q_max]
        # For kappa != 0 => f(q) ~ q^kappa
        
        if abs(kappa) < 1e-3:
            return np.random.uniform(q_min, q_max, N)
        else:
            # Inverse transform for q^(kappa)
            # F(q) = (q^(kappa+1) - q_min^(kappa+1)) / (q_max^(kappa+1) - q_min^(kappa+1))
            alpha = kappa + 1
            Qmin_alpha = q_min**alpha
            Qmax_alpha = q_max**alpha
            A = Qmax_alpha - Qmin_alpha
            u = np.random.rand(N)
            qvals = ((u*A) + Qmin_alpha)**(1./alpha)
            return qvals

    def sample_orbital_extras_vectorized(self, M1_array, logP_array, q_array, e_array,
                                        gamma_range=None,
                                        inc_mode='random'):
        """
        Vectorized version of sample_orbital_extras.
        
        Arguments:
        M1_array   : array of primary masses [Msun], shape (N,)
        logP_array : array of log10(orbital period [days]), shape (N,)
        q_array    : array of mass ratios (M2/M1), shape (N,)
        e_array    : array of eccentricities, shape (N,)
        gamma_range: (min, max) for uniform sampling of systemic velocity
        inc_mode   : 'random' => random orientation (cos i uniform in [-1,1])
                    or 'fixed90' => i = 90 deg

        Returns:
        A dictionary of arrays:
            P          : shape (N,)
            M2         : shape (N,)
            i_rad      : shape (N,)
            omega_deg  : shape (N,)
            gamma      : shape (N,)
            Tp         : shape (N,)
            K1         : shape (N,)
            K2         : shape (N,)
        """

        if gamma_range is None:
            gamma_range = self.gamma_range

        # 1) Convert logP => P in days
        P_array = 10.0 ** logP_array  # shape (N,)

        # 2) Secondary masses
        M2_array = q_array * M1_array  # shape (N,)

        # 3) Sample inclination
        if inc_mode == 'random':
            # cos i uniform in [-1,1], shape (N,)
            cos_i = np.random.uniform(-1, 1, size=M1_array.shape)
            i_array = np.arccos(cos_i)
        else:
            # e.g. fixed i=90 deg
            i_array = np.full(M1_array.shape, np.radians(90.0))

        # 4) Sample argument of periastron in [0..360], but if e=0 => set omega=90
        omega_deg_array = np.random.uniform(0, 360, size=M1_array.shape)
        zero_e_mask = (e_array == 0)
        omega_deg_array[zero_e_mask] = 90.0

        # 5) Systemic velocity gamma in [gamma_range[0]..gamma_range[1]]
        gamma_array = np.random.uniform(gamma_range[0], gamma_range[1],
                                        size=M1_array.shape)

        # 6) Time of periastron passage in [0..P_array[i]]
        #    For each star, we do random() * P_array[i]
        Tp_array = np.random.random(size=M1_array.shape) * P_array

        # 7) Compute K1, K2 using vectorized math
        #    G is the gravitational constant in km^3 Msun^-1 s^-2
        G = 4.309e-3 * 3.0857e13  # combined factor
        P_sec_array = P_array * 86400.0  # days to seconds

        # shape (N,), all vector math
        denom = (M1_array + M2_array) ** (2/3)
        sin_i = np.sin(i_array)
        factor = (2 * np.pi * G)**(1/3) * (P_sec_array ** (-1/3))

        K1_array = factor * (M2_array * sin_i) / denom
        K2_array = factor * (M1_array * sin_i) / denom

        # Return arrays in a dictionary
        return {
            'P': P_array,
            'M2': M2_array,
            'i_rad': i_array,
            'omega_deg': omega_deg_array,
            'gamma': gamma_array,
            'Tp': Tp_array,
            'K1': K1_array,
            'K2': K2_array,
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

        elif mode == "cycle":
            k = values.size
            start = self._cycle_idx.get(name, 0)
            idx = (np.arange(start, start + n) % k)
            self._cycle_idx[name] = (start + n) % k
            return values[idx]

        else:
            raise ValueError(f"Unknown fixed_values_mode: {mode}")

    def draw_M1(self, n, rng=None):
        if self.M1_values is not None:
            return self._draw_from_values("M1", self.M1_values, n, rng,
                                        mode=self.fixed_values_mode,
                                        weights=self.fixed_values_weights.get("M1"))
        return self.sample_primary_mass(n)

    def draw_logP(self, n, rng=None):
        if self.logP_values is not None:
            return self._draw_from_values("logP", self.logP_values, n, rng,
                                        mode=self.fixed_values_mode,
                                        weights=self.fixed_values_weights.get("logP"))
        return self.sample_logP(n)

    def draw_q(self, n, rng=None):
        if self.q_values is not None:
            return self._draw_from_values("q", self.q_values, n, rng,
                                        mode=self.fixed_values_mode,
                                        weights=self.fixed_values_weights.get("q"))
        return self.sample_q(n)

    def draw_e_vectorized(self, P_array, rng=None):
        """
        If e_values is provided, draw from it; otherwise use your existing vectorized sampler.
        Applies P<2 circularization and caps (global e_max + period cap) if needed.
        """
        P = np.asarray(P_array, dtype=float)
        n = P.size

        if self.e_values is None:
            # existing behavior
            return self.sample_ecc_vectorized(P)

        # Draw from fixed list
        e = self._draw_from_values("e", self.e_values, n, rng,
                                mode=self.fixed_values_mode,
                                weights=self.fixed_values_weights.get("e")).astype(float)

        # Enforce physics/caps
        # 1) circularization
        e[P < 2.0] = 0.0

        # 2) global & period-dependent caps
        cap = float(self.e_max)  # attribute per your latest refactor
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

    ######################################################################
    # MCMC implementation

    ########################
    #  PRIOR
    ########################
    @staticmethod
    def log_prior(theta):
        """
        Simple prior on f_bin: uniform between 0 and 1.
        """
        f_bin = theta[0]
        if 0.0 < f_bin < 1.0:
            return 0.0
        else:
            return -np.inf


    ########################
    #  LIKELIHOOD
    ########################
    def log_likelihood(self, theta, N_sim, dRV_real, batch_size=1000):
        """
        Compare the distribution of dRV_max between real data (dRV_real)
        and a mock sample generated with parameters from theta.

        We use Poisson-based log-likelihood in logarithmic bins.
        """
        # Unpack parameters (here only f_bin)
        f_bin = theta[0]
        n_batches = int(np.ceil(N_sim / batch_size))
        logL_total = 0.0

        # For each batch, simulate mock observations and compute a partial log-likelihood.
        for i in range(n_batches):
            # Determine number of stars in this batch (for the final batch it may be smaller)
            current_batch = batch_size if i < n_batches - 1 else (N_sim - batch_size*(n_batches-1))

            # Generate the mock sample
            # This returns a DataFrame with 'dRV_max' for each star
        
            mock_res_df = self.simulate_mock_observations(pi=0.0, kappa=0.0, eta=-0.5, gamma=-2.35,
                                                    N=current_batch, f_bin=f_bin,
                                                    save_file=False, intrinsic_sample=None)
            dRV_mock = mock_res_df['dRV_max'].values
        
            # Define log-spaced bins for the histogram
            nbins = 30
            bins = np.logspace(0.4, 3, nbins)

            # Histogram the real data and the mock data for this batch
            n_real, _ = np.histogram(dRV_real, bins=bins)
            n_mock, _ = np.histogram(dRV_mock, bins=bins)
            n_mock = n_mock.astype(float) # ensure float for division/ln

            # Scale the mock counts.
            # Note: since each batch represents only part of the total simulation, adjust scaling appropriately.
            N_obs = len(dRV_real)
            scale = N_obs / float(N_sim)  # overall scaling factor
            n_mock *= scale

            # Compute the Poisson log-likelihood for this batch
            epsilon = 1e-8
            n_mock += epsilon
            # logL = sum( n_real * ln(n_mock) - n_mock ), ignoring constants w.r.t. data
            logL_batch = 0.0
            for j in range(len(n_real)):
                # n_real[i] * ln(n_mock[i]) - n_mock[i]
                # We do not add the term -ln(n_real[i]!), 
                # because that's constant wrt model and doesn't affect the MCMC.
                logL_batch += n_real[j] * np.log(n_mock[j]) - n_mock[j]
            
            logL_total += logL_batch
        
        return logL_total

    def compute_log_likelihood_batch(self, f_bin, N_sim, dRV_real, current_batch):
        # Generate the mock sample
        # This returns a DataFrame with 'dRV_max' for each star
        mock_res_df = self.simulate_mock_observations(pi=0.0, kappa=0.0, eta=-0.5, gamma=-2.35,
                                                N=current_batch, f_bin=f_bin,
                                                save_file=False, intrinsic_sample=None)
        dRV_mock = mock_res_df['dRV_max'].values

        # Define log-spaced bins for the histogram
        nbins = 30
        bins = np.logspace(0.4, 3, nbins)

        # Histogram the real data and the mock data for this batch
        n_real, _ = np.histogram(dRV_real, bins=bins)
        n_mock, _ = np.histogram(dRV_mock, bins=bins)
        n_mock = n_mock.astype(float)

        # Scale the mock counts.
        # Note: since each batch represents only part of the total simulation, adjust scaling appropriately.
        N_obs = len(dRV_real)
        scale = N_obs / float(N_sim)  # N_sim is the overall total, you may need to pass it in or adjust logic
        n_mock *= scale

        # Compute the Poisson log-likelihood for this batch
        epsilon = 1e-8
        n_mock += epsilon
        # logL = sum( n_real * ln(n_mock) - n_mock ), ignoring constants w.r.t. data
        logL_batch = 0.0
        for j in range(len(n_real)):
            # n_real[i] * ln(n_mock[i]) - n_mock[i]
            # We do not add the term -ln(n_real[i]!), 
            # because that's constant wrt model and doesn't affect the MCMC.
            logL_batch += n_real[j] * np.log(n_mock[j]) - n_mock[j]
        return logL_batch

    def log_likelihood_parallel(self, theta, N_sim, dRV_real, batch_size=1000):
        f_bin = theta[0]
        n_batches = int(np.ceil(N_sim / batch_size))
        logL_total = 0.0

        with MP_Pool(2) as pool:
            # Create a list to hold asynchronous results.
            async_results = []
            for i in range(n_batches):
                current_batch = batch_size if i < n_batches - 1 else (N_sim - batch_size * (n_batches - 1))
                # Submit the function call asynchronously.
                async_result = pool.apply_async(self.compute_log_likelihood_batch, (f_bin, N_sim, dRV_real, current_batch))
                async_results.append(async_result)
            # Close the pool and wait for tasks to finish.
            pool.close()
            pool.join()
            # Retrieve the results and sum them.
            for res in async_results:
                logL_total += res.get()

        return logL_total

    ########################
    #  POSTERIOR
    ########################
    def log_posterior(self, theta, N, dRV_real, batch_size):
        """
        Combine prior and likelihood.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_parallel(theta, N, dRV_real, batch_size)


    ########################
    #  MCMC RUN
    ########################
    def run_mcmc(self, N, dRV_real, nwalkers=16, nsteps=2000, nthreads=4, batch_size=1000):
        """
        Example function to run an MCMC with emcee, using the Poisson 
        likelihood on a distribution of dRV_real.
        """
        ndim = 1  # We are only fitting f_bin
        # Initialize walkers
        p0 = 0.80 + 0.05 * np.random.randn(nwalkers, ndim)

        with Pool(nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim,
                self.log_posterior,
                args=[N, dRV_real, batch_size],  # pass real data's dRV distribution
                pool=pool
            )
            sampler.run_mcmc(p0, nsteps, progress=True)

        return sampler


    # Example usage (assuming you have a real_data DataFrame with 'dRV_max'):
    # dRV_real = real_data['dRV_max'].values
    #
    # sampler = run_mcmc(dRV_real, nwalkers=16, nsteps=2000, nthreads=4)
    # chain = sampler.get_chain(discard=500, thin=5, flat=True)
    # best_f_bin = np.median(chain)
    # print("Best-fit f_bin ~", best_f_bin)