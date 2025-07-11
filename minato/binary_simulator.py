import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kepler
import sys
sys.path.append('/Users/villasenor/science/github/jvillasr/MINATO')
from minato import myRC
import emcee
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocess import Pool as MP_Pool

class BinarySimulations:
    def __init__(self):
        # Default parameters
        self.M1_min = 8
        self.M1_max = 20
        self.q_min = 0.01
        self.q_max = 1.0
        self.logP_min = -0.15
        self.logP_max = 3.5
        self.gamma_range = (-1, 1)
        # self.pi = 0.0
        # self.kappa = 0.0
        # self.eta = -0.5
        # self.gamma = -2.35
        
        # Storage for data and results
        self.coverage_dict = None
        self.real_dRV = None
        self.mock_data = None
        self.sampler = None
        self.obs_results_df = None

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

    def simulate_mock_observations(self, pi=0.0, kappa=0.0, eta=-0.5, gamma=-2.35, 
                                N=100, f_bin=0.5, save_file=False, intrinsic_sample=None):
        """
        Parent function to generate (or load) the intrinsic mock sample and simulate observations.
        
        Parameters:
        pi, kappa, eta, gamma: Parameters for the binary distributions.
        N: Total number of stars.
        f_bin: Binary fraction.
        # observation_strategy: Dictionary specifying "t_array" and "rv_errors". For example:
        #                         {"t_array": np.array([0, 30, 300]),
        #                         "rv_errors": np.array([0.5, 0.5, 0.5])}
        save_file: If True, save the generated intrinsic sample to disk.
        intrinsic_sample: Optionally, an already-generated intrinsic sample (DataFrame).
        
        Returns:
        obs_results_df: A DataFrame with the simulated observation results.
        """

        if self.coverage_dict is None:
            raise ValueError("No coverage data loaded. Please load observational data using load_data before simulating observations.")

        # If no intrinsic sample is provided, try loading from file; otherwise, generate it.
        if intrinsic_sample is None:
            # print("Generating intrinsic sample ...")
            intrinsic_sample = self.generate_intrinsic_sample_vectorized(N=N, f_bin=f_bin, 
                                                            pi=pi, kappa=kappa, eta=eta, gamma=gamma,
                                                            save_file=save_file)
            # intrinsic_sample_nonvec = generate_intrinsic_sample(N=N, f_bin=f_bin, 
            #                                                 pi=pi, kappa=kappa, eta=eta, gamma=gamma,
            #                                                 save_file=save_file)
        else:
            print("Using provided intrinsic sample ...")
            intrinsic_sample = pd.read_pickle(intrinsic_sample)

        # plot_sampled_params(intrinsic_sample)

        # Now simulate observations using the chosen observing strategy.
        self.obs_results_df = self.compute_rvs(intrinsic_sample)
        return self.obs_results_df

    def generate_intrinsic_sample_vectorized(self, N=100, f_bin=0.5, pi=0.0, kappa=0.0, eta=-0.5, gamma=-2.35, save_file=False):
        # Number of binaries
        n_bin = int(np.round(f_bin * N))
        
        # 1) For the binary stars, sample all parameters in one go:
        # M1_array = self.sample_primary_mass(n_bin, Mmin=self.M1_min, Mmax=self.M1_max, gamma=gamma)      # shape (n_bin,)
        # logP_array = self.sample_logP(n_bin, pi=pi, logP_min=self.logP_min, logP_max=self.logP_max)     # shape (n_bin,)
        # q_array = self.sample_q(n_bin, kappa=kappa, q_min=self.q_min, q_max=self.q_max)           # shape (n_bin,)

        M1_array = self.sample_MOB(n_bin)
        logP_array = self.sample_logP_bh(n_bin) 
        q_array = self.sample_q_bh(n_bin)

        # Eccentricities might depend on period, so:
        P_array = 10 ** logP_array
        e_array = [self.sample_ecc(P_val, eta=eta) for P_val in P_array]  # still a loop, but only length n_bin
        tol = 1e-5
        e_array = np.asarray(e_array)
        e_array[np.abs(e_array) < tol] = 0.0

        # 2) Sample additional orbital parameters for all n_bin at once
        orb_data = self.sample_orbital_extras_vectorized(M1_array, logP_array, q_array, e_array, 
                                                    gamma_range=(-1,1)) # for BLOeM: (100,240)
        
        # 3) Build a DataFrame for binary stars:
        df_binaries = pd.DataFrame({
            'M1': M1_array,
            'M2': orb_data['M2'],
            'q': q_array,
            'i': np.degrees(orb_data['i_rad']),
            'P': orb_data['P'],
            'e': e_array,
            'Tp': orb_data['Tp'],
            'omega_deg': orb_data['omega_deg'],
            'gamma': orb_data['gamma'],
            'K1': orb_data['K1'],
            'K2': orb_data['K2'],
            'is_binary': True,
            'synthetic_ID': [f"SYN_{i:04d}" for i in range(n_bin)]
        })

        # 4) For single stars:
        n_singles = N - n_bin
        # You can sample single-star masses if needed, or just fill placeholders:
        df_singles = pd.DataFrame({
            'M1'       : self.sample_primary_mass(n_singles, Mmin=self.M1_min, Mmax=self.M1_max, gamma=gamma),
            'M2'       : np.nan,  # or zero, if you prefer
            'q'        : np.nan,
            'i'        : np.nan,
            'P'        : np.nan,
            'e'        : np.nan,
            'gamma'    : np.random.uniform(-1, 1, size=n_singles),
            'is_binary': False,
            'synthetic_ID': [f"SYN_{(n_bin + i):04d}" for i in range(n_singles)]
        })

        # 5) Combine them:
        intrinsic_df = pd.concat([df_binaries, df_singles], ignore_index=True)
        
        if save_file:
            intrinsic_df.to_pickle(f'mock_sample_N{N}_fbin{int(f_bin*100)}_pi{int(pi)}.pkl')

        return intrinsic_df

    def _compute_sigmad_vectorized(self, rv_obs, rv_errors):
        rv_obs = np.asarray(rv_obs)
        rv_errors = np.asarray(rv_errors)
        diff = np.abs(rv_obs[:, None] - rv_obs[None, :])
        err = np.sqrt(rv_errors[:, None]**2 + rv_errors[None, :]**2)
        sigma = diff / err
        return np.max(sigma)

    def compute_rvs(self, intrinsic_df):
        """
        Simulate RV observations for each star in the intrinsic sample based on the specified observing strategy.
        
        Parameters:
        intrinsic_df: DataFrame containing the intrinsic parameters of the simulated stars.
        # observation_strategy: Dictionary with keys:
        #                         "t_array": 1D array of observation times (e.g., [0, 30, 300])
        #                         "rv_errors": 1D array of RV uncertainties for each epoch.
        
        Returns:
        obs_results_df: A DataFrame with the simulated observation results.
        """
        results_obs = []

        # Make sure coverage data is available.
        if self.coverage_dict is None:
            raise ValueError("Observational data not loaded. Please use load_data to load the observational data into self.coverage_dict.")

        intrinsic_df = self.add_semi_major_axis(intrinsic_df)

        real_star_ids = list(self.coverage_dict.keys())
        
        for idx, row in intrinsic_df.iterrows():
            chosen_id = np.random.choice(real_star_ids)
            t_array, rv_errors = self.coverage_dict[chosen_id]

            rv_obs1 = None
            rv_obs2 = None # Initialize secondary RV array
            
            # Generate noise once per epoch set
            noise = np.random.normal(0, rv_errors, size=len(t_array))

            # For binaries, compute the "true" RV curve; for singles, assume constant RV.
            if row['is_binary']:
                v_orbit = self.rvcurve(t_array, row['P'], row['Tp'], row['e'], 
                                row['omega_deg'], row['gamma'], row['K1'], row['K2'], SB2=True)

                # Check if rvcurve returned one or two components
                if isinstance(v_orbit, tuple): # SB2 case
                    v1, v2 = v_orbit
                    rv_obs1 = v1 + noise
                    rv_obs2 = v2 + noise # Store secondary RVs
                else: # SB1 case (assuming rvcurve returns only v1 if SB2=False or K2=0)
                    v1 = v_orbit
                    rv_obs1 = v1 + noise
                    rv_obs2 = np.full_like(rv_obs1, np.nan) # Fill secondary with NaN for SB1

            else: # Single star case
                # print('Single star:', row['synthetic_ID'], row['gamma'])
                rv_obs1 = row['gamma'] + noise
                rv_obs2 = np.full_like(rv_obs1, np.nan) # Fill secondary with NaN for singles

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
                'P': row['P'], 'Tp': row['Tp'], 'e': row['e'], 'omega_deg': row['omega_deg'], 
                'gamma': row['gamma'], 'K1': row['K1'], 'K2': row['K2'], 
                'i_deg': row['i'], 'a': row['a'],
                'dRV_max': dRV,
                'sigma_d': sigma_detect,
                'rv_mean': rv_mean,
                'n_eps': n_eps,
                'rv_array': rv_obs1,
                'rv_array2': rv_obs2,  # Store secondary RVs
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
        fig, axs = plt.subplots(3, 3, figsize=(20, 14))

        # Plot the distribution of P
        axs[0, 0].hist(df['P'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        axs[0, 0].set_xlabel('Orbital Period (P) [days]')
        axs[0, 0].set_ylabel('Number of Binaries')
        # axs[0, 0].set_title('Distribution of Orbital Period (P)')
        axs[0, 0].grid(True)

        # Plot the distribution of M1
        axs[0, 1].hist(df['M1'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        axs[0, 1].set_xlabel('Primary Mass (M1) [M_sun]')
        axs[0, 1].set_ylabel('Number of Binaries')
        # axs[0, 1].set_title('Distribution of Primary Mass (M1)')
        axs[0, 1].grid(True)

        # Plot the distribution of Tp
        axs[0, 2].hist(df['Tp'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[0, 2].hist(intrinsic_sample_nonvec['Tp'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[0, 2].set_xlabel('Time of Periastron Passage (Tp) [days]')
        axs[0, 2].set_ylabel('Number of Binaries')
        # axs[0, 2].set_title('Distribution of Time of Periastron Passage (Tp)')
        axs[0, 2].grid(True)

        # Plot the distribution of q
        axs[1, 0].hist(df['q'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[1, 0].hist(intrinsic_sample_nonvec['q'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[1, 0].set_xlabel('Mass Ratio (q)')
        axs[1, 0].set_ylabel('Number of Binaries')
        # axs[1, 0].set_title('Distribution of Mass Ratio (q)')
        axs[1, 0].grid(True)

        # Plot the distribution of i
        axs[1, 1].hist(df['i_deg'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[1, 1].hist(intrinsic_sample_nonvec['i'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[1, 1].set_xlabel('Inclination (i) [degrees]')
        axs[1, 1].set_ylabel('Number of Binaries')
        # axs[1, 1].set_title('Distribution of Inclination (i)')
        axs[1, 1].grid(True)

        # Plot the distribution of omega_deg
        axs[1, 2].hist(df['omega_deg'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[1, 2].hist(intrinsic_sample_nonvec['omega_deg'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[1, 2].set_xlabel('Argument of Periastron (omega) [degrees]')
        axs[1, 2].set_ylabel('Number of Binaries')
        # axs[1, 2].set_title('Distribution of Argument of Periastron (omega)')
        axs[1, 2].grid(True)

        # Plot the distribution of K1
        axs[2, 0].hist(df['K1'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[2, 0].hist(intrinsic_sample_nonvec['K1'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[2, 0].set_xlabel('Radial Velocity Semi-Amplitude (K1) [km/s]')
        axs[2, 0].set_ylabel('Number of Binaries')
        # axs[2, 0].set_title('Distribution of Radial Velocity Semi-Amplitude (K1)')
        axs[2, 0].grid(True)

        # Plot the distribution of e
        axs[2, 1].hist(df['e'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[2, 1].hist(intrinsic_sample_nonvec['e'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[2, 1].set_xlabel('Eccentricity (e)')
        axs[2, 1].set_ylabel('Number of Binaries')
        # axs[2, 1].set_title('Distribution of Eccentricity (e)')
        axs[2, 1].grid(True)

        # Plot the distribution of gamma
        axs[2, 2].hist(df['gamma'].dropna(), bins=15, edgecolor='black', alpha=0.7)
        # axs[2, 2].hist(intrinsic_sample_nonvec['gamma'].dropna(), bins=15, histtype='step', edgecolor='red', alpha=0.7)
        axs[2, 2].set_xlabel('Systemic Velocity (gamma) [km/s]')
        axs[2, 2].set_ylabel('Number of Binaries')
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

    # Sampling for OB+BH population
    def sample_MOB(self, N):
        """
        Draw N OB-star masses from a truncated log-normal that peaks
        near 14 M☉ but has a substantial high-mass tail up to ~50 M☉.
        
        Distribution parameters:
        - median = 14 M☉  (mu = ln(14))
        - shape  = 0.5 dex (sigma = 0.5)
        - truncated to [8, 50] M☉
        """
        mu, sigma = np.log(14), 0.3
        M = np.random.lognormal(mean=mu, sigma=sigma, size=N)
        # rejection‐sample to enforce [8,50]
        bad = (M < 8) | (M > 50)
        while bad.any():
            M[bad] = np.random.lognormal(mean=mu, sigma=sigma, size=bad.sum())
            bad = (M < 8) | (M > 50)
        return M
    
    def sample_q_bh(self, N, mu=0.67, sigma=0.3):
        """
        Draw N OB-BH mass ratios from a truncated log-normal that peaks
        near 0.7 but has a substantial high-q tail up to ~1.7.
        """
        mu, sigma = np.log(mu), sigma
        q = np.random.lognormal(mean=mu, sigma=sigma, size=N) + np.random.lognormal(0.7, 1.5, size=N)
        # rejection‐sample to enforce [8,50]
        bad = (q < 0.3) | (q > 1.7)
        while bad.any():
            q[bad] = np.random.lognormal(mean=mu, sigma=sigma, size=bad.sum())
            bad = (q < 0.3) | (q > 1.7)
        return q
    
    def sample_logP_bh(self, N):
        """
        Approximate the bimodal log‐period PDF (Fig. 6, top):
        – Case A bump at P~5 d → log10P~0.7
        – Case B bump at P~150 d → log10P~2.2
        We take weights wA=30%, wB=70%, sigmas ≃0.2–0.3 dex.
        """
        wA, wB = 0.2, 0.8
        # draw which mode each sample comes from
        isA = np.random.rand(N) < wA
        logP = np.empty(N)
        logP[isA] = np.random.lognormal(np.log(0.9), 0.25, size=isA.sum())
        logP[~isA] = np.random.normal(2.1, 0.45, size=(~isA).sum())
        return logP
    # End of OB+BH sampling functions
    #################################

    def sample_primary_mass(self, N, Mmin=8, Mmax=16, gamma=-2.35):
        """
        Sample primary masses from a Salpeter-like IMF 
        dN/dM ~ M^gamma, with M in [Mmin, Mmax].
        """
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

    def sample_logP(self, 
        N: int,
        pi: float = 0.0,
        logP_min: float = -0.3,
        logP_max: float = 3.5) -> np.ndarray:
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

    def sample_eccentricity(self, N, eta=-0.5, e_max=0.95):
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
        if not (0.0 <= e_max < 1.0):
            raise ValueError(f"e_max must be between 0 and 0.99; got {e_max}")
        if abs(eta + 1) < 1e-9:
            raise ValueError("eta = -1 is not supported by this simple approach (logarithmic).")

        # Draw uniform random deviates u in [0,1].
        u = np.random.rand(N)

        # Inverse CDF:  u = e^(eta+1) / e_max^(eta+1)
        # => e = e_max * (u)^(1/(eta+1))
        e_samples = e_max * (u ** (1.0/(eta+1)))

        return e_samples

    def e_max(self, P):
        """
        Maximum eccentricity for a given orbital period P.
        """
        return 1.0 - (P / 2)**(-2.0/3.0)

    def sample_ecc(self, P, eta):
        """
        Draw eccentricity based on the orbital period P.
        - For P < 2 days, assume circular (e = 0).
        - For P >= 2 days, use sample_eccentricity() with rejection sampling to ensure e < e_max(P).
        """
        if P < 2:
            return 0.0  # Circularized for short-period binaries
        
        else:
            # Rejection sampling for P >= 2 days
            while True:
                e_draw = self.sample_eccentricity(1, eta=eta)
                if e_draw < self.e_max(P):
                    return e_draw[0]

    def sample_q(self, N, kappa=0.0, q_min=0.1, q_max=1.0):
        """
        Sample mass ratio q in [q_min, q_max], 
        possibly with a mild power law q^kappa.
        """
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
                                        gamma_range=(100, 240),
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