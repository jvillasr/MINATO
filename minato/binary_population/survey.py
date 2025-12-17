import numpy as np
import pandas as pd

from .population import BinaryPopulation


class BinarySurveySimulator:
    """
    Survey/cadence layer: takes an intrinsic BinaryPopulation catalogue and
    simulates observed RVs, noise, and detection metrics using real or synthetic cadence.
    """

    def __init__(self, population: BinaryPopulation):
        self.population = population
        self.coverage_dict = None
        self.obs_results_df = None
        self._last_seed = None

    def load_data(self, df: pd.DataFrame) -> dict:
        """
        Reads a pandas DataFrame with columns: ID, MJD, mean_rv_er.
        Returns a dictionary keyed by star_id, each item is a tuple of arrays (mjds, rv_errors).
        """
        required_columns = {"ID", "MJD", "mean_rv_er"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")
        coverage = {}
        for star_id, group in df.groupby("ID"):
            group_sorted = group.sort_values(by="MJD")
            mjds = group_sorted["MJD"].values
            rv_errs = group_sorted["mean_rv_er"].values
            coverage[star_id] = (mjds, rv_errs)
        self.coverage_dict = coverage
        return self.coverage_dict

    def simulate_mock_observations(
        self,
        N=100,
        f_bin=0.5,
        save_sample=False,
        intrinsic_sample=None,
        ideal_sampling=False,
        n_epochs=None,
        rv_error_common=None,
        summary_only=False,
        seed=None,
    ):
        """
        Generate (or accept) an intrinsic sample and simulate observations.
        """
        if seed is not None:
            np.random.seed(int(seed))
        self._last_seed = int(seed) if seed is not None else None

        if self.coverage_dict is None and ideal_sampling is False:
            raise ValueError("No coverage data loaded. Please load observational data using load_data before simulating observations.")

        pop = self.population
        if intrinsic_sample is None:
            intrinsic_sample = pop.generate_intrinsic_sample_vectorized(
                N=N,
                f_bin=f_bin,
                save_sample=save_sample,
            )

        self.obs_results_df = self.compute_rvs(
            intrinsic_sample,
            ideal_sampling=ideal_sampling,
            n_epochs=n_epochs,
            rv_error_common=rv_error_common,
            summary_only=summary_only,
        )
        return self.obs_results_df

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
            v1 = self.population.rvcurve(t_grid, P, Tp, e, omega_deg, gamma=0.0, K1=1.0, K2=0.0, SB2=False)
            i_max, i_min = np.argmax(v1), np.argmin(v1)
            t_ext = np.array([t_grid[i_max], t_grid[i_min]], dtype=float)

        return t_ext

    def phase_grid(self, P, Tp, n_epochs):
        """Return n_epochs times uniformly spaced in orbital phase over one period."""
        phases = np.linspace(0.0, 1.0, int(n_epochs), endpoint=False)
        return Tp + phases * P

    def _compute_sigmad_vectorized(self, rv_obs, rv_errors):
        rv_obs = np.asarray(rv_obs)
        rv_errors = np.asarray(rv_errors)
        diff = np.abs(rv_obs[:, None] - rv_obs[None, :])
        err = np.sqrt(rv_errors[:, None] ** 2 + rv_errors[None, :] ** 2)
        sigma = diff / err
        return np.max(sigma)

    def compute_rvs(self, intrinsic_df, ideal_sampling=False, n_epochs=None, rv_error_common=None, summary_only=False):
        """
        Simulate RV observations for each star in the intrinsic sample based on the specified observing strategy.
        """
        results_obs = []
        pop = self.population
        intrinsic_df = pop.add_semi_major_axis(intrinsic_df)
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

        for _, row in intrinsic_df.iterrows():
            chosen_id = "synthetic"  # default, so later 'real_ID_used' is always defined

            if row["is_binary"] and ideal_sampling is True:
                # two quadratures
                t_array = self.two_quadratures(row["P"], row["Tp"], row["e"], row["omega_deg"])

                if rv_error_common is not None:
                    rv_errors = np.full(len(t_array), float(rv_error_common), dtype=float)
                else:
                    if not has_coverage:
                        raise ValueError("For two-quadrature sampling without coverage, rv_error_common must be provided.")
                    chosen_id = np.random.choice(real_star_ids)
                    _, rv_errors_full = self.coverage_dict[chosen_id]
                    rv_errors = np.random.choice(np.asarray(rv_errors_full).ravel(), size=len(t_array), replace=True)

            elif row["is_binary"] and ideal_sampling == "phase_uniform":
                t_array = self.phase_grid(row["P"], row["Tp"], int(n_epochs))

                if rv_error_common is not None:
                    rv_errors = np.full(len(t_array), float(rv_error_common), dtype=float)
                else:
                    raise ValueError("For 'phase_uniform' sampling, rv_error_common must be provided.")

            elif (row["is_binary"] is False) and (ideal_sampling is not False):
                raise ValueError("Ideal sampling modes are only applicable to binary stars; single stars require a coverage dictionary (ideal_sampling=False).")

            elif ideal_sampling is False:
                if not has_coverage:
                    raise ValueError("Real cadence requires coverage_dict; call load_data().")
                chosen_id = np.random.choice(real_star_ids)
                t_array, rv_errors = self.coverage_dict[chosen_id]

            else:
                raise ValueError(f"Unsupported ideal_sampling value: {ideal_sampling!r}")

            # For binary stars, compute the RV curve; for single stars, use gamma.
            if row["is_binary"]:
                v_orbit = pop.rvcurve(
                    t_array,
                    row["P"],
                    row["Tp"],
                    row["e"],
                    row["omega_deg"],
                    row["gamma"],
                    row["K1"],
                    row["K2"],
                    SB2=True,
                )
            else:
                v_orbit = np.full(t_array.shape, row["gamma"], dtype=float)

            noise = np.random.normal(0, rv_errors, size=len(t_array))

            if isinstance(v_orbit, tuple):
                v1_true, v2_true = v_orbit
                rv_obs1 = v1_true + noise
                rv_obs2 = v2_true + noise
            else:
                v1_true = v_orbit
                v2_true = np.full_like(v1_true, np.nan)
                rv_obs1 = v1_true + noise
                rv_obs2 = np.full_like(rv_obs1, np.nan)

            dRV_true = np.ptp(v1_true)

            n_eps = len(t_array)
            rv_mean = np.mean(rv_obs1)
            dRV = rv_obs1.max() - rv_obs1.min()
            sigma_detect = self._compute_sigmad_vectorized(rv_obs1, rv_errors)

            if summary_only:
                results_obs.append({
                    "synthetic_ID": row["synthetic_ID"],
                    "real_ID_used": chosen_id,
                    "is_binary": row["is_binary"],
                    "dRV_max": dRV,
                    "sigma_d": sigma_detect,
                    "rv_mean": rv_mean,
                    "n_eps": n_eps,
                    "n_epochs": n_eps,
                    "rng_seed": self._last_seed,
                })
                continue

            results_obs.append({
                "synthetic_ID": row["synthetic_ID"],
                "real_ID_used": chosen_id,
                "is_binary": row["is_binary"],
                "M1": row["M1"],
                "q": row["q"],
                "P": row["P"],
                "P_drawn": row["P_drawn"],
                "P_clamped": row["P_clamped"],
                "Tp": row["Tp"],
                "e": row["e"],
                "omega_deg": row["omega_deg"],
                "gamma": row["gamma"],
                "K1": row["K1"],
                "K2": row["K2"],
                "i_deg": row["i"],
                "a": row["a"],
                "mjd_array": t_array,
                "dRV_true": dRV_true,
                "dRV_max": dRV,
                "sigma_d": sigma_detect,
                "rv_mean": rv_mean,
                "n_eps": n_eps,
                # Alias for clarity in notebooks/docs (keep n_eps for backward compatibility)
                "n_epochs": n_eps,
                "rv_true": v1_true,
                "rv_true2": v2_true,
                "rv_array": rv_obs1,
                "rv_array2": rv_obs2,
                "rv_errors": rv_errors,
                "smear_ok": row.get("smear_ok", np.nan),
                "dv_exp1_kms": row.get("dv_exp1_kms", np.nan),
                "dv_exp2_kms": row.get("dv_exp2_kms", np.nan),
                "rng_seed": self._last_seed,
            })

        return pd.DataFrame(results_obs)

    def plot_sampled_params(self, df=None):
        """
        Quick-look plots of sampled parameters. If df is None, uses the last obs_results_df.
        """
        import matplotlib.pyplot as plt  # Local import to keep base dependencies light

        if df is None:
            if self.obs_results_df is None:
                raise ValueError("No observation results available to plot.")
            df = self.obs_results_df

        fig, axs = plt.subplots(3, 3, figsize=(20, 12))
        fig.subplots_adjust(hspace=0.35)

        axs[0, 0].hist(df["P"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[0, 0].set_xlabel(r"$P_{\\rm orb}$ [d]")
        axs[0, 0].set_ylabel("Number")
        axs[0, 0].grid(True)

        axs[0, 1].hist(df["M1"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[0, 1].set_xlabel("$M_1$ [M$_{\\odot}$]")
        axs[0, 1].grid(True)

        axs[0, 2].hist(df["Tp"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[0, 2].set_xlabel("$T_p$ [d]")
        axs[0, 2].grid(True)

        axs[1, 0].hist(df["q"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[1, 0].set_xlabel("Mass Ratio ($q$)")
        axs[1, 0].set_ylabel("Number")
        axs[1, 0].grid(True)

        axs[1, 1].hist(df["i_deg"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[1, 1].set_xlabel("Inclination ($i$) [deg]")
        axs[1, 1].grid(True)

        axs[1, 2].hist(df["omega_deg"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[1, 2].set_xlabel("Argument of Periastron ($\\omega$) [deg]")
        axs[1, 2].grid(True)

        axs[2, 0].hist(df["K1"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[2, 0].set_xlabel("$K_1$ [km\\,s$^{-1}$]")
        axs[2, 0].set_ylabel("Number")
        axs[2, 0].grid(True)

        axs[2, 1].hist(df["e"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[2, 1].set_xlabel("Eccentricity ($e$)")
        axs[2, 1].grid(True)

        axs[2, 2].hist(df["gamma"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        axs[2, 2].set_xlabel("Systemic Velocity ($\\gamma$) [km\\,s$^{-1}$]")
        axs[2, 2].grid(True)

        plt.savefig("parameter_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
