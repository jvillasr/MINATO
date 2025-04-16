import os
import matplotlib
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy.io import fits
from astropy.timeseries import LombScargle
from collections import Counter
from matplotlib.ticker import StrMethodFormatter
from lmfit import Model, Parameters, models
from datetime import date
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpyro as npro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import numpy as jnp
from jax import random
import multiprocessing

# Set the number of devices to the number of available CPUs
npro.set_host_device_count(multiprocessing.cpu_count())

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

def gaussian(x, amp, cen, wid):
    """
    1-dimensional Gaussian function for absorption profiles.

    Parameters:
    x : array_like
        Input values.
    amp : float
        Amplitude of the Gaussian.
    cen : float
        Center of the Gaussian.
    wid : float
        Full width (FWHM related factor computed internally).

    Returns:
    array_like
        Gaussian evaluated at x.
    """
    return -amp * jnp.exp(-((x - cen)**2) / (2 * (wid/2.355)**2))

def lorentzian(x, amp, cen, wid):
    """
    1-dimensional Lorentzian function for absorption profiles.

    Parameters:
    x : array_like
        Input values.
    amp : float
        Amplitude of the Lorentzian.
    cen : float
        Center of the Lorentzian.
    wid : float
        Width parameter of the Lorentzian.

    Returns:
    array_like
        Lorentzian evaluated at x.
    """
    return -amp * (wid**2 / (4 * (x - cen)**2 + wid**2))

def nebu(x, amp, cen, wid):
    """
    1-dimensional Gaussian function for nebular emission profiles.

    Parameters:
    x : array_like
        Input values.
    amp : float
        Amplitude (positive) of the emission profile.
    cen : float
        Center of the profile.
    wid : float
        Full width parameter.

    Returns:
    array_like
        Nebular (emission) profile evaluated at x.
    """
    return amp * jnp.exp(-((x - cen)**2) / (2 * (wid/2.355)**2))
    
def sinu(x, A, w, phi, h): # deprecated?
    "Sinusoidal: sinu(data, amp, freq, phase, height)"
    return A*jnp.sin(w*(x-phi))+h

def compute_flux_err(wavelength, flux, wave_region=[4240, 4260]):
    """
    Estimate the flux error by computing the standard deviation in a specified wavelength region.

    Parameters:
    wavelength : array_like
        Wavelength values.
    flux : array_like
        Observed flux values.
    wave_region : list, optional
        [lower_bound, upper_bound] wavelength range for noise estimation.

    Returns:
    array_like
        Array with constant error estimated as twice the noise level.
    """
    noise_mask = (wavelength > wave_region[0]) & (wavelength < wave_region[1])
    flux_masked = flux[noise_mask]
    noise_level = np.std(flux_masked)
    flux_err = np.full_like(flux, 2 * noise_level)
    return flux_err

def read_fits(fits_file, instrument):
    """
    Read a FITS file and extract wavelength, flux, and error data based on the instrument type.

    Parameters:
    fits_file : str
        Path to the FITS file.
    instrument : str
        Instrument identifier; expected values: 'FLAMES', 'FEROS', 'HERMES', 'UVES'.

    Returns:
    tuple
        (wavelength, flux, flux_error, star_epoch, mjd)
    """
    with fits.open(fits_file) as hdul:
        # print(hdul.info())
        header = hdul[0].header
        # print(repr(header))
        try:
            if instrument == 'FLAMES':
                star_epoch = header['OBJECT'] + '_' + header['EPOCH_ID']
                mjd = header['MJD_MID']
                wave = hdul[1].data['WAVELENGTH']
                flux = hdul[1].data['SCI_NORM']
                ferr = hdul[1].data['SCI_NORM_ERR']
            elif instrument == 'FEROS':
                mjd = header['HIERARCH MBJD']
                star_epoch = header['HIERARCH TARGET NAME'] + '_' + f'{mjd:.2f}'
                wave = hdul[12].data
                flux = hdul[15].data
                ferr = None
            else:
                raise ValueError(f"Unsupported instrument: {instrument}")
        except Exception as e:
            print(f"Error reading FITS file: {e}. Please check the instrument parameter. Defaulting to FLAMES settings.")
            star_epoch = header.get('OBJECT', 'Unknown') + '_' + header.get('EPOCH_ID', 'Unknown')
            mjd = header.get('MJD_MID', 0)
            wave = hdul[1].data['WAVELENGTH']
            flux = hdul[1].data['SCI_NORM']
            ferr = hdul[1].data['SCI_NORM_ERR']

        return wave, flux, ferr, star_epoch, mjd

def read_spectra(filelist, path, file_type, instrument):
    """
    Read spectral data from a collection of files.

    Parameters:
    filelist : list or dict
        List of filenames (or a dictionary if file_type is 'dict').
    path : str
        Directory path where files are located.
    file_type : str
        File format: 'dat', 'txt', 'csv', 'fits', or 'dict'.
    instrument : str
        Instrument identifier (used when file_type is 'fits').

    Returns:
    tuple
        (wavelengths, fluxes, flux_errors, names, jds)
    """
    wavelengths, fluxes, f_errors, names, jds = [], [], [], [], []
    for spec in filelist:
        if file_type in ['dat', 'txt', 'csv']:
            names.append(spec.replace(f'.{file_type}', ''))
            try:
                df = pd.read_csv(spec, header=None, delim_whitespace=True)
                # If the file has fewer than 2 columns, try alternative separators.
                if df.shape[1] < 2:
                    for separator in [',', ';', '\t', '|']:
                        try:
                            temp_df = pd.read_csv(spec, sep=separator, header=None)
                            if temp_df.shape[1] >= 2:
                                df = temp_df
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"Error reading file {spec}: {e}")
                continue

            wavelengths.append(np.array(df[0]))
            fluxes.append(np.array(df[1]))
            if df.shape[1] >= 3:
                f_errors.append(np.array(df[2]))
            else:
                f_errors.append(compute_flux_err(df[0], df[1]))
            jds.append(None)  # Append None for non-FITS files.
        elif file_type == 'fits':
            wave, flux, ferr, star, mjd = read_fits(spec, instrument)
            wavelengths.append(wave)
            fluxes.append(flux)
            f_errors.append(ferr)
            names.append(star)
            jds.append(mjd)
        elif file_type == 'dict':
            wavelengths = filelist['wavelengths']
            fluxes = filelist['fluxes']
            f_errors = filelist['f_errors']
            names = filelist['names']
            jds = filelist['jds']

    return wavelengths, fluxes, f_errors, names, jds

def setup_star_directory_and_save_jds(names, jds, path, SB2):
    """
    Prepare a directory for star data and save the corresponding Julian Dates.

    Parameters:
    names : list
        List of star names, formatted as 'STAR_EPOCH'.
    jds : list
        List of Julian Dates for each observation.
    path : str
        Base directory to store the files.
    SB2 : bool
        If True, specifies the system as a spectroscopic binary (SB2) and modifies the path.

    Returns:
    str
        The final directory path used for saving the data.
    """
    try:
        base_name = names[0].split('_')
        star = base_name[0] + '_' + base_name[1] + '/'
    except IndexError:
        star = 'Unknown_Star/'
    path = path.replace('FITS/', '')
    if SB2:
        path = os.path.join(path, 'SB2')
    if not os.path.exists(path):
        os.makedirs(path)
    if any(jds):
        df_mjd = pd.DataFrame({'epoch': names, 'JD': jds})
        df_mjd.to_csv(os.path.join(path, 'JDs.txt'), index=False, header=False, sep='\t')
    return path

def setup_line_dictionary():
    """
    Create a dictionary of spectral lines, including regions and initial fitting parameters.

    Returns:
    dict
        A dictionary mapping spectral line identifiers to their respective properties.
    """
    lines_dic = {
        3995: { 'region': [3986, 4001], 'centre': None, 'wid_ini': 2, 'title': 'N II $\lambda$3995'},
        4009: { 'region': [4001, 4014], 'centre': [4009.2565, 0.00002], 'wid_ini': 3, 'title': 'He I $\lambda$4009'},
        4026: { 'region': [4013, 4039], 'centre': [4026.1914, 0.0010], 'wid_ini': 3, 'title': 'He I $\lambda$4026'},
        4102: { 'region': [4081, 4116], 'centre': [4101.734, 0.006], 'wid_ini': 6, 'title': 'H$\delta$'},
        4121: { 'region': [4114, 4126], 'centre': [4120.8154, 0.0012], 'wid_ini': 3, 'title': 'He I $\lambda$4121'},
        4128: { 'region': [4120, 4132], 'centre': [4128.07, 0.10], 'wid_ini': 2, 'title': 'Si II $\lambda$4128'},
        4131: { 'region': [4124, 4136], 'centre': [4130.89, 0.10], 'wid_ini': 2, 'title': 'Si II $\lambda$4131'},
        4144: { 'region': [4131, 4166], 'centre': [4143.761, 0.010], 'wid_ini': 3, 'title': 'He I $\lambda$4144'},
        4233: { 'region': [4225, 4237], 'centre': None, 'wid_ini': 2, 'title': 'Fe II $\lambda$4233'},
        4267: { 'region': [4259, 4271], 'centre': [4267.258, 0.007], 'wid_ini': 2, 'title': 'C II $\lambda$4267'},
        4340: { 'region': [4316, 4366], 'centre': [4340.472, 0.006], 'wid_ini': 7, 'title': 'H$\gamma$'},
        4388: { 'region': [4376, 4406], 'centre': [4387.9296, 0.0006], 'wid_ini': 3, 'title': 'He I $\lambda$4388'},
        4471: { 'region': [4454, 4487], 'centre': [4471.4802, 0.0015], 'wid_ini': 3, 'title': 'He I $\lambda$4471'},
        4481: { 'region': [4474, 4486], 'centre': [4481.130, 0.010], 'wid_ini': 2, 'title': 'Mg II $\lambda$4481'},
        4542: { 'region': [4533, 4548], 'centre': [4541.591, 0.010], 'wid_ini': 3, 'title': 'He II $\lambda$4542'},
        4553: { 'region': [4543, 4558], 'centre': [4552.62, 0.10], 'wid_ini': 3, 'title': 'Si III $\lambda$4553'},
        4861: { 'region': [4836, 4871], 'centre': [4861.35, 0.05], 'wid_ini': 5, 'title': 'H$\beta$'},
        4922: { 'region': [4911, 4926], 'centre': [4921.9313, 0.0005], 'wid_ini': 4, 'title': 'He I $\lambda$4922'},
        5412: { 'region': [5401, 5415], 'centre': [5411.52, 0.10], 'wid_ini': 4, 'title': 'He II $\lambda$5412'},
        5876: { 'region': [5861, 5884], 'centre': [5875.621, 0.010], 'wid_ini': 4, 'title': 'He I $\lambda$5876'},
        5890: { 'region': [5877, 5901], 'centre': [5889.951, 0.00003], 'wid_ini': 3, 'title': 'Na I $\lambda$5890'},
        6562: { 'region': [6538, 6579], 'centre': [6562.79, 0.030], 'wid_ini': 6, 'title': 'H$\alpha$'},
        6678: { 'region': [6664, 6686], 'centre': [6678.151, 0.010], 'wid_ini': 4, 'title': 'He I $\lambda$6678'},
        7774: { 'region': [7758, 7782], 'centre': [7774.17, 0.10], 'wid_ini': 3, 'title': 'O I $\lambda$7774'}
    }
    return lines_dic

def initialize_fit_variables(lines): # deprecated?
    """
    Initialize lists for storing fit parameters for each spectral line.

    Parameters:
    num_lines : int
        Number of spectral lines.

    Returns:
    tuple
        Tuple of lists for each parameter.
    """
    cen1, cen1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    amp1, amp1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    wid1, wid1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    cen2, cen2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    amp2, amp2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    wid2, wid2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    dely, sdev    = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    results, comps = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    delta_cen, chisqr = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    return (
        cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
        cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
        dely, sdev, results, comps, delta_cen, chisqr )

def setup_fits_plots(wavelengths):
    """
    Set up subplots for spectral line fitting based on the number of spectra.

    Parameters:
    wavelengths : list
        List of wavelength arrays for each spectrum.

    Returns:
    tuple
        (fig, axes) matplotlib Figure and array of Axes objects.
    """
    print('Setting up spectral-lines fit plots...')
    nplots = len(wavelengths)
    ncols = int(np.sqrt(nplots))
    nrows = nplots // ncols
    if ncols * nrows < nplots:
        nrows += 1
    print('  Number of columns:', ncols)
    print('  Number of rows:', nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols/1.2, 3 * nrows/1.2), sharey=True, sharex=True)
    fig.subplots_adjust(
        wspace=0.,   # No horizontal space between subplots
        hspace=0.,   # No vertical space between subplots
        left=0.1,
        right=0.98,
        top=0.98,
        bottom=0.10
    )
    axes = axes.flatten() if nplots > 1 else [axes]
    return fig, axes

def rv_shift_wavelength(lambda_emitted, v):
    """
    Compute the observed wavelength by applying a Doppler shift.
    
    Parameters:
    -----------
    lambda_emitted : float or array_like
        The emitted/rest wavelength.
    v : float
        The radial velocity in km/s.
        
    Returns:
    --------
    lambda_observed : float or array_like
        The observed wavelength after shifting.
    """
    c_kms = c.to('km/s').value  
    lambda_observed = lambda_emitted * (1 + (v / c_kms))
    return lambda_observed

def fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, path, K=2, shift_kms=0):
    """
    Fit SB2 (double-lined spectroscopic binary) spectral lines using a probabilistic
    model with Numpyro. The function interpolates spectral data onto a common grid,
    constructs a Bayesian model for the line profiles, and samples the posterior via
    MCMC (using NUTS).
    
    Parameters:
    -----------
    lines : list
        List of spectral line identifiers (keys from lines_dic) to be fitted.
    wavelengths : list
        List (per epoch) of wavelength arrays.
    fluxes : list
        List (per epoch) of flux arrays.
    f_errors : list
        List (per epoch) of flux error arrays.
    lines_dic : dict
        Dictionary containing spectral line regions, initial centre guesses, etc.
    Hlines : list
        List of lines (subset of `lines`) that are Hydrogen lines.
    neblines : list
        (Currently unused) List of nebular lines.
    path : str
        Path for storing output plots.
    K : int, optional
        Number of components (default 2).
    shift_kms : float, optional
        The overall velocity shift in km/s. For example, use 172 km/s for the SMC.

    Returns:
    --------
    trace : dict
        The MCMC trace (posterior samples).
    x_waves : array (JAX)
        The interpolated wavelength grid for each line and epoch.
    y_fluxes : array (JAX)
        The interpolated fluxes.
    """
    n_lines = len(lines)
    n_epochs = len(wavelengths)
    print('Number of lines:', n_lines)
    print('Number of epochs:', n_epochs)

    # Boolean mask for Hydrogen lines (will use Lorentzian instead of Gaussian)
    is_hline = jnp.array([line in Hlines for line in lines])

    # Interpolate fluxes and errors to a common grid
    x_waves_interp = []
    y_fluxes_interp = []
    y_errors_interp = []
    common_grid_length = 200  # Choose a consistent number of points for interpolation

    for line in lines:
        region_start, region_end = lines_dic[line]['region']
        # Shift the region boundaries by shift_kms
        region_start = rv_shift_wavelength(region_start, shift_kms)
        region_end = rv_shift_wavelength(region_end, shift_kms)

        x_waves_line = []
        y_fluxes_line = []
        y_errors_line = []

        for wave_set, flux_set, error_set in zip(wavelengths, fluxes, f_errors):
            mask = (wave_set > region_start) & (wave_set < region_end)
            wave_masked = wave_set[mask]
            flux_masked = flux_set[mask]
            if error_set is not None:
                error_masked = error_set[mask]
            else:
                f_err = compute_flux_err(wave_set, flux_set)
                error_masked = f_err[mask]

            # Interpolate onto a common wavelength grid for this line and epoch
            common_wavelength_grid = np.linspace(wave_masked.min(), wave_masked.max(), common_grid_length)
            interp_flux = interp1d(wave_masked, flux_masked, bounds_error=False, fill_value="extrapolate")(common_wavelength_grid)
            interp_error = interp1d(wave_masked, error_masked, bounds_error=False, fill_value="extrapolate")(common_wavelength_grid)
            x_waves_line.append(common_wavelength_grid)
            y_fluxes_line.append(interp_flux)
            y_errors_line.append(interp_error)

        x_waves_interp.append(x_waves_line)
        y_fluxes_interp.append(y_fluxes_line)
        y_errors_interp.append(y_errors_line)

    # Convert the interpolated lists to JAX arrays (all dimensions now match)
    x_waves = jnp.array(x_waves_interp)       # Shape: (n_lines, n_epochs, common_grid_length)
    y_fluxes = jnp.array(y_fluxes_interp)       # Shape: (n_lines, n_epochs, common_grid_length)
    y_errors = jnp.array(y_errors_interp)       # Shape: (n_lines, n_epochs, common_grid_length)

    # Initial guess for the rest (central) wavelength from lines_dic
    cen_ini = jnp.array([lines_dic[line]['centre'][0] for line in lines])

    # Define the probabilistic SB2 model
    def sb2_model(λ, fλ, σ_fλ, K, is_hline, Δv_means):
        """
        Numpyro model for SB2 line-profile fitting.

        Parameters:
        -----------
        λ : JAX array
            Interpolated wavelengths with shape (n_lines, n_epochs, ndata).
        fλ : JAX array
            Observed fluxes with shape (n_lines, n_epochs, ndata).
        σ_fλ : JAX array
            Flux uncertainties with shape (n_lines, n_epochs, ndata).
        K : int
            Number of velocity components.
        is_hline : JAX array
            Boolean mask for Hydrogen lines.
        Δv_means : JAX array
            Mean velocity shifts for the K components, with shape (K, 1, 1).

        Returns:
        --------
        Samples are observed via npro.sample("fλ", ...).
        """
        c_kms = c.to('km/s').value  
        nlines, nepochs, ndata = λ.shape

        # Sample continuum level with uncertainty
        logσ_ε = npro.sample('logσ_ε', dist.Uniform(-5, 0))
        σ_ε = jnp.exp(logσ_ε)
        ε = npro.sample('ε', dist.TruncatedNormal(loc=1.0, scale=σ_ε, low=0.7, high=1.1))

        # Define rest wavelengths as a parameter (one per line)
        λ_rest = npro.param("λ_rest", cen_ini)  # Shape: (n_lines,)
        
        # Sample velocity shifts for each epoch and component
        σ_Δv = 200.
        with npro.plate(f'epochs', nepochs, dim=-1):
            Δv_τk = npro.sample("Δv_τk", dist.Normal(loc=Δv_means, scale=σ_Δv))
            # Δv_τk shape: (K, nepochs)

        # Sample amplitudes and widths for each line
        amp1_min, amp1_max = 0.05, 0.3
        amp2_min = 0.01
        with npro.plate(f'lines', nlines, dim=-2):
            amp0 = npro.sample('amp0', dist.Uniform(amp1_min, amp1_max))
            amp1 = npro.sample('amp1', dist.Uniform(amp2_min, 0.75 * amp0))
            # Stack amplitudes for two components and add extra dimensions for broadcasting
            amp = jnp.stack([amp0, amp1], axis=-3)  # Shape: (2, n_lines)
            amp = amp[:, :, None]  # Shape: (2, n_lines, 1)
            
            # Sample widths for the first component and derive the second component's width.
            # Note: By enforcing wid2 = wid1 + delta_wid (with delta_wid > 0), we ensure that
            # the second component's width is always larger than the first. This constraint is
            # implemented to improve the stability of the fit and prevent label-switching issues,
            # common in multi-component SB2 spectra where lines are often misidentified.
            # For future updates: consider sampling both widths independently and applying
            # a permutation-invariant or post-hoc relabeling scheme if physical evidence suggests
            # that wid2 < wid1 is a possibility.
            wid1 = npro.sample('wid1', dist.Uniform(0.5, 5.0))
            delta_wid = npro.sample('delta_wid', dist.Uniform(0.1, 2.0))
            wid2 = wid1 + delta_wid
            wid = jnp.stack([wid1, wid2], axis=-3)  # Shape: (2, n_lines)
            wid = wid[:, :, None]  # Shape: (2, n_lines, 1)

        # Make λ_rest a deterministic variable and reshape for broadcasting
        λ0 = npro.deterministic("λ0", λ_rest)[None, :, None]  # Shape: (1, n_lines, 1)

        # Compute shifted wavelengths for each component and epoch
        μ = λ0 * (1 + Δv_τk / c_kms)  # Broadcasts: (K, n_lines, nepochs) then add an extra axis
        μ = μ[:, :, :, None]  # Final shape: (K, n_lines, nepochs, 1)

        # Prepare the observed wavelengths for model evaluation
        λ_expanded = λ[None, :, :, :]  # Shape: (1, n_lines, nepochs, ndata)
        is_hline_expanded = is_hline[None, :, None, None]  # Shape: (1, n_lines, 1, 1)

        # Compute the model profiles for each component
        gaussian_profile = gaussian(λ_expanded, amp, μ, wid)
        lorentzian_profile = lorentzian(λ_expanded, amp, μ, wid)
        # Use Lorentzian for Hydrogen lines, Gaussian otherwise:
        comp_profile = jnp.where(is_hline_expanded, lorentzian_profile, gaussian_profile)
        Ck = npro.deterministic("C_λk", comp_profile)

        # Sum over components and add continuum to yield the predicted flux
        fλ_pred = npro.deterministic("fλ_pred", ε + Ck.sum(axis=0))
        # Likelihood: compare predicted flux with observed flux
        npro.sample("fλ", dist.Normal(fλ_pred, σ_fλ), obs=fλ)

    # ------------------------
    # MCMC Sampling Procedure
    # ------------------------
    comp_sep = 500.
    Δv_means = jnp.array([shift_kms - comp_sep/2, shift_kms + comp_sep/2]).reshape(K, 1, 1)
    print(f"\nFitting with Δv_means: {Δv_means}")

    # Set a fixed random key (you can change this seed if desired)
    rng_key = random.PRNGKey(0)
    kernel = NUTS(sb2_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
    mcmc.run(rng_key, extra_fields=("potential_energy",), 
             λ=x_waves, fλ=y_fluxes, σ_fλ=y_errors, K=K, is_hline=is_hline, Δv_means=Δv_means)

    # Evaluate the mean log posterior probability (for diagnostic purposes)
    potential_energy = mcmc.get_extra_fields()['potential_energy']
    log_probs = -potential_energy  # Convert potential energy to log probability
    log_prob = np.mean(log_probs)
    print(f"Mean log posterior probability: {log_prob}")

    # Get the MCMC trace (posterior samples)
    trace = mcmc.get_samples()

    # Optionally perform prior predictive checks (set fλ_pred_plot=True to activate)
    fλ_pred_plot = False
    if fλ_pred_plot:
        num_prior_samples = 200
        prior_predictive = Predictive(sb2_model, num_samples=num_prior_samples)
        prior_samples = prior_predictive(rng_key, λ=x_waves, σ_fλ=y_errors, K=K, is_hline=is_hline, Δv_means=Δv_means)
        # Plotting code for prior predictive checks can go here...
        plt.show()

    # ------------------------
    # Plotting the Fitted Profiles
    # ------------------------
    plot_lines_fit(wavelengths, lines, x_waves, y_fluxes, n_epochs, trace, lines_dic, shift_kms, comp_sep, path)

    return trace, x_waves, y_fluxes

def plot_lines_fit(wavelengths, lines, x_waves, y_fluxes, n_epochs, trace, lines_dic, shift_kms, comp_sep, path, n_sol=100):
    """
    Plot the SB2 line-fit results based on the posterior predictions.

    Parameters:
    -----------
    wavelengths : list
        Original wavelength arrays.
    lines : list
        List of spectral line identifiers.
    x_waves : JAX array
        Interpolated wavelength grids (per line and epoch).
    y_fluxes : JAX array
        Interpolated fluxes.
    n_epochs : int
        Number of epochs.
    trace : dict
        Posterior samples from MCMC.
    lines_dic : dict
        Dictionary with spectral line details.
    shift_kms : float
        The applied velocity shift (km/s).
    comp_sep : float
        Component separation used in the model.
    path : str
        Directory path to save the plots.
    n_sol : int, optional
        Number of posterior samples to plot (default: 100).
    """
    from matplotlib.lines import Line2D 

    for idx, line in enumerate(lines):
        print('Plotting fits for line:', line)
        fig, axes = setup_fits_plots(wavelengths)
        for epoch_idx, ax in enumerate(axes.ravel()[:n_epochs]):
            # Extract posterior predictions for this line and epoch
            fλ_pred_samples = trace['fλ_pred'][-n_sol:, idx, epoch_idx, :]  # (n_sol, ndata)
            # Extract continuum predictions
            continuum_pred_samples = trace['ε'][-n_sol:, None]
            # Extract the posterior samples for each component
            fλ_pred_comp1_samples = continuum_pred_samples + trace['C_λk'][-n_sol:, 0, idx, epoch_idx, :]
            fλ_pred_comp2_samples = continuum_pred_samples + trace['C_λk'][-n_sol:, 1, idx, epoch_idx, :]

            # Plot posterior predictive samples for each component and the total prediction
            ax.plot(x_waves[idx][epoch_idx], fλ_pred_comp1_samples.T, color='C1', alpha=0.1, rasterized=True)
            ax.plot(x_waves[idx][epoch_idx], fλ_pred_comp2_samples.T, color='C0', alpha=0.1, rasterized=True)
            ax.plot(x_waves[idx][epoch_idx], fλ_pred_samples.T, color='C2', alpha=0.1, rasterized=True)
            # Plot the observed data
            ax.plot(x_waves[idx][epoch_idx], y_fluxes[idx][epoch_idx], color='k', lw=1, alpha=0.8)
            # Plot vertical lines for the rest wavelength and component shifts
            centre = rv_shift_wavelength(lines_dic[line]['centre'][0], shift_kms)
            ax.axvline(centre, color='r', linestyle='--', lw=1)
            ax.axvline(rv_shift_wavelength(lines_dic[line]['centre'][0], shift_kms - comp_sep/2), color='orange', linestyle='--', lw=1)
            ax.axvline(rv_shift_wavelength(lines_dic[line]['centre'][0], shift_kms + comp_sep/2), color='orange', linestyle='--', lw=1)
            # Annotate the epoch number
            ax.text(0.15, 0.86, f'Epoch {epoch_idx+1}', transform=ax.transAxes, fontsize=16)

        ax.set_xlim(centre - 13, centre + 13)
        # Create a custom legend
        custom_lines = [
            Line2D([0], [0], color='C2', alpha=0.5, lw=2),
            Line2D([0], [0], color='C1', alpha=0.5, lw=2),
            Line2D([0], [0], color='C0', alpha=0.5, lw=2)
        ]
        axes[0].legend(custom_lines, ['Total Prediction', 'Component 1', 'Component 2'], 
                       fontsize=14, frameon=False, borderaxespad=0.1)
        fig.supxlabel('Wavelength [Å]', fontsize=24)
        fig.supylabel('Flux', fontsize=24)
 
        plt.savefig(os.path.join(path, f'{line}_fits_SB2_.png'), dpi=300)
        plt.close()

def fit_sb1(line, wave, flux, ferr, lines_dic, Hlines, neblines, doubem, shift):
    """
    Fits a single spectral line with a profile model (Gaussian or Lorentzian) 
    plus a linear continuum using lmfit. Optionally adds one or two nebular 
    emission components if the line is known to have nebular contamination.
    
    Parameters:
        line (int): Central wavelength (or identifier) of the line to be fitted.
        wave (array-like): Wavelength array from the spectrum.
        flux (array-like): Flux array from the spectrum.
        ferr (array-like): Flux errors.
        lines_dic (dict): Dictionary with line parameters. Expected to include:
            - 'region': [min, max] wavelength limits for the line.
            - 'wid_ini': Initial guess for the line width.
            - 'centre': Initial central wavelength (or a list where the first element is used).
        Hlines (iterable): List of lines which should be fitted with a Lorentzian profile 
                           (commonly for Hydrogen lines).
        neblines (iterable): List of lines to fit with additional nebular emission components.
        doubem (iterable): List of lines that require double-peak nebular emission models.
        shift (float): Wavelength shift to apply to the central wavelength.

    Returns:
        result (ModelResult): The lmfit fit result object.
        x_wave (array-like): The subset of the wavelength array used in the fit.
        y_flux (array-like): The corresponding flux values.
        wave_region (array-like): Boolean array indicating which wavelengths were used.
    """
    # Define the wavelength region for fitting
    wave_region = (wave > lines_dic[line]['region'][0]) & (wave < lines_dic[line]['region'][1])
    x_wave = wave[wave_region]
    y_flux = flux[wave_region]

    # Initial guesses for central wavelength and width
    cen_ini = line + shift
    wid_ini = lines_dic[line]['wid_ini']

    # Define the models: continuum, profile (Gaussian or Lorentzian), and nebular emission(s)
    pars = Parameters()
    continuum = models.LinearModel(prefix='continuum_')
    gauss_model = Model(gaussian, prefix='g1_')
    loren_model = Model(lorentzian, prefix='l1_')
    nebem_model1 = Model(nebu, prefix='neb1_')
    nebem_model2 = Model(nebu, prefix='neb2_')
    
    # Setup continuum parameters, fix slope and constrain intercept near 1
    pars.update(continuum.make_params())
    pars['continuum_slope'].set(0, vary=False)
    pars['continuum_intercept'].set(1, min=0.9)
    
    # Choose the individual line model based on whether the line is in Hlines
    if line in Hlines:
        prefix = 'l1'
        indiv_mod = loren_model
    else:
        prefix = 'g1'
        indiv_mod = gauss_model
    pars.update(indiv_mod.make_params())
    # Set initial amplitude based on the difference from the minimum flux,
    # initial width from the dictionary (with some constraints),
    # and set the central wavelength to the shifted value.
    pars[f'{prefix}_amp'].set(1.0 - y_flux.min(), min=0.05, max=0.9)
    pars[f'{prefix}_wid'].set(wid_ini, min=1, max=11)
    pars[f'{prefix}_cen'].set(cen_ini, vary=True)
    
    # Combine the individual model with the continuum
    mod = indiv_mod + continuum

    # Add nebular emission components if the line is flagged as nebular
    nebem_models = [nebem_model1, nebem_model2]
    for i, nebem in enumerate(nebem_models, start=1):
        if line in neblines:
            pars.update(nebem.make_params())
            # Set the nebular amplitude based on flux variation.
            if line == 4102:
                pars[f'neb{i}_amp'].set((y_flux.max() - y_flux.min()) * 0.6, min=0.01)
            elif line == 4340:
                pars[f'neb{i}_amp'].set(y_flux.max() - y_flux.min(), min=0.01)
            else:
                pars[f'neb{i}_amp'].set((y_flux.max() - y_flux.min()) * 0.6, min=0.01)
            # Set width for nebular components with mild constraints.
            pars[f'neb{i}_wid'].set(1, min=0.05, max=3)
            # Offset the center of the nebular emission slightly differently
            # for the first and potential second component.
            if i == 2:
                if line not in doubem:
                    break  # Do not add a second nebular component if not flagged
                pars[f'neb{i}_cen'].set(cen_ini + 0.2, vary=True)
            else:
                pars[f'neb{i}_cen'].set(cen_ini - 0.2, vary=True)
            mod += nebem

    # Fit the model to the data with weights from the flux errors
    result = mod.fit(y_flux, pars, x=x_wave, weights=1 / ferr[wave_region])

    return result, x_wave, y_flux, wave_region

def mcmc_results_to_file(trace, names, jds, writer, csvfile):
    """
    Write MCMC fit results for multiple components and epochs to a CSV file.
    
    For each component (assumed to be two components) and for each epoch (from the 
    provided 'names' list) the function calculates the mean RV and its uncertainty 
    from the MCMC trace and writes these values alongside the epoch and MJD.
    
    Parameters:
        trace (dict): Dictionary containing MCMC samples (expects key 'Δv_τk').
        names (list): List of epoch identifiers (e.g., names of spectra or observation epochs).
        jds (list): List of corresponding Julian Dates (or None if unavailable).
        writer (csv.DictWriter or None): A DictWriter object or None (if first call, will be initialized).
        csvfile: An open CSV file handle to write the output.
        
    Returns:
        writer: A csv.DictWriter instance after writing the header (if initially None) and all rows.
    """
    # Loop over the two components (0 and 1, later converted to 1-based indexing)
    for i in range(2):
        # Loop over epochs (names)
        for j, epoch_name in enumerate(names):
            results_dict = {}
            results_dict['epoch'] = epoch_name
            if jds is not None and j < len(jds):
                results_dict['MJD'] = jds[j]

            # Calculate the mean RV and its error for component i at epoch j.
            # Expected trace shape for Δv_τk: (n_samples, K, ..., n_epochs)
            results_dict['mean_rv'] = np.mean(trace['Δv_τk'][:, i, :, j])
            results_dict['mean_rv_er'] = np.std(trace['Δv_τk'][:, i, :, j])

            # Components are 1-based for output (i.e., Component 1 and 2)
            results_dict['comp'] = i + 1

            # Initialize the writer if not already created
            if writer is None:
                fieldnames = results_dict.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(results_dict)

    return writer

def SLfit(spectra_list, data_path, save_path, lines, K=2, file_type='fits', instrument='FLAMES',
          plots=True, balmer=True, neblines=[], doubem=[], SB2=False, init_guess_shift=0,
          shift_kms=0, use_init_pars=False):
    """
    Perform spectral line fitting on a list of spectra. This function reads the spectral data, sets up
    the output directory, initializes line dictionaries and fit variables, and then fits each spectral
    line using either an SB2 (double-lined) or SB1 (single-lined) approach.
    
    Parameters:
        spectra_list (list): List of spectral file paths (or data objects).
        data_path (str): The directory where the spectral files are stored.
        save_path (str): The output directory for saving fit results and plots.
        lines (list): List of spectral lines (identified by wavelength or key) to be fitted.
        K (int): Number of components for the SB2 fit (default is 2).
        file_type (str): Type of the input files (e.g., 'fits', 'csv').
        instrument (str): Instrument identifier (e.g., 'FLAMES').
        plots (bool): Whether to produce and save diagnostic plots.
        balmer (bool): (Not used in this version; reserved for Balmer line special handling).
        neblines (list): List of lines known to contain nebular emission, to include extra components.
        doubem (list): List of lines where double-peak nebular emission models should be used.
        SB2 (bool): If True, perform the SB2 (double-lined) fit; otherwise, perform SB1 (single-lined).
        init_guess_shift (float): Initial wavelength shift for the fitting guess.
        shift_kms (float): Velocity shift (in km/s) to be applied to the line centres.
        use_init_pars (bool): (Reserved) Whether to use pre-defined initial parameters.
    
    Returns:
        str: The output directory path where fit results and plots were saved.
    """
    print('\n')
    print('*******************************************************************************')
    print('******************           Spectral Line fitting           ******************')
    print('*******************************************************************************\n')
    
    # If there's only one spectrum (epoch), warn and exit because RV computation requires multiple epochs
    if len(spectra_list) == 1:
        print("\n   WARNING: There is only 1 epoch to compute RVs.")
        return

    # Define default Hydrogen lines for which Lorentzian profiles may be used
    Hlines = [4102, 4340, 4861, 6562]
    print('*** SB2 set to:', SB2, '***\n')

    # Read in spectral data from the provided file list and data path
    wavelengths, fluxes, f_errors, names, jds = read_spectra(spectra_list, data_path, file_type, instrument=instrument)
    # print('names:', names)
    
    # Setup the output directory and save the JD information if available
    out_path = setup_star_directory_and_save_jds(names, jds, save_path, SB2)
    # print('Output path:', out_path)
    
    # Get the dictionary with line regions and initial parameters
    lines_dic = setup_line_dictionary()
    
    print('\n*** Fitting lines ***')
    print('---------------------')
    print('Lines to be fitted:', lines)
    
    # Initialize fit variables for each line (lists to hold fit results, uncertainties, etc.)
    (cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
     cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
     dely, sdev, results, comps, delta_cen, chisqr) = initialize_fit_variables(lines)
    
    # Open a CSV file to save summary fit values
    with open(out_path + 'fit_values.csv', 'w', newline='') as csvfile:
        writer = None
        
        print('Fitting all lines simultaneously' if SB2 else 'Fitting lines individually')
        if SB2:
            # SB2 fitting: fit all lines using the probabilistic SB2 model and write results to CSV
            result, x_wave, y_flux = fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic,
                                                      Hlines, neblines, out_path, K=K, shift_kms=shift_kms)
            writer = mcmc_results_to_file(result, names, jds, writer, csvfile)
            
            # (Optional plotting of SB2 fits is handled within fit_sb2_probmod and plot_lines_fit)
        else:
            # SB1 fitting: iterate over each line and each epoch, fit the line, and write results to CSV
            for i, line in enumerate(lines):
                # Create plots if enabled; otherwise, use a placeholder list for axes
                if plots:
                    fig, axes = setup_fits_plots(wavelengths)
                else:
                    axes = [None] * len(wavelengths)
                    
                # Process each epoch separately
                for j, (wave, flux, ferr, name, ax) in enumerate(zip(wavelengths, fluxes, f_errors, names, axes)):
                    result, x_wave, y_flux, wave_region = fit_sb1(line, wave, flux, ferr, lines_dic,
                                                                   Hlines, neblines, doubem, shift=init_guess_shift)
                    results[i].append(result)
                    chisqr[i].append(result.chisqr)
                    
                    # Get component information if available (for nebular lines)
                    if line in neblines:
                        component = result.eval_components(result.params, x=x_wave)
                        comps[i].append(component)
                    else:
                        component = None
                        comps[i].append(component)
                    
                    # Write fit statistics to a per-line text file
                    file_mode = 'w' if j == 0 else 'a'
                    with open(out_path + str(line) + '_stats.txt', file_mode) as out_file:
                        out_file.write(name + '\n')
                        out_file.write('-' * len(name.strip()) + '\n')
                        out_file.write(result.fit_report() + '\n\n')
                    
                    # Determine the appropriate prefix based on the fit type
                    if 'g1_cen' in result.params:
                        prefix = 'g1'
                    elif 'l1_cen' in result.params:
                        prefix = 'l1'
                    else:
                        raise ValueError("Unexpected fit type encountered.")
                    
                    # Build a dictionary of fit results for CSV output
                    results_dict = {
                        'epoch': name,
                        'line': line,
                        'cen1': result.params[f'{prefix}_cen'].value,
                        'cen1_er': result.params[f'{prefix}_cen'].stderr,
                        'amp1': result.params[f'{prefix}_amp'].value,
                        'amp1_er': result.params[f'{prefix}_amp'].stderr,
                        'wid1': result.params[f'{prefix}_wid'].value,
                        'wid1_er': result.params[f'{prefix}_wid'].stderr,
                        'chisqr': result.chisqr
                    }
                    if results_dict['cen1_er'] is None or results_dict['amp1_er'] is None:
                        print('Error: Uncertainty computation failed for line', line, 'epoch', j + 1)
                    # Optionally compute a 3-sigma uncertainty region (if available)
                    if results_dict['cen1_er'] not in [0, None]:
                        dely = result.eval_uncertainty(sigma=3)
                    else:
                        dely = None
                    
                    # Initialize CSV writer if needed and write the current row
                    if writer is None:
                        fieldnames = results_dict.keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                    writer.writerow(results_dict)
                    
                    # Plotting of the fit results if enabled
                    if plots:
                        xx = np.linspace(min(x_wave), max(x_wave), 500)
                        result_new = result.eval(x=xx)
                        init_new = result.eval(params=result.init_params, x=xx)
                        ax.plot(x_wave, y_flux, 'k-', lw=3, ms=4, zorder=1,
                                label=str(name.replace('./', '').replace('_', ' ')))
                        ax.plot(xx, init_new, '--', c='grey', zorder=5)
                        ax.plot(xx, result_new, 'r-', lw=2, zorder=4)
                        
                        if component is not None:
                            if line in Hlines:
                                ax.plot(x_wave, component['continuum_'] + component['l1_'],
                                        '--', zorder=3, c='limegreen', lw=2)
                            elif SB2 and line not in Hlines:
                                ax.plot(x_wave, component['continuum_'] + component['g1_'],
                                        '--', zorder=3, c='blue', lw=2)
                                ax.plot(x_wave, component['continuum_'] + component['g2_'],
                                        '--', zorder=3, c='blue', lw=2)
                            if line in neblines:
                                ax.plot(x_wave, component['continuum_'] + component['neb1_'],
                                        '--', zorder=3, c='orange', lw=2)
                            if line in neblines and not line in Hlines and not SB2:
                                ax.plot(x_wave, component['continuum_'] + component['g1_'],
                                        '--', zorder=3, c='limegreen', lw=2)
                        
                        if dely is not None:
                            ax.fill_between(x_wave, result.best_fit - dely, result.best_fit + dely,
                                            zorder=2, color="#ABABAB", alpha=0.5)
                        ax.set_ylim(0.9 * y_flux.min(), 1.1 * y_flux.max())
                # Save the figure for the current line after processing all epochs
                if plots:
                    fig.supxlabel('Wavelength', size=24)
                    fig.supylabel('Flux', size=24)
                    filename = out_path + str(line) + '_fits.png'
                    plt.savefig(filename, bbox_inches='tight', dpi=150)
                    plt.close()
        
    return out_path

class GetRVs:
    """
    Class for extracting, processing, and analyzing radial velocity (RV) measurements
    from SLfit spectral line fitting results.

    This class performs a full RV analysis workflow that includes:
      - Loading SLfit fit values from a CSV file and computing percentage errors for key
        parameters (e.g., centroids, amplitudes, and widths) for the primary (and secondary,
        if SB2) components.
      - Computing radial velocities from the measured line centers by comparing them
        with pre-defined rest wavelengths (with error propagation).
      - Generating diagnostic plots for individual spectral lines, including errorbar plots 
        per epoch and an overall RV plot for each component.
      - Grouping and statistically analyzing the percentage errors to identify the best 
        spectral lines using an outlier detection approach based on the median absolute 
        deviation (MAD).
      - Removing bad epochs and lines based on user-defined error thresholds.
      - Calculating weighted mean radial velocities for each epoch from the selected best lines.
      - Printing and writing comprehensive summary statistics and final RV measurements to
        various output files.

    Attributes:
        fit_values (str): Filename of the CSV file containing SLfit fit results.
        path (str): Directory path for reading input files and writing output results.
        JDfile (str): Filename containing the Julian Date (MJD) information.
        balmer (bool): Flag indicating whether Balmer-specific processing should be applied.
        SB2 (bool): Flag indicating whether the analysis pertains to double-lined (SB2) spectra.
        use_lines (list): Optional list of spectral lines to consider; if None, all available lines are used.
        lines_ok_thrsld (float): Threshold for acceptable line error percentages.
        epochs_ok_thrsld (float): Threshold for acceptable epoch errors.
        min_sep (float): Minimum separation (e.g., in wavelength) required between lines.
        print_output (bool): If True, prints debug and status messages.
        random_eps (bool): Flag for using random epsilon values (experimental).
        rndm_eps_n (int): Number of random epsilon values to try.
        rndm_eps_exc (list): List of lines to exclude from random epsilon sampling.
        plots (bool): Flag indicating whether to generate diagnostic plots.
        error_type (str): Key to determine which error metric is used for selection (e.g., 'wid1_percer').
        rm_epochs (bool): If True, epochs with large errors are removed from further analysis.
        df_SLfit (pd.DataFrame): DataFrame containing the SLfit results.
        lines (ndarray): Unique spectral line identifiers present in the SLfit results.
        nepochs (int): Number of epochs calculated from the SLfit data.

    Methods:
        outlier_killer(data, thresh, print_output):
            Identifies inliers and outliers in a data series using a MAD test.
        weighted_mean(data, errors):
            Computes the weighted mean and error of a set of values given uncertainties.
        compute_rvs(lines, lambda_rest_dict):
            Computes radial velocities (and errors) by comparing fitted line centers with rest wavelengths.
        print_error_stats(grouped_error, error_type):
            Prints mean and median statistics for grouped error data.
        write_error_stats(out, grouped_errors, stat_type):
            Writes summary error statistics to an output file.
        select_lines(error_type, lines):
            Selects the best spectral lines based on error metrics.
        remove_bad_epochs(df, metric, error_type, epochs_ok_thrsld):
            Identifies and returns indices of epochs with excessive errors.
        compute_weighted_mean_rvs(rvs_dict, lines, rm_OBs_idx):
            Computes weighted mean radial velocities per epoch from the best spectral lines.
        print_and_write_results(lines, line_avg1, total_mean_rv1, nepochs, line_avg2, total_mean_rv2):
            Prints and writes summary RV results to a file.
        compute():
            Orchestrates the full RV analysis workflow:
              - Defines rest wavelengths.
              - Computes per-line RVs.
              - Generates diagnostic plots of RVs per line.
              - Groups errors and selects the best lines.
              - Removes problematic epochs.
              - Computes weighted mean RVs and writes final results.
    """

    def __init__(self, fit_values, path, JDfile, balmer=False, SB2=False, use_lines=None,
                 lines_ok_thrsld=2, epochs_ok_thrsld=2, min_sep=2, print_output=True,
                 random_eps=False, rndm_eps_n=29, rndm_eps_exc=[], plots=True,
                 error_type='wid1_percer', rm_epochs=True):
        """
        Initialize the GetRVs instance and load the SLfit results. Computes percentage errors
        for the primary (and secondary, if SB2) components.

        Parameters:
            fit_values (str): Filename of the CSV file with SLfit fit values.
            path (str): Directory path for reading and writing output.
            JDfile (str): Filename for Julian Date data.
            balmer (bool): Whether to apply Balmer-specific settings.
            SB2 (bool): Whether the fits correspond to double-lined spectra.
            use_lines (list or None): Specific lines to use; if None, all lines are considered.
            lines_ok_thrsld (float): Threshold for acceptable line errors.
            epochs_ok_thrsld (float): Threshold for acceptable epoch errors.
            min_sep (float): Minimum required separation between lines.
            print_output (bool): Whether to print output messages.
            random_eps (bool): Experimental flag for using random epsilon values.
            rndm_eps_n (int): Number of random epsilon values.
            rndm_eps_exc (list): Lines to exclude from random epsilon sampling.
            plots (bool): Whether to generate plots.
            error_type (str): Which error metric to use (e.g., 'wid1_percer').
            rm_epochs (bool): Whether to remove epochs with large errors.
        """
        self.fit_values = fit_values
        self.path = path
        self.JDfile = JDfile
        self.balmer = balmer
        self.SB2 = SB2
        self.use_lines = use_lines
        self.lines_ok_thrsld = lines_ok_thrsld
        self.epochs_ok_thrsld = epochs_ok_thrsld
        self.min_sep = min_sep
        self.print_output = print_output
        self.random_eps = random_eps
        self.rndm_eps_n = rndm_eps_n
        self.rndm_eps_exc = rndm_eps_exc
        self.plots = plots
        self.error_type = error_type
        self.rm_epochs = rm_epochs

        # Get the current date for reference
        date_current = str(date.today())

        # Load the SLfit results CSV file
        self.df_SLfit = pd.read_csv(self.path + self.fit_values)
        if self.df_SLfit.isnull().values.any():
            print("Warning: NaN values found in df_SLfit")
        # Compute percentage errors for primary component parameters
        for param in ['cen1', 'amp1', 'wid1']:
            self.df_SLfit[f'{param}_percer'] = np.where(
                self.df_SLfit[f'{param}_er'] != 0,
                np.abs(100 * self.df_SLfit[f'{param}_er'] / self.df_SLfit[param]),
                np.nan
            )
        # For SB2, compute percentage errors for secondary component parameters
        if self.SB2:
            for param in ['cen2', 'amp2', 'wid2']:
                self.df_SLfit[f'{param}_percer'] = np.where(
                    self.df_SLfit[f'{param}_er'] != 0,
                    np.abs(100 * self.df_SLfit[f'{param}_er'] / self.df_SLfit[param]),
                    np.nan
                )

        # Store the unique lines from the fits and calculate the number of epochs
        self.lines = self.df_SLfit['line'].unique()
        print('lines from SLfit results:', self.lines)
        self.nepochs = len(self.df_SLfit) // len(self.lines)

    def outlier_killer(self, data, thresh=2, print_output=None):
        """
        Identify inliers and outliers from a data array using a MAD-based test.
        
        Parameters:
            data (array-like): Numerical data to test.
            thresh (float): Threshold multiplier for the MAD.
            print_output (bool or None): Whether to print details (default uses self.print_output).
        
        Returns:
            tuple: (inliers, outliers) where both are lists of indices.
        """
        if print_output is None:
            print_output = self.print_output
        diff = data - np.nanmedian(data)
        # Output debug information to file and/or console
        with open(self.path + 'rv_stats.txt', 'a') as f:
            if print_output:
                print('     mean(x)        =', f'{np.nanmean(data):.3f}')
                print('     mean(x)        =', f'{np.nanmean(data):.3f}', file=f)
                print('     median(x)      =', f'{np.nanmedian(data):.3f}')
                print('     median(x)      =', f'{np.nanmedian(data):.3f}', file=f)
                print('     x-median(x)    =', [f'{x:.3f}' for x in diff])
                print('     x-median(x)    =', [f'{x:.3f}' for x in diff], file=f)
                print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff])
                print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff], file=f)
            mad = 1.4826 * np.nanmedian(np.abs(diff))
            if print_output:
                print('     mad =', f'{mad:.3f}')
                print('     mad =', f'{mad:.3f}', file=f)
                print('     abs(x-median(x)) <', f'{thresh * mad:.3f}', '(thresh*mad) =',
                      [abs(x) < thresh * mad for x in diff])
                print('     abs(x-median(x)) <', f'{thresh * mad:.3f}', '(thresh*mad) =',
                      [abs(x) < thresh * mad for x in diff], file=f)
        inliers = [i for i, x in enumerate(data) if abs(x - np.nanmedian(data)) < thresh * mad]
        outliers = [i for i in range(len(data)) if i not in inliers]
        return inliers, outliers

    @staticmethod
    def weighted_mean(data, errors):
        """
        Compute a weighted mean and its error based on input values and uncertainties.
        
        Parameters:
            data (list or array): Values for which the mean is computed.
            errors (list or array): Corresponding uncertainties.
        
        Returns:
            tuple: (weighted mean, weighted error)
        """
        weights = [1 / (dx ** 2) for dx in errors]
        mean = sum(wa * a for a, wa in zip(data, weights)) / sum(weights)
        mean_err = np.sqrt(sum((da * wa) ** 2 for da, wa in zip(errors, weights))) / sum(weights)
        return mean, mean_err

    def compute_rvs(self, lines, lambda_rest_dict):
        """
        Compute radial velocities (and uncertainties) for each spectral line.
        
        The method compares fitted central wavelengths (and errors) to the provided
        rest wavelengths, propagating errors appropriately. For SB2 fits, it computes
        results for the secondary as well.
        
        Parameters:
            lines (list): List of line identifiers.
            lambda_rest_dict (dict): Dictionary mapping each line to a tuple
                                     (lambda_rest, lambda_rest_error).
        
        Returns:
            dict: For each line, a sub-dictionary with keys 'rv1' and 'rv1_er' (and if SB2,
                  also 'rv2' and 'rv2_er').
        """
        c_kms = c.to('km/s').value
        rvs = {}
        for line in lines:
            lambda_rest, lambda_r_er = lambda_rest_dict[line]
            current_line_values = self.df_SLfit[self.df_SLfit['line'] == line]
            # Calculate the radial velocity for the first component
            dlambda1 = current_line_values['cen1'].values - lambda_rest
            dlambda1_er = np.sqrt(current_line_values['cen1_er'].values ** 2 + lambda_r_er ** 2)
            rv1 = dlambda1 * c_kms / lambda_rest
            rv1_er = np.sqrt((dlambda1_er / dlambda1) ** 2 + (lambda_r_er / lambda_rest) ** 2) * np.abs(rv1)
            rvs[line] = {'rv1': rv1, 'rv1_er': rv1_er}
            # If SB2, calculate the radial velocity for the second component
            if self.SB2:
                dlambda2 = np.abs(current_line_values['cen2'].values - lambda_rest)
                dlambda2_er = np.sqrt(current_line_values['cen2_er'].values ** 2 + lambda_r_er ** 2)
                rv2 = dlambda2 * c_kms / lambda_rest
                rv2_er = np.sqrt((dlambda2_er / dlambda2) ** 2 + (lambda_r_er / lambda_rest) ** 2) * rv2
                rvs[line].update({'rv2': rv2, 'rv2_er': rv2_er})
        return rvs

    @staticmethod
    def print_error_stats(grouped_error, error_type):
        """
        Print the mean and median values for a grouped error dictionary.
        
        Parameters:
            grouped_error (dict): Error values grouped by some line or parameter.
            error_type (str): A label describing the error type.
        """
        mean_values = [f'{np.mean(value):6.3f}' for value in grouped_error.values()]
        median_values = [f'{np.median(value):6.3f}' for value in grouped_error.values()]
        print(f'   mean({error_type})  ', ' '.join(mean_values))
        print(f'   median({error_type})', ' '.join(median_values))

    def write_error_stats(self, out, grouped_errors, stat_type):
        """
        Write summarized error statistics to an output file.
        
        Parameters:
            out (file-like): Open file handle to which the stats will be written.
            grouped_errors (dict): Grouped error data for each line.
            stat_type (str): Indicates whether using mean or median (e.g., "Mean").
        """
        stat_func = np.nanmean if stat_type == 'Mean' else np.nanmedian
        out.write(f' {stat_type} of percentual errors\n')
        out.write('   Lines   wid1    cen1   amp1   |    wid2    cen2   amp2 \n')
        out.write('   -------------------------------------------------------\n')
        for line in grouped_errors['cen1_percer'].keys():
            stats = ' '.join([f'{stat_func(grouped_errors[error_type][line]):7.3f}' 
                              for error_type in grouped_errors])
            out.write(f'   {line}: {stats}\n')
        out.write('\n')

    def select_lines(self, error_type, lines):
        """
        Select the best lines for RV computation based on error criteria.
        
        If no preferred lines are specified via use_lines, the method filters out those with
        NaN errors and then identifies outliers based on the median error.
        
        Parameters:
            error_type (pd.Series or pd.DataFrame): Error data (e.g., percentage errors) for each line.
            lines (list): List of candidate line identifiers.
        
        Returns:
            tuple: (best_lines, best_lines_index, rm_lines_idx) where best_lines is a list of selected lines,
                   best_lines_index are indices of selected lines, and rm_lines_idx are indices of removed lines.
        """
        if not self.use_lines:
            # Only consider lines with no NaN values in their errors
            nonan_lines = [line for line, x in error_type.items() if np.isnan(x).sum() == 0]
            best_lines_index, rm_lines_idx = self.outlier_killer(
                [np.nanmedian(error_type.loc[i]) for i in nonan_lines],
                thresh=self.lines_ok_thrsld
            )
            best_lines = [nonan_lines[x] for x in best_lines_index]
        else:
            best_lines_index = [i for i, line in enumerate(lines) if line in self.use_lines]
            rm_lines_idx = [i for i, line in enumerate(lines) if line not in self.use_lines]
            best_lines = [lines[x] for x in best_lines_index]
        return best_lines, best_lines_index, rm_lines_idx

    def remove_bad_epochs(self, df, metric='mean', error_type='cen', epochs_ok_thrsld=None):
        """
        Identify epochs with high errors that should be removed from subsequent RV computations.
        
        This method computes the per-epoch error (using either the mean or median) for a selected parameter,
        then uses the outlier_killer to mark epochs that exceed the threshold.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the SLfit results.
            metric (str): 'mean' or 'median' to compute the summary error statistic.
            error_type (str): Key for the error type; valid options include 'wid', 'rvs', 'cen', or 'amp'.
            epochs_ok_thrsld (float, optional): Threshold for acceptable error values (if not provided,
                                                uses self.epochs_ok_thrsld).
        
        Returns:
            list: Indices of the epochs to remove.
        """
        error_list = []
        err_type_dic = {'wid': 'wid1_er', 'rvs': 'sigma_rv', 'cen': 'cen1_er', 'amp': 'amp1_er'}
        if self.print_output:
            print('\n' + '-' * 34 + '\n   Removing epochs with large errors:')
            print('     using error_type:', error_type)
        if not epochs_ok_thrsld:
            epochs_ok_thrsld = self.epochs_ok_thrsld
        with open(self.path + 'rv_stats.txt', 'a') as out:
            out.write(' Removing epochs with large errors:\n')
            rm_OBs_idx = []
            epochs_unique = df['epoch'].unique()
            for i, epoch in enumerate(epochs_unique):
                df_epoch = df[df['epoch'] == epoch]
                error = df_epoch[err_type_dic[error_type]].mean() if metric == 'mean' else df_epoch[err_type_dic[error_type]].median()
                error_list.append(error)
                # If the error type is 'rvs' and there is any NaN, mark this epoch for removal
                if error_type == 'rvs' and df_epoch.isna().any().any():
                    rm_OBs_idx.append(i)
            if self.print_output:
                print('\n   Applying outlier_killer to remove epochs')
            rm_OBs_idx += self.outlier_killer(error_list, thresh=epochs_ok_thrsld)[1]
            if self.print_output:
                print(f'   Indices of epochs to be removed: {rm_OBs_idx}')
            out.write(f'   Indices of epochs to be removed: {rm_OBs_idx}\n\n')
        return rm_OBs_idx

    def compute_weighted_mean_rvs(self, rvs_dict, lines, rm_OBs_idx):
        """
        Compute the weighted mean radial velocities for each epoch based on measurements from the best lines.
        
        For each line, bad epochs (determined by rm_OBs_idx) are removed before computing per-line
        averages. Then, a weighted mean over the selected lines is computed for each epoch.
        
        Parameters:
            rvs_dict (dict): Dictionary of RV measurements for each line.
            lines (list): Lines to consider.
            rm_OBs_idx (list): Indices of epochs to remove.
        
        Returns:
            tuple: (wmean_rvs1, wmean_rvs2, line_avg1, line_avg2) where wmean_rvs1 and wmean_rvs2
                   are dictionaries mapping each epoch to its weighted mean RV (primary and secondary),
                   and line_avg1 and line_avg2 are per-line average RVs.
        """
        wmean_rvs1, wmean_rvs2 = {}, {}
        line_avg1, line_avg2 = {}, {}
        # Process primary component RVs
        for line, rvs in rvs_dict.items():
            if line not in lines:
                continue
            rv1 = np.delete(rvs['rv1'], rm_OBs_idx)
            rv1_er = np.delete(rvs['rv1_er'], rm_OBs_idx)
            line_avg1[line] = {'mean': np.mean(rv1), 'std': np.std(rv1)}
            rvs_dict[line]['rv1'] = rv1
            rvs_dict[line]['rv1_er'] = rv1_er
        # Compute weighted mean for primary RVs for each epoch
        for epoch in range(len(rv1)):
            weighted_mean1, weighted_error1 = GetRVs.weighted_mean(
                [rvs_dict[line]['rv1'][epoch] for line in lines],
                [rvs_dict[line]['rv1_er'][epoch] for line in lines]
            )
            wmean_rvs1[epoch] = {'value': weighted_mean1, 'error': weighted_error1}
        # Process secondary RVs if SB2 is enabled
        if self.SB2:
            for line, rvs in rvs_dict.items():
                if line not in lines:
                    continue
                rv2 = np.delete(rvs['rv2'], rm_OBs_idx)
                rv2_er = np.delete(rvs['rv2_er'], rm_OBs_idx)
                line_avg2[line] = {'mean': np.mean(rv2), 'std': np.std(rv2)}
                rvs_dict[line]['rv2'] = rv2
                rvs_dict[line]['rv2_er'] = rv2_er
            for epoch in range(len(rv2)):
                weighted_mean2, weighted_error2 = GetRVs.weighted_mean(
                    [rvs_dict[line]['rv2'][epoch] for line in lines],
                    [rvs_dict[line]['rv2_er'][epoch] for line in lines]
                )
                wmean_rvs2[epoch] = {'value': weighted_mean2, 'error': weighted_error2}
        return wmean_rvs1, wmean_rvs2, line_avg1, line_avg2

    def print_and_write_results(self, lines, line_avg1, total_mean_rv1, nepochs, line_avg2=None, total_mean_rv2=None):
        """
        Generate formatted text output summarizing per-line and overall RV results and write to file.
        
        Parameters:
            lines (list): Spectral line identifiers.
            line_avg1 (dict): Per-line average primary RVs.
            total_mean_rv1 (dict): Overall weighted mean and standard deviation for primary RVs.
            nepochs (int): Total number of epochs.
            line_avg2 (dict, optional): Per-line average secondary RVs (if SB2).
            total_mean_rv2 (dict, optional): Overall weighted mean and standard deviation for secondary RVs (if SB2).
        """
        rows = []
        rows.append(f'RV mean of the {nepochs} epochs for each line:')
        rows.append('---------------------------------------')
        for line in lines:
            mean = line_avg1[line]['mean']
            std = line_avg1[line]['std']
            rows.append(f'   - {line}: {mean:.3f} +/- {std:.3f}')
        if self.SB2:
            rows.append('   Component 2:')
            for line in lines:
                mean = line_avg2[line]['mean']
                std = line_avg2[line]['std']
                rows.append(f'   - {line}: {mean:.3f} +/- {std:.3f}')
        rows.append('')
        rows.append(f'Weighted mean RV of the {nepochs} epochs:')
        rows.append('------------------------------------------')
        mean1, std1 = total_mean_rv1['mean'], total_mean_rv1['std']
        rows.append(f'   Primary  : {mean1:.3f}, std dev = {std1:.3f}')
        if self.SB2:
            mean2, std2 = total_mean_rv2['mean'], total_mean_rv2['std']
            rows.append(f'   Secondary: {mean2:.3f}, std dev = {std2:.3f}')
        rows.append('')
        if self.print_output:
            print('\n'.join(rows))
        with open(self.path + 'rv_stats.txt', 'a') as out:
            out.write('\n'.join(rows))

    def compute(self):
        """
        Orchestrates the full radial velocity (RV) analysis workflow for single-lined (SB1)
        spectra. 
        Note: SB2-related processing in GetRVs is now deprecated, as the new SB2 
        probabilistic method computes RVs directly.

        This method performs the following steps:
          - Prints header information.
          - Defines a dictionary of rest wavelengths (in air).
          - Computes per-line RVs (with error propagation) from the SLfit results.
          - Generates an errorbar plot of RVs per spectral line.
          - Groups percentage errors per line and selects the best lines using a MAD-based outlier test.
          - Writes statistical error analyses to a file.
          - Reads and processes JD information.
          - Removes bad/noisy epochs based on error thresholds.
          - Computes weighted mean RVs per epoch from the selected best lines.
          - Prints and writes summary RV results and saves final RV values to file.
          - Generates a final diagnostic plot of RVs versus Julian Date.
        
        Returns:
            pd.DataFrame: A DataFrame containing the final RV results.
        """
        # Print header information
        if self.print_output:
            print('\n' + '*' * 79)
            print('******************                RV Analysis                ******************')
            print('*' * 79 + '\n')

        # Define rest wavelengths dictionary (wavelength and its error)
        lambda_rest_dict = {
            4009: [4009.2565, 0.00002], 4026: [4026.1914, 0.0010],
            4102: [4101.734, 0.006],     4121: [4120.8154, 0.0012],
            4128: [4128.07, 0.10],       4131: [4130.89, 0.10],
            4144: [4143.761, 0.010],     4267: [4267.258, 0.007],
            4340: [4340.472, 0.006],     4388: [4387.9296, 0.0006],
            4471: [4471.4802, 0.0015],   4481: [4481.130, 0.010],
            4542: [4541.591, 0.010],     4553: [4552.62, 0.10],
            4713: [4713.1457, 0.0006],
            4861: [4861.35, 0.05],       4922: [4921.9313, 0.0005],
            5412: [5411.52, 0.10],       5876: [5875.621, 0.010],
            5890: [5889.951, 0.00003],   6562: [6562.79, 0.030],
            6678: [6678.151, 0.010],     7774: [7774.17, 0.10]
        }

        if self.print_output:
            print('*** SB2 set to:', self.SB2, '***\n')
            print('\n*** Computing Radial Velocities ***')
            print('-----------------------------------')

        # Compute per-line RVs based on the SLfit results and rest wavelengths.
        rvs_dict = self.compute_rvs(self.lines, lambda_rest_dict)

        # Plot RVs for each spectral line
        fig, ax = plt.subplots()
        markers = ['o', 'v', '^', '<', '>', 's', 'X', '*', 'D', 'H']
        for line, marker in zip(self.lines, markers):
            rv1 = rvs_dict[line]['rv1']
            rv1_er = rvs_dict[line]['rv1_er']
            ax.errorbar(range(len(rv1)), rv1, yerr=rv1_er, fmt=marker,
                        color='dodgerblue', label=f'Comp. 1 {line}', alpha=0.5)
            if self.SB2:
                rv2 = rvs_dict[line]['rv2']
                rv2_er = rvs_dict[line]['rv2_er']
                ax.errorbar(range(len(rv2)), rv2, yerr=rv2_er, fmt=marker,
                            color='darkorange', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Radial Velocity (km/s)')
        ax.legend(loc='lower left', fontsize=8)
        plt.savefig(self.path + 'rvs_per_line.png', bbox_inches='tight', dpi=300)
        plt.close()

        #################################################################
        #                Selecting lines with the best fits
        #################################################################
        primary_error_types = ['wid1_percer', 'cen1_percer', 'amp1_percer']
        secondary_error_types = ['wid2_percer', 'cen2_percer', 'amp2_percer'] if self.SB2 else []
        all_error_types = primary_error_types + secondary_error_types
        grouped_errors = {error_type: self.df_SLfit.groupby('line')[error_type].apply(list)
                          for error_type in all_error_types}

        if self.print_output:
            print('\n*** Choosing the best lines ***\n-------------------------------')
            print_lines = [str(line) for line in grouped_errors['cen1_percer'].keys()]
            print(' Primary:' + ' ' * 14, '   '.join(print_lines))
            for error_type in primary_error_types:
                GetRVs.print_error_stats(grouped_errors[error_type], error_type)
            if self.SB2:
                print(' Secondary:')
                for error_type in secondary_error_types:
                    GetRVs.print_error_stats(grouped_errors[error_type], error_type)

        # Write statistical error analysis to file.
        with open(self.path + 'rv_stats.txt', 'w') as out:
            out.write(' ********************************\n')
            out.write('  Statistical analysis of errors \n')
            out.write(' ********************************\n\n')
            for stat_type in ['Mean', 'Median']:
                self.write_error_stats(out, grouped_errors, stat_type)

        if self.print_output:
            print('\n   --------------------------------------------')
            if not self.use_lines:
                print('   Applying outlier_killer to remove bad lines:')
            else:
                print('   Selecting lines determined by user:')

        # Select the best lines based on the specified error metric.
        best_lines, best_lines_index, rm_lines_idx = self.select_lines(grouped_errors[self.error_type], self.lines)
        if self.print_output:
            print('\n   These are the best lines:', best_lines)
        nlines = len(best_lines)
        with open(self.path + 'rv_stats.txt', 'a') as out:
            out.write('\n')
            out.write(f' Lines with the best fitted profile according to the median {self.error_type} criterion:\n')
            out.write(' --------------------------------------------------------------------------\n')
            for i in range(nlines):
                if i < nlines - 1:
                    out.write(f'   {best_lines_index[i]}: {best_lines[i]}, ')
                else:
                    out.write(f'   {best_lines_index[i]}: {best_lines[i]}\n')
            out.write('\n')


        #################################################################
        #       Removing lines with inverted components (deprecated)
        #################################################################
        if self.SB2 and len(best_lines) > 2:
            if self.print_output:
                print('\n   --------------------------------------------')
                print('   Removing lines with inverted components:')
            failed_lines_indices = []
            for epoch in range(self.nepochs):
                rv1s = [rvs_dict[line]['rv1'][epoch] for line in best_lines]
                _, failed_idx = self.outlier_killer(rv1s, thresh=self.lines_ok_thrsld, print_output=False)
                failed_lines_indices.extend(failed_idx)
            failed_lines_counts = Counter(failed_lines_indices)
            print(f'   {failed_lines_counts}')
            threshold = 0.60 * self.nepochs  # Remove lines that fail >60% of epochs
            lines_to_remove = [best_lines[i] for i, count in failed_lines_counts.items() if count > threshold]
            print(f"   Lines to remove: {lines_to_remove}")
            best_lines = [line for line in best_lines if line not in lines_to_remove]
            print(f"   Remaining lines: {best_lines}")

        #################################################################
        #                  Removing bad/noisy epochs
        #################################################################
        df_rv = pd.read_csv(self.JDfile, names=['epoch', 'MJD'], sep='\s+').replace({'.fits': ''}, regex=True)
        df_rv2 = pd.read_csv(self.JDfile, names=['epoch', 'MJD'], sep='\s+').replace({'.fits': ''}, regex=True)
        rv1_values, rv2_values = [], []
        for line in best_lines:
            df_rv[f'rv_{line}'] = rvs_dict[line]['rv1']
            df_rv[f'rv_{line}_er'] = rvs_dict[line]['rv1_er']
            rv1_values.append(rvs_dict[line]['rv1'])
            if self.SB2:
                df_rv2[f'rv_{line}'] = rvs_dict[line]['rv2']
                df_rv2[f'rv_{line}_er'] = rvs_dict[line]['rv2_er']
                rv2_values.append(rvs_dict[line]['rv2'])
        df_rv['comp'] = 1
        df_rv['sigma_rv'] = np.std(np.stack(rv1_values), axis=0)
        error_par = 'wid'
        if self.SB2:
            df_rv2['comp'] = 2
            df_rv2['sigma_rv'] = np.std(np.stack(rv2_values), axis=0)
            error_par = 'cen'

        # Remove bad/noisy epochs if specified.
        if self.rm_epochs:
            if self.print_output:
                print('\n*** Choosing best epochs for analysis ***\n---------------------------------------------')
            if len(best_lines) > 1:
                rm_OBs_idx = self.remove_bad_epochs(self.df_SLfit, metric='median', error_type=error_par)
                rm_OBs_idx += self.remove_bad_epochs(df_rv, metric='median', error_type='rvs', epochs_ok_thrsld=10)
                final_nepochs = self.nepochs - len(rm_OBs_idx)
            else:
                rm_OBs_idx = []
                final_nepochs = self.nepochs
        else:
            rm_OBs_idx = []
            final_nepochs = self.nepochs

        #################################################################
        #                Computing weighted mean RVs
        #################################################################
        if self.print_output:
            print('\n*** Calculating the RV weighted mean for each epoch  ***')
            print('--------------------------------------------------------')
        wmean_rv1, wmean_rv2, line_avg1, line_avg2 = self.compute_weighted_mean_rvs(rvs_dict, best_lines, rm_OBs_idx)
        total_rv1 = {'mean': np.mean([wmean_rv1[i]['value'] for i in wmean_rv1.keys()]),
                     'std': np.std([wmean_rv1[i]['value'] for i in wmean_rv1.keys()])}
        if self.SB2:
            total_rv2 = {'mean': np.mean([wmean_rv2[i]['value'] for i in wmean_rv2.keys()]),
                         'std': np.std([wmean_rv2[i]['value'] for i in wmean_rv2.keys()])}
        else:
            total_rv2 = None

        self.print_and_write_results(best_lines, line_avg1, total_rv1, final_nepochs,
                                     line_avg2=line_avg2, total_mean_rv2=total_rv2)

        #################################################################
        #                Writing RVs to file RVs.txt
        #################################################################
        if self.rm_epochs and rm_OBs_idx:
            df_rv.drop(rm_OBs_idx, inplace=True)
            df_rv.reset_index(drop=True, inplace=True)
        df_rv['mean_rv'] = [wmean_rv1[i]['value'] for i in range(len(wmean_rv1))]
        df_rv['mean_rv_er'] = [wmean_rv1[i]['error'] for i in range(len(wmean_rv1))]
        if self.SB2:
            if self.rm_epochs and rm_OBs_idx:
                df_rv2.drop(rm_OBs_idx, inplace=True)
                df_rv2.reset_index(drop=True, inplace=True)
            df_rv2['mean_rv'] = [wmean_rv2[i]['value'] for i in range(len(wmean_rv2))]
            df_rv2['mean_rv_er'] = [wmean_rv2[i]['error'] for i in range(len(wmean_rv2))]
            df_rv = pd.concat([df_rv, df_rv2], ignore_index=True)

        with open(self.path + 'RVs1.txt', 'w') as fo:
            fo.write(df_rv.to_string(formatters={'MJD': '{:.8f}'.format}, index=False))

        #################################################################
        #        Plotting RVs per spectral line and weighted mean
        #################################################################
        data = pd.read_csv(self.path + 'RVs1.txt', delim_whitespace=True)
        primary_data = data[data['comp'] == 1]
        secondary_data = data[data['comp'] == 2]
        fig, ax = plt.subplots(figsize=(8, 6))
        rv_lines = [f'rv_{line}' for line in best_lines]
        rv_er = [f'rv_{line}_er' for line in best_lines]
        for i, (rv_line, marker) in enumerate(zip(rv_lines, markers)):
            ax.errorbar(primary_data['MJD'], primary_data[rv_line], yerr=primary_data[rv_er[i]],
                        fmt=marker, color='dodgerblue', fillstyle='none',
                        label=f'Comp. 1 {best_lines[i]}', alpha=0.5)
        for i, (rv_line, marker) in enumerate(zip(rv_lines, markers)):
            ax.errorbar(secondary_data['MJD'], secondary_data[rv_line], yerr=secondary_data[rv_er[i]],
                        fmt=marker, color='darkorange', fillstyle='none', alpha=0.5)
        ax.errorbar(primary_data['MJD'], primary_data['mean_rv'], fmt='s', color='dodgerblue',
                    alpha=0.5, label='Primary weighted mean')
        ax.errorbar(secondary_data['MJD'], secondary_data['mean_rv'], fmt='s', color='darkorange',
                    alpha=0.5, label='Secondary weighted mean')
        ax.set_xlabel('Julian Date')
        ax.set_ylabel('Mean Radial Velocity')
        ax.legend(fontsize=8)
        plt.savefig(self.path + 'RVs1.png', bbox_inches='tight', dpi=300)
        plt.close()

        return df_rv

def get_peaks(power, frequency, fal_50pc, fal_1pc, fal_01pc, minP=1.1):
    """
    Identify significant peaks in a periodogram based on false alarm levels.
    
    The function determines a minimum power threshold (peaks_min_h) based on the
    maximum power relative to the given false alarm levels and then finds peaks 
    within specified frequency regions.
    
    Parameters:
        power (array-like): The Lomb–Scargle power spectrum.
        frequency (array-like): The corresponding frequency grid (1/days).
        fal_50pc (float): The false alarm level corresponding to a 50% probability.
        fal_1pc (float): The false alarm level corresponding to a 1% probability.
        fal_01pc (float): The false alarm level corresponding to a 0.1% probability.
        minP (float, optional): The minimum period to consider (default is 1.1 days).
    
    Returns:
        tuple:
            - freq_peaks (np.array): Frequencies at which significant peaks occur.
            - peri_peaks (np.array): Corresponding periods (in days) computed as 1/frequency.
            - peaks (np.array): Indices of the detected peaks in the `power` array.
    """
    if power.max() > fal_01pc:
        peaks_min_h = float(f'{fal_1pc:.3f}')
    elif power.max() <= fal_01pc and power.max() > fal_1pc:
        peaks_min_h = float(f'{0.6 * fal_1pc:.3f}')
    else:
        peaks_min_h = float(f'{fal_50pc:.3f}')
    
    # Define index boundaries corresponding to periods
    freq_index1 = np.argmin(np.abs(frequency - 1/5))  # approx period = 5 days
    freq_index2 = np.argmin(np.abs(frequency - 1/minP))
    
    # Plot the periodogram (could be commented out in production)
    # plt.plot(frequency, power)
    
    # Find peaks in two frequency ranges and merge the results
    peaks1, _ = find_peaks(power[:freq_index1], height=peaks_min_h, distance=1000)
    peaks2, _ = find_peaks(power[:freq_index2], height=peaks_min_h, distance=5000)
    peaks = np.unique(np.concatenate((peaks1, peaks2)))
    
    freq_peaks = frequency[peaks]
    peri_peaks = 1 / frequency[peaks]
    
    return freq_peaks, peri_peaks, peaks

def run_LS(hjd, rv, rv_err=None, probabilities=[0.5, 0.01, 0.001], method='bootstrap',
           P_ini=1.2, P_end=500, samples_per_peak=5000):
    """
    Run a Lomb–Scargle periodogram on provided time series data.
    
    This function computes the Lomb–Scargle power spectrum and obtains the false alarm 
    levels for the specified probability thresholds. It uses the 'fast' method for power 
    computation.
    
    Parameters:
        hjd (array-like): Times of observation (HJD or MJD).
        rv (array-like): Radial velocity measurements.
        rv_err (array-like, optional): Measurement uncertainties. If None, errors are ignored.
        probabilities (list, optional): List of false alarm probability thresholds (default: [0.5, 0.01, 0.001]).
        method (str, optional): The method used for computing false alarm levels (default is 'bootstrap').
        P_ini (float, optional): The minimum period (in days) to consider (default is 1.2).
        P_end (float, optional): The maximum period (in days) to consider (default is 500).
        samples_per_peak (int, optional): Oversampling factor per peak (default is 5000).
    
    Returns:
        tuple:
            - frequency (np.array): The frequency grid over which power was computed (1/days).
            - power (np.array): The computed Lomb–Scargle power spectrum.
            - fap (float): The false alarm probability of the highest peak.
            - fal (list): The false alarm levels corresponding to the specified probabilities.
    """
    if rv_err is None:
        ls = LombScargle(hjd, rv, normalization='model')
    else:
        ls = LombScargle(hjd, rv, rv_err, normalization='model')
    fal = ls.false_alarm_level(probabilities, method=method)
    frequency, power = ls.autopower(method='fast', minimum_frequency=1/P_end,
                                    maximum_frequency=1/P_ini, samples_per_peak=samples_per_peak)
    fap = ls.false_alarm_probability(power.max(), method=method)
    return frequency, power, fap, fal

def lomb_scargle(df, path, SB2=False, print_output=True, plots=True, best_lines=False, Pfold=True, fold_rv_curve=True):
    """
    Perform Lomb–Scargle period analysis on a DataFrame of radial velocities.
    
    This function reads the input DataFrame (from getrvs()), determines which column to use for times
    (HJD/MJD), and computes the Lomb–Scargle periodogram for the primary (and, if SB2==True, secondary)
    component. It saves outputs (text and plots) to a dedicated 'LS' directory within the provided path.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing RV measurements and associated metadata.
        path (str): Directory path for saving LS outputs.
        SB2 (bool, optional): If True, perform secondary analysis for SB2 stars (default is False).
        print_output (bool, optional): If True, print intermediate outputs (default is True).
        plots (bool, optional): If True, generate diagnostic plots (default is True).
        best_lines (bool, optional): (Reserved) Whether to use best-lines selection (default is False).
        Pfold (bool, optional): If True, perform RV curve folding (default is True).
        fold_rv_curve (bool, optional): If True, compute phase-folded RV curves (default is True).
    
    Returns:
        dict: A dictionary (`ls_results`) containing periodogram outputs, including frequency grid,
              power spectrum, false alarm probabilities and levels, detected peaks, and best period estimates.
    """
    # Ensure LS output directory exists
    ls_path = os.path.join(path, 'LS')
    if not os.path.exists(ls_path):
        os.makedirs(ls_path)
    
    # Select times and RVs from DataFrame; try both 'JD' and 'MJD'
    try:
        hjd1, rv1 = df['JD'][df['comp'] == 1], df['mean_rv'][df['comp'] == 1]
    except:
        hjd1, rv1 = df['MJD'][df['comp'] == 1], df['mean_rv'][df['comp'] == 1]
    
    rv1_err = df['mean_rv_er'][df['comp'] == 1] if 'mean_rv_er' in df.columns else None
    starname = df['epoch'][0].split('_')[0] + '_' + df['epoch'][0].split('_')[1]
    nepochs = len(hjd1)
    
    # Write header information
    with open(os.path.join(ls_path, 'ls_output.txt'), 'w') as lsout:
        lsout.write(' ***************************************\n')
        lsout.write('   LS output for star ' + starname + '\n')
        lsout.write(' ***************************************\n\n')
    
    #################################################################
    #                Running the Lomb-Scargle periodogram
    #################################################################
    frequency1, power1, fap1, fal1 = run_LS(hjd1, rv1, rv1_err)
    fal1_50pc, fal1_1pc, fal1_01pc = fal1[0].value, fal1[1].value, fal1[2].value
    freq1_at_max_power = frequency1[np.argmax(power1)]
    period1_at_max_power = 1 / freq1_at_max_power
    
    ls_results = {'freq': {1: frequency1},
                  'power': {1: power1},
                  'fap': {1: fap1},
                  'fal_50%': {1: fal1_50pc},
                  'fal_1%': {1: fal1_1pc},
                  'fal_01%': {1: fal1_01pc},
                  'max_freq': {1: freq1_at_max_power},
                  'max_period': {1: period1_at_max_power},
                  'peaks': {},
                  'best_period': {},
                  'best_P_pow': {},
                  'ind': {},
                  'freq_peaks': {},
                  'peri_peaks': {},
                  'pow_over_fal01': {},
                  'pow_over_fal1': {}}
    
    if SB2:
        # Secondary component analysis (if applicable)
        hjd2 = df['MJD'][df['comp'] == 2]
        rv2 = df['mean_rv'][df['comp'] == 2]
        rv2_err = df['mean_rv_er'][df['comp'] == 2]
        frequency2, power2, fap2, fal2 = run_LS(hjd2, rv2, rv2_err)
        fal2_50pc, fal2_1pc, fal2_01pc = fal2[0].value, fal2[1].value, fal2[2].value
        freq2_at_max_power = frequency2[np.argmax(power2)]
        period2_at_max_power = 1 / freq2_at_max_power
        ls_results['freq'][2] = frequency2
        ls_results['power'][2] = power2
        ls_results['fap'][2] = fap2
        ls_results['fal_50%'][2] = fal2_50pc
        ls_results['fal_1%'][2] = fal2_1pc
        ls_results['fal_01%'][2] = fal2_01pc
        ls_results['max_freq'][2] = freq2_at_max_power
        ls_results['max_period'][2] = period2_at_max_power
    
    # Write some summary info to ls_output.txt
    fapper1 = fap1 * 100
    if print_output:
        print('   False alarm levels:', [f'{x:.3f}' for x in fal1])
        print('   FAP of highest peak:', f'{fap1:.5f}', ' (x100: ', f'{fapper1:.5f}', ')')
    with open(os.path.join(ls_path, 'ls_output.txt'), 'a') as lsout:
        lsout.write(' False alarm levels: ' + ' '.join([f'{x:.3f}' for x in fal1]) + '\n')
        lsout.write(' FAP of highest peak: ' + f'{fap1:.5f}' + '\n')
        lsout.write(' FAP of highest peak x100: ' + f'{fapper1:.5f}' + '\n')
    
    #################################################################
    #                        Finding peaks
    #################################################################
    freq_peaks1, peri_peaks1, peaks1 = get_peaks(power1, frequency1, fal1_50pc, fal1_1pc, fal1_01pc)
    ls_results['freq_peaks'][1] = freq_peaks1
    ls_results['peri_peaks'][1] = peri_peaks1
    ls_results['peaks'][1] = peaks1
    
    if SB2:
        freq_peaks2, peri_peaks2, peaks2 = get_peaks(power2, frequency2, fal2_50pc, fal2_1pc, fal2_01pc)
        ls_results['freq_peaks'][2] = freq_peaks2
        ls_results['peri_peaks'][2] = peri_peaks2
        ls_results['peaks'][2] = peaks2
        
        # Test period from the difference between RV components
        rv_abs = np.abs(rv1 - rv2.reset_index(drop=True))
        if rv1_err is not None:
            rv_abs_er = np.sqrt(rv1_err**2 + rv2_err.reset_index(drop=True)**2)
        else:
            rv_abs_er = None
        frequency3, power3, fap3, fal3 = run_LS(hjd1, rv_abs, rv_abs_er)
        fal3_50pc, fal3_1pc, fal3_01pc = fal3[0], fal3[1], fal3[2]
        freq3_at_max_power = frequency3[np.argmax(power3)]
        period3_at_max_power = 1 / freq3_at_max_power
        freq_peaks3, peri_peaks3, peaks3 = get_peaks(power3, frequency3, fal3_50pc, fal3_1pc, fal3_01pc)
        ls_results['freq_peaks'][3] = freq_peaks3
        ls_results['peri_peaks'][3] = peri_peaks3
        ls_results['peaks'][3] = peaks3
        ls_results['max_freq'][3] = freq3_at_max_power
        ls_results['max_period'][3] = period3_at_max_power

    #################################################################
    #           Printing the results and writing to file
    #################################################################
    if period1_at_max_power < 1.1:
        if peri_peaks1.size > 0:
            best_period = peri_peaks1[np.argmax(power1[peaks1])]
            best_pow = power1[peaks1].max()
        elif SB2 and peri_peaks2.size > 0:
            best_period = peri_peaks2[np.argmax(power2[peaks2])]
            best_pow = power2[peaks2].max()
        else:
            best_period = period1_at_max_power
            best_pow = power1.max()
    else:
        best_period = period1_at_max_power
        best_pow = power1.max()
    ls_results['best_period'][1] = best_period
    ls_results['best_P_pow'][1] = best_pow

    if print_output:
        print("   Best frequency                  :  {0:.3f}".format(freq1_at_max_power))
        print('   ***********************************************')
        print("   Best Period                     :  {0:.8f} days".format(best_period))
        print('   ***********************************************')
        if SB2:
            print("   Best Period from secondary      :  {0:.8f} days".format(period2_at_max_power))
            print("   Period from |RV1-RV2|           :  {0:.8f} days".format(period3_at_max_power), 'correct period = P1 or ', period1_at_max_power/2)
        print('   Other periods:')
        print('     peaks                         : ', [f'{x:.3f}' for x in power1[peaks1]])
        print('     frequencies                   : ', [f'{x:.5f}' for x in freq_peaks1])
        print('     periods                       : ', [f'{x:.3f}' for x in peri_peaks1])
    
    with open(os.path.join(ls_path, 'ls_output.txt'), 'a') as lsout:
        lsout.write("\n Best frequency                  :  {0:.3f}\n".format(freq1_at_max_power))
        lsout.write(' ****************************************************\n')
        lsout.write(" Best Period                     :  {0:.8f} days\n".format(best_period))
        lsout.write(' ****************************************************\n')
        if SB2:
            lsout.write(" Best Period from secondary      :  {0:.8f} days\n".format(period2_at_max_power))
            lsout.write(" Period from |RV1-RV2|           :  {0:.8f} days".format(period3_at_max_power) +
                        ' correct period = P1 or ' + str(period1_at_max_power/2) + '\n')
        lsout.write(' Other periods:\n')
        lsout.write('   peaks                         : ')
        for peak in power1[peaks1]:
            lsout.write('     ' + f'{peak:7.3f}')
        lsout.write('\n   frequencies                   : ')
        for freq in freq_peaks1:
            lsout.write('     ' + f'{freq:7.3f}')
        lsout.write('\n   periods                       : ')
        for per in peri_peaks1:
            lsout.write('     ' + f'{per:7.3f}')
        if SB2:
            lsout.write('\n from secondary:\n')
            lsout.write('   peaks                         : ')
            for peak in power2[peaks2]:
                lsout.write('     ' + f'{peak:7.3f}')
            lsout.write('\n   frequencies                   : ')
            for freq in freq_peaks2:
                lsout.write('     ' + f'{freq:7.3f}')
            lsout.write('\n   periods                       : ')
            for per in peri_peaks2:
                lsout.write('     ' + f'{per:7.3f}')
        lsout.write('\n')
    
    #################################################################
    #           Setting quality index for the periodogram
    #################################################################
    indi = []
    maxpower = power1.max()
    for LS_pow, peri in zip(power1[peaks1], peri_peaks1):
        maxpower_maxfal = LS_pow / fal1[2]
        maxpower_maxfal2 = LS_pow / fal1[1]
        if print_output:
            print('   fal1/P                          : ', f'{maxpower_maxfal:.2f}')
            print('   fal2/P                          : ', f'{maxpower_maxfal2:.2f}')
        with open(os.path.join(ls_path, 'ls_output.txt'), 'a') as lsout:
            lsout.write(' fal1/P                          :  ' + f'{maxpower_maxfal:.2f}' + '\n')
            lsout.write(' fal2/P                          :  ' + f'{maxpower_maxfal2:.2f}' + '\n')
        conditions = [
            (maxpower > fal1_01pc),                   # FAL 0.1%
            (fal1_01pc >= maxpower > fal1_1pc),         # FAL 1%
            (fal1_1pc >= maxpower > fal1_50pc),         # FAL 50%
            (maxpower <= fal1_50pc)                     # Below 50% FAL
        ]
        indices = [0, 1, 2, 3]
        indi = [index for condition, index in zip(conditions, indices) if condition]
    ind = indi[0] if indi else 4
    maxpower_maxfal = maxpower / fal1[2].value
    maxpower_maxfal2 = maxpower / fal1[1].value
    if print_output:
        print('\n   Classification index            : ', ind)
        print('   maxpower                        : ', f'{maxpower:.2f}')
        print('   fal1                            : ', f'{fal1[2]:.2f}')
        print('   maxpower_maxfal                 : ', f'{maxpower_maxfal:.2f}')
    with open(os.path.join(ls_path, 'ls_output.txt'), 'a') as lsout:
        lsout.write(' Classification index            :  ' + str(ind) + '\n')
        lsout.write(' maxpower                        :  ' + f'{maxpower:.2f}' + '\n')
        lsout.write(' fal1                            :  ' + f'{fal1[2]:.2f}' + '\n')
        lsout.write(' maxpower_maxfal                 :  ' + f'{maxpower_maxfal:.2f}' + '\n')
    
    ls_results['ind'][1] = ind
    ls_results['max_power'][1] = maxpower
    ls_results['pow_over_fal01'][1] = maxpower_maxfal
    ls_results['pow_over_fal1'][1] = maxpower_maxfal2
    
    #################################################################
    #                  Plotting the periodogram
    #################################################################
    if plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(hjd1, rv1, yerr=rv1_err, fmt='o', color='dodgerblue')
        ax.set(xlabel='MJD', ylabel='RV [km/s]')
        plt.savefig(f'{path}LS/RVs_MJD_{starname}.png', dpi=300, bbox_inches='tight')
        plt.close()

        bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        if SB2==True:
            f_list, pow_list, comp_list, Per_list, fal_list, peak_list = [frequency1, frequency2, frequency3], \
                        [power1, power2, power3], ['primary', 'secondary', 'subtracted'], \
                        [best_period, period2_at_max_power, period3_at_max_power], [fal1, fal2, fal3], [peaks1, peaks2, peaks3]
        else:
            f_list, pow_list, comp_list, Per_list, fal_list, peak_list = [frequency1], \
                [power1], ['primary'], [best_period], [fal1], [peaks1]
        for frequency, power, comp, best_per, fal, peaks in \
                zip(f_list, pow_list, comp_list, Per_list, fal_list, peak_list):
            if not any(np.isnan(power)):
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
                ax.plot(1/frequency, power, 'k-', alpha=0.5)
                ax.plot(1/frequency[peaks], power[peaks], "ob", label='prominence=1')
                ax.yaxis.set_label_coords(-0.09, 0.5)
                ax.set(xlim=(0.3, 1300), xlabel='Period [d]', ylabel='Lomb-Scargle Power')
                if np.isfinite(power1.max()):
                    ax.set(ylim=(-0.03*power.max(), power.max()+0.1*power.max()))
                plt.xscale('log')
                tickLabels = map(str, bins)
                ax.set_xticks(bins)
                ax.set_xticklabels(tickLabels)
                ax.plot( (0.5, 800), (fal[0], fal[0]), '--r', lw=1.2)
                ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
                ax.plot( (0.5, 800), (fal[2], fal[2]), '--g', lw=1.2)
                ax.text( 100, fal[2]+0.01, '0.1\% fap', fontsize=16)
                ax.text( 100, fal[1]+0.01, '1\% fap', fontsize=16)
                ax.text( 100, fal[0]+0.01, '50\% fap', fontsize=16)
                if power.max()+0.1*power.max() >= 10:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
                else:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
                plt.title(starname+' Periodogram $-$ Best Period: {0:.4f}'.format(best_per)+' d')
                if best_lines:
                    labels = ['lines ='+str(best_lines), 'n epochs ='+str(nepochs)]
                    leg = plt.legend(labels, loc='best', markerscale=0, handletextpad=0, handlelength=0)
                    for item in leg.legendHandles:
                        item.set_visible(False)
                plt.savefig(f'{path}LS/LS_{starname}_period_{comp}_{str(len(rv1))}_epochs.png', dpi=300)
                plt.close()

                # Plot for paper
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
                ax.plot(1/frequency, power, 'k-', alpha=0.5)
                ax.yaxis.set_label_coords(-0.09, 0.5)
                ax.set(xlim=(0.3, 1300), xlabel='Period [d]', ylabel='Lomb-Scargle Power')
                if np.isfinite(power1.max()):
                    ax.set(ylim=(-0.03*power.max(), power.max()+0.1*power.max()))
                plt.xscale('log')
                if power1[peaks1].size > 0 and (power1[peaks1].max() < maxpower):
                    ax.set(ylim=(0-0.01*power1[peaks1].max(), power1[peaks1].max()+0.2*power1[peaks1].max()))
                tickLabels = map(str, bins)
                ax.set_xticks(bins)
                ax.set_xticklabels(tickLabels)
                ax.plot( (0.5, 800), (fal[0], fal[0]), '-r', lw=1.2)
                ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
                ax.plot( (0.5, 800), (fal[2], fal[2]), ':g', lw=1.2)
                if power.max()+0.1*power.max() >= 10:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
                else:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
                plt.title(starname+' '+comp)
                plt.savefig(f'{path}LS/{starname}_paper_LS_{comp}.pdf')
                plt.close()

        # vs grid points
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10)
        ax.plot(power1)
        ax.plot(peaks1, power1[peaks1], "ob")
        ax.set(xlabel='Number of points', ylabel='Lomb-Scargle Power')
        if np.isfinite(power1.max()):
            ax.set_ylim(0, power1.max()+0.1*power1.max())
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(which='both', width=0.6, labelsize=14)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.savefig(f'{path}LS/LS_{starname}_points.png', dpi=300)
        plt.close()

        # vs frequency
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10)
        ax.plot(frequency1, power1)
        ax.plot(frequency1[peaks1], power1[peaks1], "ob")
        ax.vlines(np.abs(freq1_at_max_power-1), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(freq1_at_max_power-2), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(freq1_at_max_power+1), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(freq1_at_max_power+2), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.text( freq1_at_max_power+0.03, power1.max(), r'$\mathrm{f}_0$', fontsize=14)
        ax.text( np.abs(freq1_at_max_power-1)-0.03, power1.max(), r'$\left| \mathrm{f}_0-1\right|$', fontsize=14, horizontalalignment='right')
        ax.text( np.abs(freq1_at_max_power-2)-0.03, power1.max(), r'$\left| \mathrm{f}_0-2\right|$', fontsize=14, horizontalalignment='right')
        ax.text( np.abs(freq1_at_max_power+1)+0.03, power1.max(), r'$\left| \mathrm{f}_0+1\right|$', fontsize=14, horizontalalignment='left')
        ax.text( np.abs(freq1_at_max_power+2)+0.03, power1.max(), r'$\left| \mathrm{f}_0+2\right|$', fontsize=14, horizontalalignment='left')
        ax.set(xlabel='Frequency (1/d)', ylabel='Lomb-Scargle Power')
        if np.isfinite(power1.max()):
            ax.set(ylim=(0, power1.max()+0.1*power1.max()))
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(which='both', width=0.6, labelsize=14)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.savefig(f'{path}LS/LS_{starname}_frequency.pdf')
        plt.close()

        # Stack the arrays in columns
        pow_spec = np.column_stack((frequency1, power1))
        # Save the data to a text file
        np.savetxt(path+'LS/power_spectrum.txt', pow_spec)
    
    #################################################################
    #               Compute phases of the observations
    #################################################################
    if fold_rv_curve:
        if Pfold:
            print('peri_peaks1 size:', peri_peaks1.size)
            if peri_peaks1.size == 0:
                if SB2:
                    if peri_peaks2.size > 0:
                        periods = peri_peaks2
                    else:
                        print('  Warning: No periods found for primary nor secondary star')
                        periods = [period1_at_max_power]
                else:
                    print('  Warning: No periods found for primary star')
                    periods = [period1_at_max_power]
            elif (peri_peaks1.size < 6) & (peri_peaks1.size >0):
                periods = peri_peaks1
            else:
                periods = [best_period]
            print('periods:', periods)
            
            for period in periods:
                print(f'\nComputing phases for period={period}, from {len(periods)} periods')
                fine_phase = np.linspace(-0.5, 1.5, 1000)
                if SB2:
                    print('\nComputing phases for the secondary star\n')
                    results2, phase, vel1, vel1_err, vel2, vel2_err = phase_rv_curve(hjd1, rv1, rv1_err, rv2, rv2_err, period=period)

                    fine_preds1 = []
                    fine_preds2 = []
                    
                    for i in range(results2['K1'].shape[0]):
                        # Extract parameters
                        K1 = results2['K1'][i]
                        K2 = results2['K2'][i]
                        phi0 = results2['phi0'][i]
                        gamma = results2['gamma'][i]

                        # Compute predictions
                        pred1 = K1 * np.sin(2 * np.pi * fine_phase + phi0) + gamma
                        pred2 = -K2 * np.sin(2 * np.pi * fine_phase + phi0) + gamma

                        fine_preds1.append(pred1)
                        fine_preds2.append(pred2)

                    fine_preds1 = np.array(fine_preds1)
                    fine_preds2 = np.array(fine_preds2)
                else:
                    results1, phase, vel1, vel1_err  = phase_rv_curve(hjd1, rv1, rv1_err, period=period)
                    fine_preds1, fine_preds2 = [], []
                    for i in range(results1['amplitude'].shape[0]):
                        fine_pred1 = results1['amplitude'][i] * np.sin(2 * np.pi * fine_phase + results1['phase_shift'][i]) \
                            + results1['height'][i]
                        fine_preds1.append(fine_pred1)
                    fine_preds1 = np.array(fine_preds1)

                print('\n*** Plotting phased RV curve ***\n-------------------------------')
                n_lines = 200
                fig, ax = plt.subplots(figsize=(8, 6))
                for pred in fine_preds1[-n_lines:]:
                    ax.plot(fine_phase, pred, rasterized=True, color='C1', alpha=0.05)
                ax.errorbar(phase, vel1, yerr=vel1_err, color='k', fmt='o', label='Data')
                if SB2:
                    for pred in fine_preds2[-n_lines:]:
                        ax.plot(fine_phase, pred, rasterized=True, color='C0', alpha=0.05)
                    ax.errorbar(phase, vel2, yerr=vel2_err, color='C2', fmt='^', label='Data')
                ax.set_xlabel('Phase')
                ax.set_ylabel('Radial Velocity [km\,s$^{-1}$]')  
                if SB2:
                    plt.savefig(f'{path}LS/{starname}_sinu_fit_SB2_P={period:.2f}.png', bbox_inches='tight', dpi=300)
                else:
                    plt.savefig(f'{path}LS/{starname}_sinu_fit_SB1_P={period:.2f}.png', bbox_inches='tight', dpi=300)
                plt.close()
    
    return ls_results

def fit_sinusoidal_probmod(times, rvs, rv_errors):
    """
    Probabilistic model for fitting a sinusoidal RV curve with a fixed frequency (2π per cycle).
    
    This function defines a Numpyro model that samples amplitude, phase shift, and baseline 
    (height) parameters to reproduce the observed RV variation as a sinusoid. It uses the NUTS
    sampler and returns the trace of posterior samples.
    
    Parameters:
        times (array-like): Array of observation times or phases.
        rvs (array-like): Radial velocity measurements.
        rv_errors (array-like): Measurement uncertainties for the RVs.
    
    Returns:
        dict: Posterior samples (trace) of the model parameters.
    """
    def sinu_model(times=None, rvs=None):
        fixed_frequency = 2 * jnp.pi
        # Model parameters
        amplitude = npro.sample('amplitude', dist.Uniform(0, 500))
        phase_shift = npro.sample('phase_shift', dist.Uniform(-jnp.pi, jnp.pi))
        height = npro.sample('height', dist.Uniform(50, 250))
        # Sinusoidal model with fixed frequency
        pred = amplitude * jnp.sin(fixed_frequency * times + phase_shift) + height
        # Likelihood
        npro.deterministic('pred', pred)
        if rv_errors is not None:
            npro.sample('obs', dist.Normal(pred, jnp.array(rv_errors)), obs=rvs)
    rng_key = random.PRNGKey(0)
    kernel = NUTS(sinu_model)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000)
    mcmc.run(rng_key, times=jnp.array(times), rvs=jnp.array(rvs))
    mcmc.print_summary()
    return mcmc.get_samples()

def fit_sinusoidal_probmod_sb2(phase, rv1, rv1_err, rv2, rv2_err, amp_max=500, height_min=50, height_max=250):
    """
    Probabilistic model for sinusoidally fitting RV curves for SB2 binary stars.
    
    In this model, a fixed frequency is assumed, with shared parameters for phase offset 
    (phi0) and systemic velocity (gamma) and separate amplitudes (K1 and K2) for the primary
    and secondary components. The model returns RV predictions for both components.
    
    Parameters:
        phase (array-like): Phases of observation.
        rv1 (array-like): Radial velocities for the primary component.
        rv1_err (array-like): Uncertainties for the primary RVs.
        rv2 (array-like): Radial velocities for the secondary component.
        rv2_err (array-like): Uncertainties for the secondary RVs.
        amp_max (float, optional): Maximum amplitude allowed (default=500).
        height_min (float, optional): Minimum systemic velocity (default=50).
        height_max (float, optional): Maximum systemic velocity (default=250).
    
    Returns:
        dict: Posterior samples (trace) of the model parameters.
    """
    def sb2_model(phase, rv1, rv1_err, rv2, rv2_err):
        fixed_frequency = 2 * jnp.pi
        # Shared parameters
        phi0 = npro.sample('phi0', dist.Uniform(-jnp.pi, jnp.pi))
        gamma = npro.sample('gamma', dist.Uniform(height_min, height_max))
        # Distinct parameters
        K1 = npro.sample('K1', dist.Uniform(0, amp_max))
        K2 = npro.sample('K2', dist.Uniform(0, amp_max))
        # Predicted RVs
        pred_rv1 = K1 * jnp.sin(fixed_frequency * phase + phi0) + gamma
        pred_rv2 = -K2 * jnp.sin(fixed_frequency * phase + phi0) + gamma
        # Likelihood
        npro.sample('obs1', dist.Normal(pred_rv1, rv1_err), obs=rv1)
        npro.sample('obs2', dist.Normal(pred_rv2, rv2_err), obs=rv2)
    rng_key = random.PRNGKey(0)
    kernel = NUTS(sb2_model)
    mcmc = MCMC(kernel, num_warmup=3000, num_samples=3000, num_chains=4)
    mcmc.run(rng_key, phase=phase, rv1=rv1, rv1_err=rv1_err, rv2=rv2, rv2_err=rv2_err)
    mcmc.print_summary()
    return mcmc.get_samples()

def phase_rv_curve(time, rv1, rv1_er=None, rv2=None, rv2_er=None, period=None, print_output=True, plots=True):
    """
    Compute phases for the RV observations and prepare data for sinusoidal fitting.
    
    Given a period, this function computes the phase for each observation (time modulo period),
    sorts the data, and duplicates the phase data shifted by -1 and +1 to allow for a continuous 
    view over a phase range of [-0.5, 1.5]. It returns the expanded phase array and sorted RVs 
    (and errors). If secondary RVs are provided, they are similarly processed.
    
    Parameters:
        time (array-like): Observation times.
        rv1 (array-like): Primary RV measurements.
        rv1_er (array-like, optional): Errors on the primary RVs.
        rv2 (array-like, optional): Secondary RV measurements (if available).
        rv2_er (array-like, optional): Errors for the secondary RVs.
        period (float): The period to use in phasing.
        print_output (bool, optional): If True, print status messages.
        plots (bool, optional): If True, generate plots (not directly used here).
    
    Returns:
        If secondary RVs are provided:
            tuple: (result, phase_expanded, rv1_expanded, rv1_err_expanded, rv2_expanded, rv2_err_expanded)
        Otherwise:
            tuple: (result, phase_expanded, rv1_expanded, rv1_err_expanded)
        Where 'result' is the output of the sinusoidal fitting function.
    """
    if print_output:
        print('\n*** Computing phases ***')
        print('------------------------')
    print('  period = ', period)
    if not period:
        print('  No period provided for primary star')
        return
    
    # Compute phases mod period
    phase = (np.array(time) / period) % 1
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    rv1_sorted = np.array(rv1.reset_index(drop=True))[sort_idx]
    rv1_er_sorted = np.array(rv1_er.reset_index(drop=True))[sort_idx] if rv1_er is not None else None
    if rv2 is not None:
        rv2_sorted = np.array(rv2.reset_index(drop=True))[sort_idx]
        rv2_er_sorted = np.array(rv2_er.reset_index(drop=True))[sort_idx] if rv2_er is not None else None
    
    # Duplicate phases shifted by -1 and +1 for continuity
    phase_neg = phase_sorted - 1
    phase_pos = phase_sorted + 1
    rv1_neg = rv1_sorted.copy()
    rv1_pos = rv1_sorted.copy()
    rv1_err_neg = rv1_er_sorted.copy() if rv1_er_sorted is not None else None
    rv1_err_pos = rv1_er_sorted.copy() if rv1_er_sorted is not None else None
    if rv2 is not None:
        rv2_neg = rv2_sorted.copy()
        rv2_pos = rv2_sorted.copy()
        rv2_err_neg = rv2_er_sorted.copy() if rv2_er_sorted is not None else None
        rv2_err_pos = rv2_er_sorted.copy() if rv2_er_sorted is not None else None
    
    phase_all = np.concatenate([phase_neg, phase_sorted, phase_pos])
    rv1_all = np.concatenate([rv1_neg, rv1_sorted, rv1_pos])
    rv1_err_all = np.concatenate([rv1_err_neg, rv1_er_sorted, rv1_err_pos]) if rv1_er_sorted is not None else None
    if rv2 is not None:
        rv2_all = np.concatenate([rv2_neg, rv2_sorted, rv2_pos])
        rv2_err_all = np.concatenate([rv2_err_neg, rv2_er_sorted, rv2_err_pos]) if rv2_er_sorted is not None else None
    
    # Select data in the phase interval [-0.5, 1.5]
    mask = (phase_all >= -0.5) & (phase_all <= 1.5)
    phase_expanded = phase_all[mask]
    rv1_expanded = rv1_all[mask]
    rv1_err_expanded = rv1_err_all[mask] if rv1_err_all is not None else None
    if rv2 is not None:
        rv2_expanded = rv2_all[mask]
        rv2_err_expanded = rv2_err_all[mask] if rv2_err_all is not None else None
    
    # Fit a sinusoidal model to the phased data
    if rv2 is not None:
        result = fit_sinusoidal_probmod_sb2(phase_expanded, rv1_expanded, rv1_err_expanded, rv2_expanded, rv2_err_expanded)
        return result, phase_expanded, rv1_expanded, rv1_err_expanded, rv2_expanded, rv2_err_expanded
    else:
        result = fit_sinusoidal_probmod(phase_expanded, rv1_expanded, rv1_err_expanded)
        return result, phase_expanded, rv1_expanded, rv1_err_expanded