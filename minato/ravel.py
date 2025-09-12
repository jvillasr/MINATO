import os
os.environ["JAX_ENABLE_X64"] = "True"
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
from scipy.special import wofz as scipy_wofz
import multiprocessing
import numpyro as npro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import find_valid_initial_params
from numpyro.infer.initialization import init_to_uniform 
import jax
from jax import numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from exojax.special.faddeeva import rewofz
print(f"JAX 64-bit enabled: {jax.config.jax_enable_x64}") # Verify
# Set the number of devices to the number of available CPUs
npro.set_host_device_count(multiprocessing.cpu_count())
import corner
from scipy.stats import gaussian_kde
from astropy.time import Time
import gc

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

def trace_mean(trc):
    """
    Return dictionary of posterior means from trace as a numpy array.
    """
    return {k: np.asarray(v).mean(axis=0) for k, v in trc.items()}

def get_chi2(observed_flux, model_flux, flux_error):
    """
    Compute chi^2 for each epoch across all lines.

    In:
    observed_flux : array, shape (n_lines, n_epochs, n_data)
    model_flux    : array, shape (n_lines, n_epochs, n_data)
    flux_error    : array, shape (n_lines, n_epochs, n_data)

    Out:
    chi2 : array, shape (n_epochs,)
        Chi^2 value for each epoch.
    """
    observed_flux = np.asarray(observed_flux, dtype=float)
    model_flux    = np.asarray(model_flux,    dtype=float)
    flux_error    = np.asarray(flux_error,    dtype=float)

    # Standard chi^2 formula, summed over lines and data points
    chi2_values = np.sum(
        ((observed_flux - model_flux) / flux_error) ** 2,
        axis=(0, 2)
    )
    return chi2_values

def hdi_summary(x, cred=0.68):
    """
    Determine the Highest Density Interval - containing (cred*100) % of samples - from MCMC posterior.
    In:
    x    : array-like oflength (n_samples)
        Flattened RV samples from MCMC posterior result (via fit_sb2_probmod, fit_sb1_probmod).
    cred : float, optional, default = 0.68
        Fraction of samples that must be contained in the credible interval for the HDI/mode-aware analysis.
    
    Out:
    Center, half-width of HDI for each epoch [km/s].
    """
    x = np.sort(np.asarray(x).ravel()) # sort RV samples from posterior
    n = x.size
    if n == 0:
        raise ValueError('No samples in trace!')
    k = max(1, int(np.ceil(cred * n))) # get # of samples that constitute 68% of total samples
    widths = x[k-1:] - x[:n-k+1] # get all possible RV-space window lengths that encompass k samples
    j = int(np.argmin(widths)) # get minimum window length (highest density interval for this cred)
    lo, hi = x[j], x[j + k - 1]
    return 0.5 * (lo + hi), 0.5 * (hi - lo) # return center, half-width of interval

def summarize_mode_1d(samples, cred=0.68, min_sep_sigma=3, min_frac=0.2, kmeans_iters=20):
    """
    Mode-aware summary with automatic choice:
      - If the marginal looks unimodal/ambiguous -> return HDI (center, half-width).
      - If clearly bimodal -> choose the cluster with the higher peak in the 1d marginalized posterior
        & return that cluster's median ± inner-68% half-width.
      - If the HDI center already lies inside the HDI (68%) band, default to HDI results.
    
    In:
    samples : array-like of length (n_samples)
        The posterior RV samples for one line & epoch.
    cred : float, optional
        Fraction of samples that must be contained in the credible interval for the HDI/mode-aware analysis.
    min_sep_sigma : float, optional
        Defining a combined σ as the σs of both modes added in quadrature, this parameter determines
        the minimum separation (in terms of the combined σ) for which the mode-aware summary will be utilized.
        This can be fine-tuned by considering the results of the RV cornerplots.
    min_frac: float on interval (0,1), optional
        Minimum threshold of fraction of total samples that must be present in the dominant mode.
    kmeans_iters : int
        Number of iterations executed in attempting to clearly define mode centers.

    Out:
    center, half-width of dominant RV mode [km/s] : floats
    posterior_method : str ('HDI' or 'MODE-AWARE')
        Flags which method was used in analyzing posterior to write out to fit_values.csv.
    """
    x = np.asarray(samples).ravel() # flatten sample
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        raise ValueError("No samples in trace!")

    # Fetch Highest Density Interval (appropriate when unimodal/ambiguous)
    hdi_center, hdi_half = hdi_summary(x, cred=cred)

    # Check for bimodality
    c_low, c_high = np.percentile(x, [33, 67]) # centers at 33rd & 67th percentile
    labels = None
    for _ in range(kmeans_iters):
        # assign each sample to nearest centroid
        labels = (np.abs(x - c_high) < np.abs(x - c_low)).astype(int) #  False (label 0) near low, True (label 1) near high)
        if labels.sum() == 0 or labels.sum() == n: # All samples assigned to one label
            break
        new_low  = np.median(x[labels == 0]) # median of all labels nearer 33rd percentile
        new_high = np.median(x[labels == 1]) # median of all labels nearer 67th percentile
        if np.allclose([new_low, new_high], [c_low, c_high]): # break if centers = medians after current iteration
            c_low, c_high = new_low, new_high
            break
        c_low, c_high = new_low, new_high # update centers of bimodal dist.

    # If no usable split, use HDI
    if labels is None or labels.sum() == 0 or labels.sum() == n:
        return hdi_center, hdi_half, 'HDI'

    # Analyze two separated modes
    m0, m1 = (labels == 0), (labels == 1)
    x0, x1 = x[m0], x[m1]
    frac0, frac1 = x0.size/n, x1.size/n

    # Separability test (gap vs pooled blur) + dominant fraction CITE
    combined_sigmas = np.sqrt(np.var(x0, ddof=1) + np.var(x1, ddof=1) + 1e-12)
    gap  = abs(c_high - c_low)
    separated = (gap >= min_sep_sigma*combined_sigmas) and (max(frac0, frac1) >= min_frac)
    if not separated:
        return hdi_center, hdi_half, 'HDI'

    # --- Choose the cluster with the higher peak height (via KDE at medians) ---
    kde = gaussian_kde(x, bw_method='silverman') # or 'scott'?
    med0, med1 = np.median(x0), np.median(x1)

    p0, p1 = kde([med0, med1])  # KDE peak heights at each mode's median

    # Pick mode with the higher peak height
    if p1 > p0:
        x_main, center = x1, med1
    else:
        x_main, center = x0, med0

    half = 0.5 * (np.percentile(x_main, 84) - np.percentile(x_main, 16))

    # If HDI center already sits inside this dominant 68% band, prefer HDI
    if (center - half) <= hdi_center <= (center + half):
        return hdi_center, hdi_half, 'HDI'

    return center, half, 'MODE-AWARE'

def make_rv_corner_for_component(dv, k, path, diagnostics=True):
    """
    Build a corner plot for the RVs of a single component k,
    overlaying mode-aware and HDI summaries on the 1d marginalized posteriors.
    In:
    dv : array of length n_samples
        RV samples from MCMC trace for one epoch
    k : int (0 = primary component, 1 = secondary component)
        Select the component to plot (assume formatting of SB2 output trace).
    path : str
        Output path to save figure to.
    diagnostics : bool
        Add annotations to each epoch indicated which method (HDI/mode aware) was used.
    """
    # Build matrix of samples: columns = epochs for this component
    n_samp, K, _, T = dv.shape
    param_names = [f"RV_{k+1}_{t+1}" for t in range(T)]
    cols = [dv[:, k, 0, t] for t in range(T)]
    samples_k = np.vstack(cols).T
    ndim = samples_k.shape[1]

    # Corner plot
    fig = corner.corner(
        samples_k,
        labels=param_names,
        show_titles=True,
        title_kwargs=dict(fontsize=10),
        label_kwargs=dict(fontsize=16),
        plot_datapoints=True,
        smooth=0.0,
        quiet=True
    )

    # Overlays: mode-aware vs HDI on the diagonal
    mode_centers, mode_lo, mode_hi, hdi_centers, hdi_lo, hdi_hi, methods = [], [], [], [], [], [], []

    for j in range(ndim):
        x = samples_k[:, j]
        # Mode-aware (dominant-mode) summary
        val, err, method = summarize_mode_1d(x)
        mode_centers.append(val)
        mode_lo.append(val - err)
        mode_hi.append(val + err)
        methods.append(method)

        # Global HDI (shortest 68% interval)
        c_hdi, e_hdi = hdi_summary(x, cred=0.68)
        hdi_centers.append(c_hdi)
        hdi_lo.append(c_hdi - e_hdi)
        hdi_hi.append(c_hdi + e_hdi)

    axes = np.array(fig.axes).reshape((ndim, ndim))
    for j in range(ndim):
        ax = axes[j, j]
        # Mode-aware
        ax.axvline(mode_centers[j], color='C3', lw=2, alpha=0.95, label='Mode-aware center')
        ax.axvspan(mode_lo[j], mode_hi[j], color='C3', alpha=0.20, label='Mode-aware 68%')
        # HDI
        ax.axvline(hdi_centers[j], color='0.25', lw=1.2, ls='--', alpha=0.9, label='Global HDI center')
        ax.axvspan(hdi_lo[j], hdi_hi[j], color='0.25', alpha=0.10, label='Global HDI 68%')

        # Small annotation
        if diagnostics == True:
            ax.text(
                1.05, 0.95,
                f"Modal {mode_centers[j]:.1f}±{(mode_hi[j]-mode_lo[j])/2:.1f}\n"
                f"HDI  {hdi_centers[j]:.1f}±{(hdi_hi[j]-hdi_lo[j])/2:.1f}\n"
                f"method = {methods[j]}",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2)
            )

    # Single legend in top-left panel
    axes[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                      fontsize=8, frameon=True, framealpha=0.6)

    fig.suptitle(f"RV posteriors (component {k+1})",
                 y=0.995, fontsize=35)
    fig.savefig(os.path.join(path, f"corner_rv_comp{k+1}.pdf"),
                bbox_inches='tight')
    plt.close()


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

@jit
def voigt(x, amp, cen, wid_G, wid_L):
    """
    Voigt profile using exojax.special.rewofz, assuming multi-dimensional input support.
    Ensures explicit broadcasting before calculations.

    Parameters:
    x : jax.Array (e.g., shape (1, n_lines, nepochs, ndata))
    amp : jax.Array (e.g., shape (K, n_lines, 1, 1))
    cen : jax.Array (e.g., shape (K, n_lines, nepochs, 1))
    wid_G : jax.Array (e.g., shape (K, n_lines, 1, 1))
    wid_L : jax.Array (e.g., shape (K, n_lines, 1, 1))

    Returns:
    jax.Array with broadcasted shape (e.g., (K, n_lines, nepochs, ndata))
    """
    # Type casting (optional, ensures consistency)
    x = x.astype(jnp.float64)
    cen = cen.astype(jnp.float64)
    wid_G = wid_G.astype(jnp.float64)
    wid_L = wid_L.astype(jnp.float64)
    amp = amp.astype(jnp.float64)

    # Constants
    sqrt2 = jnp.sqrt(2.0)
    sqrt2pi = jnp.sqrt(2.0 * jnp.pi)
    sigma_factor = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) # FWHM_G = sigma * factor

    # Widths conversion
    sigma = wid_G / sigma_factor
    gamma = wid_L / 2.0 # HWHM_L
    sigma = jnp.maximum(sigma, 1e-10) # Avoid division by zero

    # print("shapes →  x", x.shape,
    #     "amp", amp.shape, "cen", cen.shape,
    #     "wid_G", wid_G.shape, "wid_L", wid_L.shape)

    # --- Explicitly broadcast all inputs to a common target shape ---
    try:
        # Determine target shape by broadcasting all relevant inputs together
        target_shape = jnp.broadcast_shapes(x.shape, cen.shape, sigma.shape, gamma.shape, amp.shape)
    except ValueError as e:
        # Print shapes if broadcasting fails to help debugging
        print(f"Error broadcasting shapes in voigt_exojax:")
        print(f"  x.shape: {x.shape}")
        print(f"  cen.shape: {cen.shape}")
        print(f"  sigma.shape: {sigma.shape}")
        print(f"  gamma.shape: {gamma.shape}")
        print(f"  amp.shape: {amp.shape}")
        raise e

    # Broadcast arrays to the target shape
    x_b = jnp.broadcast_to(x, target_shape)
    cen_b = jnp.broadcast_to(cen, target_shape)
    sigma_b = jnp.broadcast_to(sigma, target_shape)
    gamma_b = jnp.broadcast_to(gamma, target_shape)
    amp_b = jnp.broadcast_to(amp, target_shape)
    # --- End Broadcasting ---

    # Scaled coordinates for rewofz (using broadcasted arrays)
    x_scaled = (x_b - cen_b) / (sigma_b * sqrt2)
    y_scaled = gamma_b / (sigma_b * sqrt2)


    # --- Flatten inputs for rewofz ---
    original_shape = x_scaled.shape # Store the 4D shape
    x_flat = x_scaled.reshape(-1)   # Flatten to 1D
    y_flat = y_scaled.reshape(-1)   # Flatten to 1D
    # --- End Flattening ---

    # --- Define a function that calls rewofz for SCALAR inputs ---
    # We need to handle the potential complex output and take the real part
    def scalar_rewofz(scalar_x, scalar_y):
        res = rewofz(scalar_x, scalar_y)
        # Check if the result is complex (some implementations might return complex)
        # and return the real part if necessary.
        # If rewofz always returns real, this check can be simplified/removed.
        return jnp.real(res) # Take real part safely

    # --- Use vmap to apply scalar_rewofz element-wise ---
    # Map over the first (and only) dimension of x_flat and y_flat
    vectorized_rewofz = vmap(scalar_rewofz, in_axes=(0, 0))

    try:
        w_flat = vectorized_rewofz(x_flat, y_flat) # Apply vmapped function
    except Exception as e:
        print(f"Error calling vmapped rewofz:")
        print(f"  x_flat.shape: {x_flat.shape}")
        print(f"  y_flat.shape: {y_flat.shape}")
        raise e
    # --- End vmap call ---

    # --- Reshape result back to 4D ---
    try:
        w = w_flat.reshape(original_shape)
    except ValueError as e:
        print(f"Error reshaping vmapped rewofz output:")
        print(f"  w_flat.shape: {w_flat.shape}")
        print(f"  target original_shape: {original_shape}")
        raise e
    # --- End Reshaping ---

    # Calculate Voigt profile value (using 4D arrays)
    V = w / (sigma_b * sqrt2pi)

    # --- Peak value (x_scaled = 0) ---
    zero_flat = jnp.zeros_like(x_flat)
    try:
        # Apply the same vmapped function for the peak
        w0_flat = vectorized_rewofz(zero_flat, y_flat)
    except Exception as e:
        print(f"Error calling vmapped rewofz for peak value:")
        print(f"  zero_flat.shape: {zero_flat.shape}")
        print(f"  y_flat.shape: {y_flat.shape}")
        raise e

    # Reshape peak value back to 4D
    try:
        w0 = w0_flat.reshape(original_shape)
    except ValueError as e:
        print(f"Error reshaping vmapped rewofz peak output:")
        print(f"  w0_flat.shape: {w0_flat.shape}")
        print(f"  target original_shape: {original_shape}")
        raise e

    V0 = w0 / (sigma_b * sqrt2pi)
    V0 = jnp.maximum(V0, 1e-15)
    # --- End Peak Calculation ---

    # Final scaling: Returns absorption profile
    return -amp_b * (V / V0)

SQRT_2   = jnp.sqrt(2.0)
SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)
LN2      = jnp.log(2.0)
SIGMA_FAC = 2.0 * jnp.sqrt(2.0 * LN2)          # FWHM_G = sigma * SIGMA_FAC

@jax.jit
def pseudo_voigt(x, amp, cen, fwhm_G, fwhm_L):
    """
    Pseudo-Voigt profile (Thompson-Cox-Hastings cubic η, Olivero-Longbothum FWHM).

    Shapes follow the broadcasting convention used in the original `voigt`.
    All arrays are promoted to float64 for consistency.
    """
    # --- promote to float64 (once JIT-compiled this is negligible) -----------
    x   = x.astype(jnp.float64)
    cen = cen.astype(jnp.float64)
    fwhm_G = fwhm_G.astype(jnp.float64)
    fwhm_L = fwhm_L.astype(jnp.float64)
    amp = amp.astype(jnp.float64)

    # --- Gaussian & Lorentzian widths ----------------------------------------
    sigma  = fwhm_G / SIGMA_FAC           # Gaussian σ
    gamma  = fwhm_L / 2.0                 # Lorentzian HWHM

    # Olivero–Longbothum approximation of the *Voigt* FWHM
    fG5 = fwhm_G**5
    fL5 = fwhm_L**5
    fV  = (fG5
           + 2.69269 * fwhm_G**4 * fwhm_L
           + 2.42843 * fwhm_G**3 * fwhm_L**2
           + 4.47163 * fwhm_G**2 * fwhm_L**3
           + 0.07842 * fwhm_G    * fwhm_L**4
           + fL5) ** 0.2

    # Thompson-Cox-Hastings mixing coefficient η(y) ;  y = fL / fV
    y   = fwhm_L / fV
    eta = 1.36603 * y - 0.47719 * y**2 + 0.11116 * y**3
    eta = jnp.clip(eta, 0.0, 1.0)         # numerical safety

    # --- profiles, unit peak height ------------------------------------------
    x_c = x - cen                         # (K,n_lines,n_epochs,n_data)

    G = jnp.exp(- (x_c**2) / (2.0 * sigma**2))            # Gaussian, G(0)=1
    L = (gamma**2) / (x_c**2 + gamma**2)                  # Lorentzian, L(0)=1

    pV = (1.0 - eta) * G + eta * L                        # unit-height pVoigt

    # --- scale to requested amplitude (absorption) ---------------------------
    return -amp * pV                                       # negative = absorption

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
        header = hdul[0].header
        try:
            if instrument == 'FLAMES':
                # BLOeM
                # star_epoch = header['OBJECT'] + '_' + header['EPOCH_ID']
                # mjd = header['MJD_MID']
                # wave = hdul[1].data['WAVELENGTH']
                # flux = hdul[1].data['SCI_NORM']
                # ferr = hdul[1].data['SCI_NORM_ERR']
                # BBC
                star_epoch = header['BBC_NAME']
                mjd = header['MJD_OB']
                lengthx =  header['NAXIS1']
                refx    =  header['CRPIX1']
                stepx   =  header['CDELT1']
                startx  =  header['CRVAL1']
                wave = (np.arange(lengthx) - (refx - 1))*stepx + startx 
                flux = hdul[2].data
                ferr = hdul[3].data

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

def read_spectra(filelist, path, file_type, instrument=None, SB2=False):
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
    if file_type == 'dict':
        # require these keys, or error out
        required = ['wavelengths','fluxes','f_errors','names','jds']
        missing = [k for k in required if k not in filelist]
        if missing:
            raise KeyError(f"read_spectra(dict): missing keys {missing}")
        wavelengths = filelist['wavelengths']
        fluxes      = filelist['fluxes']
        f_errors    = filelist['f_errors']
        names       = filelist['names']
        jds         = filelist['jds']
    else:
        # Text and FITS files are processed one-by-one
        wavelengths, fluxes, f_errors, names, jds = [], [], [], [], []
        for spec in filelist:
            if file_type in ['dat', 'txt', 'csv']:
                names.append(spec.replace(f'.{file_type}', ''))
                try:
                    # read everything as rows of strings
                    df = pd.read_csv(spec, header=None, sep=r'\s+', dtype=str)
                    # keep only rows where first column is numeric (drop header lines)
                    mask = pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()
                    df = df[mask].reset_index(drop=True).astype(float)

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

            elif file_type == 'fits':
                if instrument is None:
                    raise ValueError(
                        "Instrument must be provided when reading FITS files. "
                        "Currently supported instruments: 'FLAMES', 'FEROS'."
                    )
                wave, flux, ferr, star, mjd = read_fits(spec, instrument)
                wavelengths.append(wave)
                fluxes.append(flux)
                f_errors.append(ferr)
                names.append(star)
                jds.append(mjd)

    if file_type in ['dat', 'txt', 'csv']:
        # Check if JDs.txt file with observation times exists:
        try:
            with open(os.path.join(path, 'JDs.txt'), 'r') as f:
                df_jds = pd.read_csv(f, header=None, sep=r'\s+', dtype=str)
                jds = df_jds[1].tolist()
        except FileNotFoundError:
            print(f"JDs.txt file not found in {path}.")
            jds.append(None)

    return wavelengths, fluxes, f_errors, names, jds

# def setup_star_directory_and_save_jds(names, jds, path, SB2):
#     """
#     Prepare a directory for star data and save the corresponding Julian Dates.

#     Parameters:
#     names : list
#         List of star names.
#     jds : list
#         List of Julian Dates for each observation.
#     path : str
#         Base directory to store the files.
#     SB2 : bool
#         If True, specifies the system as a spectroscopic binary (SB2) and modifies the path.

#     Returns:
#     str
#         The final directory path used for saving the data.
#     """
#     try:
#         basename = names[0].split('.')[0]  
#         star = basename + '/'
#     except IndexError:
#         star = 'Unknown_Star/'
#     star_path = os.path.join(path, star)
#     # if SB2:
#     #    star_path = os.path.join(star_path, 'SB2/')
#     if not os.path.exists(star_path):
#         os.makedirs(star_path)
#     if any(jds):
#         df_mjd = pd.DataFrame({'epoch': names, 'JD': jds})
#         df_mjd.to_csv(os.path.join(star_path, 'JDs.txt'), index=False, header=False, sep='\t')
#     return star_path

def setup_star_directory_and_save_jds(names, jds, path, SB2):
    """
    Prepare a directory for star data and save the corresponding Julian Dates.

    Parameters:
    names : list
        List of star names.
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
    import os
    import pandas as pd

    star_path = path
    if not os.path.exists(star_path):
        os.makedirs(star_path)
    if any(jds):
        clean_names = [os.path.basename(n) for n in names]
        df_mjd = pd.DataFrame({'epoch': clean_names, 'JD': jds})
        df_mjd.to_csv(os.path.join(star_path, 'JDs.txt'),
                      index=False, header=False, sep='\t')

    return star_path


def setup_line_dictionary():
    """
    Create a dictionary of spectral lines, including regions and initial fitting parameters.

    * All values under the 'centre' key are Ritz vacuum wavelengths (Å) taken from the
      NIST Atomic Spectra Database (ASD).
    * The 'air' key gives the corresponding air wavelengths (Å) converted with the
      Edlén (1966) refractive index of standard air.
    * For the He I blends at 4026 Å, 4121 Å, 4471 Å and 4713 Å the listed vacuum and air
      wavelengths are Aₖᵢ-weighted centroids of the unresolved fine-structure components
      (weights from the Einstein-A coefficients tabulated in NIST).  Uncertainties reflect
      the scatter among those components (4-5 x 10-5 Å); the intrinsic Ritz uncertainties
      are <= 5 x 10-6 Å and thus negligible here.
    * He II wavelengths computed from the NIST He II “Levels” table (dataset L3620c107, level 
      uncertainties included). Fine-structure doublets were J-averaged with statistical weights. 
      Vacuum→air conversion via Edlén (1966).

    Returns
    -------
    dict
        A dictionary mapping spectral-line identifiers to their respective properties.
    """
    lines_dic = {
        # --- Balmer centres of gravity (NIST) --------------------------------
        4102: {  # Hδ
            'region': [4080, 4122], 'centre': [4102.8991, 0.0024], 'air': [4101.7414, 0.0024], 'wid_ini': 6, 'title': 'H$\delta$'},
        4340: {  # Hγ
            'region': [4316, 4366], 'centre': [4341.691, 0.003],   'air': [4340.471, 0.003],   'wid_ini': 7, 'title': 'H$\gamma$'},
        4861: {  # Hβ
            'region': [4836, 4871], 'centre': [4862.691, 0.003],   'air': [4861.333, 0.003],   'wid_ini': 6, 'title': 'H$\beta$'},
        6562: {  # Hα
            'region': [6538, 6579], 'centre': [6564.632, 0.007],   'air': [6562.819, 0.007],   'wid_ini': 6, 'title': 'H$\alpha$'
        },
        # --- He I lines ------------------------------------------------
        4009: { 'region': [4001, 4014], 'centre': [4010.3899037, 0.0000011], 'air': [4009.256516, 0.000020], 'wid_ini': 3, 'title': 'He I $\lambda$4009'},
        4026: { 'region': [4013, 4039], 'centre': [4027.36003, 0.00004],     'air': [4026.22221, 0.00004],   'wid_ini': 3, 'title': 'He I $\lambda$4026'},
        4121: { 'region': [4114, 4126], 'centre': [4121.99776, 0.00002],     'air': [4120.83518, 0.00002],   'wid_ini': 3, 'title': 'He I $\lambda$4121'},
        4144: { 'region': [4131, 4166], 'centre': [4144.9276502, 0.0000012], 'air': [4143.761, 0.010],       'wid_ini': 3, 'title': 'He I $\lambda$4144'},
        4388: { 'region': [4373, 4403], 'centre': [4389.1619053, 0.0000013], 'air': [4387.9296, 0.0006],     'wid_ini': 3, 'title': 'He I $\lambda$4388'},
        4471: { 'region': [4454, 4487], 'centre': [4472.77309, 0.00003],     'air': [4471.51829, 0.00003],   'wid_ini': 3, 'title': 'He I $\lambda$4471'},
        4713: { 'region': [4697, 4728], 'centre': [4714.48979, 0.00005],     'air': [4713.17111, 0.00005],   'wid_ini': 3, 'title': 'He I $\lambda$4713'},
        4922: { 'region': [4911, 4933], 'centre': [4923.3050740, 0.0000017], 'air': [4921.931036, 0.000025], 'wid_ini': 3, 'title': 'He I $\lambda$4922'},
        5876: { 'region': [5863, 5889], 'centre': [5877.31599825, 0.0000007],'air': [5875.687499, 0.000030], 'wid_ini': 3, 'title': 'He I $\lambda$5876'},
        6678: { 'region': [6658, 6698], 'centre': [6679.995639, 0.000003],   'air': [6678.15174, 0.00003],   'wid_ini': 3, 'title': 'He I $\lambda$6678'},
        # --- He II lines ------------------------------------------------
        4542: { 'region': [4533, 4551], 'centre': [4542.912549, 0.000132], 'air': [4541.63924, 0.000132],    'wid_ini': 3, 'title': 'He II $\lambda$4542'},
        4686: { 'region': [4672, 4700], 'centre': [4687.015247, 0.000091], 'air': [4685.70384 , 0.000091],   'wid_ini': 4, 'title': 'He II $\lambda$4686'},
        5412: { 'region': [5401, 5415], 'centre': [5413.083151, 0.000262], 'air': [5411.57874 , 0.000262],   'wid_ini': 4, 'title': 'He II $\lambda$5412'},
        # --- Other lines ------------------------------------------------
        3995: { 'region': [3986, 4001], 'centre': None,                                 'wid_ini': 2, 'title': 'N II $\lambda$3995'},
        4089: { 'region': [4075, 4095], 'centre': [4090.016, 0.1],   'air':   [4088.862, 0.10], 'wid_ini': 2, 'title': 'Si IV $\lambda$4089'},
        4128: { 'region': [4120, 4132], 'centre': [4129.218, 0.003], 'air': [4128.054, 0.003],            'wid_ini': 2, 'title': 'Si II $\lambda$4128'},
        4131: { 'region': [4124, 4136], 'centre': [4132.059, 0.003], 'air': [4130.894, 0.003],            'wid_ini': 2, 'title': 'Si II $\lambda$4131'},
        4233: { 'region': [4225, 4237], 'centre': [], 'air': None,                       'wid_ini': 2, 'title': 'Fe II $\lambda$4233'},
        4267: { 'region': [4259, 4271], 'centre': [], 'air': [4267.258, 0.007],          'wid_ini': 2, 'title': 'C II $\lambda$4267'},
        4481: { 'region': [4474, 4486], 'centre': [4482.4766, 0.0003], 'air': [4481.2192, 0.003],          'wid_ini': 2, 'title': 'Mg II $\lambda$4481'},
        4553: { 'region': [4543, 4558], 'centre': [4553.898, 0.001], 'air': [4552.622, 0.001], 'wid_ini': 3, 'title': 'Si III $\lambda$4553'},
        5890: { 'region': [5875, 5905], 'centre': [], 'air': [5889.951, 0.00003],        'wid_ini': 3, 'title': 'Na I $\lambda$5890'},
        7774: { 'region': [7758, 7782], 'centre': [], 'air': [7774.17, 0.10],            'wid_ini': 3, 'title': 'O I $\lambda$7774'}
    }
    return lines_dic

def deprecated_setup_line_dictionary():
    """
    Create a dictionary of spectral lines, including regions and initial fitting parameters.
    Lines centroids are given in vacuum wavelengths (Angstroms). Alternative, the 'air' key  provides air wavelengths. 

    Returns:
    dict
        A dictionary mapping spectral line identifiers to their respective properties.
    """
    lines_dic = {
        3995: { 'region': [3986, 4001], 'centre': None, 'wid_ini': 2, 'title': 'N II $\lambda$3995'},
        4009: { 'region': [4001, 4014], 'centre': [4010.3899037, 0.0000011], 'air': [4009.256516, 0.000020], 'wid_ini': 3, 'title': 'He I $\lambda$4009'},
        4026: { 'region': [4013, 4039], 'centre': [4027.3238176, 0.0000003], 'air': [4026.184368, 0.000020], 'wid_ini': 3, 'title': 'He I $\lambda$4026'},
        4089: { 'region': [4075, 4095], 'centre': [4090.016, 0.1], 'air': [4088.862, 0.10],              'wid_ini': 2, 'title': 'Si IV $\lambda$4089'},
        4102: { 'region': [4080, 4122], 'centre': [4102.92068748, 0.00000008], 'air': [4101.734, 0.006], 'wid_ini': 6, 'title': 'H$\delta$'},
        4121: { 'region': [4114, 4126], 'centre': [4121.9733416, 0.0000017], 'air': [4120.8154, 0.0012], 'wid_ini': 3, 'title': 'He I $\lambda$4121'},
        4128: { 'region': [4120, 4132], 'centre': [], 'air': [4128.07, 0.10],                            'wid_ini': 2, 'title': 'Si II $\lambda$4128'},
        4131: { 'region': [4124, 4136], 'centre': [], 'air': [4130.89, 0.10],                            'wid_ini': 2, 'title': 'Si II $\lambda$4131'},
        4144: { 'region': [4131, 4166], 'centre': [4144.9276502, 0.0000012], 'air': [4143.761, 0.010],   'wid_ini': 3, 'title': 'He I $\lambda$4144'},
        4233: { 'region': [4225, 4237], 'centre': [], 'air': None,                                       'wid_ini': 2, 'title': 'Fe II $\lambda$4233'},
        4267: { 'region': [4259, 4271], 'centre': [], 'air': [4267.258, 0.007],                          'wid_ini': 2, 'title': 'C II $\lambda$4267'},
        4340: { 'region': [4316, 4366], 'centre': [4341.714690, 0.000004], 'air': [4340.472, 0.006],     'wid_ini': 7, 'title': 'H$\gamma$'},
        4388: { 'region': [4373, 4403], 'centre': [4389.1619053, 0.0000013], 'air': [4387.9296, 0.0006], 'wid_ini': 3, 'title': 'He I $\lambda$4388'},
        4471: { 'region': [4454, 4487], 'centre': [4472.7291049, 0.0000004], 'air': [4471.4802, 0.0015], 'wid_ini': 3, 'title': 'He I $\lambda$4471'},
        4481: { 'region': [4474, 4486], 'centre': [], 'air': [4481.130, 0.010],                          'wid_ini': 2, 'title': 'Mg II $\lambda$4481'},
        4542: { 'region': [4533, 4551], 'centre': [], 'air': [4541.591, 0.010],                          'wid_ini': 3, 'title': 'He II $\lambda$4542'},
        4553: { 'region': [4543, 4558], 'centre': [4553.898, 0.1], 'air': [4552.62, 0.10],               'wid_ini': 3, 'title': 'Si III $\lambda$4553'},
        4861: { 'region': [4836, 4871], 'centre': [], 'air': [4861.35, 0.05],     'wid_ini': 5, 'title': 'H$\beta$'},
        4922: { 'region': [4911, 4933], 'centre': [], 'air': [4921.9313, 0.0005], 'wid_ini': 4, 'title': 'He I $\lambda$4922'},
        5412: { 'region': [5401, 5415], 'centre': [], 'air': [5411.52, 0.10],     'wid_ini': 4, 'title': 'He II $\lambda$5412'},
        5876: { 'region': [5863, 5889], 'centre': [], 'air': [5875.621, 0.010],   'wid_ini': 4, 'title': 'He I $\lambda$5876'},
        5890: { 'region': [5875, 5905], 'centre': [], 'air': [5889.951, 0.00003], 'wid_ini': 3, 'title': 'Na I $\lambda$5890'},
        6562: { 'region': [6538, 6579], 'centre': [], 'air': [6562.79, 0.030],    'wid_ini': 6, 'title': 'H$\alpha$'},
        6678: { 'region': [6658, 6698], 'centre': [], 'air': [6678.151, 0.010],   'wid_ini': 4, 'title': 'He I $\lambda$6678'},
        7774: { 'region': [7758, 7782], 'centre': [], 'air': [7774.17, 0.10],     'wid_ini': 3, 'title': 'O I $\lambda$7774'}
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
    nplots = len(wavelengths)
    ncols = int(np.sqrt(nplots))
    nrows = nplots // ncols
    if ncols * nrows < nplots:
        nrows += 1
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

def fit_sb1_probmod(lines, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, path,
                    shift_kms=0, wavelength_type='air', rm_epochs=None, profile='Voigt', cornerplot=True):
    """
    Fit SB1 (single-lined spectroscopic binary) spectral lines using a probabilistic
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
    sigma_prior : float
        Sigma on Gaussian priors for RV fitting in the second MCMC run [km/s].
    K : int, optional
        Number of components (default 2).
    shift_kms : float, optional
        The overall velocity shift in km/s. For example, use 172 km/s for the SMC.
    wavelength_type : str, optional
        Type of wavelength to use ('air' or 'vacuum'). Default is 'air'.
    rm_epochs : list, optional
        The indices (0-based) of the epochs to remove from the fitting.
    profile : str, optional ('Voigt' or 'Gaussian)
        Profile type used for all components in the spectral line fitting.
    chi2_plots : bool
        Enable/disable seperate plotting of first, second, and final (stitched) MCMC results.
    cornerplot : bool
        Enable/disable automatic cornerplot of RV posteriors for each epoch.

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

    # Determine the key to use based on the chosen wavelength type
    key = 'centre' if wavelength_type == 'vacuum' else 'air'

    # Boolean mask for Hydrogen lines (will use Lorentzian instead of Gaussian)
    is_hline = jnp.array([line in Hlines for line in lines])

    # Interpolate fluxes and errors to a common grid
    x_waves_interp, y_fluxes_interp, y_errors_interp = [], [], []
    common_grid_length = 200  # Choose a consistent number of points for interpolation

    for line in lines:
        region_start, region_end = lines_dic[line]['region']
        # Shift the region boundaries by shift_kms
        region_start = rv_shift_wavelength(region_start, shift_kms)
        region_end = rv_shift_wavelength(region_end, shift_kms)

        x_waves_line, y_fluxes_line, y_errors_line = [], [], []

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
    x_waves = jnp.array(x_waves_interp)
    y_fluxes = jnp.array(y_fluxes_interp)
    y_errors = jnp.array(y_errors_interp)

    # Remove bad epochs along the second axis (axis=1)
    if rm_epochs is not None:
        x_waves = jnp.delete(x_waves, jnp.array(rm_epochs), axis=1)
        y_fluxes = jnp.delete(y_fluxes, jnp.array(rm_epochs), axis=1)
        y_errors = jnp.delete(y_errors, jnp.array(rm_epochs), axis=1)

    # Initial guess for the rest (central) wavelength from lines_dic
    cen_ini = jnp.array([lines_dic[line][key][0] for line in lines])

    # Define the probabilistic SB1 model
    def sb1_model(λ, fλ, σ_fλ, is_hline):
        c_kms = c.to('km/s').value
        nlines, nepochs, ndata = λ.shape

        # Sample continuum level with uncertainty
        logσ_ε = npro.sample('logσ_ε', dist.Uniform(-5, 0))
        σ_ε = jnp.exp(logσ_ε)
        ε = npro.sample('ε', dist.TruncatedNormal(loc=1.0, scale=σ_ε, low=0.7, high=1.1))

        # Define rest wavelengths as a parameter (one per line)
        λ_rest = npro.param("λ_rest", cen_ini)

        σ_Δv = 500. # standard deviation (spread of prior on velocity) to supply to random var. in prob. NumPyro model?

        with npro.plate('epochs', nepochs, dim=-2):
            Δv = npro.sample("Δv", dist.Uniform(-σ_Δv, σ_Δv))
        # make it 1-D for broadcasting 
        Δv = jnp.squeeze(Δv, axis=-1)

        with npro.plate('lines', nlines, dim=-3):
            amp   = npro.sample('amp',   dist.TruncatedNormal(loc=0.18, scale=0.06, low=0.02, high=0.80))   # (L,1,1)
            wid_G = npro.sample('wid_G', dist.Uniform(0.5, 5.0))   # (L,1,1)
            wid_L = npro.sample('wid_L', dist.Uniform(0.1, 3.0))   # (L,1,1)
            wid   = npro.sample('wid',   dist.Uniform(0.5, 5.0))   # (L,1,1)

        # Make λ_rest a deterministic variable and reshape for broadcasting
        λ0 = npro.deterministic("λ0", λ_rest[:, None, None])   # (L,1,1)

        # Compute shifted wavelengths for each epoch
        μ = λ0 * (1 + Δv[None, :, None] / c_kms)               # (L,E,1)

        # Prepare the observed wavelengths for model evaluation
        λ_expanded = λ                                         # (L,E,N)
        is_hline_expanded = is_hline[:, None, None]            # (L,1,1)

        # Compute the model profiles
        G = gaussian(λ_expanded, amp, μ, wid_G)                # (L,E,N)
        L = lorentzian(λ_expanded, amp, μ, wid_L)             # (L,E,N)
        V  = pseudo_voigt(λ_expanded, amp, μ, wid_G, wid_L)    # (L,E,N)

        # Use Lorentzian for Hydrogen lines, Gaussian or Voigt otherwise:
        if profile =='Voigt':
            comp = jnp.where(is_hline_expanded, L, V)             # (L,E,N)
        elif profile =='Gaussian':
            comp = jnp.where(is_hline_expanded, L, G)  
        else:
            raise ValueError('profile argument must be one of Voigt or Gaussian')

        # Sum component profile and add continuum to yield the predicted flux
        fλ_pred = npro.deterministic("fλ_pred", ε + comp)      # (L,E,N)

        # Likelihood: compare predicted flux with observed flux
        npro.sample("fλ", dist.StudentT(df=8, loc=fλ_pred, scale=σ_fλ), obs=fλ)

    # ------------------------
    # MCMC Sampling Procedure
    # ------------------------
    rng_key = random.PRNGKey(0)
    kernel = NUTS(sb1_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_chains=4, num_samples=2000)
    mcmc.run(rng_key, extra_fields=("potential_energy",),
             λ=x_waves, fλ=y_fluxes, σ_fλ=y_errors, is_hline=is_hline)

    potential_energy = mcmc.get_extra_fields()['potential_energy']
    log_probs = -potential_energy
    log_prob = np.mean(log_probs)

    trace = mcmc.get_samples(group_by_chain=False)

    # If specific epochs have bene provided to discard from the fitting procedure
    if rm_epochs is not None:
        n_epochs = n_epochs - len(rm_epochs)
    plot_lines_fit_sb1(wavelengths, lines, x_waves, y_fluxes, n_epochs, trace, lines_dic, shift_kms, path)

    # Output cornerplot of recovered RV posteriors
    if cornerplot:
        dv = np.asarray(trace["Δv"])   # shape (S, E)
        S, E, _ = dv.shape
        data = jnp.squeeze(dv)
        n_samp, E, _ = dv.shape
        param_names = [f"RV_{e+1}" for e in range(E)]
        cols = [dv[:, e, 0] for e in range(E)]
        samples_k = np.vstack(cols).T
        ndim = samples_k.shape[1]

        # Corner plot
        fig = corner.corner(
            samples_k,
            labels=param_names,
            show_titles=True,
            title_kwargs=dict(fontsize=10),
            label_kwargs=dict(fontsize=16),
            plot_datapoints=True,
            smooth=0.0,
            quiet=True
        )

        out_png = os.path.join(path, "cornerplot.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return trace, x_waves, y_fluxes

def plot_lines_fit_sb1(wavelengths, lines, x_waves, y_fluxes, n_epochs, trace, lines_dic, shift_kms, path, n_sol=100):
    """
    Plot the SB1 line-fit results based on the posterior predictions by plotting the best-fitting sample
    across all epochs & lines, as determined by a χ2 test.

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
    path : str
        Directory path to save the plots.
    n_sol : int
        Number of posterior samples ot plot
    """
    from matplotlib.lines import Line2D 

    for idx, line in enumerate(lines):
        print('Plotting fits for line:', line)
        fig, axes = setup_fits_plots(wavelengths)
        for epoch_idx, ax in enumerate(axes.ravel()[:n_epochs]):
            f_pred = trace['fλ_pred']
            # If it came out (S, epochs, lines, N), swap to (S, lines, epochs, N)
            if f_pred.shape[1] == n_epochs and f_pred.shape[2] == len(lines):
                f_pred = np.swapaxes(f_pred, 1, 2)

            fλ_pred_samples = f_pred[-n_sol:, idx, epoch_idx, :]
            continuum_pred_samples = trace['ε'][-n_sol:, None]
            ax.plot(x_waves[idx][epoch_idx], fλ_pred_samples.T, color='orangered', alpha=0.1, rasterized=True)
            # Plot the observed data
            ax.plot(x_waves[idx][epoch_idx], y_fluxes[idx][epoch_idx], color='k', lw=1, alpha=0.8)
            # Plot vertical lines for the rest wavelength and component shifts
            centre = rv_shift_wavelength(lines_dic[line]['air'][0], shift_kms)
            ax.axvline(centre, color='r', linestyle='--', lw=1)
            ax.axvline(rv_shift_wavelength(lines_dic[line]['air'][0], shift_kms), color='orange', linestyle='--', lw=1)
            ax.axvline(rv_shift_wavelength(lines_dic[line]['air'][0], shift_kms), color='orange', linestyle='--', lw=1)
            # Annotate the epoch number
            ax.text(0.1, 0.1, f'Epoch {epoch_idx+1}', transform=ax.transAxes, fontsize=16)

        ax.set_xlim(centre - 13, centre + 13)
        fig.supxlabel('Wavelength [Å]', fontsize=24, y=-0.015)
        fig.supylabel('Flux', fontsize=24, x=-0.01)
        
        custom_lines = [
            Line2D([0], [0], color='orangered', lw=2)
        ]
        fig.subplots_adjust(bottom=0.1)

        # Legend formatting
        fig.legend(
            custom_lines,
            ['Prediction'],
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),  # closer to bottom edge
            ncol=3,
            frameon=False,
            fontsize=14,
            borderaxespad=0.0,
            columnspacing=1.5,
            handlelength=2.5,
        )

        plt.savefig(os.path.join(path, f'{line}_fits_SB1_.png'), dpi=300, bbox_inches='tight')
        plt.close()

def mcmc_results_to_file_sb1(trace, names, jds, writer, csvfile, rm_epochs=None):
    """
    Write MCMC fit results for an SB1 (single component) to a CSV file.
    
    The function utilizes either HDI or mode-aware analysis (see  summarize_mode_1d) to define RV results from 
    the posterior, and writes out to a file called fit_values.csv.
    
    Parameters:
        trace (dict): Dictionary containing MCMC samples (expects key 'Δv_τk').
        names (list): List of epoch identifiers (e.g., names of spectra or observation epochs).
        jds (list): List of corresponding Julian Dates (or None if unavailable).
        writer (csv.DictWriter or None): A DictWriter object or None (if first call, will be initialized).
        csvfile: An open CSV file handle to write the output.
        rm_epochs (list): indices of epochs discounted from the fitting procedure.
        
    Returns:
        writer: A csv.DictWriter instance after writing the header (if initially None) and all rows.
    """
    # Loop over the two components (0 and 1, later converted to 1-based indexing)
    if rm_epochs is not None:
        names = [x for i, x in enumerate(names) if i not in rm_epochs]
        jds = [x for i, x in enumerate(jds) if i not in rm_epochs]

    # Loop over epochs (names)
    for j, epoch_name in enumerate(names):
        results_dict = {}
        results_dict['epoch'] = epoch_name
        if jds is not None and j < len(jds):
            results_dict['MJD'] = jds[j]

        # Calculate the mean RV and its error at epoch j.
        vals = (np.asarray(trace['Δv'][:, j]))
        rv_val, rv_err, mode = summarize_mode_1d(vals)
        results_dict['mean_rv'] = rv_val
        results_dict['mean_rv_er'] = rv_err
        results_dict['posterior_method'] = mode # method used to fetch results from posterior

        # Initialize the writer if not already created
        if writer is None:
            fieldnames = results_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(results_dict)

    return writer

def fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, path, sigma_prior, K=2, shift_kms=0,
                    wavelength_type='air', rm_epochs=None, profile='Voigt', chi2_plots=False):
    """
    Fit SB2 (double-lined spectroscopic binary) spectral lines using a probabilistic
    model with Numpyro. The function interpolates spectral data onto a common grid,
    constructs a Bayesian model for the line profiles, and samples the posterior via
    MCMC (using NUTS). To improve robustness against switching, a second model is constructed
    that re-samples only the RV posteriors. The best-fitting result between both runs is 
    determined via a χ2 comparison.
    
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
    sigma_prior : float
        Sigma on Gaussian priors for RV fitting in the second MCMC run [km/s].
    K : int, optional
        Number of components (default 2).
    shift_kms : float, optional
        The overall velocity shift in km/s. For example, use 172 km/s for the SMC.
    wavelength_type : str, optional
        Type of wavelength to use ('air' or 'vacuum'). Default is 'air'.
    rm_epochs : list, optional
        The indices (0-based) of the epochs to remove from the fitting.
    profile : str, optional ('Voigt' or 'Gaussian)
        Profile type used for all components in the spectral line fitting.
    chi2_plots : bool
        Enable/disable seperate plotting of first, second, and final (stitched) MCMC results.

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

    # Determine the key to use based on the chosen wavelength type
    key = 'centre' if wavelength_type == 'vacuum' else 'air'

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

    # Remove bad epochs along the second axis (axis=1)
    if rm_epochs is not None:
        x_waves = jnp.delete(x_waves, jnp.array(rm_epochs), axis=1)
        y_fluxes = jnp.delete(y_fluxes, jnp.array(rm_epochs), axis=1)
        y_errors = jnp.delete(y_errors, jnp.array(rm_epochs), axis=1)

    # Initial guess for the rest (central) wavelength from lines_dic
    cen_ini = jnp.array([lines_dic[line][key][0] for line in lines])

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

        with npro.plate(f'lines', nlines, dim=-2):
            # Primary amplitude
            amp0 = npro.sample("amp0", dist.TruncatedNormal(loc=0.18, scale=0.06, low=0.02, high=0.40))
            # Depth ratio
            amp_ratio = npro.sample("amp_ratio", dist.TruncatedNormal(loc=0.60, scale=0.15, low=0.25, high=0.95))
            amp1 = amp_ratio * amp0

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
            
            # Sample widths to use with new Voigt model
            # Sample Gaussian FWHM for component 1
            wid_G1 = npro.sample('wid_G1', dist.Uniform(0.5, 5.0)) # Adjust prior as needed
            # Sample Lorentzian FWHM for component 1
            wid_L1 = npro.sample('wid_L1', dist.Uniform(0.1, 3.0)) # Adjust prior as needed

            # Constrain widths for component 2 (example: wid2 > wid1)
            delta_wid_G = npro.sample('delta_wid_G', dist.Uniform(0.1, 2.0))
            delta_wid_L = npro.sample('delta_wid_L', dist.Uniform(0.05, 1.0))
            wid_G2 = wid_G1 + delta_wid_G
            wid_L2 = wid_L1 + delta_wid_L

            # Stack widths for two components and add extra dimensions for broadcasting
            wid_G = jnp.stack([wid_G1, wid_G2], axis=-3)  # Shape: (2, n_lines)
            wid_L = jnp.stack([wid_L1, wid_L2], axis=-3)  # Shape: (2, n_lines)
            wid_G = wid_G[:, :, None]  # Shape: (2, n_lines, 1)
            wid_L = wid_L[:, :, None]  # Shape: (2, n_lines, 1)

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
        voigt_profile = pseudo_voigt(λ_expanded, amp, μ, wid_G, wid_L)

        # Use Lorentzian for Hydrogen lines, Gaussian or Voigt otherwise:
        if profile == 'Voigt':
            comp_profile = jnp.where(is_hline_expanded, lorentzian_profile, voigt_profile)
        elif profile == 'Gaussian':
            comp_profile = jnp.where(is_hline_expanded, lorentzian_profile, gaussian_profile)
        else:
            raise ValueError('Profile to fit must be one of Voigt or Gaussian')

        Ck = npro.deterministic("C_λk", comp_profile)

        # Sum over components and add continuum to yield the predicted flux
        fλ_pred = npro.deterministic("fλ_pred", ε + Ck.sum(axis=0))

        # Likelihood: compare predicted flux with observed flux
        npro.sample("fλ", dist.StudentT(df=8, loc=fλ_pred, scale=σ_fλ), obs=fλ)
    
    # ------------------------
    # MCMC Sampling Procedure
    # ------------------------
    comp_sep = 200. # initially proposed component separation
    Δv_means = jnp.array([shift_kms - comp_sep/2, shift_kms + comp_sep/2]).reshape(K, 1, 1)
    print(f"\nFitting with profile: {profile}, Δv_means: {Δv_means}")

    # ------------------------
    # FIRST MCMC
    # ------------------------
    rng_key = random.PRNGKey(0)
    kernel = NUTS(sb2_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_chains=4, num_samples=2000)
    mcmc.run(rng_key, extra_fields=("potential_energy",), 
             λ=x_waves, fλ=y_fluxes, σ_fλ=y_errors, K=K, is_hline=is_hline, Δv_means=Δv_means)

    trace1 = mcmc.get_samples()

    # ---------------------------------------------------------
    # Build frozen dict from first MCMC posterior means
    # ---------------------------------------------------------
    def get_post_median(name):
        return jnp.array(np.median(trace1[name], axis=0))

    frozen = {
        "logσ_ε":      get_post_median("logσ_ε"),     # scalar
        "ε":           get_post_median("ε"),          # scalar
        "amp0":        get_post_median("amp0"),       # (n_lines,)
        "amp_ratio":   get_post_median("amp_ratio"),  # (n_lines,)
        "wid1":        get_post_median("wid1"),       # (n_lines,)
        "delta_wid":   get_post_median("delta_wid"),  # (n_lines,)
        "wid_G1":      get_post_median("wid_G1"),     # (n_lines,)
        "wid_L1":      get_post_median("wid_L1"),     # (n_lines,)
        "delta_wid_G": get_post_median("delta_wid_G"),# (n_lines,)
        "delta_wid_L": get_post_median("delta_wid_L") # (n_lines,)
        # DOES NOT include 'Δv_τk'
    }

    # RV priors for second MCMC run, switched relative to the first
    dv_prior = np.squeeze(np.mean(np.asarray(trace1["Δv_τk"]), axis=0))  # (K,E)
    if dv_prior.ndim != 2:
        raise ValueError(f"dv_prior shape {dv_prior.shape}, expected (2, E)")
    dv_prior_sw = dv_prior[::-1, :]  # swapping component RVs

    # ------------------------------------
    # RV-only model with swapped prior
    # ------------------------------------
    def rv_only_model(λ, fλ, σ_fλ, K, is_hline,
                    dv_prior_sw, sigma_prior):
        """
        Numpyro model for SB2 RV-only fitting.

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

        Returns:
        --------
        Samples are observed via npro.sample("fλ", ...).
        """
        nlines, nepochs, ndata = λ.shape
        c_kms = 299792.458

        # Switched prior
        with npro.plate("epochs", nepochs, dim=-1):
            Δv_τk = npro.sample("Δv_τk",
                dist.Normal(loc=jnp.array(dv_prior_sw), scale=sigma_prior)  # dv_prior_sw: (K, E)
            )
        Δv_τk = Δv_τk[:, None, :]

        # From the first 
        logσ_ε = npro.sample('logσ_ε', dist.Uniform(-5, 0))
        σ_ε    = jnp.exp(logσ_ε)
        ε      = npro.sample('ε', dist.TruncatedNormal(loc=1.0, scale=σ_ε, low=0.7, high=1.1))

        # wavelengths param (constant)
        λ_rest = npro.param("λ_rest", cen_ini)  # (n_lines,)

        # as before:
        with npro.plate('lines', nlines, dim=-2):
            amp0      = npro.sample("amp0", dist.TruncatedNormal(loc=0.18, scale=0.06, low=0.02, high=0.40))
            amp_ratio = npro.sample("amp_ratio", dist.TruncatedNormal(loc=0.60, scale=0.15, low=0.25, high=0.95))

            wid1      = npro.sample('wid1', dist.Uniform(0.5, 5.0))
            delta_wid = npro.sample('delta_wid', dist.Uniform(0.1, 2.0))

            wid_G1      = npro.sample('wid_G1', dist.Uniform(0.5, 5.0))
            wid_L1      = npro.sample('wid_L1', dist.Uniform(0.1, 3.0))
            delta_wid_G = npro.sample('delta_wid_G', dist.Uniform(0.1, 2.0))
            delta_wid_L = npro.sample('delta_wid_L', dist.Uniform(0.05, 1.0))

            # build shapes exactly as in run 1
            amp1 = amp_ratio * amp0
            amp  = jnp.stack([amp0, amp1], axis=-3)[:, :, None]     # (K, n_lines, 1)

            wid2 = wid1 + delta_wid
            wid  = jnp.stack([wid1, wid2], axis=-3)[:, :, None]     # (K, n_lines, 1)

            wid_G2 = wid_G1 + delta_wid_G
            wid_L2 = wid_L1 + delta_wid_L
            wid_G  = jnp.stack([wid_G1, wid_G2], axis=-3)[:, :, None]
            wid_L  = jnp.stack([wid_L1, wid_L2], axis=-3)[:, :, None]

        # centers and likelihood
        λ0 = λ_rest[None, :, None]                               # (1, n_lines, 1)
        μ  = (λ0 * (1 + Δv_τk / c_kms))[:, :, :, None]            # (K, n_lines, n_epochs, 1)

        λ_expanded        = λ[None, :, :, :]                     # (1, n_lines, n_epochs, ndata)
        is_hline_expanded = is_hline[None, :, None, None]        # (1, n_lines, 1, 1)

        gaussian_profile   = gaussian(λ_expanded, amp, μ, wid)
        lorentzian_profile = lorentzian(λ_expanded, amp, μ, wid)
        voigt_profile      = pseudo_voigt(λ_expanded, amp, μ, wid_G, wid_L)

        if profile == 'Voigt':
            comp_profile = jnp.where(is_hline_expanded, lorentzian_profile, voigt_profile)
        elif profile == 'Gaussian':
            comp_profile = jnp.where(is_hline_expanded, lorentzian_profile, gaussian_profile)
        else:
            raise ValueError('Profile to fit must be one of Voigt or Gaussian')
            
        Ck = npro.deterministic("C_λk", comp_profile)

        fλ_pred = npro.deterministic("fλ_pred", ε + comp_profile.sum(axis=0))
        # npro.sample("fλ", dist.Normal(fλ_pred, σ_fλ), obs=fλ)
        npro.sample("fλ", dist.StudentT(df=8, loc=fλ_pred, scale=σ_fλ), obs=fλ)

    # ----------------------------------------------------------------
    # SECOND MCMC: condition on frozen samples and refit only Δv_τk
    # ----------------------------------------------------------------
    conditioned_rv_model = handlers.condition(rv_only_model, frozen)

    rng_key2 = random.PRNGKey(2)
    kernel2  = NUTS(conditioned_rv_model)
    mcmc2    = MCMC(kernel2, num_warmup=1000, num_chains=4, num_samples=2000)
    mcmc2.run(
        rng_key2, extra_fields=("potential_energy",),
        λ=x_waves, fλ=y_fluxes, σ_fλ=y_errors, K=K, is_hline=is_hline, 
        dv_prior_sw=dv_prior_sw, sigma_prior=sigma_prior
    )

    trace2 = mcmc2.get_samples()

    # Give trace2 'ε' array from trace1 for plotting
    eps_scalar = float(np.asarray(frozen["ε"]))
    n2 = trace2["Δv_τk"].shape[0] # nsamples in trace2
    trace2["ε"] = np.full((n2,), eps_scalar, dtype=float)

    # Extract chi2 from best n_sols models
    n_sols = 100

    # MCMC1
    f1  = np.asarray(trace1["fλ_pred"])    # (n_samples, n_lines, n_epochs, n_points)
    dv1 = np.asarray(trace1["Δv_τk"])      # (n_samples, K, 1, n_epochs)
    S, L, E, N = f1.shape
    model_result_orig = np.empty((L, E, N), dtype=f1.dtype)
    for epoch in range(E):
        rv0 = dv1[:, 0, 0, epoch]
        center, _, _ = summarize_mode_1d(rv0)
        sols = np.argsort(np.abs(rv0 - center))[:n_sols] # get solutions closest to the center of the posterior mode
        model_result_orig[:, epoch, :] = f1[sols, :, epoch, :].mean(axis=0)
    
    #MCMC2
    f2 = np.asarray(trace2["fλ_pred"])     # (n_samples, n_lines, n_epochs, n_points)
    dv2 = np.asarray(trace2["Δv_τk"])      # (n_samples, K, n_epochs)
    model_result_switched = np.empty((L, E, N), dtype=f2.dtype)
    for epoch in range(E):
        rv0 = dv2[:, 0, epoch]
        center, _, _ = summarize_mode_1d(rv0)
        sols = np.argsort(np.abs(rv0 - center))[:n_sols]
        model_result_switched[:, epoch, :] = f2[sols, :, epoch, :].mean(axis=0)

    # Get chi2 values of both fits
    chi2_orig = get_chi2(y_fluxes, model_result_orig, y_errors)
    chi2_switched = get_chi2(y_fluxes, model_result_switched, y_errors)

    # Where epochs should be switched
    use_switched = chi2_switched < chi2_orig # (n_epochs)

    n_samp1 = trace1["Δv_τk"].shape[0]
    n_samp2 = trace2["Δv_τk"].shape[0]
    N = min(n_samp1, n_samp2)  # common sample count

    # Randomly subsample to the same length for fair stitching in final sample
    rng = np.random.default_rng(123)
    idx1 = rng.choice(n_samp1, size=N, replace=(n_samp1 < N))
    idx2 = rng.choice(n_samp2, size=N, replace=(n_samp2 < N))

    # Get all non-deterministic sites (widths, amplitudes) from first MCMC run
    stitched = {}
    for k, v in trace1.items():
        if k in ("Δv_τk", "fλ_pred", "C_λk"):   # Exclude sites affected by switch
            continue
        stitched[k] = np.asarray(v)[idx1]

    # RV samples from both MCMC runs
    dv1 = np.asarray(trace1["Δv_τk"])[idx1]  # (n_samples, K, 1, n_epochs)
    dv2 = np.asarray(trace2["Δv_τk"])[idx2] # (n_samples, K, n_epochs)
    dv2 = dv2[:, :, None, :] # (n_samples, K, 1, n_epochs)
    assert dv1.shape[2] == 1 and dv2.shape[2] == 1, f"Unexpected Δv_τk shapes: {dv1.shape}, {dv2.shape}"

    # Stitch on EPOCH axis!
    mask = np.asarray(use_switched, dtype=bool)[None, None, None, :]  # (1, 1, 1, n_epochs)
    dv_stitched = np.where(mask, dv2, dv1)  # (n_samples, K, 1, n_epochs)

    stitched["Δv_τk"] = dv_stitched

    # Recompute deterministics for final, stitched samples
    def _model_predictive(λ, σ_fλ, K, is_hline, Δv_means):
        return sb2_model(λ=λ, fλ=None, σ_fλ=σ_fλ, K=K, is_hline=is_hline, Δv_means=Δv_means)

    pred = Predictive(
        _model_predictive,
        posterior_samples=stitched,
        return_sites=("fλ_pred", "C_λk", "ε")
    )

    rng_key_pred = random.PRNGKey(42) # anything
    preds = pred(
        rng_key_pred,
        λ=x_waves,
        σ_fλ=y_errors,
        K=K,
        is_hline=is_hline,
        Δv_means=Δv_means
    )

    # Merge the recomputed deterministics for final trace
    stitched.update({k: np.asarray(v) for k, v in preds.items()})    

    # ------------------------
    # 4) Plot 
    # ------------------------
    plot_path = path + 'Line_plots'
    os.makedirs(plot_path, exist_ok=True)

    if rm_epochs is not None:
        n_epochs = n_epochs - len(rm_epochs)
        
    if chi2_plots:
        plot_lines_fit(wavelengths, lines, x_waves, y_fluxes, y_errors, n_epochs, trace1, lines_dic, shift_kms, comp_sep, plot_path, chi2_1=chi2_orig, chi2_2=chi2_switched, type_name='original', show_chi2=True) # MCMC1 result
        plot_lines_fit(wavelengths, lines, x_waves, y_fluxes, y_errors, n_epochs, trace2, lines_dic, shift_kms, comp_sep, plot_path, chi2_1=chi2_orig, chi2_2=chi2_switched, type_name='switched', show_chi2=True) # MCMC2 result
        plot_lines_fit(wavelengths, lines, x_waves, y_fluxes, y_errors, n_epochs, stitched, lines_dic, shift_kms, comp_sep, plot_path, chi2_1=chi2_orig, chi2_2=chi2_switched, type_name='final', show_chi2=True) # stitched final result
    else:
        plot_lines_fit(wavelengths, lines, x_waves, y_fluxes, y_errors, n_epochs, stitched, lines_dic, shift_kms, comp_sep, plot_path, chi2_1=chi2_orig, chi2_2=chi2_switched, type_name='final', show_chi2=False)
    return stitched, x_waves, y_fluxes

def plot_lines_fit(
    wavelengths, lines, x_waves, y_fluxes, y_errors, n_epochs, trace, lines_dic, shift_kms, comp_sep, path,
    chi2_1, chi2_2, type_name, show_chi2=True):
    """
    Plot SB2 line-fit results using ONE global best posterior sample (s_star),
    chosen by minimizing total χ² across all epochs. For each epoch, we plot
    that sample's direct predictions (total and components). No reference epoch
    or template shifting is used.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredText

    f_pred = np.asarray(trace['fλ_pred'])   # (S, L, E, N) total flux prediction
    C_lk   = np.asarray(trace['C_λk'])      # (S, 2, L, E, N) components profiles (no continuum)
    eps    = np.asarray(trace['ε'])         # (S,) continuum

    S, L, E, N = f_pred.shape

    # stack data & weights for chi^2 determination
    y_arr = np.stack([np.asarray(y_fluxes[li]) for li in range(L)], axis=0)  # (L, E, N)
    y_err = np.stack([np.asarray(y_errors[li]) for li in range(L)], axis=0)  # (L, E, N)
    w     = 1.0 / np.clip(y_err, 1e-12, None)**2

    # per-sample, per-epoch chi^2 across all lines
    resid = f_pred - y_arr[None, ...]                     # (S, L, E, N)
    chi2  = np.sum(resid * resid * w[None, ...], axis=(1, 3))  # (S, E)

    # choose single global best sample (min total chi^2 over epochs)
    s_star = int(np.argmin(np.sum(chi2, axis=1)))

    # ======================== PLOT ========================
    for idx, line in enumerate(lines):
        print(f'Plotting {type_name} fits for line:', line)
        fig, axes = setup_fits_plots(wavelengths)

        centre = rv_shift_wavelength(lines_dic[line]['air'][0], shift_kms)

        for e, ax in enumerate(axes.ravel()[:n_epochs]):
            x = x_waves[idx, e, :]
            y = y_fluxes[idx, e, :]

            # s_star results
            tot  = f_pred[s_star, idx, e, :] # total prediction
            cont = float(np.squeeze(eps[s_star]))
            c1   = cont + C_lk[s_star, 0, idx, e, :] # component 1 prediction (profile + continuum)
            c2   = cont + C_lk[s_star, 1, idx, e, :] # component 2 prediction (profile + continuum)

            # plot data
            ax.plot(x, y, color='k', lw=1, alpha=0.9, antialiased=False, zorder=5)

            # plot models
            ax.plot(x, c1, alpha=0.95, color='C1', antialiased=False, lw=2, zorder=2)
            ax.plot(x, c2, alpha=0.95, color='C0', antialiased=False, lw=2, zorder=1)
            ax.plot(x, tot, alpha=0.95, color='C2', antialiased=False, lw=3)

            # reference line, labels
            ax.axvline(centre, color='r', linestyle='--', lw=1, alpha=0.8)
            ax.set_xlim(centre - 13, centre + 13)
            ax.text(0.1, 0.1, f'Epoch {e+1}', transform=ax.transAxes, fontsize=16)
            ax.set_rasterization_zorder(0)

            if show_chi2: # OPTIONAL: Display results of chi^2 comparison between MCMC runs from fit_sb2_probmod
                if chi2_1[e] < chi2_2[e]:
                    text_box = AnchoredText(f'  χ2 \n *{chi2_1[e]:.2f}*\n  {chi2_2[e]:.2f}', frameon=True, loc=4, pad=0.5)
                else:
                    text_box = AnchoredText(f'  χ2 \n  {chi2_1[e]:.2f} \n*{chi2_2[e]:.2f}*', frameon=True, loc=4, pad=0.5)
                plt.setp(text_box.patch, facecolor='white', alpha=0.5)
                ax.add_artist(text_box)

        # Legend
        custom_lines = [
            Line2D([0], [0], color='C2', lw=2),
            Line2D([0], [0], color='C1', lw=2),
            Line2D([0], [0], color='C0', lw=2)
        ]
        fig.legend(
            custom_lines,
            ['Total Prediction', 'Component 1', 'Component 2'],
            loc='lower center',
            bbox_to_anchor=(0.5, 0.03),
            ncol=3,
            frameon=False,
            fontsize=14,
            borderaxespad=0.0,
            columnspacing=1.5,
            handlelength=2.5,
        )

        fig.supxlabel('Wavelength [Å]', fontsize=24, y=-0.005)
        fig.supylabel('Flux', fontsize=24, x=0.01)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.118, wspace=0, hspace=0)
        plt.savefig(os.path.join(path, f'{type_name}_{line}_fits_SB2_.png'), dpi=400, bbox_inches='tight')
        plt.close()

def mcmc_results_to_file(trace, names, jds, writer, csvfile, rm_epochs):
    """
    Write MCMC fit results for multiple components and epochs to a CSV file.
    
    For each component (assumed to be two components) and for each epoch (from the 
    provided 'names' list) the function calculates the mean RV and its uncertainty 
    from the MCMC trace and writes these values alongside the epoch, MJD, and strategy
    used to extract values from the posterior (HDI or mode-aware, see hdi_summary and summarize_mode_1d).
    
    Parameters:
        trace (dict): Dictionary containing MCMC samples (expects key 'Δv_τk').
        names (list): List of epoch identifiers (e.g., names of spectra or observation epochs).
        jds (list): List of corresponding Julian Dates (or None if unavailable).
        writer (csv.DictWriter or None): A DictWriter object or None (if first call, will be initialized).
        csvfile: An open CSV file handle to write the output.
        rm_epochs (list): indices of epochs discounted from the fitting procedure.
        
    Returns:
        writer: A csv.DictWriter instance after writing the header (if initially None) and all rows.
    """
    if rm_epochs is not None:
        names = [x for i, x in enumerate(names) if i not in rm_epochs]
        jds = [x for i, x in enumerate(jds) if i not in rm_epochs]

    n_epochs_to_write = len(names)
    # Loop over the two components (0 and 1, output as 1 and 2)
    for i in range(2):
        for j in range(n_epochs_to_write):
            results_dict = {}
            results_dict['epoch'] = names[j]
            if jds is not None and j < len(jds):
                results_dict['MJD'] = jds[j]

            # Extract 1D samples for component i, epoch j
            vals = np.squeeze(np.asarray(trace['Δv_τk'][:, i, :, j]))
            rv_val, rv_err, mode = summarize_mode_1d(vals)

            results_dict['mean_rv'] = rv_val
            results_dict['mean_rv_er'] = rv_err
            results_dict['comp'] = i + 1
            results_dict['posterior_method'] = mode

            if writer is None:
                fieldnames = results_dict.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(results_dict)

    return writer

def SLfit(spectra_list, data_path, save_path, lines, K=2, file_type='fits', instrument='FLAMES',
          plots=True, balmer=True, neblines=[], doubem=[], SB2=False, init_guess_shift=0, sigma_prior=20,
          shift_kms=0, use_init_pars=False, rm_epochs=None, cornerplots=True, chi2_plots=False, profile='Voigt'):
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
    wavelengths, fluxes, f_errors, names, jds = read_spectra(spectra_list, data_path, file_type, instrument=instrument, SB2=SB2)
    
    # Setup the output directory and save the JD information if available
    out_path = setup_star_directory_and_save_jds(names, jds, save_path, SB2)
    
    # Get the dictionary with line regions and initial parameters
    lines_dic = setup_line_dictionary()

    # Verify that user‑requested lines exist in the dictionary
    missing = [ln for ln in lines if ln not in lines_dic]
    if missing:
        print("Error: Unknown spectral line identifier(s):", missing)
        print("Available lines are:", sorted(lines_dic.keys()))
        raise ValueError(f"Please choose from the available lines or add your own. Missing: {missing}")

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
        
        print('Fitting all lines simultaneously')
        if SB2:
            # SB2 fitting: fit all lines using the probabilistic SB2 model and write results to CSV
            result, x_wave, y_flux = fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic,
                                                      Hlines, neblines, out_path, K=K, shift_kms=shift_kms, 
                                                      rm_epochs=rm_epochs, chi2_plots=chi2_plots, profile=profile, sigma_prior=sigma_prior)
            writer = mcmc_results_to_file(result, names, jds, writer, csvfile, rm_epochs=rm_epochs)

            if cornerplots == True:
                for k in range(K):   # typically K=2, but robust if different
                    make_rv_corner_for_component(result['Δv_τk'], k, out_path)
            
            # (Optional plotting of SB2 fits is handled within fit_sb2_probmod and plot_lines_fit)
        else:
            # SB1 fitting: using probabilistic method
            result, x_waves, y_fluxes = fit_sb1_probmod(lines, wavelengths, fluxes, f_errors, lines_dic,
                                                        Hlines, neblines, out_path, shift_kms=shift_kms,
                                                        rm_epochs=rm_epochs, profile=profile)
            writer = mcmc_results_to_file_sb1(result, names, jds, writer, csvfile, rm_epochs=rm_epochs)

        plt.close('all')
        
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

    def __init__(self, fit_values, path, JDfile, balmer=False, SB2=False, lines_dic=None,  
                 wavelength_type='air', use_lines=None, lines_ok_thrsld=3, epochs_ok_thrsld=3, 
                 print_output=True, random_eps=False, rndm_eps_n=29, rndm_eps_exc=[], 
                 plots=True, error_type='wid1_percer', rm_epochs=False):
        """
        Initialize the GetRVs instance and load the SLfit results. Computes percentage errors
        for the primary (and secondary, if SB2) components.

        Parameters:
            fit_values (str): Filename of the CSV file with SLfit fit values.
            path (str): Directory path for reading and writing output.
            JDfile (str): Filename for Julian Date data.
            balmer (bool): Whether to apply Balmer-specific settings.
            SB2 (bool): Whether the fits correspond to double-lined spectra.
            lines_dic (dict, optional): A dictionary mapping spectral line IDs to properties.
                                        If None, the default dictionary from
                                        `ravel.setup_line_dictionary()` is used. Defaults to None.
            wavelength_type (str): Specifies whether to use 'vacuum' or 'air' rest wavelengths 
                                    from lines_dic. Defaults to 'air'.
            use_lines (list or None): Specific lines to use; if None, all lines are considered.
            lines_ok_thrsld (float): Threshold for acceptable line errors.
            epochs_ok_thrsld (float): Threshold for acceptable epoch errors.
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
        self.lines_dic = lines_dic
        self.wavelength_type = wavelength_type
        self.use_lines = use_lines
        self.lines_ok_thrsld = lines_ok_thrsld
        self.epochs_ok_thrsld = epochs_ok_thrsld
        self.print_output = print_output
        self.random_eps = random_eps
        self.rndm_eps_n = rndm_eps_n
        self.rndm_eps_exc = rndm_eps_exc
        self.plots = plots
        self.error_type = error_type
        self.rm_epochs = rm_epochs

        # Get the current date for reference
        date_current = str(date.today())

        if wavelength_type not in ['vacuum', 'air']:
            raise ValueError("wavelength_type must be either 'vacuum' or 'air'.")

        if lines_dic is None:
            # Import here if GetRVs is in the same file as setup_line_dictionary
            # Or adjust import path if it's elsewhere
            try:
                # Assuming setup_line_dictionary is accessible in the current scope
                # If GetRVs is in ravel.py, this works directly.
                # If not, you might need 'from . import setup_line_dictionary' or similar
                self.lines_dic = setup_line_dictionary()
                print("Using default line dictionary from ravel.setup_line_dictionary().")
            except NameError:
                # Handle case where the function isn't directly available
                # This might require a specific import based on your project structure
                raise ImportError("Could not find setup_line_dictionary(). "
                                  "Ensure it's imported or provide a lines_dic.")
        else:
            # Use the user-provided dictionary
            self.lines_dic = lines_dic
            print("Using user-provided line dictionary.")

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

    # def outlier_killer(self, data, thresh=2, print_output=None):
    #     """
    #     Identify inliers and outliers from a data array using a MAD-based test.
        
    #     Parameters:
    #         data (array-like): Numerical data to test.
    #         thresh (float): Threshold multiplier for the MAD.
    #         print_output (bool or None): Whether to print details (default uses self.print_output).
        
    #     Returns:
    #         tuple: (inliers, outliers) where both are lists of indices.
    #     """
    #     if print_output is None:
    #         print_output = self.print_output
    #     diff = data - np.nanmedian(data)
    #     # Output debug information to file and/or console
    #     with open(self.path + 'rv_stats.txt', 'a') as f:
    #         if print_output:
    #             print('     mean(x)        =', f'{np.nanmean(data):.3f}')
    #             print('     mean(x)        =', f'{np.nanmean(data):.3f}', file=f)
    #             print('     median(x)      =', f'{np.nanmedian(data):.3f}')
    #             print('     median(x)      =', f'{np.nanmedian(data):.3f}', file=f)
    #             print('     x-median(x)    =', [f'{x:.3f}' for x in diff])
    #             print('     x-median(x)    =', [f'{x:.3f}' for x in diff], file=f)
    #             print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff])
    #             print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff], file=f)
    #         mad = 1.4826 * np.nanmedian(np.abs(diff))
    #         if print_output:
    #             print('     mad =', f'{mad:.3f}')
    #             print('     mad =', f'{mad:.3f}', file=f)
    #             print('     abs(x-median(x)) <', f'{thresh * mad:.3f}', '(thresh*mad) =',
    #                   [abs(x) < thresh * mad for x in diff])
    #             print('     abs(x-median(x)) <', f'{thresh * mad:.3f}', '(thresh*mad) =',
    #                   [abs(x) < thresh * mad for x in diff], file=f)
    #     inliers = [i for i, x in enumerate(data) if abs(x - np.nanmedian(data)) < thresh * mad]
    #     outliers = [i for i in range(len(data)) if i not in inliers]
    #     return inliers, outliers

    def outlier_killer(self, data, thresh=2, max_iter=5, print_output=None):
        """
        Iterative MAD-based clipper with diagnostics.

        Parameters
        ----------
        data : array-like
            1-D numeric input (NaNs are ignored automatically).
        thresh : float
            Clipping level in units of MAD.  Typical = 2–3.
        max_iter : int
            Maximum number of clipping rounds (safety valve).
        print_output : bool or None
            Verbose output flag.  If None, falls back to self.print_output.

        Returns
        -------
        inliers, outliers : list[int]
            Index lists referring to *data*.
        """
        if print_output is None:
            print_output = getattr(self, "print_output", False)

        data = np.asarray(data, dtype=float)
        idx_all = np.arange(data.size)       # master index tracker
        mask    = np.isfinite(data)          # start by discarding NaNs
        out_tot = []                         # store every removed index

        # ---------------------------------------------------------------------
        # open the diagnostics file once; append everything that follows
        # ---------------------------------------------------------------------
        diag_path = os.path.join(self.path, "rv_stats.txt")
        with open(diag_path, "a") as log:

            def _writeln(*args, **kwargs):
                """Helper: write both to console and file when print_output is on."""
                if print_output:
                    print(*args, **kwargs)
                    print(*args, **kwargs, file=log)

            _writeln("\n=== outlier_killer call ===============================")
            _writeln(f"input data: {[round(x,3) for x in data]}")
            _writeln(f"threshold  : {thresh} MAD   (max_iter = {max_iter})")

            for it in range(1, max_iter + 1):
                work = data[mask]
                if work.size == 0:
                    break

                med = np.nanmedian(work)
                mad = 1.4826 * np.nanmedian(np.abs(work - med))

                _writeln(f"\n-- iteration {it} --")
                _writeln(f"   n (working)      = {work.size}")
                _writeln(f"   median           = {med:.3f}")
                _writeln(f"   MAD              = {mad:.3f}")

                if mad == 0:                       # all remaining points identical
                    _writeln("   MAD == 0 → stop.")
                    break

                z = np.abs((work - med) / mad)
                bad_local = np.where(z > thresh)[0]             # indices in *work*
                if bad_local.size == 0:
                    _writeln("   no points exceed threshold → finished.")
                    break

                bad_global = idx_all[mask][bad_local]           # map to original
                _writeln(f"   removing indices {bad_global.tolist()}  "
                        f"(|z| = {z[bad_local].round(2).tolist()})")

                mask[bad_global] = False          # mark as outlier
                out_tot.extend(bad_global.tolist())

            # summary
            _writeln("\n   ==> kept indices   :", idx_all[mask].tolist())
            _writeln("   ==> clipped indices :", out_tot)
            _writeln("========================================================\n")

        return idx_all[mask].tolist(), out_tot

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

    def compute_rvs(self, lines):
        """
        Compute radial velocities (and uncertainties) for each spectral line.
        
        The method compares fitted central wavelengths (and errors) to the provided
        rest wavelengths, propagating errors appropriately. 
        
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
        # Determine the key to use based on the chosen wavelength type
        key = 'centre' if self.wavelength_type == 'vacuum' else 'air'

        for line in lines:

            # Select rest wavelength and error from self.lines_dic
            if line not in self.lines_dic:
                print(f"  Warning: Line {line} not found in lines_dic. Skipping RV computation.")
                continue
            line_info = self.lines_dic[line]

            # Check if the required key exists and has valid data (list with at least 2 elements)
            if key not in line_info or not isinstance(line_info[key], list) or len(line_info[key]) < 2:
                print(f"  Warning: Wavelength type '{self.wavelength_type}' ('{key}' key) "
                      f"not available, empty, or incomplete for line {line}. Skipping RV computation.")
                continue

            lambda_rest, lambda_r_er = line_info[key] # Get wavelength and its error

            # Check if rest wavelength is valid
            if lambda_rest is None or lambda_rest <= 0:
                 print(f"  Warning: Invalid rest wavelength ({lambda_rest}) for line {line} and type '{self.wavelength_type}'. Skipping.")
                 continue
            # Ensure error is a non-negative float, default to 0.1 if None or invalid
            if lambda_r_er is None or not isinstance(lambda_r_er, (int, float)) or lambda_r_er < 0:
                lambda_r_er = 0.1

            current_line_values = self.df_SLfit[self.df_SLfit['line'] == line]
            if current_line_values.empty:
                 print(f"  Warning: No SLfit data found for line {line}. Skipping RV computation.")
                 continue

            # Ensure fitted center and error columns exist and handle potential NaNs
            if 'cen1' not in current_line_values.columns or 'cen1_er' not in current_line_values.columns:
                 print(f"  Warning: Missing 'cen1' or 'cen1_er' column for line {line}. Skipping primary RV.")
                 continue
            
            cen1_vals = current_line_values['cen1'].values
            cen1_er_vals = current_line_values['cen1_er'].values
            dlambda1 = cen1_vals - lambda_rest
            dlambda1_er = np.sqrt(cen1_er_vals**2 + lambda_r_er**2)
            rv1 = dlambda1 * c_kms / lambda_rest
            rv1_er = np.sqrt((dlambda1_er / dlambda1) ** 2 + (lambda_r_er / lambda_rest) ** 2) * np.abs(rv1)
            rvs[line] = {'rv1': rv1, 'rv1_er': rv1_er}

        return rvs

    @staticmethod
    def print_error_stats(grouped_error, error_type):
        """
        Print the mean and median values for a grouped error dictionary.
        
        Parameters:
            grouped_error (dict): Error values grouped by spectral line.
            error_type (str): A label describing the error type.
        """
        mean_values = [f'{np.mean(value):6.3f}' for value in grouped_error]
        median_values = [f'{np.median(value):6.3f}' for value in grouped_error]
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

        if self.print_output:
            print('*** SB2 set to:', self.SB2, '***\n')
            print('\n*** Computing Radial Velocities ***')
            print('-----------------------------------')

        # Compute per-line RVs based on the SLfit results and rest wavelengths.
        rvs_dict = self.compute_rvs(self.lines)

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
                print(error_type)
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
        data = pd.read_csv(self.path + 'RVs1.txt', sep=r'\s+')
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

def lomb_scargle(df, path, SB2=False, print_output=True, plots=True, best_lines=False, Pfold=True, fold_rv_curve=True, 
                 save_power_spectrum=False, starname=None):
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
        save_power_spectrum (bool, optional): If True, save the power spectrum data to a file (default is False).
    
    Returns:
        dict: A dictionary (`ls_results`) containing periodogram outputs, including frequency grid,
              power spectrum, false alarm probabilities and levels, detected peaks, and best period estimates.
    """
    # Ensure LS output directory exists
    ls_path = os.path.join(path, 'LS')
    if not os.path.exists(ls_path):
        os.makedirs(ls_path)
    
    # Select times and RVs from DataFrame; try both 'JD' and 'MJD'
    if 'comp' not in df.columns:
        hjd1, rv1 = df['MJD'], df['mean_rv']
        rv1_err = df['mean_rv_er'] if 'mean_rv_er' in df.columns else None
    else:
        try:
            hjd1, rv1 = df['JD'][df['comp'] == 1], df['mean_rv'][df['comp'] == 1]
            rv1_err = df['mean_rv_er'][df['comp'] == 1] if 'mean_rv_er' in df.columns else None
        except:
            hjd1, rv1 = df['MJD'][df['comp'] == 1], df['mean_rv'][df['comp'] == 1]
            rv1_err = df['mean_rv_er'][df['comp'] == 1] if 'mean_rv_er' in df.columns else None
    
    if starname == None:
        starname = df['epoch'][0].split('_')[0] + '_' + df['epoch'][0].split('_')[1]
        starname = starname.split('/')[-1]
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
                  'max_power': {},
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
        plt.tight_layout()
        plt.savefig(f'{path}LS/RVs_MJD_{starname}.png', dpi=300)
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
                plt.tight_layout()
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
                plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig(f'{path}LS/LS_{starname}_points.png', dpi=300, bbox_inches='tight')
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
        plt.tight_layout()
        plt.savefig(f'{path}LS/LS_{starname}_frequency.pdf')
        plt.close()

        if save_power_spectrum:
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
                    phase_results = results2
                else:
                    results1, phase, vel1, vel1_err  = phase_rv_curve(hjd1, rv1, rv1_err, period=period)
                    fine_preds1, fine_preds2 = [], []
                    for i in range(results1['K'].shape[0]):
                        fine_pred1 = results1['K'][i] * np.sin(2 * np.pi * fine_phase + results1['phase_shift'][i]) \
                            + results1['gamma'][i]
                        fine_preds1.append(fine_pred1)
                    fine_preds1 = np.array(fine_preds1)
                    phase_results = results1

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
    
    return ls_results, phase_results

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
        # amplitude = npro.sample('amplitude', dist.Uniform(0, 500))
        amplitude = npro.sample('K', dist.Normal(0, 500))
        phase_shift = npro.sample('phase_shift', dist.Uniform(-jnp.pi, jnp.pi))
        # height = npro.sample('height', dist.Uniform(50, 250))
        rv_min, rv_max = jnp.min(rvs), jnp.max(rvs)
        height = npro.sample('gamma', dist.Uniform(rv_min-10, rv_max+10))
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

def fit_sinusoidal_probmod_sb2(phase, rv1, rv1_err, rv2, rv2_err, amp_max=500):
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
        rv_min, rv_max = jnp.min(rv1), jnp.max(rv1)
        gamma = npro.sample('gamma', dist.Uniform(rv_min-10, rv_max+10))
        # Distinct parameters
        K1 = npro.sample('K1', dist.HalfNormal(amp_max)) # Ensures K1 > 0
        delta_K = npro.sample('delta_K', dist.HalfNormal(amp_max))
        K2 = npro.deterministic('K2', K1 + delta_K) # Ensures K2 > K1 > 0
        # K1 = npro.sample('K1', dist.Normal(0, amp_max))
        # K2 = npro.sample('K2', dist.Normal(0, amp_max))
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

def plot_pair_scatter(samples, savepath=None):
    """ Diagnostic for bimodal posterior
        Plots each RV1 vs. RV2 from each trace for each epoch of MCMC result.
    """
    rv1, rv2 = samples['rv1'], samples['rv2']
    n_samples, n_epochs = rv1.shape
    cmap = plt.cm.get_cmap("tab10", n_epochs)

    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = min(rv1.min(), rv2.min())
    vmax = max(rv1.max(), rv2.max())
    for t in range(n_epochs):
        ax.scatter(rv1[:, t], rv2[:, t], s=4, alpha=0.5, color=cmap(t), label=f'Epoch {t+1}')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=0.6)
    ax.set_xlabel("RV1 [km/s]")
    ax.set_ylabel("RV2 [km/s]")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_title("RV1 vs RV2 for all epochs")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.savefig(savepath + f'pair_scatter.png', bbox_inches='tight')
    plt.close(fig)

def plot_sign_trace(samples, savepath=None, ax=None):
    """ Diagnostic for bimodal posteriod
        Plots the sign of RV1-RV2 for each sample in each epoch to detect evidence of sign switching.
    """
    rv1, rv2 = samples['rv1'], samples['rv2']
    n_samples, n_epochs = rv1.shape
    sign_matrix = np.sign(rv1 - rv2)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("tab10", n_epochs)

    for t in range(n_epochs):
        ax.plot(sign_matrix[:, t], label=f'Epoch {t+1}', color=cmap(t), lw=0.6)

    ax.set_xlabel("MCMC draw")
    ax.set_ylabel("sign of ΔRV")
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, 0, 1])
    ax.axhline(0, ls='--', color='gray', lw=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    ax.set_title("Sign of RV1 − RV2")

    plt.savefig(savepath + f'sign_trace.png', bbox_inches='tight')
    plt.close(fig)
