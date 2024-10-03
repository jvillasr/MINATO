import sys
import copy
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
from matplotlib.lines import Line2D
from lmfit import Model, Parameters, models, minimize, report_fit
from datetime import date
from scipy.optimize import basinhopping
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpyro as npro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
from jax import numpy as jnp
from jax import random

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return -amp*jnp.exp(-(x-cen)**2 /(2*(wid/2.355)**2))

def lorentzian(x, amp, cen, wid):
    "1-d lorentzian: lorentzian(x, amp, cen, wid)"
    return -amp*(wid**2/( 4*(x-cen)**2 + wid**2 ))

def nebu(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return amp*jnp.exp(-(x-cen)**2 /(2*(wid/2.355)**2))

def double_gaussian_with_continuum(x, amp1, cen1, wid1, amp2, cen2, wid2, cont):
    return (-amp1 * np.exp(-(x - cen1)**2 / (2*(wid1/2.355)**2)) -
            amp2 * np.exp(-(x - cen2)**2 / (2*(wid2/2.355)**2)) + cont)
    
def sinu(x, A, w, phi, h):
    "Sinusoidal: sinu(data, amp, freq, phase, height)"
    return A*jnp.sin(w*(x-phi))+h

def read_fits(fits_file):
    with fits.open(fits_file) as hdul:
        # hdul.info()
        header = hdul[0].header
        star_epoch = header['OBJECT']+'_'+ header['EPOCH_ID']
        mjd = header['MJD_MID']
        wave = hdul[1].data['WAVELENGTH']
        flux = hdul[1].data['SCI_NORM']
        ferr = hdul[1].data['SCI_NORM_ERR']
        return wave, flux, ferr, star_epoch, mjd

def read_spectra(filelist, path, file_type):
    wavelengths, fluxes, f_errors, names, jds = [], [], [], [], []
    for spec in filelist:
        if file_type in ['dat', 'txt', 'csv']:
            names.append(spec.replace(f'.{file_type}', ''))
            df = pd.read_csv(path + spec, sep='\t', header=None)
            wavelengths.append(np.array(df[0]))
            fluxes.append(np.array(df[1]))
            try:
                f_errors.append(np.array(df[2]))
            except:
                f_errors.append(1 / (np.array(df[1])) ** 2)
            jds = None
        elif file_type == 'fits':
            
            wave, flux, ferr, star, mjd = read_fits(spec)
            wavelengths.append(wave)
            fluxes.append(flux)
            f_errors.append(ferr)
            names.append(star)
            jds.append(mjd)
    return wavelengths, fluxes, f_errors, names, jds

def setup_star_directory_and_save_jds(names, jds, path, SB2):
    star = names[0].split('_')[0] + '_' + names[0].split('_')[1] + '/'
    path = path.replace('FITS/', '') + star
    if SB2==True:
        path = path + 'SB2_2/'
    if not os.path.exists(path):
        os.makedirs(path)
    if jds:    
        df_mjd = pd.DataFrame()
        df_mjd['epoch'] = names
        df_mjd['JD'] = jds
        df_mjd.to_csv(path + 'JDs.txt', index=False, header=False, sep='\t')
    return path

def setup_line_dictionary():
    lines_dic = {
        3995: { 'region':[3990, 4005], 'centre':None, 'wid_ini':2, 'title':'N II $\lambda$3995'},
        4009: { 'region':[4005, 4018], 'centre':[4009.2565, 0.00002], 'wid_ini':3, 'title':'He I $\lambda$4009'},
        4026: { 'region':[4017, 4043], 'centre':[4026.1914, 0.0010], 'wid_ini':3, 'title':'He I $\lambda$4026'},
        4102: { 'region':[4085, 4120], 'centre':[4101.734, 0.006], 'wid_ini':5, 'title':'H$\delta$'},
        4121: { 'region':[4118, 4130], 'centre':[4120.8154, 0.0012], 'wid_ini':3, 'title':'He I $\lambda$4121'},
        4128: { 'region':[4124, 4136], 'centre':[4128.07, 0.10], 'wid_ini':2, 'title':'Si II $\lambda$4128'},
        4131: { 'region':[4128, 4140], 'centre':[4130.89, 0.10], 'wid_ini':2, 'title':'Si II $\lambda$4131'},
        4144: { 'region':[4135, 4160], 'centre':[4143.761, 0.010], 'wid_ini':3, 'title':'He I $\lambda$4144'},
        4233: { 'region':[4229, 4241], 'centre':None, 'wid_ini':2, 'title':'Fe II $\lambda$4233'},
        4267: { 'region':[4263, 4275], 'centre':[4267.258, 0.007], 'wid_ini':2, 'title':'C II $\lambda$4267'},
        4340: { 'region':[4320, 4360], 'centre':[4340.472, 0.006], 'wid_ini':6, 'title':'H$\gamma$'},
        4388: { 'region':[4380, 4405], 'centre':[4387.9296, 0.0006], 'wid_ini':3, 'title':'He I $\lambda$4388'},
        4471: { 'region':[4462, 4487], 'centre':[4471.4802, 0.0015], 'wid_ini':3, 'title':'He I $\lambda$4471'},
        4481: { 'region':[4478, 4490], 'centre':[4481.130, 0.010], 'wid_ini':2, 'title':'Mg II $\lambda$4481'},
        4542: { 'region':[4537, 4552], 'centre':[4541.591, 0.010], 'wid_ini':3, 'title':'He II $\lambda$4542'},
        4553: { 'region':[4547, 4562], 'centre':[4552.62, 0.10], 'wid_ini':3, 'title':'Si III $\lambda$4553'},
        4861: { 'region':[4840, 4875], 'centre':[4861.35, 0.05], 'wid_ini':5, 'title':'H$\beta$'}, 
        4922: { 'region':[4915, 4930], 'centre':[4921.9313, 0.0005], 'wid_ini':4, 'title':'He I $\lambda$4922'}, 
        5412: { 'region':[5405, 5419], 'centre':[5411.52, 0,10], 'wid_ini':4, 'title':'He II $\lambda$5412'},
        5876: { 'region':[5865, 5888], 'centre':[5875.621, 0.010], 'wid_ini':4, 'title':'He I $\lambda$5876'},  
        5890: { 'region':[5881, 5905], 'centre':[5889.951, 0.00003], 'wid_ini':3, 'title':'Na I $\lambda$5890'}, 
        6562: { 'region':[6542, 6583], 'centre':[6562.79, 0.030], 'wid_ini':5, 'title':'H$\alpha$'}, 
        6678: { 'region':[6668, 6690], 'centre':[6678.151, 0.010], 'wid_ini':4, 'title':'He I $\lambda$6678'}, 
        7774: { 'region':[7762, 7786], 'centre':[7774.17, 0,10], 'wid_ini':3, 'title':'O I $\lambda$7774'}
    }
    return lines_dic

def initialize_fit_variables(lines):
    cen1, cen1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    amp1, amp1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    wid1, wid1_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    cen2, cen2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    amp2, amp2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    wid2, wid2_er = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    dely, sdev = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    results, comps = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    delta_cen, chisqr = [[] for _ in range(len(lines))], [[] for _ in range(len(lines))]
    return (
        cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
        cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
        dely, sdev, results, comps, delta_cen, chisqr
    )

def setup_fits_plots(wavelengths):
    nplots = len(wavelengths)
    ncols = int(np.sqrt(nplots))
    nrows = nplots // ncols
    if ncols * nrows < nplots:
        nrows += 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0., hspace=0.)
    axes = axes.flatten()
    return fig, axes

def fit_sb2(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, axes, path):
    # Trim the data to the region of interest
    region_start = lines_dic[line]['region'][0]
    region_end = lines_dic[line]['region'][1]
    x_waves = [wave[(wave > region_start) & (wave < region_end)] for wave in wavelengths]
    y_fluxes = [flux[(wave > region_start) & (wave < region_end)] for flux, wave in zip(fluxes, wavelengths)]
    y_errors = [f_err[(wave > region_start) & (wave < region_end)] for f_err, wave in zip(f_errors, wavelengths)]

    # Initial guess for the central wavelength and width
    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']
    amp_ini = 0.9 - min([flux.min() for flux in y_fluxes])

    # Define the objective function for global optimization
    def objective(params, x, data, errors):
        resid = []
        for i, (x_vals, data_vals, err_vals) in enumerate(zip(x, data, errors)):
            cen1 = np.float64(params[f'cen1_{i+1}'].value)
            cen2 = np.float64(params[f'cen2_{i+1}'].value)
            model = double_gaussian_with_continuum(x_vals, np.float64(params['amp1'].value), cen1, np.float64(params['wid1'].value), 
                                                np.float64(params['amp2'].value), cen2, np.float64(params['wid2'].value), np.float64(params['cont'].value))
            resid.append((data_vals - model) / err_vals)  # Weighted residuals
        return np.concatenate(resid)  # Return the weighted residual array

    # Scalar objective function for basinhopping
    def scalar_objective(params, x, data, errors):
        return np.sum(objective(params, x, data, errors)**2)  # Sum of squares of weighted residuals

    # Initialize parameters for each epoch
    def initialize_params(wavelengths):
        params = Parameters()
        params.add('amp1', value=amp_ini, min=0.05, max=1)
        params.add('wid1', value=wid_ini, min=0.5, max=10.0)
        params.add('amp2', value=amp_ini*0.7, min=0.05, max=1)
        params.add('wid2', value=wid_ini, min=0.5, max=10.0)
        params.add('cont', value=1.0, min=0.9, max=1.1)
        for i in range(len(wavelengths)):
            params.add(f'cen1_{i+1}', value=np.random.uniform(cen_ini-5, cen_ini+5), min=cen_ini-10, max=cen_ini+10)
            params.add(f'cen2_{i+1}', value=np.random.uniform(cen_ini-5, cen_ini+5), min=cen_ini-10, max=cen_ini+10)
        return params

    # Perform global optimization using Basin Hopping
    def global_optimization(wavelengths, fluxes, flux_errors, params):
        initial_params = params_to_array(params)
        minimizer_kwargs = {"method": "L-BFGS-B", "args": (wavelengths, fluxes, flux_errors, params)}
        result = basinhopping(scalar_objective_wrapper, initial_params, minimizer_kwargs=minimizer_kwargs, niter=100)
        array_to_params(result.x, params)
        return params

    # Perform local optimization using lmfit
    def local_optimization(wavelengths, fluxes, flux_errors, params):
        result = minimize(objective, params, args=(wavelengths, fluxes, flux_errors))
        return result

    # Convert lmfit Parameters to a flat array
    def params_to_array(params):
        return np.array([param.value for param in params.values()], dtype=np.float64)

    # Convert a flat array back to lmfit Parameters
    def array_to_params(param_array, params):
        for i, (name, param) in enumerate(params.items()):
            param.set(value=np.float64(param_array[i]))

    # Wrapper function for the scalar objective to use with scipy.optimize
    def scalar_objective_wrapper(param_array, x, data, errors, params):
        array_to_params(param_array, params)
        return scalar_objective(params, x, data, errors)  # Return the scalar value

    # Initialize and optimize parameters
    print(list(x_waves[7]), list(y_fluxes[7]))
    params = initialize_params(x_waves)
    params = global_optimization(x_waves, y_fluxes, y_errors, params)
    result = local_optimization(x_waves, y_fluxes, y_errors, params)

    # Plot the results
    # for i, (x_vals, data_vals) in enumerate(zip(x_waves, y_fluxes)):
    #     ax = axes[i]
    #     ax.plot(x_vals, data_vals, '-', label=f'Dataset {i+1}')
    #     model_fit = double_gaussian_with_continuum(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value, 
    #                                             result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value, result.params['cont'].value)
    #     component1 = gaussian(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value)
    #     component2 = gaussian(x_vals, result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value)
        
    #     ax.plot(x_vals, model_fit, '-', label=f'Fit {i+1}')
    #     ax.plot(x_vals, component1 + result.params['cont'].value, '--', label=f'Component 1 - Dataset {i+1}')
    #     ax.plot(x_vals, component2 + result.params['cont'].value, '--', label=f'Component 2 - Dataset {i+1}')
    #     # ax.legend()
    #     ax.set_xlabel('Wavelength')
    #     ax.set_ylabel('Flux')
    #     ax.set_title(f'Epoch {i+1}')

    # plt.tight_layout()
    # plt.show()
    # plt.show()

    # Save fit results
    with open(path + str(line) + '_stats.txt', 'w') as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file we created.
        report_fit(result)
        sys.stdout = original_stdout  # Reset the standard output to its original value
    
    # Print optimized parameters
    # for name, param in result.params.items():
    #     print(f"{name}: {param.value}")

    return result, x_waves, y_fluxes

def fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, path, K=2):
    '''
    Probabilistic model for SB2 line profile fitting. It uses Numpyro for Bayesian inference, 
    sampling from the posterior distribution of the model parameters using MCMC with the NUTS algorithm. 
    The model includes plates for vectorized computations over epochs and wavelengths. 
    '''
    n_lines = len(lines)
    n_epochs = len(wavelengths)
    print('Number of lines:', n_lines)
    print('Number of epochs:', n_epochs)

    # Check if lines are Hydrogen lines
    is_hline = jnp.array([line in Hlines for line in lines])

    # Interpolate fluxes and errors to the same length, but with different wavelength values
    x_waves_interp = []
    y_fluxes_interp = []
    y_errors_interp = []

    # Choose a consistent number of points for interpolation (e.g., 1000 points)
    common_grid_length = 200

    for i, line in enumerate(lines):
        region_start, region_end = lines_dic[line]['region']
        x_waves = []
        y_fluxes = []
        y_errors = []

        for wave_set, flux_set, error_set in zip(wavelengths, fluxes, f_errors):
            # Mask the regions
            mask = (wave_set > region_start) & (wave_set < region_end)
            wave_masked = wave_set[mask]
            flux_masked = flux_set[mask]
            error_masked = error_set[mask]

            # Interpolate to a common grid of the same length
            common_wavelength_grid = np.linspace(wave_masked.min(), wave_masked.max(), common_grid_length)
            interp_flux = interp1d(wave_masked, flux_masked, bounds_error=False, fill_value="extrapolate")(common_wavelength_grid)
            interp_error = interp1d(wave_masked, error_masked, bounds_error=False, fill_value="extrapolate")(common_wavelength_grid)

            x_waves.append(common_wavelength_grid)
            y_fluxes.append(interp_flux)
            y_errors.append(interp_error)

        x_waves_interp.append(x_waves)
        y_fluxes_interp.append(y_fluxes)
        y_errors_interp.append(y_errors)

    # Convert to JAX arrays (now all have the same length)
    x_waves = jnp.array(x_waves_interp)
    y_fluxes = jnp.array(y_fluxes_interp)
    y_errors = jnp.array(y_errors_interp)

    # Initial guess for the central wavelength
    cen_ini = jnp.array([lines_dic[line]['centre'][0] for line in lines])

    def sb2_model(λ=None, fλ=None, σ_fλ=None, K=K, is_hline=None):
        c_kms = c.to('km/s').value   
        # Get the number of lines, epochs, and wavelengths points
        nlines, nepochs, ndata = λ.shape

        # Setting central rest wavelength (λ_rest) for each line
        λ_rest = npro.param("λ_rest", cen_ini)
        λ_rest = λ_rest[None, :, None]   # Shape (1, nlines, 1)

        # prior on velocity shift
        Δv = npro.param('Δv', 0)
        σ_Δv = npro.param('σ_Δv', 500)

        # Continuum level ε with uncertainty
        logσ_ε = npro.sample('logσ_ε', dist.Uniform(-5, 0))
        σ_ε = jnp.exp(logσ_ε)
        ε = npro.sample('ε', dist.Normal(1.0, σ_ε))

        #amplitude prior
        #Ak_0 = [.5] * 2 + [0.1] * (K - 2)

        with npro.plate(f"k=1..{K}", K, dim=-3): # Component plate
            # Sample velocity shifts for each component and epoch
            Δv_τk = npro.sample("Δv_τk", dist.Uniform(Δv, σ_Δv), sample_shape=(nepochs,))
            Δv_τk_expanded = Δv_τk[:, :, :, jnp.newaxis]             # Shape: (K, 1, nepochs, 1)

            with npro.plate(f'λ=1..{nlines}', nlines, dim=-2): # Lines plate
                # Sample the amplitude and width for each component and line
                amp = npro.sample('𝛼_kλ', dist.Uniform(0.05, 0.5))
                amp = amp[:, :, :, jnp.newaxis]                      # Shape: (K, nlines, 1, 1)
                wid = npro.sample('σ_kλ', dist.Uniform(0.5, 3))
                wid = wid[:, :, :, jnp.newaxis]                      # Shape: (K, nlines, 1, 1)

                # Making λ_rest a deterministic variable
                λ0 = npro.deterministic("λ0", λ_rest)
                λ0 = λ0[:, :, :, jnp.newaxis]                        # Shape: (1, nlines, 1, 1)

                # Compute the shifted wavelengths
                μ = λ0 * (1 + Δv_τk_expanded / c_kms)

                # Expand wavelength and is_hline for broadcasting
                λ = λ[jnp.newaxis, :, :, :]                          # Shape: (1, nlines, nepochs, ndata)                
                is_hline = is_hline[None, :, None, None]             # Shape: (1, nlines, 1, 1)

                # Compute both profiles
                gaussian_profile = gaussian(λ, amp, μ, wid)
                lorentzian_profile = lorentzian(λ, amp, μ, wid)

                with npro.plate(f'τ=1..{nepochs}', nepochs, dim=-1): # Epoch plate

                    # Select the appropriate profile
                    comp = jnp.where(is_hline, lorentzian_profile, gaussian_profile)

                    Ck = npro.deterministic("C_λk", comp)
                    fλ_pred = npro.deterministic("fλ_pred", ε + Ck.sum(axis=0))
        
        npro.sample("fλ", dist.Normal(fλ_pred, σ_fλ), obs=fλ)

    # rendered_model = npro.render_model(model2, model_args=(x_waves, y_fluxes, y_errors), model_kwargs={'K':K, 'is_hline': is_hline},
    #              render_distributions=True, render_params=True)
    # rendered_model.render(filename=path+'output_graph.gv', format='png')

    rng_key = random.PRNGKey(0)
    # model_init = initialize_model(rng_key, sb2_model, model_args=(x_waves, y_fluxes, y_errors))
    kernel = NUTS(sb2_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
    mcmc.run(rng_key, λ=jnp.array(x_waves), fλ=jnp.array(y_fluxes), σ_fλ=jnp.array(y_errors), K=K, is_hline=is_hline)
    mcmc.print_summary()
    trace = mcmc.get_samples()
    # print(trace.keys())
    # for key in trace:
    #     print(f"{key}: {trace[key].shape}")

    n_sol = 100
    for idx, line in enumerate(lines): 
        fig, axes = setup_fits_plots(wavelengths)
        for epoch_idx, (epoch, ax) in enumerate(zip(range(n_epochs), axes.ravel())):
            # Extract the posterior samples for the total prediction
            fλ_pred_samples = trace['fλ_pred'][-n_sol:, idx, epoch, :]  # Shape: (n_sol, ndata)
            # Extract the posterior samples for the continuum
            continuum_pred_samples = trace['ε'][-n_sol:, None]
            # Extract the posterior samples for each component
            fλ_pred_comp1_samples = continuum_pred_samples + trace['C_λk'][-n_sol:, 0, idx, epoch, :]
            fλ_pred_comp2_samples = continuum_pred_samples + trace['C_λk'][-n_sol:, 1, idx, epoch, :]
            
            # Plot the posterior predictive samples without labels
            ax.plot(x_waves[idx][epoch], fλ_pred_comp1_samples.T, rasterized=True, color='C0', alpha=0.1)
            ax.plot(x_waves[idx][epoch], fλ_pred_comp2_samples.T, rasterized=True, color='C1', alpha=0.1)
            ax.plot(x_waves[idx][epoch], fλ_pred_samples.T, rasterized=True, color='C2', alpha=0.1)

            # Plot the observed data without label
            ax.plot(x_waves[idx][epoch], y_fluxes[idx][epoch], color='k', lw=1, alpha=0.8)
            
        # Create custom legend entries
        custom_lines = [
            Line2D([0], [0], color='C2', alpha=0.5, lw=2),
            Line2D([0], [0], color='C0', alpha=0.5, lw=2),
            Line2D([0], [0], color='C1', alpha=0.5, lw=2),
            Line2D([0], [0], color='k', lw=2)
        ]
        axes[0].legend(custom_lines, ['Total Prediction', 'Component 1', 'Component 2', 'Observed Data'], fontsize=10)
        fig.supxlabel('Wavelength [\AA]', size=22)
        fig.supylabel('Flux', size=22)  
        plt.savefig(path + f'{line}_fits_SB2_.png', bbox_inches='tight', dpi=150)
        plt.close()

    return trace, x_waves, y_fluxes

def fit_sb1(line, wave, flux, ferr, lines_dic, Hlines, neblines, doubem, shift):  
    """
    Fits a Gaussian or Lorentzian profile to a spectral line.

    Parameters:
    line (int): Label (wavelength) of the line to fit.
    wave (array-like): Spectrum wavelength.
    flux (array-like): Spectrum flux.
    ferr (array-like): The flux errors.
    lines_dic (dict): A dictionary containing information about the lines.
    Hlines (array-like): A list of lines to be fitted with a Lorentzian profile.
    neblines (array-like): A list of lines to be fitted with a nebular emission model.
    doubem (array-like): A list of lines to be fitted with two nebular emission models.
    shift (float): The shift to apply to the central wavelength of the line.

    Returns:
    result (ModelResult): The result of the fit.
    x_wave (array-like): The wavelengths used in the fit.
    y_flux (array-like): The fluxes used in the fit.
    wave_region (array-like): A boolean array indicating the wavelengths used in the fit.
    """
    wave_region = np.logical_and(wave > lines_dic[line]['region'][0], wave < lines_dic[line]['region'][1])

    x_wave = wave[wave_region]
    y_flux = flux[wave_region]

    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']

    pars = Parameters()
    gauss1 = Model(gaussian, prefix='g1_')
    loren1 = Model(lorentzian, prefix='l1_')
    nebem1 = Model(nebu, prefix='neb1_')
    nebem2 = Model(nebu, prefix='neb2_')
    cont = models.LinearModel(prefix='continuum_')

    # Define parameters and set initial values
    pars.update(cont.make_params())
    pars['continuum_slope'].set(0, vary=False)
    pars['continuum_intercept'].set(1, min=0.9)

    if line in Hlines:
        prefix = 'l1'
        indiv_mod = loren1
    else:
        prefix = 'g1'
        indiv_mod = gauss1

    pars.update(indiv_mod.make_params())
    pars[f'{prefix}_amp'].set(1.05-y_flux.min(), min=0.01, max=2. )
    pars[f'{prefix}_wid'].set(wid_ini, min=1, max=11)
    pars[f'{prefix}_cen'].set(cen_ini, min=cen_ini-4, max=cen_ini+4)
    mod = indiv_mod + cont
    # if line==5890:
    #     pars.add('g1_wid', value=wid_ini, min=0.1, max=4.)
    #     pars.update(gauss2.make_params())
    #     pars.add('g2_cen', value=cen_ini+6., vary=True)
    #     pars.add('g2_amp', 0.8*(1-y_flux.min()), min=0.05, max=0.5)
    #     pars.add('g2_wid', value=wid_ini-1.5, min=0.1, max=4.)
    #     mod = gauss1 + gauss2 + cont

    # Define a list of nebular emission models
    nebem_models = [nebem1, nebem2]
    # Loop over the nebular emission models
    for i, nebem in enumerate(nebem_models, start=1):
        if line in neblines:
            pars.update(nebem.make_params())
            # if y_flux.max() < 1.2:
            #     pars['neb_amp'].set((1-y_flux.min())/2, min=0.001, max=0.5)
            # else:
            #     pars['neb_amp'].set((y_flux.max()-y_flux.min())/1.1, min=0.05)#, max=65)
            pars[f'neb{i}_amp'].set(y_flux.max()-y_flux.min(), min=0.01)
            pars[f'neb{i}_wid'].set(1, min=0.05, max=2)
            if i == 2:
                pars[f'neb{i}_cen'].set(cen_ini+0.2)
            else:
                pars[f'neb{i}_cen'].set(cen_ini-0.2)
            mod += nebem
            if i == 2 and line not in doubem:
                break

    result = mod.fit(y_flux, pars, x=x_wave, weights=1/ferr[wave_region])

    return result, x_wave, y_flux, wave_region

def sb2_results_to_file(result, wavelengths, names, line, writer, csvfile):
    for i in range(len(wavelengths)):
        results_dict = {}
        for j in range(1, 3):
            prefix = f'cen{j}_{i+1}'
            if prefix in result.params:
                component_dict = {
                    'epoch': names[i],
                    'line': line,
                    f'cen{j}': result.params[f'{prefix}'].value,
                    f'cen{j}_er': result.params[f'{prefix}'].stderr,
                }
                results_dict.update(component_dict)

        # Add common parameters
        common_params = ['amp1', 'wid1', 'amp2', 'wid2']
        for param in common_params:
            if param in result.params:
                results_dict[param] = result.params[param].value
                results_dict[f'{param}_er'] = result.params[param].stderr

        # If writer is None, this is the first dictionary, so get the fieldnames from its keys
        if writer is None:
            fieldnames = results_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(results_dict)

    return writer

# def mcmc_results_to_file(trace, names, line, writer, csvfile):
#     # samples = mcmc.get_samples()
#     samples = trace
#     print(samples)
#     # for key in samples.keys():
#     #     print(f"{key}: {samples[key]}")
#     # print(samples.keys()
#     # print({(key, type(key)) for key in samples.keys()})
#     results_dict = {}
#     for i in range(len(names)):
#         results_dict['epoch'] = names[i]
#         results_dict['line'] = line
#         for j in range(1, 3):
#             prefix = f'mean{j}'
#             if prefix in samples:

#                 # Calculate mean and standard deviation for each epoch
#                 means = np.mean(samples[prefix], axis=0)
#                 stds = np.std(samples[prefix], axis=0)

#                 # Add the mean and standard deviation for the current epoch to the dictionary
#                 results_dict[f'cen{j}'] = means[i, 0]
#                 results_dict[f'cen{j}_er'] = stds[i, 0]

#             else:
#                 print(f"Prefix {prefix} not found in samples")

#         # print(f"Results dict: {results_dict}")
#         # Add common parameters
#         common_params = ['amp1', 'wid1', 'amp2', 'wid2']
#         for param in common_params:
#             if param in samples:
#                 results_dict[param] = np.mean(samples[param])
#                 results_dict[f'{param}_er'] = np.std(samples[param])

#         # If writer is None, this is the first dictionary, so get the fieldnames from its keys
#         if writer is None:
#             fieldnames = results_dict.keys()
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#         writer.writerow(results_dict)

#     return writer

def mcmc_results_to_file(trace, names, jds, writer, csvfile):
    # Loop over the components
    for i in range(2):  # components are 0-based
        # Loop over the epochs
        for j in range(len(names)):
            # Initialize a new dictionary for this component and epoch
            results_dict = {}

            # Add the epoch and MJD to the dictionary
            results_dict['epoch'] = names[j]
            results_dict['MJD'] = jds[j]

            # Add the RV and its uncertainty to the dictionary
            results_dict['mean_rv'] = np.mean(trace['Δv_τk'][:, i, 0, j])
            results_dict['mean_rv_er'] = np.std(trace['Δv_τk'][:, i, 0, j])

            # Add the component to the dictionary
            results_dict['comp'] = i + 1  # +1 because components are 1-based

            # Write the dictionary to the CSV file
            if writer is None:
                # If writer is None, this is the first dictionary, so get the fieldnames from its keys
                fieldnames = results_dict.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(results_dict)

    return writer



def SLfit(spectra_list, path, lines, K=2, file_type='fits', plots=True, balmer=True, neblines=[4102, 4340], doubem=[], SB2=False, shift=0, use_init_pars=False):
    '''
    spectra_list
    path
    lines:        lines to be fitted
    file_type
    plots
    balmer
    neblines
    doubem
    SB2
    shift:        wavelength shift to be applied for initial guess used for the fit
    '''

    print('\n')
    print( '*******************************************************************************' )
    print( '******************           Spectral Line fitting           ******************' )
    print( '*******************************************************************************\n' )

    if len(spectra_list) == 1:
        print("\n   WARNING: There is only 1 epoch to compute RVs.")
        return    
    # doubem = doubem[0]
    Hlines = [4102, 4340, 6562]

    print('*** SB2 set to: ', SB2, ' ***\n')
    
    wavelengths, fluxes, f_errors, names, jds = read_spectra(spectra_list, path, file_type)

    path = setup_star_directory_and_save_jds(names, jds, path, SB2)

    lines_dic = setup_line_dictionary()

    print( '\n')
    print( '*** Fitting lines ***')
    print( '---------------------')

    print('  these are the lines: ', lines)

    (
        cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
        cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
        dely, sdev, results, comps, delta_cen, chisqr
    ) = initialize_fit_variables(lines)

    with open(path + 'fit_values.csv', 'w', newline='') as csvfile:
        writer = None

        print('Fitting all lines simultaneously')
        # if plots:
        #     fig, axes = setup_fits_plots(wavelengths)
        # else:
        #     axes = [None] * len(wavelengths)
        if SB2:
            result, x_wave, y_flux = fit_sb2_probmod(lines, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, path, K=K)   
            # result, x_wave, y_flux = fit_sb2(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, axes, path)
            # writer = sb2_results_to_file(result, wavelengths, names, line, writer, csvfile)
            # writer = mcmc_results_to_file(result, names, line, writer, csvfile)
            # Iterate over each line to write the results into the CSV
            # for i, line in enumerate(lines):
            writer = mcmc_results_to_file(result, names, jds, writer, csvfile)
                
                # if plots:
                #     for i, (x_vals, y_vals, ax) in enumerate(zip(x_wave, y_flux, axes)):
                #         ax.plot(x_vals, y_vals, 'k-', label=str(names[i].replace('./','').replace('_',' ')))
                #         model_fit = double_gaussian_with_continuum(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value, 
                #                         result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value, result.params['cont'].value)
                #         component1 = gaussian(x_vals, result.params['amp1'].value, result.params[f'cen1_{i+1}'].value, result.params['wid1'].value)
                #         component2 = gaussian(x_vals, result.params['amp2'].value, result.params[f'cen2_{i+1}'].value, result.params['wid2'].value)
                #         ax.plot(x_vals, component1 + result.params['cont'].value, '--', c='limegreen', label=f'Component 1 - Epoch {i+1}')
                #         ax.plot(x_vals, component2 + result.params['cont'].value, '--', c='dodgerblue', label=f'Component 2 - Epoch {i+1}')
                #         ax.plot(x_vals, model_fit, 'r-', label=f'Fit {i+1}')
                #         # if line in neblines:
                #         #     ax.plot(x_wave, component['continuum_']+component['neb_'], '--', zorder=3, c='orange', lw=2)#, label='nebular emision')
                #         # if dely is not None:
                #         #     ax.fill_between(x_wave, result.best_fit-dely, result.best_fit+dely, zorder=2, color="#ABABAB", alpha=0.5)
                #         # ax.legend(loc='upper center',fontsize=10, handlelength=0, handletextpad=0.4, borderaxespad=0., frameon=False)
                #         # ax.legend(fontsize=14) 
                #         # ax.set_ylim(0.9*y_flux.min(), 1.1*y_flux.max())
                #     fig.supxlabel('Wavelength [\AA]', size=22)
                #     fig.supylabel('Flux', size=22)
                #     plt.savefig(path+str(line)+'_fits_SB2_.png', bbox_inches='tight', dpi=150)
                #     plt.close()
        else:
            for line in lines:
                for j, (wave, flux, ferr, name, ax) in enumerate(zip(wavelengths, fluxes, f_errors, names, axes)):
                    result, x_wave, y_flux, wave_region = fit_sb1(line, wave, flux, ferr, lines_dic, Hlines, neblines, doubem, shift)
                    
                    results[i].append(result)
                    chisqr[i].append(result.chisqr)
                    if line in neblines or SB2==True:
                        component = result.eval_components(result.params, x=x_wave)
                        comps[i].append(component)
                    else:
                        component = None
                        comps[i].append(component)
                    with open(path + str(line) + '_stats.txt', 'w') as out_file:
                        out_file.write(name+'\n')
                        out_file.write('-' * len(name.strip()) + '\n')
                        out_file.write(result.fit_report()+'\n\n')

                    if 'g1_cen' in result.params:
                        prefix = 'g1'
                    elif 'l1_cen' in result.params:
                        prefix = 'l1'
                    else:
                        raise ValueError("Unexpected fit type")

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
                        print('errors computation failed for line ', line, 'epoch ', j+1)
                    if results_dict['cen1_er'] != 0 and results_dict['cen1_er'] is not None:
                        dely = result.eval_uncertainty(sigma=3)
                    else:
                        dely = None

                    # If writer is None, this is the first dictionary, so get the fieldnames from its keys
                    if writer is None:
                        fieldnames = results_dict.keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                    writer.writerow(results_dict)

                    '''
                    Plotting 1st fits of the lines
                    '''
                    if plots:
                        xx = np.linspace(min(x_wave), max(x_wave), 500)
                        result_new = result.eval(x=xx)
                        init_new = result.eval(params=result.init_params, x=xx)
                        ax.plot(x_wave, y_flux, 'k-', lw=3, ms=4, zorder=1, label=str(name.replace('./','').replace('_',' ')))
                        ax.plot(xx, init_new, '--', c='grey', zorder=5)
                        ax.plot(xx, result_new, 'r-', lw=2, zorder=4)

                        if component is not None:
                            if line in Hlines:
                                ax.plot(x_wave, component['continuum_']+component['l1_'], '--', zorder=3, c='limegreen', lw=2)#, label='spectral line')
                            if SB2==True and not line in Hlines:
                                ax.plot(x_wave, component['continuum_']+component['g1_'], '--', zorder=3, c='blue', lw=2)#, label='spectral line')
                                ax.plot(x_wave, component['continuum_']+component['g2_'], '--', zorder=3, c='blue', lw=2)#, label='spectral line')
                            if line in neblines:
                                ax.plot(x_wave, component['continuum_']+component['neb_'], '--', zorder=3, c='orange', lw=2)#, label='nebular emision')
                            if line in neblines and not line in Hlines and SB2==False:
                                ax.plot(x_wave, component['continuum_']+component['g1_'], '--', zorder=3, c='limegreen', lw=2)#, label='spectral line')

                        if dely is not None:
                            ax.fill_between(x_wave, result.best_fit-dely, result.best_fit+dely, zorder=2, color="#ABABAB", alpha=0.5)

                        # ax.legend(loc='upper center',fontsize=10, handlelength=0, handletextpad=0.4, borderaxespad=0., frameon=False)

                        ax.set_ylim(0.9*y_flux.min(), 1.1*y_flux.max())
            if plots:        
                fig.supxlabel('Wavelength', size=22)
                fig.supylabel('Flux', size=22)
                if SB2==True:
                    plt.savefig(path+str(line)+'_fits_0_.png', bbox_inches='tight', dpi=150)
                else:
                    plt.savefig(path+str(line)+'_fits.png', bbox_inches='tight', dpi=150)
                plt.close()

    return path

class GetRVs:
    def __init__(self, fit_values, path, JDfile, balmer=False, SB2=False, use_lines=None, 
               lines_ok_thrsld=2, epochs_ok_thrsld=2, min_sep=2, print_output=True, 
               random_eps=False, rndm_eps_n=29, rndm_eps_exc=[], plots=True, 
               error_type='wid1_percer', rm_epochs=True):
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

        fecha = str(date.today()) # '2017-12-26'

        # Reading data from fit_states.csv
        # print(path+fit_values)
        self.df_SLfit = pd.read_csv(self.path+self.fit_values)

        # Calculate percentage errors
        if self.df_SLfit.isnull().values.any():
            print("Warning: NaN values found in df_SLfit")    
        for param in ['cen1', 'amp1', 'wid1']:
            self.df_SLfit[f'{param}_percer'] = np.where(self.df_SLfit[f'{param}_er'] != 0, np.abs(100 * self.df_SLfit[f'{param}_er'] / self.df_SLfit[param]), np.nan)

        if self.SB2:
            for param in ['cen2', 'amp2', 'wid2']:
                self.df_SLfit[f'{param}_percer'] = np.where(self.df_SLfit[f'{param}_er'] != 0, np.abs(100 * self.df_SLfit[f'{param}_er'] / self.df_SLfit[param]), np.nan)

        # Get unique lines
        self.lines = self.df_SLfit['line'].unique()
        print('lines from SLfit results:', self.lines)

        self.nepochs = len(self.df_SLfit) // len(self.lines)

    def outlier_killer(self, data, thresh=2, print_output=None):
        ''' returns the positions of elements in a list that
            are not outliers. Based on a median-absolute-deviation
            (MAD) test '''
        if print_output is None:
            print_output = self.print_output
        # diff = abs(data - np.nanmedian(data))
        diff = data - np.nanmedian(data)
        # diff = data - np.nanmean(data)
        with open(self.path+'rv_stats.txt', 'a') as f:
            if print_output==True:
                print('     mean(x)        =', f'{np.nanmean(data):.3f}')
                print('     mean(x)        =', f'{np.nanmean(data):.3f}', file=f)
                print('     median(x)        =', f'{np.nanmedian(data):.3f}')
                print('     median(x)        =', f'{np.nanmedian(data):.3f}', file=f)
                print('     x-median(x)      =', [f'{x:.3f}' for x in diff])
                print('     x-median(x)      =', [f'{x:.3f}' for x in diff], file=f)
                print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff])
                print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff], file=f)
            mad = 1.4826*np.nanmedian(abs(diff))
            if print_output==True:
                print('     mad =', f'{mad:.3f}')
                print('     mad =', f'{mad:.3f}', file=f)
                print('     abs(x-median(x)) <', (f'{thresh*mad:.3f}'), '(thresh*mad) =', [abs(x)<thresh*mad for x in diff])
                print('     abs(x-median(x)) <', (f'{thresh*mad:.3f}'), '(thresh*mad) =', [abs(x)<thresh*mad for x in diff], file=f)
        inliers, outliers = [], []
        for i in range(len(data)):
            # if diff[i] < thresh*mad:
            if abs(diff[i]) < thresh*mad:
                inliers.append(i)
            else:
                outliers.append(i)
        return inliers, outliers

    @staticmethod
    def weighted_mean(data, errors):
        weights = [1/(dx**2) for dx in errors]
        mean = sum([wa*a for a,wa in zip(data, weights)])/sum(weights)
        mean_err = np.sqrt(sum( [(da*wa)**2 for da,wa in zip(errors, weights)] ))/sum(weights)
        return mean, mean_err

    def compute_rvs(self, lines, lambda_rest_dict):
        c_kms = c.to('km/s').value   
        rvs = {}

        for line in lines:
            # print(f'Computing RVs for line {line}')
            lambda_rest, lambda_r_er = lambda_rest_dict[line]
            current_line_values = self.df_SLfit[self.df_SLfit['line'] == line]

            dlambda1 = current_line_values['cen1'].values - lambda_rest
            dlambda1_er = np.sqrt(current_line_values['cen1_er'].values**2 + lambda_r_er**2)
            rv1 = dlambda1 * c_kms / lambda_rest
            # print('rv1:', rv1)
            rv1_er = np.sqrt((dlambda1_er / dlambda1)**2 + (lambda_r_er / lambda_rest)**2) * np.abs(rv1)
            # print('rv1_er:', rv1_er)
            rvs[line] = {'rv1': rv1, 'rv1_er': rv1_er}

            if self.SB2:
                dlambda2 = np.abs(current_line_values['cen2'].values - lambda_rest)
                dlambda2_er = np.sqrt(current_line_values['cen2_er'].values**2 + lambda_r_er**2)
                rv2 = dlambda2 * c_kms / lambda_rest
                rv2_er = np.sqrt((dlambda2_er / dlambda2)**2 + (lambda_r_er / lambda_rest)**2) * rv2

                rvs[line].update({'rv2': rv2, 'rv2_er': rv2_er})

        return rvs

    @staticmethod
    def print_error_stats(grouped_error, error_type):
        mean_values = [f'{np.mean(value):6.3f}' for value in grouped_error.values]
        median_values = [f'{np.median(value):6.3f}' for value in grouped_error.values]
        print(f'   mean({error_type})  ', ' '.join(mean_values))
        print(f'   median({error_type})', ' '.join(median_values))

    def write_error_stats(self, out, grouped_errors, stat_type):
        stat_func = np.nanmean if stat_type == 'Mean' else np.nanmedian
        out.write(f' {stat_type} of percentual errors\n')
        out.write('   Lines   wid1    cen1   amp1   |    wid2    cen2   amp2 \n')
        out.write('   -------------------------------------------------------\n')
        for line in grouped_errors['cen1_percer'].keys():
            stats = ' '.join([f'{stat_func(grouped_errors[error_type][line]):7.3f}' for error_type in grouped_errors])
            out.write(f'   {line}: {stats}\n')
        out.write('\n')

    # def select_lines(self, error_type, lines):
    #     if not self.use_lines:
    #         best_lines_index, rm_lines_idx = self.outlier_killer([np.nanmedian(x) for x in error_type], thresh=self.lines_ok_thrsld)
    #     else:
    #         best_lines_index = [i for i, line in enumerate(lines) if line in self.use_lines]
    #         rm_lines_idx = [i for i, line in enumerate(lines) if line not in self.use_lines]

    #     best_lines = [lines[x] for x in best_lines_index]
    #     return best_lines, best_lines_index, rm_lines_idx
        
    def select_lines(self, error_type, lines):
        if not self.use_lines:
            nonan_lines = []
            for line, x in error_type.iteritems():
                if np.isnan(x).sum() <= 1:
                    nonan_lines.append(line)
            best_lines_index, rm_lines_idx = self.outlier_killer([np.nanmedian(error_type.loc[i]) for i in nonan_lines], thresh=self.lines_ok_thrsld)
            best_lines = [nonan_lines[x] for x in best_lines_index]
        else:
            best_lines_index = [i for i, line in enumerate(lines) if line in self.use_lines]
            rm_lines_idx = [i for i, line in enumerate(lines) if line not in self.use_lines]
            best_lines = [lines[x] for x in best_lines_index]
        
        return best_lines, best_lines_index, rm_lines_idx

    def remove_bad_epochs(self, df, metric='mean', error_type='cen', epochs_ok_thrsld=None):
        
        error_list = []
        err_type_dic={'wid':'wid1_er', 'rvs':'sigma_rv', 'cen':'cen1_er', 'amp':'amp1_er'}
        self.print_output and print('\n   '+'-'*34+'\n   Removing epochs with large errors:')
        self.print_output and print('     using error_type:', error_type)
        if not epochs_ok_thrsld:
            epochs_ok_thrsld = self.epochs_ok_thrsld
        with open(self.path+'rv_stats.txt', 'a') as out:
            out.write(' Removing epochs with large errors:\n')
            rm_OBs_idx = []
            epochs_unique = df['epoch'].unique()
            for i, epoch in enumerate(epochs_unique):
                df_epoch = df[df['epoch'] == epoch]
                error = df_epoch[err_type_dic[error_type]].mean() if metric == 'mean' \
                    else df_epoch[err_type_dic[error_type]].median()
                error_list.append(error)

                if error_type == 'rvs' and df_epoch.isna().any().any():
                    rm_OBs_idx.append(i)

            self.print_output and print('\n   Applying outlier_killer to remove epochs')
            rm_OBs_idx += self.outlier_killer(error_list, thresh=epochs_ok_thrsld)[1]

            self.print_output and print(f'   Indices of epochs to be removed: {[x for x in rm_OBs_idx]}')
            out.write(f'   Indices of epochs to be removed: {[x for x in rm_OBs_idx]}\n\n')

            return rm_OBs_idx

    def compute_weighted_mean_rvs(self, rvs_dict, lines, rm_OBs_idx):
        wmean_rvs1, wmean_rvs2 = {}, {}
        line_avg1, line_avg2 = {}, {}
        # Filter the RVs and errors for the best lines and remove bad epochs
        for line, rvs in rvs_dict.items():
            if line not in lines:
                continue
            rv1 = np.delete(rvs['rv1'], rm_OBs_idx)
            rv1_er = np.delete(rvs['rv1_er'], rm_OBs_idx)
            line_avg1[line] = {'mean': np.mean(rv1), 'std': np.std(rv1)}
            rvs_dict[line]['rv1'] = rv1
            rvs_dict[line]['rv1_er'] = rv1_er
        # Compute the weighted mean RV for the primary for each epoch
        for epoch in range(len(rv1)):
            weighted_mean1, weighted_error1 = GetRVs.weighted_mean([rvs_dict[line]['rv1'][epoch] for line in lines], 
                                                                   [rvs_dict[line]['rv1_er'][epoch] for line in lines])
            wmean_rvs1[epoch] = {'value': weighted_mean1, 'error': weighted_error1}

        # Same for the secondary if SB2
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
                weighted_mean2, weighted_error2 = GetRVs.weighted_mean([rvs_dict[line]['rv2'][epoch] for line in lines], 
                                                                       [rvs_dict[line]['rv2_er'][epoch] for line in lines])
                wmean_rvs2[epoch] = {'value': weighted_mean2, 'error': weighted_error2}

        return wmean_rvs1, wmean_rvs2, line_avg1, line_avg2

    def print_and_write_results(self, lines, line_avg1, total_mean_rv1, nepochs, line_avg2=None, total_mean_rv2=None):
        # Prepare the lines to print/write
        rows = []
        rows.append(f'RV mean of the {nepochs} epochs for each line:')
        rows.append('---------------------------------------')
        # print('line_avg1:', line_avg1)
        # print('line_avg2:', line_avg2)
        for line in lines:
            mean = line_avg1[line]['mean']
            std = line_avg1[line]['std']
            rows.append(f'   - {line}: {mean:.3f} +/- {std:.3f}')
        if self.SB2:
            rows.append(f'   Component 2:') 
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
        with open(self.path+'rv_stats.txt', 'a') as out:
            out.write('\n'.join(rows))

    def compute(self):
        '''
        
        '''
        
        if self.print_output==True:
            print('\n')
            print( '*******************************************************************************' )
            print( '******************                RV Analisys                ******************' )
            print( '*******************************************************************************\n' )

        # All wavelengths in air
        lambda_rest_dict =  {4009: [4009.2565, 0.00002], 4026: [4026.1914, 0.0010],
                    4102: [4101.734, 0.006],    4121: [4120.8154, 0.0012],
                    4128: [4128.07, 0.10],      4131: [4130.89, 0.10],
                    4144: [4143.761, 0.010],    4267: [4267.258, 0.007], 
                    4340: [4340.472, 0.006],    4388: [4387.9296, 0.0006],  
                    4471: [4471.4802, 0.0015],  4481: [4481.130, 0.010],    
                    4542: [4541.591, 0.010],    4553: [4552.62, 0.10], 
                    4713: [4713.1457, 0.0006],      
                    4861: [4861.35, 0.05],      4922: [4921.9313, 0.0005],
                    5412: [5411.52, 0,10],      5876: [5875.621, 0.010],
                    5890: [5889.951, 0.00003],  6562: [6562.79, 0.030],
                    6678: [6678.151, 0.010],    7774: [7774.17, 0,10]
                    }

        if self.print_output==True:
            print('*** SB2 set to: ', self.SB2, ' ***\n')
    
        if self.print_output==True:
            print( '\n')
            print( '*** Computing Radial velocities ***')
            print( '-----------------------------------')

        rvs_dict = self.compute_rvs(self.lines, lambda_rest_dict)
 
        # RV plot per spectral line from rvs_dict
        fig, ax = plt.subplots()
        markers = ['o', 'v', '^', '<', '>', 's', 'X', '*', 'D', 'H']
        for line, marker in zip(self.lines, markers):
            rv1 = rvs_dict[line]['rv1']
            rv1_er = rvs_dict[line]['rv1_er']
            ax.errorbar(range(len(rv1)), rv1, yerr=rv1_er, fmt=marker, color='dodgerblue', label=f'Comp. 1 {line}', alpha=0.5)
            if self.SB2:
                rv2 = rvs_dict[line]['rv2']
                rv2_er = rvs_dict[line]['rv2_er']
                ax.errorbar(range(len(rv2)), rv2, yerr=rv2_er, fmt=marker, color='darkorange', alpha=0.5)# , label=f'Comp. 2 {line}'
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Radial Velocity (km/s)')
        ax.legend(loc='lower left', fontsize=8)
        plt.savefig(self.path+'rvs_per_line.png', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()

        #################################################################
        #                Selecting lines with the best fits
        #################################################################

        primary_error_types = ['wid1_percer', 'cen1_percer', 'amp1_percer']
        secondary_error_types = ['wid2_percer', 'cen2_percer', 'amp2_percer'] if self.SB2 else []
        
        grouped_errors = {error_type: self.df_SLfit.groupby('line')[error_type].apply(list) for error_type in primary_error_types + secondary_error_types}

        if self.print_output:
            print('\n*** Choosing the best lines ***\n-------------------------------')

            print_lines = [str(line) for line in grouped_errors['cen1_percer'].keys()]
            print(' Primary:'+' ' * 14, '   '.join(print_lines))
            for error_type in primary_error_types:
                GetRVs.print_error_stats(grouped_errors[error_type], error_type)

            if self.SB2:
                print(' Secondary:')
                for error_type in secondary_error_types:
                    GetRVs.print_error_stats(grouped_errors[error_type], error_type)

        with open(self.path+'rv_stats.txt', 'w') as out:
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

        # if self.SB2:
        #     error_type = 'cen2_percer'
        # for line in self.lines:
        #     print(line, grouped_errors[self.error_type][line])
        #     print('mean:', np.nanmean(grouped_errors[self.error_type][line]))
        #     print('median:', np.nanmedian(grouped_errors[self.error_type][line]))
        # Selecting the best lines
        best_lines, best_lines_index, rm_lines_idx = self.select_lines(grouped_errors[self.error_type], self.lines)
        if self.print_output:
            print('\n   These are the best lines: ', best_lines)

        nlines = len(best_lines)

        with open(self.path+'rv_stats.txt', 'a') as out:
            out.write('\n')
            out.write(f' Lines with the best fitted profile according to the median {self.error_type} criterion:\n')
            out.write(' --------------------------------------------------------------------------\n')
            for i in range(nlines):
                if i<range(nlines)[-1]:
                    out.write('   '+str(best_lines_index[i])+': '+str(best_lines[i])+', ')
                else:
                    out.write('   '+str(best_lines_index[i])+': '+str(best_lines[i])+'\n')
            out.write('\n')

        #################################################################
        #                Removing lines with inverted components
        #################################################################
        if self.SB2:
            if len(best_lines) > 2:
                if self.print_output:
                    print('\n   --------------------------------------------')
                    print('   Removing lines with inverted components:')

                failed_lines_indices = []
                for epoch in range(self.nepochs):
                    rv1s = [rvs_dict[line]['rv1'][epoch] for line in best_lines]
                    _, failed_lines_index = self.outlier_killer(rv1s, thresh=self.lines_ok_thrsld, print_output=False)
                    failed_lines_indices.extend(failed_lines_index)
                    failed_lines_counts = Counter(failed_lines_indices)
                print(f'   {failed_lines_counts}')

                threshold_percentage = 0.60  # 60%
                threshold = threshold_percentage * self.nepochs
                # Find the lines that failed the test more than the threshold number of times
                lines_to_remove = [line for line, count in failed_lines_counts.items() if count > threshold]
                lines_to_remove = [best_lines[i] for i in lines_to_remove]
                # Remove the lines from the list of lines
                print(f"   Lines to remove: {lines_to_remove}")
                best_lines = [line for line in best_lines if line not in lines_to_remove]
                print(f"   Remaining lines: {best_lines}")

        #################################################################
        #                  Removing bad/noisy epochs
        #################################################################
        df_rv = pd.read_csv(self.JDfile, names = ['epoch', 'MJD'], sep='\s+')
        df_rv = df_rv.replace({'.fits':''}, regex=True)
        df_rv2 = pd.read_csv(self.JDfile, names = ['epoch', 'MJD'], sep='\s+')
        df_rv2 = df_rv2.replace({'.fits':''}, regex=True)

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
        if self.print_output==True:
            print( '\n')
            print( '*** Calculating the RV weighted mean for each epoch  ***')
            print( '--------------------------------------------------------')

        wmean_rv1, wmean_rv2, line_avg1, line_avg2 = self.compute_weighted_mean_rvs(rvs_dict, best_lines, rm_OBs_idx)
        total_rv1, total_rv2 = {}, {}
        total_rv1['mean'] = np.mean([wmean_rv1[i]['value'] for i in wmean_rv1.keys()])
        total_rv1['std'] = np.std([wmean_rv1[i]['value'] for i in wmean_rv1.keys()])
        if self.SB2:
            total_rv2['mean'] = np.mean([wmean_rv2[i]['value'] for i in wmean_rv2.keys()])
            total_rv2['std'] = np.std([wmean_rv2[i]['value'] for i in wmean_rv2.keys()])

        # Printing/Writing
        self.print_and_write_results(best_lines, line_avg1, total_rv1, final_nepochs, line_avg2=line_avg2, total_mean_rv2=total_rv2)
        
        #################################################################
        #                Writing RVs to file RVs.txt
        #################################################################
        if self.rm_epochs and rm_OBs_idx:
            df_rv.drop(rm_OBs_idx, inplace=True)
            df_rv.reset_index(drop=True, inplace=True)
            # rv1_values = [np.delete(rv, rm_OBs_idx) for rv in rv1_values]
        df_rv['mean_rv'] = [wmean_rv1[i]['value'] for i in range(len(wmean_rv1))]
        df_rv['mean_rv_er'] = [wmean_rv1[i]['error'] for i in range(len(wmean_rv1))]

        if self.SB2:
            if self.rm_epochs and rm_OBs_idx:
                df_rv2.drop(rm_OBs_idx, inplace=True)
                df_rv2.reset_index(drop=True, inplace=True)
            df_rv2['mean_rv'] = [wmean_rv2[i]['value'] for i in range(len(wmean_rv2))]
            df_rv2['mean_rv_er'] = [wmean_rv2[i]['error'] for i in range(len(wmean_rv2))]
            df_rv = pd.concat([df_rv, df_rv2], ignore_index=True)

        with open(self.path+'RVs1.txt', 'w') as fo:
            fo.write(df_rv.to_string(formatters={'MJD': '{:.8f}'.format}, index=False))

        #################################################################
        #        Plotting RVs per spectral line and weighted mean
        #################################################################
        data = pd.read_csv(self.path+'RVs1.txt', delim_whitespace=True)
        primary_data = data[data['comp'] == 1]
        secondary_data = data[data['comp'] == 2]
        fig, ax = plt.subplots(figsize=(8, 6))
        rv_lines = [f'rv_{line}' for line in best_lines]
        rv_er = [f'rv_{line}_er' for line in best_lines]
        markers = ['o', 'v', '^', '<', '>', 's', 'X', '*', 'D', 'H']
        for i, (rv_line, marker) in enumerate(zip(rv_lines, markers)):
            ax.errorbar(primary_data['MJD'], primary_data[rv_line], yerr=primary_data[rv_er[i]], fmt=marker, 
                        color='dodgerblue', fillstyle='none', label=f'Comp. 1 {best_lines[i]}', alpha=0.5)
        for i, (rv_line, marker) in enumerate(zip(rv_lines, markers)):
            ax.errorbar(secondary_data['MJD'], secondary_data[rv_line], yerr=secondary_data[rv_er[i]], fmt=marker, 
                        color='darkorange', fillstyle='none', alpha=0.5)# label=f'Comp. 1 {best_lines[i]}', 
        ax.errorbar(primary_data['MJD'], primary_data['mean_rv'], fmt='s', color='dodgerblue', alpha=0.5, label=f'Primary weighted mean')
        ax.errorbar(secondary_data['MJD'], secondary_data['mean_rv'], fmt='s', color='darkorange', alpha=0.5, label=f'Secondary weighted mean')
        ax.set_xlabel('Julian Date')
        ax.set_ylabel('Mean Radial Velocity')
        ax.legend(fontsize=8)
        plt.savefig(self.path+'RVs1.png', bbox_inches='tight', dpi=300)
        plt.close()

        return df_rv

def get_peaks(power, frequency, fal_50pc, fal_1pc, fal_01pc, minP=1.1):
    if power.max() > fal_01pc:
        peaks_min_h = float(f'{fal_1pc:.3f}')
    elif power.max() <= fal_01pc and power.max() > fal_1pc:
        peaks_min_h = float(f'{0.6*fal_1pc:.3f}')
    else:
        peaks_min_h = float(f'{fal_50pc:.3f}')
   
    freq_index1 = np.argmin(np.abs(frequency - 1/5))
    freq_index2 = np.argmin(np.abs(frequency - 1/minP))
    plt.plot(frequency, power)
    peaks1, _ = find_peaks(power[:freq_index1], height=peaks_min_h, distance=1000)
    peaks2, _ = find_peaks(power[:freq_index2], height=peaks_min_h, distance=5000)

    peaks = np.unique(np.concatenate((peaks1, peaks2)))

    freq_peaks = frequency[peaks]
    peri_peaks = 1/frequency[peaks]

    return freq_peaks, peri_peaks, peaks

def run_LS(hjd, rv, rv_err=None, probabilities = [0.5, 0.01, 0.001], method='bootstrap', P_ini=0.4, P_end=1000, samples_per_peak=2000):
    if rv_err is None:
    # if rv_err.any() == False:
        ls = LombScargle(hjd, rv, normalization='model')
    else:
        ls = LombScargle(hjd, rv, rv_err, normalization='model')
    fal = ls.false_alarm_level(probabilities, method=method)
    frequency, power = ls.autopower(method='fast', minimum_frequency=1/P_end, #0.001=1000
                                                    maximum_frequency=1/P_ini, # 0.9=1.1, 1.5=0.67, 2=0.5, 4=0.25
                                                    samples_per_peak=samples_per_peak) # 2000
    fap = ls.false_alarm_probability(power.max(), method=method)

    return frequency, power, fap, fal

def lomb_scargle(df, path, probabilities = [0.5, 0.01, 0.001], SB2=False, print_output=True, plots=True, best_lines=False, Pfold=True):
    '''
    df: dataframe with the RVs, output from getrvs()
    '''
    # print(df)
    if not os.path.exists(path+'LS'):
        os.makedirs(path+'LS')

    hjd1, rv1 = df['MJD'][df['comp']==1], df['mean_rv'][df['comp']==1]
    if 'mean_rv_er' in df.columns:
        rv1_err = df['mean_rv_er'][df['comp']==1]
    else:
        rv1_err = None
    
    starname = df['epoch'][0].split('_')[0]+'_'+df['epoch'][0].split('_')[1]
    nepochs=len(hjd1)

    with open(path+'LS/ls_output.txt', 'w') as lsout:
        lsout.write(' ***************************************\n')
        lsout.write('   LS output for star '+starname+'\n')
        lsout.write(' ***************************************\n')
        lsout.write('\n')

    #################################################################
    #                Running the Lomb-Scargle periodogram
    #################################################################
    ls_results = {'freq': {}, 'power': {}, 'fap': {}, 'fal_50%': {}, 'fal_1%': {}, 
                  'fal_01%': {}, 'max_freq': {}, 'max_period': {}, 'max_power': {}, 
                  'peaks': {}, 'best_period': {}, 'best_P_pow': {}, 'ind': {}, 'freq_peaks': {}, 
                  'peri_peaks': {}, 'pow_over_fal01': {}, 'pow_over_fal1': {}}    
    if rv1_err is not None:
        frequency1, power1, fap1, fal1 = run_LS(hjd1, rv1, rv1_err)
    else:
        frequency1, power1, fap1, fal1 = run_LS(hjd1, rv1)
        
    fal1_50pc, fal1_1pc, fal1_01pc = fal1[0].value, fal1[1].value, fal1[2].value
    freq1_at_max_power = frequency1[np.argmax(power1)]
    period1_at_max_power = 1/freq1_at_max_power
    
    ls_results['freq'][1] = frequency1
    ls_results['power'][1] = power1
    ls_results['fap'][1] = fap1
    ls_results['fal_50%'][1] = fal1_50pc
    ls_results['fal_1%'][1] = fal1_1pc
    ls_results['fal_01%'][1] = fal1_01pc
    ls_results['max_freq'][1] = freq1_at_max_power
    ls_results['max_period'][1] = period1_at_max_power
    if SB2:
        hjd2, rv2, rv2_err = df['MJD'][df['comp']==2], df['mean_rv'][df['comp']==2], df['mean_rv_er'][df['comp']==2]
        frequency2, power2, fap2, fal2 = run_LS(hjd2, rv2, rv2_err)
        fal2_50pc, fal2_1pc, fal2_01pc = fal2[0].value, fal2[1].value, fal2[2].value
        freq2_at_max_power = frequency2[np.argmax(power2)]
        period2_at_max_power = 1/freq2_at_max_power
        
        ls_results['freq'][2] = frequency2
        ls_results['power'][2] = power2
        ls_results['fap'][2] = fap2
        ls_results['fal_50%'][2] = fal2_50pc
        ls_results['fal_1%'][2] = fal2_1pc
        ls_results['fal_01%'][2] = fal2_01pc
        ls_results['max_freq'][2] = freq2_at_max_power
        ls_results['max_period'][2] = period2_at_max_power

    fapper1 = fap1*100
    if print_output==True:
        print('   these are the false alarm levels: ', [f'{x:.3f}' for x in fal1])
        print('   FAP of the highest peak         : ', f'{fap1:.5f}')
        print('   FAP of the highest peak x100    : ', f'{fapper1:.5f}')
    with open(path+'LS/ls_output.txt', 'a') as lsout:
        lsout.write(' these are the false alarm levels: ')
        for fal in fal1:
            lsout.write(str(f'{fal:.3f}')+'  ')
        lsout.write('\n')
        lsout.write(' FAP of the highest peak         : '+ f'{fap1:.5f}'+'\n')
        lsout.write(' FAP of the highest peak x100    : '+ f'{fapper1:.5f}'+'\n')

    #################################################################
    #                        Finding peaks
    #################################################################
    freq_peaks1, peri_peaks1, peaks1 = get_peaks(power1, frequency1, fal1_50pc, fal1_1pc, fal1_01pc)
    ls_results['freq_peaks'][1] = freq_peaks1
    ls_results['peri_peaks'][1] = peri_peaks1
    ls_results['peaks'][1] = peaks1
    
    if SB2:
        freq_peaks2, peri_peaks2, peaks2 = get_peaks(power2, frequency2, fal2_50pc, fal2_1pc, fal2_01pc)
        
        # test |RV1-RV2|
        rv_abs = np.abs(rv1-rv2.reset_index(drop=True))
        if rv1_err is not None:
            rv_abs_er = np.sqrt(rv1_err**2+rv2_err.reset_index(drop=True)**2)
        else:
            rv_abs_er = None

        frequency3, power3, fap3, fal3 = run_LS(hjd1, rv_abs, rv_abs_er)

        fal3_50pc, fal3_1pc, fal3_01pc = fal3[0], fal3[1], fal3[2]
        freq3_at_max_power = frequency3[np.argmax(power3)]
        period3_at_max_power = 1/freq3_at_max_power
        freq_peaks3, peri_peaks3, peaks3 = get_peaks(power3, frequency3, fal3_50pc, fal3_1pc, fal3_01pc)

        ls_results['freq_peaks'][2] = freq_peaks2
        ls_results['peri_peaks'][2] = peri_peaks2
        ls_results['peaks'][2] = peaks2

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

    if print_output==True:
        print("   Best frequency                  :  {0:.3f}".format(freq1_at_max_power))
        print('   ***********************************************')
        print("   Best Period                     :  {0:.8f} days".format(best_period))
        print('   ***********************************************')
        if SB2==True:
            print("   Best Period from secondary      :  {0:.8f} days".format(period2_at_max_power))
            print("   Period from |RV1-RV2|           :  {0:.8f} days".format(period3_at_max_power), 'correct period = P1 or ', period1_at_max_power/2)

        print('   Other periods:')
        print('     peaks                         : ', [f'{x:.3f}' for x in power1[peaks1]])
        print('     positions                     : ', [x for x in peaks1 ])
        # print('     positions                     : ', [f'{x:.3f}' for x in np.where(power1==power[peaks1]) ])
        # print('/////////////// these are the periods', periods)
        print('     frequencies                   : ', [f'{x:.5f}' for x in freq_peaks1])
        print('     periods                       : ', [f'{x:.3f}' for x in peri_peaks1])
        if SB2==True:
            print('   from secondary:')
            print('     peaks                         : ', [f'{x:.3f}' for x in power2[peaks2]])
            print('     frequencies                   : ', [f'{x:.3f}' for x in freq_peaks2])
            print('     periods                       : ', [f'{x:.3f}' for x in peri_peaks2])
    with open(path+'LS/ls_output.txt', 'a') as lsout:
        lsout.write("\n Best frequency                  :  {0:.3f}".format(freq1_at_max_power)+'\n')
        lsout.write(' ****************************************************'+'\n')
        lsout.write(" Best Period                     :  {0:.8f} days".format(best_period)+'\n')
        lsout.write(' ****************************************************'+'\n')
        if SB2==True:
            lsout.write(" Best Period from secondary      :  {0:.8f} days".format(period2_at_max_power)+'\n')
            lsout.write(" Period from |RV1-RV2|           :  {0:.8f} days".format(period3_at_max_power)+ ' correct period = P1 or '+ str(period1_at_max_power/2)+'\n')
        lsout.write(' Other periods:'+'\n')
        lsout.write('   peaks                         : ')
        for peak in power1[peaks1]:
            lsout.write('     '+f'{peak:7.3f}')
        lsout.write('\n   frequencies                   : ')
        for freq in freq_peaks1:
            lsout.write('     '+f'{freq:7.3f}')
        lsout.write('\n   periods                       : ')
        for per in peri_peaks1:
            lsout.write('     '+f'{per:7.3f}')
        if SB2==True:
            lsout.write('\n from secondary:\n')
            lsout.write('   peaks                         : ')
            for peak in power2[peaks2]:
                lsout.write('     '+f'{peak:7.3f}')
            lsout.write('\n   frequencies                   : ')
            for freq in freq_peaks2:
                lsout.write('     '+f'{freq:7.3f}')
            lsout.write('\n   periods                       : ')
            for per in peri_peaks2:
                lsout.write('     '+f'{per:7.3f}')
        lsout.write('\n')

    #################################################################
    #            Setting quality index for the periodogram
    #################################################################
    indi = []
    maxpower = power1.max()
    for LS_pow, peri in zip(power1[peaks1], peri_peaks1):
        maxpower_maxfal = LS_pow/fal1[2]
        maxpower_maxfal2 = LS_pow/fal1[1]
        if print_output==True:
            print('   fal1/P                          : ', f'{maxpower_maxfal:.2f}')
            print('   fal2/P                          : ', f'{maxpower_maxfal2:.2f}')
        with open(path+'LS/ls_output.txt', 'a') as lsout:
            lsout.write(' fal1/P                          :  '+ f'{maxpower_maxfal:.2f}'+'\n')
            lsout.write(' fal2/P                          :  '+ f'{maxpower_maxfal2:.2f}'+'\n')

        conditions = [
            (maxpower > fal1_01pc),                 # FAL 0.1%
            (fal1_01pc >= maxpower > fal1_1pc),     # FAL 1%
            (fal1_1pc >= maxpower > fal1_50pc),     # FAL 50%
            (maxpower <= fal1_50pc)                 # Below 50% FAL
        ]
        indices = [0, 1, 2, 3] 

        # Get the index for each condition that is True
        indi = [index for condition, index in zip(conditions, indices) if condition]

    # If no conditions were True, set ind to 4
    if not indi:
        ind = 4
    else:
        ind = indi
        
    maxpower_maxfal = maxpower/fal1[2].value
    maxpower_maxfal2 = maxpower/fal1[1].value
    if print_output==True:
        # print(indi)
        print('\n   Classification index            : ', ind)
        print('   maxpower                        : ', f'{maxpower:.2f}')
        print('   fal1                            : ', f'{fal1[2]:.2f}')
        print('   maxpower_maxfal                 : ', f'{maxpower_maxfal:.2f}')
    with open(path+'LS/ls_output.txt', 'a') as lsout:
        lsout.write(' Classification index            :  '+str(ind)+'\n')
        lsout.write(' maxpower                        :  '+f'{maxpower:.2f}'+'\n')
        lsout.write(' fal1                            :  '+f'{fal1[2]:.2f}'+'\n')
        lsout.write(' maxpower_maxfal                 :  '+f'{maxpower_maxfal:.2f}'+'\n')

    ls_results['ind'][1] = ind
    ls_results['max_power'][1] = maxpower
    ls_results['pow_over_fal01'][1] = maxpower_maxfal
    ls_results['pow_over_fal1'][1] = maxpower_maxfal2
    
    #################################################################
    #                  Plotting the periodogram
    #################################################################
    # vs Period
    if plots==True:

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
                if power1[peaks1] and (power1[peaks1].max() < maxpower):
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
    #               Compute phases of the obsevations
    #################################################################
    if Pfold:
        # print(peri_peaks1)
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
        elif peri_peaks1.size < 4 & peri_peaks1.size >0:
            periods = peri_peaks1
        else:
            periods = [best_period]
            
        
        for period in periods:
            print(f'Computing phases for period={period}, from {len(periods)} periods')
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

                # for i in range(results2['amplitude'].shape[0]):
                #     fine_pred2 = results2['amplitude'][i] * np.sin(2 * np.pi * fine_phase + results2['phase_shift'][i]) \
                #         + results2['height'][i]
                #     fine_preds2.append(fine_pred2)
                # fine_preds2 = np.array(fine_preds2)  

            else:
                results1, phase, vel1, vel1_err  = phase_rv_curve(hjd1, rv1, period=period, rv_err=rv1_err)

                # fine_phase = np.linspace(phase.min(), phase.max(), 1000)

                fine_preds1, fine_preds2 = [], []
                for i in range(results1['amplitude'].shape[0]):
                    # fine_pred1 = results1['amplitude'][i] * np.sin(results1['frequency'][i] * (fine_phase - results1['phase'][i])) \
                    #     + results1['height'][i]
                    fine_pred1 = results1['amplitude'][i] * np.sin(2 * np.pi * fine_phase + results1['phase_shift'][i]) \
                        + results1['height'][i]

                    fine_preds1.append(fine_pred1)
                fine_preds1 = np.array(fine_preds1)


            # Plotting the phased RV curve
            print('\n*** Plotting phased RV curve ***\n-------------------------------')
            n_lines = 200
            fig, ax = plt.subplots(figsize=(8, 6))
            # ax.plot(phase, result['pred'][-n_lines:].T, rasterized=True, color='C2', alpha=0.1)
            for pred in fine_preds1[-n_lines:]:
                ax.plot(fine_phase, pred, rasterized=True, color='C0', alpha=0.05)
            ax.errorbar(phase, vel1, yerr=vel1_err, color='C2', fmt='o', label='Data')
            if SB2:
                for pred in fine_preds2[-n_lines:]:
                    ax.plot(fine_phase, pred, rasterized=True, color='C1', alpha=0.05)
                ax.errorbar(phase, vel2, yerr=vel2_err, color='k', fmt='^', label='Data')
            ax.set_xlabel('Phase')
            ax.set_ylabel('Radial Velocity [km\,s$^{-1}$]')  
            if SB2:
                plt.savefig(f'{path}LS/{starname}_sinu_fit_SB2_P={period:.2f}.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig(f'{path}LS/{starname}_sinu_fit_SB1_P={period:.2f}.png', bbox_inches='tight', dpi=300)
            plt.close()

    #################################################################
    #                      Return the results
    #################################################################
    # if SB2==True:
    #     return [best_period, period2_at_max_power, period3_at_max_power], [frequency1, frequency2, frequency3],  \
    #             [power1, power2, power3], [power1.max(), power2.max(), power3.max()], [fal1, fal2, fal3], \
    #             [peaks1, peaks2, peaks3], sorted(peri_peaks1), peri_peaks2, indi, [fapper1, maxpower_maxfal, maxpower_maxfal2]
    # else:
    #     return [best_period], [frequency1], [power1], [maxpower], [fal1], [peaks1], sorted(peri_peaks1), ind, [fapper1, maxpower_maxfal, maxpower_maxfal2]
    return ls_results


def fit_sinusoidal_probmod(times, rvs, rv_errors):
    '''
    Probabilistic model for sinusoidal fitting with fixed frequency.
    '''
    def sinu_model(times=None, rvs=None):
        # Fixed frequency based on phasing
        fixed_frequency = 2 * jnp.pi  # 2π radians per phase cycle
        
        # Model parameters
        amplitude = npro.sample('amplitude', dist.Uniform(0, 500))
        phase_shift = npro.sample('phase_shift', dist.Uniform(-jnp.pi, jnp.pi))
        height = npro.sample('height', dist.Uniform(50, 250))
        
        # Sinusoidal model with fixed frequency
        pred = amplitude * jnp.sin(fixed_frequency * times + phase_shift) + height
        
        # Likelihood
        model = npro.deterministic('pred', pred)
        if rv_errors is not None:
            npro.sample('obs', dist.Normal(model, jnp.array(rv_errors)), obs=rvs)
    
    rng_key = random.PRNGKey(0)
    kernel = NUTS(sinu_model)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000)  # Adjusted for efficiency
    mcmc.run(rng_key, times=jnp.array(times), rvs=jnp.array(rvs))
    mcmc.print_summary()
    trace = mcmc.get_samples()

    return trace

def fit_sinusoidal_probmod_sb2(phase, rv1, rv1_err, rv2, rv2_err, amp_max=500, height_min=50, height_max=250):
    '''
    Probabilistic model for sinusoidal fitting of SB2 binary stars.
    '''
    def sb2_model(phase, rv1, rv1_err, rv2, rv2_err):
        fixed_frequency = 2 * jnp.pi  # Fixed frequency for phased data
        
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
    trace = mcmc.get_samples()
    
    return trace

def phase_rv_curve(time, rv1, rv1_er=None, rv2=None, rv2_er=None, period=None, print_output=True, plots=True):
    '''
    Compute phases of the obsevations and models
    '''
    if print_output==True:
        print( '\n*** Computing phases ***')
        print( '------------------------')
    print('  period = ', period)
    if not period:
        print('  No periods provided for primary star')
        return
    
    # Compute phase
    phase = (np.array(time) / period) % 1

    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    rv1_sorted = np.array(rv1.reset_index(drop=True))[sort_idx]
    if rv1_er is not None:
        rv1_er_sorted = np.array(rv1_er.reset_index(drop=True))[sort_idx]
    else:
        rv1_er_sorted = None
    if rv2 is not None:
        rv2_sorted = np.array(rv2.reset_index(drop=True))[sort_idx]
        if rv2_er is not None:
            rv2_er_sorted = np.array(rv2_er.reset_index(drop=True))[sort_idx]
        else:
            rv2_er_sorted = None

    # Duplicate data shifted by -1 and +1 to cover [-1,0) and [1,2)
    phase_neg = phase_sorted - 1  # Shifted to [-1,0)
    phase_pos = phase_sorted + 1  # Shifted to [1,2)
    
    rv1_neg = rv1_sorted.copy()
    rv1_pos = rv1_sorted.copy()
    rv1_err_neg = rv1_er_sorted.copy()
    rv1_err_pos = rv1_er_sorted.copy()
    if rv2 is not None:
        rv2_neg = rv2_sorted.copy()
        rv2_pos = rv2_sorted.copy()
        rv2_err_neg = rv2_er_sorted.copy()
        rv2_err_pos = rv2_er_sorted.copy()
    
    # Concatenate all data
    phase_all = np.concatenate([phase_neg, phase_sorted, phase_pos])
    rv1_all = np.concatenate([rv1_neg, rv1_sorted, rv1_pos])
    rv1_err_all = np.concatenate([rv1_err_neg, rv1_er_sorted, rv1_err_pos])
    if rv2 is not None:
        rv2_all = np.concatenate([rv2_neg, rv2_sorted, rv2_pos])
        rv2_err_all = np.concatenate([rv2_err_neg, rv2_er_sorted, rv2_err_pos])
    
    # Select phases within [-0.5, 1.5]
    mask = (phase_all >= -0.5) & (phase_all <= 1.5)
    phase_expanded = phase_all[mask]
    rv1_expanded = rv1_all[mask]
    rv1_err_expanded = rv1_err_all[mask]
    if rv2 is not None:
        rv2_expanded = rv2_all[mask]
        rv2_err_expanded = rv2_err_all[mask]

    # Fitting a sinusoidal model
    if rv2 is not None:
        result = fit_sinusoidal_probmod_sb2(phase_expanded, rv1_expanded, rv1_err_expanded, rv2_expanded, rv2_err_expanded)    
        return result, phase_expanded, rv1_expanded, rv1_err_expanded, rv2_expanded, rv2_err_expanded
    else:
        result = fit_sinusoidal_probmod(phase_expanded, rv1_expanded, rv1_err_expanded)
        return result, phase_expanded, rv1_expanded, rv1_err_expanded

    
