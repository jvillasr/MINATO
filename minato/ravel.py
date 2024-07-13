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
from lmfit import Model, Parameters, models, minimize, report_fit
from datetime import date
# from iteration_utilities import duplicates, unique_everseen
from scipy.optimize import basinhopping
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpyro as npro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
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
        path = path + 'SB2/'
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
        3995: { 'region':[3990, 4005], 'wid_ini':2, 'title':'N II $\lambda$3995'},
        4009: { 'region':[4005, 4018], 'wid_ini':3, 'title':'He I $\lambda$4009'},
        4026: { 'region':[4017, 4043], 'wid_ini':3, 'title':'He I $\lambda$4026'},
        4102: { 'region':[4085, 4120], 'wid_ini':5, 'title':'H$\delta$'},
        4121: { 'region':[4118, 4130], 'wid_ini':3, 'title':'He I $\lambda$4121'},
        4128: { 'region':[4124, 4136], 'wid_ini':2, 'title':'Si II $\lambda$4128'},
        4131: { 'region':[4128, 4140], 'wid_ini':2, 'title':'Si II $\lambda$4131'},
        4144: { 'region':[4135, 4160], 'wid_ini':4, 'title':'He I $\lambda$4144'},
        4233: { 'region':[4229, 4241], 'wid_ini':2, 'title':'Fe II $\lambda$4233'},
        4267: { 'region':[4263, 4275], 'wid_ini':2, 'title':'C II $\lambda$4267'},
        4340: { 'region':[4320, 4360], 'wid_ini':6, 'title':'H$\gamma$'},
        4388: { 'region':[4380, 4405], 'wid_ini':4, 'title':'He I $\lambda$4388'},
        4471: { 'region':[4462, 4487], 'wid_ini':4, 'title':'He I $\lambda$4471'},
        4481: { 'region':[4478, 4490], 'wid_ini':2, 'title':'Mg II $\lambda$4481'},
        4542: { 'region':[4537, 4552], 'wid_ini':3, 'title':'He II $\lambda$4542'},
        4553: { 'region':[4547, 4562], 'wid_ini':3, 'title':'Si III $\lambda$4553'},
        4861: { 'region':[4840, 4875], 'wid_ini':5, 'title':'H$\beta$'}, 
        4922: { 'region':[4915, 4930], 'wid_ini':4, 'title':'He I $\lambda$4922'}, 
        5412: { 'region':[5405, 5419], 'wid_ini':4, 'title':'He II $\lambda$5412'},
        5876: { 'region':[5865, 5888], 'wid_ini':4, 'title':'He I $\lambda$5876'},  
        5890: { 'region':[5881, 5905], 'wid_ini':3, 'title':'Na I $\lambda$5890'}, 
        6562: { 'region':[6542, 6583], 'wid_ini':5, 'title':'H$\alpha$'}, 
        6678: { 'region':[6668, 6690], 'wid_ini':4, 'title':'He I $\lambda$6678'}, 
        7774: { 'region':[7762, 7786], 'wid_ini':3, 'title':'O I $\lambda$7774'}
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows), sharey=True, sharex=True)
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

def fit_sb2_probmod(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, fig, axes, path):
    '''
    Probabilistic model for SB2 line profile fitting. It uses Numpyro for Bayesian inference, 
    sampling from the posterior distribution of the model parameters using MCMC with the NUTS algorithm. 
    The model includes plates for vectorized computations over epochs and wavelengths. 
    '''
    # Trim the data to the region of interest
    region_start = lines_dic[line]['region'][0]
    region_end = lines_dic[line]['region'][1]
    x_waves = [wave[(wave > region_start) & (wave < region_end)] for wave in wavelengths]
    y_fluxes = [flux[(wave > region_start) & (wave < region_end)] for flux, wave in zip(fluxes, wavelengths)]
    y_errors = [f_err[(wave > region_start) & (wave < region_end)] for f_err, wave in zip(f_errors[:7], wavelengths)]

    # Initial guess for the central wavelength and width
    cen_ini = line+shift
    wid_ini = lines_dic[line]['wid_ini']
    amp_ini = 0.9 - min([flux.min() for flux in y_fluxes])
    
    def sb2_model(wavelengths=None, fluxes=None):

        n_epoch, n_wavelength = fluxes.shape

        #spectral window
        cen = npro.param("cen", (cen_ini))
        # delta_cen = npro.param("delta_cen", (wavelengths[0].max() - wavelengths[0].min()) / 8)
        # cen = npro.param("cen", (wavelengths[0].max() + wavelengths[0].min())/2)
        delta_cen = npro.param("delta_cen", (wavelengths[0].max() - wavelengths[0].min()))
        # print('centre =', cen, '+/-', delta_cen)

        # Model parameters
        continuum = npro.sample('continuum', dist.Normal(1, 0.1))
        amp1 = npro.sample('amp1', dist.Uniform(0, 1))
        amp2 = npro.sample('amp2', dist.Uniform(0, amp1))
        wid1 = npro.sample('wid1', dist.Uniform(0.5, 11))
        wid2 = npro.sample('wid2', dist.Uniform(0.5, 11))
        # print(f'amp1_ini = {amp_ini} +/- {amp_ini*0.2}')
        # print(f'amp2_ini = {amp_ini*0.6} +/- {amp_ini*0.6*0.2}')
        # amp1 = npro.sample('amp1', dist.Normal(amp_ini, amp_ini*0.2))
        # amp2 = npro.sample('amp2', dist.Normal(amp_ini*0.6, amp_ini*0.6*0.2))
        # wid1 = npro.sample('wid1', dist.Normal(wid_ini, wid_ini*0.2))
        # wid2 = npro.sample('wid2', dist.Normal(wid_ini, wid_ini*0.2))
        # logsig = npro.sample('logsig', dist.Normal(-2, 1))

        with npro.plate(f'epoch=1..{n_epoch}', n_epoch, dim=-2):
            mean1 = npro.sample('mean1', dist.Uniform(cen - 0.2 * delta_cen, cen + 0.2 * delta_cen))
            mean2 = npro.sample('mean2', dist.Uniform(cen - 0.2 * delta_cen, cen + 0.2 * delta_cen))
            with npro.plate(f'wavelength=1..{n_wavelength}', n_wavelength, dim=-1):
                if line in Hlines:
                    # comp1 = -amp1 * jnp.exp(dist.Cauchy(mean1, wid1).log_prob(wavelengths))
                    # comp2 = -amp2 * jnp.exp(dist.Cauchy(mean2, wid2).log_prob(wavelengths))
                    comp1 = lorentzian(wavelengths, amp1, mean1, wid1)
                    comp2 = lorentzian(wavelengths, amp2, mean2, wid2)
                else:
                    # comp1 = -amp1 * jnp.exp(dist.Normal(mean1, wid1).log_prob(wavelengths))
                    # comp2 = -amp2 * jnp.exp(dist.Normal(mean2, wid2).log_prob(wavelengths))
                    comp1 = gaussian(wavelengths, amp1, mean1, wid1)
                    comp2 = gaussian(wavelengths, amp2, mean2, wid2)
                pred = continuum + comp1 + comp2
                npro.deterministic('pred_1', continuum + comp1)
                npro.deterministic('pred_2', continuum + comp2)
                model = npro.deterministic('pred', pred)
                npro.sample('obs', dist.Normal(model, 0.05), obs=fluxes)
     
    rng_key = random.PRNGKey(0)

    kernel = NUTS(sb2_model)
    mcmc = MCMC(kernel, 
                num_warmup=5000, 
                num_samples=5000)

    mcmc.run(rng_key, wavelengths=jnp.array(x_waves), fluxes=jnp.array(y_fluxes))
    mcmc.print_summary()
    trace = mcmc.get_samples()

    n_lines = 100
    n_epochs = len(x_waves)
    for epoch, ax in zip(range(n_epochs), axes.ravel()):
        ax.plot(x_waves[epoch], trace['pred'][-n_lines:, epoch, :].T, rasterized=True, color='C2', alpha=0.1)
        ax.plot(x_waves[epoch], trace['pred_1'][-n_lines:, epoch, :].T, rasterized=True, color='C0', alpha=0.1)
        ax.plot(x_waves[epoch], trace['pred_2'][-n_lines:, epoch, :].T, rasterized=True, color='C1', alpha=0.1)
        ax.plot(x_waves[epoch], y_fluxes[epoch], color='k', lw=1, alpha=0.8)
        
    fig.supxlabel('Wavelength', size=22)
    fig.supylabel('Flux', size=22)  
    plt.savefig(path+str(line)+'_fits_SB2_.png', bbox_inches='tight', dpi=150)
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
    pars[f'{prefix}_amp'].set(1.2-y_flux.min(), min=0.01, max=2. )
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

def mcmc_results_to_file(trace, names, line, writer, csvfile):
    # samples = mcmc.get_samples()
    samples = trace
    # print(samples)
    # for key in samples.keys():
    #     print(f"{key}: {samples[key]}")
    # print(samples.keys()
    # print({(key, type(key)) for key in samples.keys()})
    results_dict = {}
    for i in range(len(names)):
        results_dict['epoch'] = names[i]
        results_dict['line'] = line
        for j in range(1, 3):
            prefix = f'mean{j}'
            if prefix in samples:

                # Calculate mean and standard deviation for each epoch
                means = np.mean(samples[prefix], axis=0)
                stds = np.std(samples[prefix], axis=0)

                # Add the mean and standard deviation for the current epoch to the dictionary
                results_dict[f'cen{j}'] = means[i, 0]
                results_dict[f'cen{j}_er'] = stds[i, 0]

            else:
                print(f"Prefix {prefix} not found in samples")

        # print(f"Results dict: {results_dict}")
        # Add common parameters
        common_params = ['amp1', 'wid1', 'amp2', 'wid2']
        for param in common_params:
            if param in samples:
                results_dict[param] = np.mean(samples[param])
                results_dict[f'{param}_er'] = np.std(samples[param])

        # If writer is None, this is the first dictionary, so get the fieldnames from its keys
        if writer is None:
            fieldnames = results_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(results_dict)

    return writer

def SLfit(spectra_list, path, lines, file_type='fits', plots=True, balmer=True, neblines=[4102, 4340], doubem=[], SB2=False, shift=0, use_init_pars=False):
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

    if len(spectra_list) == 1:
        print("\n   WARNING: There is only 1 epoch to compute RVs.")
        return    
    # doubem = doubem[0]
    Hlines = [4102, 4340, 6562]

    print('*** SB2 set to: ', SB2, ' ***\n')
    
    wavelengths, fluxes, f_errors, names, jds = read_spectra(spectra_list, path, file_type)

    path = setup_star_directory_and_save_jds(names, jds, path, SB2)

    lines_dic = setup_line_dictionary()

    print('these are the lines: ', lines)

    (
        cen1, cen1_er, amp1, amp1_er, wid1, wid1_er, 
        cen2, cen2_er, amp2, amp2_er, wid2, wid2_er, 
        dely, sdev, results, comps, delta_cen, chisqr
    ) = initialize_fit_variables(lines)

    with open(path + 'fit_values.csv', 'w', newline='') as csvfile:
        writer = None

        for i, line in enumerate(lines):
            print('Fitting line ', line)
            if plots:
                fig, axes = setup_fits_plots(wavelengths)
            else:
                axes = [None] * len(wavelengths)
            if SB2:
                result, x_wave, y_flux = fit_sb2_probmod(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, fig, axes, path)
                # result, x_wave, y_flux = fit_sb2(line, wavelengths, fluxes, f_errors, lines_dic, Hlines, neblines, shift, axes, path)
                # writer = sb2_results_to_file(result, wavelengths, names, line, writer, csvfile)
                writer = mcmc_results_to_file(result, names, line, writer, csvfile)
                
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
        if print_output==True:
            print('     mean(x)        =', f'{np.nanmean(data):.3f}')
            print('     median(x)        =', f'{np.nanmedian(data):.3f}')
            print('     x-median(x)      =', [f'{x:.3f}' for x in diff])
            print('     abs(x-median(x)) =', [f'{abs(x):.3f}' for x in diff])
        mad = 1.4826*np.nanmedian(abs(diff))
        if print_output==True:
            print('     mad =', f'{mad:.3f}')
            print('     abs(x-median(x)) <', (f'{thresh*mad:.3f}'), '(thresh*mad) =', [abs(x)<thresh*mad for x in diff])
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
            lambda_rest, lambda_r_er = lambda_rest_dict[line]
            current_line_values = self.df_SLfit[self.df_SLfit['line'] == line]

            dlambda1 = current_line_values['cen1'].values - lambda_rest
            dlambda1_er = np.sqrt(current_line_values['cen1_er'].values**2 + lambda_r_er**2)
            rv1 = dlambda1 * c_kms / lambda_rest
            rv1_er = np.sqrt((dlambda1_er / dlambda1)**2 + (lambda_r_er / lambda_rest)**2) * rv1

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
        out.write('   Lines   amp1    cen1   wid1   |    amp2    cen2   wid2 \n')
        out.write('   -------------------------------------------------------\n')
        for line in grouped_errors['cen1_percer'].keys():
            stats = ' '.join([f'{stat_func(grouped_errors[error_type][line]):7.3f}' for error_type in grouped_errors])
            out.write(f'   {line}: {stats}\n')
        out.write('\n')

    def select_lines(self, error_type, lines):
        if not self.use_lines:
            best_lines_index, rm_lines_idx = self.outlier_killer([np.nanmedian(x) for x in error_type], thresh=self.lines_ok_thrsld)
        else:
            best_lines_index = [i for i, line in enumerate(lines) if line in self.use_lines]
            rm_lines_idx = [i for i, line in enumerate(lines) if line not in self.use_lines]

        best_lines = [lines[x] for x in best_lines_index]
        return best_lines, best_lines_index, rm_lines_idx

    def remove_bad_epochs(self, lines_index, metric='mean', error_type='cen'):
        self.print_output and print('\n*** Choosing best epochs for analysis ***\n---------------------------------------------')
        error_list = []
        err_type_dic={'wid':'wid1_er', 'rvs':'rvs1_er', 'cen':'cen1_er', 'amp':'amp1_er'}
        self.print_output and print('\n   '+'-'*34+'\n   Removing epochs with large errors:')
        with open(self.path+'rv_stats.txt', 'a') as out:
            out.write(' Removing epochs with large errors:\n')

            for epoch in self.df_SLfit['epoch'].unique():
                df_epoch = self.df_SLfit[self.df_SLfit['epoch'] == epoch]
                error = df_epoch[err_type_dic[error_type]].mean() if metric == 'mean' \
                    else df_epoch[err_type_dic[error_type]].median()
                error_list.append(error)

            self.print_output and print('\n   Applying outlier_killer to remove epochs')
            rm_OBs_idx = self.outlier_killer(error_list, thresh=self.epochs_ok_thrsld)[1]

            self.print_output and print(f'   epochs removed: {[x+1 for x in rm_OBs_idx]}')
            out.write(f'   epochs removed: {[x+1 for x in rm_OBs_idx]}\n\n')

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
                line_avg2[line] = {'mean': np.mean(rv1), 'std': np.std(rv1)}
            for epoch in range(len(rv2)):
                weighted_mean2, weighted_error2 = GetRVs.weighted_mean([rvs_dict[line]['rv2'][epoch] for line in lines], 
                                                                       [rvs_dict[line]['rv2_er'][epoch] for line in lines])
                wmean_rvs2[epoch] = {'value': weighted_mean2, 'error': weighted_error2}

        return wmean_rvs1, wmean_rvs2, line_avg1, line_avg2

    def print_and_write_results(self, lines, line_avg1, total_mean_rv1, line_avg2=None, total_mean_rv2=None):
        # Prepare the lines to print/write
        rows = []
        rows.append(f'RV mean of the {self.nepochs} epochs for each line:')
        rows.append('---------------------------------------')
        print('line_avg1:', line_avg1)
        print('line_avg2:', line_avg2)
        for line in lines:
            mean = line_avg1[line]['mean']
            std = line_avg1[line]['std']
            rows.append(f'   - {line}: {mean:.3f} +/- {std:.3f}')
        if self.SB2:
            for line in lines:
                mean = line_avg2[line]['mean']
                std = line_avg2[line]['std']
                rows.append(f'   - {line}: {mean:.3f} +/- {std:.3f}')                
        rows.append('')
        rows.append(f'Weighted mean RV of the {self.nepochs} epochs:')
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
        markers = ['o', 'v', '^', '<', '>', 's']
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


        primary_error_types = ['wid1_percer', 'cen1_percer', 'amp1_percer']
        secondary_error_types = ['wid2_percer', 'cen2_percer', 'amp2_percer'] if self.SB2 else []
        
        grouped_errors = {error_type: self.df_SLfit.groupby('line')[error_type].apply(list) for error_type in primary_error_types + secondary_error_types}
        # for line, values in grouped_errors['cen2_percer'].items():
        #     print(f"Line {line}: {values}")
        # print(grouped_errors['cen2_percer'][4026])


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
    
        '''
        Selecting the line-rejection type
        '''
        if self.print_output:
            print('\n   --------------------------------------------')
            if not self.use_lines:
                print('   Applying outlier_killer to remove bad lines:')
            else:
                print('   Selecting lines determined by user:')

        if self.SB2:
            error_type = 'cen2_percer'

        
        best_lines, best_lines_index, rm_lines_idx = self.select_lines(grouped_errors[error_type], self.lines)
        if self.print_output:
            print('\n   These are the best lines: ', best_lines)

        nlines = len(best_lines)

        with open(self.path+'rv_stats.txt', 'a') as out:
            out.write('\n')
            out.write(f' Lines with the best fitted profile according to the median {error_type} criterion:\n')
            out.write(' --------------------------------------------------------------------------\n')
            for i in range(nlines):
                if i<range(nlines)[-1]:
                    out.write('   '+str(best_lines_index[i])+': '+str(best_lines[i])+', ')
                else:
                    out.write('   '+str(best_lines_index[i])+': '+str(best_lines[i])+'\n')
            out.write('\n')



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

        if self.rm_epochs:
            # removing bad/noisy epochs    
            rm_OBs_idx = self.remove_bad_epochs(best_lines_index, metric='mean', error_type='cen')

        ########################################################################

        if self.print_output==True:
            print( '\n')
            print( '*** Calculating the RV weighted mean for each epoch  ***')
            print( '--------------------------------------------------------')

        wmean_rv1, wmean_rv2, line_avg1, line_avg2 = self.compute_weighted_mean_rvs(rvs_dict, best_lines, rm_OBs_idx)
        total_rv1, total_rv2 = {}, {}
        total_rv1['mean'] = np.mean([wmean_rv1[i]['value'] for i in wmean_rv1.keys()])
        total_rv1['std'] = np.std([wmean_rv1[i]['value'] for i in wmean_rv1.keys()])
        total_rv2['mean'] = np.mean([wmean_rv2[i]['value'] for i in wmean_rv2.keys()])
        total_rv2['std'] = np.std([wmean_rv2[i]['value'] for i in wmean_rv2.keys()])

        # Printing/Writing
        self.print_and_write_results(best_lines, line_avg1, total_rv1, line_avg2=line_avg2, total_mean_rv2=total_rv2)

        
        #################################################################
        #                Writing RVs to file RVs.txt
        #################################################################
        df_rv = pd.read_csv(self.JDfile, names = ['epoch', 'MJD'], sep='\s+')
        df_rv = df_rv.replace({'BBC_':''}, regex=True).replace({'.fits':''}, regex=True)
        if self.rm_epochs and rm_OBs_idx:
            df_rv.drop(rm_OBs_idx, inplace=True)
            df_rv.reset_index(drop=True, inplace=True)
        rv1_values = []
        for line in best_lines:
            df_rv[f'rv_{line}'] = rvs_dict[line]['rv1']
            df_rv[f'rv_{line}_er'] = rvs_dict[line]['rv1_er']
            rv1_values.append(rvs_dict[line]['rv1'])
        df_rv['mean_rv'] = [wmean_rv1[i]['value'] for i in range(len(wmean_rv1))]
        df_rv['mean_rv_er'] = [wmean_rv1[i]['error'] for i in range(len(wmean_rv1))]
        df_rv['sigma_rv'] = np.std(np.stack(rv1_values), axis=0)
        df_rv['comp'] = 1

        if self.SB2:
            df_rv2 = pd.read_csv(self.JDfile, names = ['epoch', 'MJD'], sep='\s+')
            df_rv2 = df_rv2.replace({'BBC_':''}, regex=True).replace({'.fits':''}, regex=True)
            if self.rm_epochs and rm_OBs_idx:
                df_rv2.drop(rm_OBs_idx, inplace=True)
                df_rv2.reset_index(drop=True, inplace=True)
            rv2_values = []
            for line in best_lines:
                df_rv2[f'rv_{line}'] = rvs_dict[line]['rv2']
                df_rv2[f'rv_{line}_er'] = rvs_dict[line]['rv2_er']
                rv2_values.append(rvs_dict[line]['rv2'])
            df_rv2['mean_rv'] = [wmean_rv2[i]['value'] for i in range(len(wmean_rv2))]
            df_rv2['mean_rv_er'] = [wmean_rv2[i]['error'] for i in range(len(wmean_rv2))]
            df_rv2['sigma_rv'] = np.std(np.stack(rv2_values), axis=0)
            df_rv2['comp'] = 2
            df_rv = pd.concat([df_rv, df_rv2], ignore_index=True)

        with open(self.path+'RVs1.txt', 'w') as fo:
            fo.write(df_rv.to_string(formatters={'MJD': '{:.8f}'.format}, index=False))

        # RV plot per spectral line
        data = pd.read_csv(self.path+'RVs1.txt', delim_whitespace=True)
        primary_data = data[data['comp'] == 1]
        secondary_data = data[data['comp'] == 2]
        fig, ax = plt.subplots()
        rv_lines = [f'rv_{line}' for line in best_lines]
        markers = ['o', 'v', '^', '<', '>', 's']
        for rv_line, marker in zip(rv_lines, markers):
            ax.errorbar(primary_data['MJD'], primary_data[rv_line], fmt=marker, 
                        color='dodgerblue', label=f'Primary {rv_line}, alpha=0.5')
        for rv_line, marker in zip(rv_lines, markers):
            ax.errorbar(secondary_data['MJD'], secondary_data[rv_line], fmt=marker, 
                        color='darkorange', label=f'Secondary {rv_line}, alpha=0.5')
        ax.set_xlabel('Julian Date')
        ax.set_ylabel('Mean Radial Velocity')
        ax.legend(fontsize=12)
        plt.savefig(self.path+'RVs1.png', bbox_inches='tight', dpi=150)
        plt.show()
        plt.show()

        return df_rv

def lomb_scargle(df, path, probabilities = [0.5, 0.01, 0.001], SB2=False, rv2=False, rv2_err=False, print_output=False, plots=False, best_lines=False, min_sep=False):
    '''
    df: dataframe with the RVs, output from getrvs()
    '''
    if not os.path.exists(path+'LS'):
        os.makedirs(path+'LS')

    hjd, rv, rv_err = df['MJD'], df['mean_rv'], df['mean_rv_er']
    starname = df['epoch'][0].split('_')[0]+'_'+df['epoch'][0].split('_')[1]
    print('starname:', starname)
    nepochs=len(hjd)

    lsout = open(path+'LS/ls_output.txt', 'w')
    lsout.write(' *************************\n')
    lsout.write('   LS output for star '+starname+'\n')
    lsout.write(' *************************\n')
    lsout.write('\n')

    # probabilities = [0.5, 0.01, 0.001]
    # print(rv)
    #ls = LombScargle(df_rv['HJD'], df_rv['mean_rv'], df_rv['sigma_rv'])
    # print(type(rv_err), rv_err)
    if rv_err.any() == False:
        ls1 = LombScargle(hjd, rv, normalization='model')
    else:
        ls1 = LombScargle(hjd, rv, rv_err, normalization='model')
    #method = 'baluev'
    method = 'bootstrap'
    fal1 = ls1.false_alarm_level(probabilities, method=method)
    if print_output==True:
        print('   these are the false alarm levels: ', [f'{x:.3f}' for x in fal1])
    lsout.write(' these are the false alarm levels: ')
    for fal in fal1:
        lsout.write(str(f'{fal:.3f}')+'  ')
    lsout.write('\n')
    # t0 = time.time()
    frequency1, power1 = ls1.autopower(method='fast', minimum_frequency=0.001, #0.001=1000
                                                maximum_frequency=2.5, # 0.9=1.1, 1.5=0.67, 2=0.5, 4=0.25
                                                samples_per_peak=2000) # 2000
    # t1 = time.time()
    # print('# LS calculation time: ', t1-t0)
    fap1 = ls1.false_alarm_probability(power1.max(), method=method)
    # t2 = time.time()
    # print('# FAP calculation time: ', t2-t1)
    fapper1 = fap1*100
    if print_output==True:
        print('   FAP of the highest peak         : ', f'{fap1:.5f}')
        print('   FAP of the highest peak x100    : ', f'{fapper1:.5f}')
    lsout.write(' FAP of the highest peak         : '+ f'{fap1:.5f}'+'\n')
    lsout.write(' FAP of the highest peak x100    : '+ f'{fapper1:.5f}'+'\n')

    max_freq1 = frequency1[np.argmax(power1)]
    # print('max_freq1 =', max_freq1)

    fal_50pc, fal_1pc, fal_01pc = fal1[0], fal1[1], fal1[2] 
    if power1.max() > fal_01pc:
        peaks_min_h = float(f'{fal_1pc:.3f}')
    elif power1.max() <= fal_01pc and power1.max() > fal_1pc:
        peaks_min_h = float(f'{0.6*fal_1pc:.3f}')
    else:
        peaks_min_h = float(f'{fal_50pc:.3f}')
    # print(peaks_min_h)
    
    freq_index1 = np.argmin(np.abs(frequency1 - 1/5))
    freq_index2 = np.argmin(np.abs(frequency1 - 1/1.1))
    peaks1a, _ = find_peaks(power1[:freq_index1], height=peaks_min_h, distance=1000)
    peaks1b, _ = find_peaks(power1[:freq_index2], height=peaks_min_h, distance=5000)
    # print(peaks1a)
    # print(peaks1b)
    peaks1 = np.unique(np.concatenate((peaks1a, peaks1b)))
    # print(peaks1)
    # peaks1, _ = find_peaks(power1, height=0.8*fal1[1], distance=20000) # height=0.8*fal1[1], distance=20000
    # peaks1, _ = find_peaks(power1, height=0.5*fal1[2], distance=1000) # height=0.8*fal1[1], distance=20000
    freq1 = frequency1[peaks1]
    peri1 = 1/frequency1[peaks1]

    if 1/max_freq1 < 1.2 and any(1/freq1 > 1.2):
        for i in range(len(1/freq1)):
            if 1/freq1[i]==max(1/freq1):
                best_freq1 = freq1[i]
                LS_power1 = power1[peaks1][i]
    else:
        best_freq1 = max_freq1
        LS_power1 = max(power1)
    best_period1 = 1/best_freq1
    # print('freq1 =', freq1)
    # print('peri1 =', peri1)
    # print('best_freq1 =', best_freq1)
    # print('best_period1 =', best_period1)

    if len(peri1)>0:
        max_peri1 = max(peri1)
        max_peri1_idx = peri1.tolist().index(max_peri1)
        # print('max_peri_idx', max_peri_idx)
        max_peri1_pow = power1[peaks1][max_peri1_idx]
        # print('max_peri_pow', max_peri_pow)
        # print('fal[1]/max_peri_pow', fal[1]/max_peri_pow)
        allP = sorted(peri1)
    if len(peri1)==0:
        peri1 = [best_period1]

    # periods.append(1/frequency1[peaks1])
    # frequencies.append(frequency1[peaks1])

    if SB2==True:
        if rv2_err == False:
            ls2 = LombScargle(hjd, rv2, normalization='model')
        else:
            ls2 = LombScargle(hjd, rv2, rv2_err, normalization='model')
        fal2 = ls2.false_alarm_level(probabilities, method=method)
        frequency2, power2 = ls2.autopower(method='fast', minimum_frequency=0.001, #0.001=1000
                                                    maximum_frequency=2.5, # 0.9=1.1, 1.5=0.67, 2=0.5
                                                    samples_per_peak=2000)
        fap2 = ls2.false_alarm_probability(power2.max(), method=method)
        fapper2 = fap2*100
        LS_power2 = max(power2)
        best_freq2 = frequency2[np.argmax(power2)]
        best_period2 = 1/best_freq2
        peaks2_min_h = float(f'{0.8*fal1[1]:.3f}')
        # print(peaks2_min_h)
        peaks2, _ = find_peaks(power2, height=peaks2_min_h, distance=20000) # distance=20000
        # peaks2, _ = find_peaks(power2, height=0.5*fal2[2], distance=1000) # distance=20000
        peri2 = 1/frequency2[peaks2]
        freq2 = frequency2[peaks2]
        if best_period2 < 1.1:
            if print_output==True:
                print('\n')
                print('   ### Warning: Secondary period possible alias, selecting most similar period to P$\_$1')
                print('\n')
            lsout.write(' ### Warning: Secondary period possible alias, selecting most similar period to P$\_$1 \n')
            dif = []
            for i in range(len(freq2)):
                dif.append(np.abs(best_freq1-freq2[i]))
            for i in range(len(freq2)):
                if np.abs(best_freq1-freq2[i]) == np.min(dif):
                    best_freq2   = freq2[i]
                    best_period2 = 1/freq2[i]
        # test |RV1-RV2|
        rv_abs = np.abs(rv-rv2)
        rv_abs_er = np.sqrt(rv_err**2+rv2_err**2)
        ls3 = LombScargle(hjd, rv_abs, rv_abs_er, normalization='model')
        fal3 = ls3.false_alarm_level(probabilities, method=method)
        frequency3, power3 = ls3.autopower(method='fast', minimum_frequency=0.001, #0.001=1000
                                                    maximum_frequency=2.5, # 0.9=1.1, 1.5=0.67, 2=0.5
                                                    samples_per_peak=2000)
        fap3 = ls3.false_alarm_probability(power3.max(), method=method)
        LS_power3 = max(power3)
        best_freq3 = frequency3[np.argmax(power3)]
        best_period3 = 1/best_freq3
        peaks3_min_h = float(f'{0.8*fal3[1]:.3f}')
        peaks3, _ = find_peaks(power3, height=peaks3_min_h, distance=20000) # distance=20000


    # t3 = time.time()
    # print('# Peaks calculation time and all before prints: ', t3-t2)
    # print("   number of periods               :  {0}".format(len(frequency1)))
    if print_output==True:
        print("   Best frequency                  :  {0:.3f}".format(best_freq1))
        print('   ***********************************************')
        print("   Best Period                     :  {0:.8f} days".format(best_period1))
        print('   ***********************************************')
        if SB2==True:
            print("   Best Period from secondary      :  {0:.8f} days".format(best_period2))
            print("   Period from |RV1-RV2|           :  {0:.8f} days".format(best_period3), 'correct period = P1 or ', best_period1/2)

        print('   Other periods:')
        print('     peaks                         : ', [f'{x:.3f}' for x in power1[peaks1]])
        print('     positions                     : ', [x for x in peaks1 ])
        # print('     positions                     : ', [f'{x:.3f}' for x in np.where(power1==power[peaks1]) ])
        # print('/////////////// these are the periods', periods)
        print('     frequencies                   : ', [f'{x:.5f}' for x in freq1])
        print('     periods                       : ', [f'{x:.3f}' for x in peri1])
        if SB2==True:
            print('   from secondary:')
            print('     peaks                         : ', [f'{x:.3f}' for x in power2[peaks2]])
            print('     frequencies                   : ', [f'{x:.3f}' for x in freq2])
            print('     periods                       : ', [f'{x:.3f}' for x in peri2])
    lsout.write("\n Best frequency                  :  {0:.3f}".format(best_freq1)+'\n')
    lsout.write(' ****************************************************'+'\n')
    lsout.write(" Best Period                     :  {0:.8f} days".format(best_period1)+'\n')
    lsout.write(' ****************************************************'+'\n')
    if SB2==True:
        lsout.write(" Best Period from secondary      :  {0:.8f} days".format(best_period2)+'\n')
        lsout.write(" Period from |RV1-RV2|           :  {0:.8f} days".format(best_period3)+ 'correct period = P1 or '+ str(best_period1/2)+'\n')
    lsout.write(' Other periods:'+'\n')
    lsout.write('   peaks                         : ')
    for peak in power1[peaks1]:
        lsout.write('     '+f'{peak:7.3f}')
    lsout.write('\n   frequencies                   : ')
    for freq in freq1:
        lsout.write('     '+f'{freq:7.3f}')
    lsout.write('\n   periods                       : ')
    for per in peri1:
        lsout.write('     '+f'{per:7.3f}')
    if SB2==True:
        lsout.write('\n from secondary:\n')
        lsout.write('   peaks                         : ')
        for peak in power2[peaks2]:
            lsout.write('     '+f'{peak:7.3f}')
        lsout.write('\n   frequencies                   : ')
        for freq in freq2:
            lsout.write('     '+f'{freq:7.3f}')
        lsout.write('\n   periods                       : ')
        for per in peri2:
            lsout.write('     '+f'{per:7.3f}')
    lsout.write('\n')
    # print(freq[2]-freq[1], freq[0])
    # print('beat period (f2-f1) = ',2*(freq[2]-freq[1]))
    # print('beat period (f1-f0) = ',(freq[1]-freq[0]))
    # print('beat period (f3-f2) = ',(freq[3]-freq[2]))
    # print('beat period 2/(f2-f1) = ',2/(freq[2]-freq[1]))
    # print("Best Period: {0:.8f} days".format(best_period))

   # print('allP: ', allP)
    #print('number of local maxima: '+str(len(mm[0])))
    #yy = [power[i] for i in mm]
    # t4 = time.time()
    # print('# Printing/writing time: ', t4-t3)

    indi = []
    for LS_pow, peri in zip(power1[peaks1], peri1):

        maxpower_maxfal = fal1[2]/LS_pow
        maxpower_maxfal2 = fal1[1]/LS_pow
        if print_output==True:
            print('   fal1/P                          : ', f'{maxpower_maxfal:.2f}')
            print('   fal2/P                          : ', f'{maxpower_maxfal2:.2f}')
            # print('fal1[2]/LS_pow :', fal1[2], '/', LS_pow, ' = ', fal1[2]/LS_pow)
        lsout.write(' fal1/P                          :  '+ f'{maxpower_maxfal:.2f}'+'\n')
        lsout.write(' fal2/P                          :  '+ f'{maxpower_maxfal2:.2f}'+'\n')
        # print('maxpower_maxfal =', maxpower_maxfal)
        # print('maxpower_maxfal2 =', maxpower_maxfal2)
        # print('peri =', peri)
        # print('peri1 =', peri1, ' len(peri1) =', len(peri1))
        # print('allP[-1] =', allP[-1])
        # print('fal1[1]/LS_power1', fal1[1]/LS_power1)
        # print('fal1[1]/max_peri1_pow', fal1[1]/max_peri1_pow)

        if len(peri1)>0:
            if maxpower_maxfal < 0.2 and peri >= 1.1:
                indi.append(0)
            elif maxpower_maxfal >= 0.2 and maxpower_maxfal < 0.6 and peri >= 1.1:
                indi.append(1)
            elif maxpower_maxfal >= 0.6 and maxpower_maxfal < 0.95 and peri >= 1.1:
                indi.append(2)
            elif peri >= 1.05 and maxpower_maxfal >= 0.95 and maxpower_maxfal2 < 0.95:
                indi.append(3)
            elif peri < 1.05 and sorted(peri1)[-1] > 1.05 and maxpower_maxfal < 1.0 and fal1[1]/LS_power1 < 1.0:
                indi.append(3)
            elif peri >= 1.05 and maxpower_maxfal >= 0.9 and maxpower_maxfal2 < 1.2:
                indi.append(4)
            elif peri < 1.05 and sorted(peri1)[-1] > 1.05 and maxpower_maxfal2 > 1.1 and len(peri1) > 1 and fal1[1]/LS_power1 < 1.25:
                indi.append(4)
            elif peri < 1.05 and sorted(peri1)[-1] > 1.05 and maxpower_maxfal2 < 1.1 and len(peri1) > 1 and fal1[1]/LS_power1 < 1.25:
                indi.append(4)
            elif peri >= 1.05 and maxpower_maxfal > 1.5 and maxpower_maxfal2 < 1.5:
                indi.append(5)
            elif peri < 1.05 and maxpower_maxfal2 < 1.1 and len(peri1) < 2:
                indi.append(5)
            elif peri < 1.05 and maxpower_maxfal2 < 1.1 and len(peri1) > 1 and sorted(peri1)[-1] < 1.05:
                indi.append(5)
            elif peri < 1.05 and maxpower_maxfal2 > 1.1 and len(peri1) > 1 and sorted(peri1)[-1] > 1.05 and fal1[1]/max_peri1_pow > 1.1:
                indi.append(5)
            else:
                indi.append(6)
           # for i in range(len(df_Bstars)):
           #     if df_Bstars['VFTS'][i] == star:
           #         if df_Bstars['SB2'][i] in ['SB2', 'SB2?']:
           #             ind = 7
        else:
            indi.append(6)
        if peri == best_period1:
            ind = indi[-1]
        # print('indi=',indi)
        # print('best_period1=',best_period1)
        # print('\n')
    if not indi:
        ind = 6
    elif all(x==6 for x in indi):
        ind = 6
    # print('indi =', indi)
    maxpower_maxfal = fal1[2]/LS_power1
    maxpower_maxfal2 = fal1[1]/LS_power1
    if print_output==True:
        # print(indi)
        print('\n   Classification index            : ', ind)
        print('   LS_power1                       : ', f'{LS_power1:.2f}')
        print('   fal1                            : ', f'{fal1[2]:.2f}')
        print('   maxpower_maxfal                 : ', f'{maxpower_maxfal:.2f}')
    lsout.write(' Classification index            :  '+str(ind)+'\n')
    # t5 = time.time()
    # print('# Classification index calculation time: ', t5-t4)
   # print('  fal1/power: ', maxpower_maxfal)
   # print('  fal2/power: ', maxpower_maxfal2)


    '''
    Plot periodogram
    '''
    # vs Period
    if plots==True:
        bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        if SB2==True:
            f_list, pow_list, comp_list, Per_list, fal_list, peak_list = [frequency1, frequency2, frequency3], \
                        [power1, power2, power3], ['primary', 'secondary', 'subtracted'], \
                        [best_period1, best_period2, best_period3], [fal1, fal2, fal3], [peaks1, peaks2, peaks3]
        else:
            f_list, pow_list, comp_list, Per_list, fal_list, peak_list = [frequency1], \
                [power1], ['primary'], [best_period1], [fal1], [peaks1]
        for frequency, power, comp, best_period, fal, peaks in \
                zip(f_list, pow_list, comp_list, Per_list, fal_list, peak_list):
            if not any(np.isnan(power)):
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
                ax.plot(1/frequency, power, 'k-', alpha=0.5)
                #ax.plot(1/frequency[peaks0], power[peaks0], "xr", label='distance=200')
                ax.plot(1/frequency[peaks], power[peaks], "ob", label='prominence=1')
                #ax.plot(1/frequency[peaks2], power[peaks2], "vg", label='width=2000')
                #ax.plot(1/frequency[peaks3], power[peaks3], "xk", label='threshold=2')
                #ax.plot(1/frequency[peaks4], power[peaks4], "xk", label='height=1')
                #ax.plot(1/frequency[mm], power[mm], 'ro')
                ax.yaxis.set_label_coords(-0.09, 0.5)
                ax.set(xlim=(0.3, 1300), ylim=(-0.03*power.max(), power.max()+0.1*power.max()), #x2=426
                # ax.set(xlim=(0.4, 600), ylim=(0, 35), #x2=426
                   xlabel='Period (days)',
                   ylabel='Lomb-Scargle Power');
                # ax.tick_params(which='both', width=0.6, labelsize=14)
                plt.xscale('log')
                #ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                tickLabels = map(str, bins)
                ax.set_xticks(bins)
                ax.set_xticklabels(tickLabels)
                # ax.set_yticks(range(0,35,5))
                ax.plot( (0.5, 800), (fal[0], fal[0]), '--r', lw=1.2)
                ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
                ax.plot( (0.5, 800), (fal[2], fal[2]), '--g', lw=1.2)
                ax.text( 100, fal[2]+0.01, '0.1\% fap', fontsize=16)
                ax.text( 100, fal[1]+0.01, '1\% fap', fontsize=16)
                ax.text( 100, fal[0]+0.01, '50\% fap', fontsize=16)
                # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                if power.max()+0.1*power.max() >= 10:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
                else:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
                plt.title(starname+' Periodogram $-$ Best Period: {0:.4f}'.format(best_period)+' d')
                if best_lines and min_sep:
                    if SB2==True:
                        labels = ['lines ='+str(best_lines), 'min sep ='+str(min_sep)]
                    else:
                        labels = ['lines ='+str(best_lines), 'n epochs ='+str(nepochs)]
                    leg = plt.legend(labels, loc='best', markerscale=0, handletextpad=0, handlelength=0)
                    for item in leg.legendHandles:
                        item.set_visible(False)
                # plt.tight_layout()
                plt.savefig(path+'LS/LS-fap-BBB_'+starname+'_'+comp+'_periods_'+str(len(rv))+'_epochs.pdf')
                plt.close()

                # Plot for paper
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
                ax.plot(1/frequency, power, 'k-', alpha=0.5)
                ax.yaxis.set_label_coords(-0.09, 0.5)
                ax.set(xlim=(0.3, 1300), ylim=(-0.03*power.max(), power.max()+0.1*power.max()),
                   xlabel='Period (days)',
                   ylabel='Lomb-Scargle Power');
                plt.xscale('log')
                tickLabels = map(str, bins)
                ax.set_xticks(bins)
                ax.set_xticklabels(tickLabels)
                ax.plot( (0.5, 800), (fal[0], fal[0]), '-r', lw=1.2)
                ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
                ax.plot( (0.5, 800), (fal[2], fal[2]), ':g', lw=1.2)
                # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                if power.max()+0.1*power.max() >= 10:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
                else:
                    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
                # plt.title('VFTS '+starname)
                plt.title(starname+' '+comp)
                plt.savefig(path+'LS/BBC_'+starname+'_paper_LS_'+comp+'.png')
                # plt.show()
                plt.close()

        # t6 = time.time()
        # print('# Normal plot + paper plot time: ', t6-t5)

        # vs grid points
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10)
        ax.plot(power1)
        ax.plot(peaks1, power1[peaks1], "ob")
        #ax.set(xlim=(0.4, 600), ylim=(0, power.max()+0.1*power.max()), #x2=426
        #ax.set(xlim=(0.4, 600), ylim=(0, 15), #x2=426
        ax.set(ylim=(0, power1.max()+0.1*power1.max()),
           xlabel='number of points',
           ylabel='Lomb-Scargle Power');
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(which='both', width=0.6, labelsize=14)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.savefig(path+'LS/LS-fap-BBC_'+starname+'_points.pdf')
        plt.close()

        # vs frequency
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10)
        ax.plot(frequency1, power1)
        ax.plot(frequency1[peaks1], power1[peaks1], "ob")
        ax.vlines(np.abs(best_freq1-1), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(best_freq1-2), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(best_freq1+1), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.vlines(np.abs(best_freq1+2), 0, power1.max()+0.1*power1.max(), colors='green', linestyles='dashed')
        ax.text( best_freq1+0.03, power1.max(), r'$\mathrm{f}_0$', fontsize=14)
        ax.text( np.abs(best_freq1-1)-0.03, power1.max(), r'$\left| \mathrm{f}_0-1\right|$', fontsize=14, horizontalalignment='right')
        ax.text( np.abs(best_freq1-2)-0.03, power1.max(), r'$\left| \mathrm{f}_0-2\right|$', fontsize=14, horizontalalignment='right')
        ax.text( np.abs(best_freq1+1)+0.03, power1.max(), r'$\left| \mathrm{f}_0+1\right|$', fontsize=14, horizontalalignment='left')
        ax.text( np.abs(best_freq1+2)+0.03, power1.max(), r'$\left| \mathrm{f}_0+2\right|$', fontsize=14, horizontalalignment='left')
        # ax.set(xlim=(-0.1, 2.3), ylim=(0, power1.max()+0.1*power1.max()), #x2=426
        #ax.set(xlim=(0.4, 600), ylim=(0, 15), #x2=426
        ax.set(ylim=(0, power1.max()+0.1*power1.max()),
           xlabel='frequency (1/d)',
           ylabel='Lomb-Scargle Power');
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.tick_params(which='both', width=0.6, labelsize=14)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.savefig(path+'LS/LS-fap-BBC_'+starname+'_frequencies.pdf')
        plt.close()

        # Stack the arrays in columns
        pow_spec = np.column_stack((frequency1, power1))
        # Save the data to a text file
        np.savetxt(path+'LS/power_spectrum.txt', pow_spec)

    if SB2==True:
        return [best_period1, best_period2, best_period3], [frequency1, frequency2, frequency3], [power1, power2, power3], [LS_power1, LS_power2, LS_power3], [fal1, fal2, fal3], [peaks1, peaks2, peaks3], sorted(peri1), peri2, indi, [fapper1, maxpower_maxfal, maxpower_maxfal2]
    else:
        return [best_period1], [frequency1], [power1], [LS_power1], [fal1], [peaks1], sorted(peri1), ind, [fapper1, maxpower_maxfal, maxpower_maxfal2]

def phase_rv_curve(df, periods, path, SB2=False, print_output=True, plots=True):

    def sinu(self, x, A, w, phi, h):
        "Sinusoidal: sinu(data, amp, freq, phase, height)"
        return A*np.sin(w*(x-phi))+h

    '''
    Compute phases of the obsevations and models
    '''
    if print_output==True:
        print( '\n*** Computing phases ***')
        print( '------------------------')

    hjd, rv, rv_err = df['JD'], df['mean_rv'], df['mean_rv_er']
    name = df['epoch'][0].split('_')[0]+'_'+df['epoch'][0].split('_')[1]
    # print('starname:', starname)

    for Per in periods:
        phase = [ x/Per % 1 for x in hjd]
        ini = min(phase)
        fin = max(phase)

        df['phase'] = phase
        df_phase1 = df.sort_values('phase', ascending=True).reset_index(drop=True)
        df_phase2 = df_phase1.copy(deep=True)
        for i in range(len(df_phase2)):
            df_phase2.loc[i, 'phase']=df_phase1['phase'][i]+1
        df_phase = pd.concat([df_phase1, df_phase2], ignore_index=True)
        # Create a finer grid of phase values
        fine_phase = np.linspace(df_phase['phase'].min(), df_phase['phase'].max(), 1000)

        # Fitting a sinusoidal curve
        sinumodel = Model(sinu)

        phase_shift= [df_phase['phase'].iloc[i] for i,x in enumerate(df_phase['mean_rv']) if df_phase['mean_rv'].iloc[i]==max(df_phase['mean_rv'])][0]
        pars = Parameters()
        pars.add('A', value=(df_phase['mean_rv'].max()-df_phase['mean_rv'].min())/2 )
        pars.add('w', value=2*np.pi)
        pars.add('phi', value=(1-phase_shift+0.25)*2*np.pi)
        pars.add('h', value=(df_phase['mean_rv'].max()-df_phase['mean_rv'].min())/2 +df_phase['mean_rv'].min())

        res_sinu = sinumodel.fit(df_phase['mean_rv'], pars, x=df_phase['phase'], weights=1/df_phase['mean_rv_er'])
        # Calculate the 3-sigma uncertainty
        dely = 3 * res_sinu.params['A'].stderr
        # dely0 = res_sinu.eval_uncertainty(sigma=3)
        best_pars=res_sinu.best_values
        init_pars=res_sinu.init_values
        # Interpolate the best_fit values over the finer grid
        interp_func = interp1d(df_phase['phase'], res_sinu.best_fit, kind='cubic')
        fine_best_fit = interp_func(fine_phase)        
        # print('chi2 = ', res_sinu.chisqr)
        # print('Per = ', Per)
        out_sinu = open(path+'LS/'+name+f'_sinu_stats_{Per}.txt', 'w')
        out_sinu.write(name+'\n')
        out_sinu.write(res_sinu.fit_report()+'\n')
        out_sinu.close()

        sinu_chi2 = res_sinu.chisqr
        dely_dif = 2*dely
        mean_dely = np.mean(dely_dif)
        syst_RV = best_pars['h']
        syst_RV_er = res_sinu.params['h'].stderr
        # delta_RV = 2*best_pars['A']
        # delta_RV_er = 2*res_sinu.params['A'].stderr

        xx = np.linspace(df_phase['phase'].iloc[0], df_phase['phase'].iloc[-1], num=100, endpoint=True)
        #spl3 = UnivariateSpline(df_phase['phase'], df_phase['mean_rv'], df_phase['sigma_rv'], s=5e20)
        #diff = spl3(df_phase['phase']) - df_phase['mean_rv']
        if SB2==True:
            # for period2 in peri2:
            #     if (Per>1.1 and Per<80 and np.abs(Per-period2)<0.5) or (Per>=80 and np.abs(Per-period2)<2):
                # if f'{Per:.2f}'==f'{best_period2:.2f}':
                    # print(Per, ' - ', period2, ' = ', np.abs(Per-period2))
                    # phase = [ x/best_period2 % 1 for x in df_rv['HJD']]
                    # ini = min(phase)
                    # fin = max(phase)
            df['phase'] = phase
            df2_phase1 = df.sort_values('phase', ascending=True).reset_index(drop=True)
            df2_phase2 = df2_phase1.copy(deep=True)

            for i in range(len(df2_phase2)):
                df2_phase2.loc[i, 'phase']=df2_phase1['phase'][i]+1
            df2_phase = pd.concat([df2_phase1, df2_phase2], ignore_index=True)
            phase_shift2= [df2_phase['phase'].iloc[i] for i,x in enumerate(df2_phase['mean_rv']) if df2_phase['mean_rv'].iloc[i]==max(df2_phase['mean_rv'])][0]
            pars = Parameters()
            pars.add('A', value=(df2_phase['mean_rv'].max()-df2_phase['mean_rv'].min())/2 )
            pars.add('w', value=2*np.pi)
            pars.add('phi', value=(1-phase_shift2+0.25)*2*np.pi)
            pars.add('h', value=(df2_phase['mean_rv'].max()-df2_phase['mean_rv'].min())/2 +df2_phase['mean_rv'].min())

            res_sinu2 = sinumodel.fit(df2_phase['mean_rv'], pars, x=df2_phase['phase'], weights=1/df2_phase['mean_rv_er'])
            #res_sinu = sinumodel.fit(df_phase['mean_rv'], x=df_phase['phase'], A=100, w=5, phi=0.2, h=250)
            dely2 = res_sinu2.eval_uncertainty(sigma=3)
            best_pars2=res_sinu2.best_values
            init_pars2=res_sinu2.init_values
            xx2 = np.linspace(df2_phase['phase'].iloc[0], df2_phase['phase'].iloc[-1], num=100, endpoint=True)
            # if Per == LS_P[0]:
            out_sinu2 = open(path+'LS/'+name+'_sinu_stats2.txt', 'w')
            out_sinu2.write(name+'\n')
            out_sinu2.write(res_sinu2.fit_report()+'\n')
            out_sinu2.close()

        '''
        Plot the phased data, model and residuals
        '''
        if plots==True:
            #print('ploting fit to phased data')
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
            fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.11, hspace=0.)

            #ax[0].plot(xx, spl3(xx), 'g-',lw=6, label='Spline weighted', alpha=0.5)
            ax[0].errorbar(df_phase['phase'], df_phase['mean_rv'], df_phase['mean_rv_er'], fmt='.k',ms=12, ecolor='gray', label='data')
            # ax[0].plot(df_phase['phase'], res_sinu.init_fit, '--', c='gray', label='init')
            #ax[0].plot(df_phase['phase'], res_sinu.best_fit, 'r-', lw=2, label='best-fit')
            # ax[0].plot(xx, sinu(xx, init_pars['A'], init_pars['w'], init_pars['phi'], init_pars['h']), '--', c='gray', label='init')
            ax[0].plot(xx, sinu(xx, best_pars['A'], best_pars['w'], best_pars['phi'], best_pars['h']), '-', c='darkorange', lw=2, label='best-fit')
            # ax[0].fill_between(df_phase['phase'], res_sinu.best_fit-dely, res_sinu.best_fit+dely, color='gray', alpha=0.3, label='$3\sigma$ uncert.')#color="#ABABAB"
            ax[0].fill_between(fine_phase, fine_best_fit-dely, fine_best_fit+dely, color='darkorange', alpha=0.2, label='$3\sigma$ uncert.')
            #res_sinu.plot_fit(ax=ax[0], datafmt='ko', fitfmt='r-', initfmt='--', show_init=True)
            res_sinu.plot_residuals(ax=ax[1], datafmt='ko')
            #plt.plot(phase,diff,'ok', alpha=0.5)
            if SB2==True:
                # delta_per = []
                # for period2 in peri2:
                #     delta_per.append(np.abs(Per-period2))
                # for k, dif in enumerate(delta_per):
                #     if dif == min(delta_per) and peri2[k]>1.1 and Per>1.1:
                #         print(f'{Per:.2f}', '=', f'{peri2[k]:.2f}')
                ax[0].errorbar(df2_phase['phase'], df2_phase['mean_rv'], df2_phase['mean_rv_er'], fmt='.g',ms=12, ecolor='green', label='data2')
                ax[0].plot(xx2, sinu(xx2, best_pars2['A'], best_pars2['w'], best_pars2['phi'], best_pars2['h']), 'b-', lw=2, label='best-fit2')
                # ax[0].fill_between(df2_phase['phase'], res_sinu2.best_fit-dely2, res_sinu2.best_fit+dely2, color='green', alpha=0.3, label='$3\sigma$ uncert.')#color="#ABABAB"
                res_sinu2.plot_residuals(ax=ax[1], datafmt='go')

            ax[0].set_title('P = '+str(f'{Per:.4f}')+'(d)')
            ax[0].set(ylabel='Radial Velocity (km s$^{-1}$)')

            ax[1].set(xlabel='Phase')
            ax[1].set(ylabel='Residuals')
            ax[0].legend(loc='best',markerscale=1., frameon=False)
            plt.savefig(path+'LS/'+name+'_velcurve_P'+str(f'{Per:.4f}')+'.pdf')
            plt.close()

            # if Per == LS_P[0]:
            # Plot for paper
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.11)
            ax.errorbar(df_phase['phase'], df_phase['mean_rv'], df_phase['mean_rv_er'], fmt='ok', ms=6, ecolor='black', label='data')
            ax.plot(xx, sinu(xx, best_pars['A'], best_pars['w'], best_pars['phi'], best_pars['h']), 'r-', lw=2, label='best-fit')
            # ax.plot(df_phase['phase'], res_sinu.init_fit, '--', c='gray', label='init')
            plt.axhline(syst_RV, color='gray', linestyle=':',linewidth=1.5, label='systemic RV = {0:.2f}'.format(syst_RV))
            # if SB2==True:
            #     delta_per = []
            #     for period2 in peri2:
            #         delta_per.append(np.abs(Per-period2))
            #     for k, dif in enumerate(delta_per):
            #         if dif == min(delta_per):
            #         # if Per == peri2[k]:
            #             print(dif, min(delta_per))
            #             print(f'{Per:.2f}', '=', f'{peri2[k]:.2f}')
            #             ax.errorbar(df2_phase['phase'], df2_phase['mean_rv'], df2_phase['mean_rv_er'], fmt='sg', ms=6, ecolor='green', label='data2') # mfc='none',
            #             ax.plot(xx2, sinu(xx2, best_pars2['A'], best_pars2['w'], best_pars2['phi'], best_pars2['h']), 'b--', lw=2, label='best-fit2')
            # ax.set_title('P = '+str(f'{Per:.4f}')+'(d)',fontsize=22)
            ax.set_title(name)

            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()
            x_min = x1-(x2-x1)*0.05
            x_max = x2+(x2-x1)*0.05
            y_min = y1-(y2-y1)*0.1
            y_max = y2+(y2-y1)*0.1

            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                    ylabel='Radial Velocity (km s$^{-1}$)',
                    xlabel='Orbital Phase')
            # ax[0].yaxis.label.set_size(22)
            # ax.legend(loc='best',numpoints=1,markerscale=1.,fontsize=12, handletextpad=0.5,borderaxespad=0.3, borderpad=0.8, frameon=False)
            if SB2==True:
                plt.savefig(path+'LS/'+name+'_paper_RV_curve_SB2.pdf')
            else:
                plt.savefig(path+'LS/'+name+'_paper_RV_curve_SB1.pdf')
            plt.close()













































        # # sys.exit()
        # '''
        # Lomb-Scargle
        # '''
        # if print_output==True:
        #     print( '*** Computing L-S ***')
        #     print( '---------------------')
        # if SB2==True:
        #     LS_P, LS_f, LS_Pow, LS_peak, LS_FAL, LS_peaks, allP, peri2, indi, LS_faps = lomb_scargle(df_rv['HJD'], df_rv['mean_rv'], df_rv['mean_rv_er'], \
        #     path, star, SB2=SB2, rv2=df_rv2['mean_rv'], rv2_err=df_rv2['mean_rv_er'], print_output=print_output, plots=plots, \
        #     best_lines=best_lines, min_sep=min_sep)
        # else:
        #     LS_P, LS_f, LS_Pow, LS_peak, LS_FAL, LS_peaks, allP, indi, LS_faps = lomb_scargle(df_rv['HJD'], df_rv['mean_rv'], df_rv['mean_rv_er'], \
        #     path, star, SB2=SB2, print_output=print_output, plots=plots, \
        #     best_lines=best_lines, min_sep=min_sep)
        #     # t7 = time.time()
        #     # print('# points + freq plot time: ', t7-t6)
        # #####################################################################

        # '''
        # Fit to periodogram peak for period error
        # '''
        # Perr=False
        # if Perr==True:
        #     pars = Parameters()
        #     usemod = 'loren'
        #     if usemod == 'gauss':
        #         prefix = 'g_'
        #         gauss = Model(gaussian, prefix=prefix)
        #         pars.update(gauss.make_params())
        #         mod = gauss
        #     if usemod == 'loren':
        #         prefix = 'l_'
        #         loren = Model(lorentzian, prefix=prefix)
        #         pars.update(loren.make_params())
        #         mod = loren

        #     # cont = models.LinearModel(prefix='continuum_')
        #     # pars.update(cont.make_params())
        #     # pars['continuum_slope'].set(0, vary=False)
        #     # pars['continuum_intercept'].set(1, min=0.9)

        #     if peaks1.any():
        #         if print_output==True:
        #             print('   # peaks1 = ', peaks1)
        #         pars.add(prefix+'amp', value=-np.max(power1[peaks1]), max=-np.max(power1[peaks1])*0.8)
        #     else:
        #         if print_output==True:
        #             print('   # peaks1 is empty')
        #         pars.add(prefix+'amp', value=-np.max(power1), max=-np.max(power1)*0.8)
        #     pars.add(prefix+'wid', value=0.001, max=0.003)
        #     pars.add(prefix+'cen', value=best_freq1, vary=True)
        #     mod = mod
        #     # flim = np.logical_and(frequency1 > 0.0025, frequency1 < 0.0031)
        #     results = mod.fit(power1, pars, x=frequency1)
        #     # results = mod.fit(power1[flim], pars, x=frequency1[flim])
        #     out_file = open(path+'LS/freqfit_stats.txt', 'w')
        #     out_file.write(results.fit_report()+'\n')
        #     out_file.close()
        #     # print(results.fit_report())
        #     freq_fwhm = results.params[prefix+'wid'].value
        #     P_err = (1/best_freq1**2)*(freq_fwhm/2)
        #     if print_output==True:
        #         print('   width of the peak               : ', f'{freq_fwhm:.5f}')
        #         print('   Error on the period             : ', f'{P_err:.5f}')
        #         print('   Final period                    : ', f'{best_period1:.4f}', '+/-', f'{P_err:.4f}')
        #     lsout.write('\n')
        #     lsout.write(' Width of the peak               :  '+ f'{freq_fwhm:.5f}'+'\n')
        #     lsout.write(' Error on the period             :  '+ f'{P_err:.5f}'+'\n')
        #     lsout.write(' Final period                    :  '+ f'{best_period1:.4f}'+' +/- '+f'{P_err:.4f}'+'\n')

        #     if SB2==True:
        #         if peaks2.any():
        #             if print_output==True:
        #                 print('   # peaks2 = ', peaks2)
        #             pars.add(prefix+'amp', value=-np.max(power2[peaks2]), max=-np.max(power2[peaks2])*0.8)
        #         else:
        #             if print_output==True:
        #                 print('   # peaks2 is empty')
        #             pars.add(prefix+'amp', value=-np.max(power2), max=-np.max(power2)*0.8)
        #         # pars.add(prefix+'amp', value=-np.max(power2[peaks2]), max=-np.max(power2[peaks2])*0.8)
        #         pars.add(prefix+'wid', value=0.001, max=0.003)
        #         pars.add(prefix+'cen', value=best_freq2, vary=True)
        #         mod = mod
        #         # flim = np.logical_and(frequency1 > 0.0025, frequency1 < 0.0031)
        #         results2 = mod.fit(power2, pars, x=frequency2)
        #         # results = mod.fit(power1[flim], pars, x=frequency1[flim])
        #         out_file2 = open(path+'LS/freqfit_stats_SB2.txt', 'w')
        #         out_file2.write(results2.fit_report()+'\n')
        #         out_file2.close()
        #         # print(results.fit_report())
        #         freq_fwhm2 = results2.params[prefix+'wid'].value
        #         P_err2 = (1/best_freq2**2)*(freq_fwhm2/2)
        #         if print_output==True:
        #             print('   width of the peak               : ', f'{freq_fwhm2:.5f}')
        #             print('   Error on the period             : ', f'{P_err2:.5f}')
        #             print('   Final period                    : ', f'{best_period2:.4f}', '+/-', f'{P_err2:.4f}')
        #         lsout.write('\n # For secondary \n')
        #         lsout.write(' Width of the peak               :  '+ f'{freq_fwhm2:.5f}'+'\n')
        #         lsout.write(' Error on the period             :  '+ f'{P_err2:.5f}'+'\n')
        #         lsout.write(' Final period                    :  '+ f'{best_period2:.4f}'+' +/- '+f'{P_err2:.4f}'+'\n')

        #     if plots==True:
        #         if best_period1 > 100:
        #             x = 2
        #         else:
        #             x = 1.5
        #         fig, ax = plt.subplots(figsize=(8, 6))
        #         fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        #         ax.plot(frequency1, power1, 'k-', alpha=0.5)
        #         ax.plot(frequency1, results.init_fit, '--', c='grey')
        #         # ax.plot(frequency1[flim], results.init_fit, '--', c='grey')
        #         ax.plot(frequency1, results.best_fit, 'r-', lw=2)
        #         # ax.plot(frequency1[flim], results.best_fit, 'r-', lw=2)
        #         ax.set(xlim=(best_freq1-0.001*x, best_freq1+0.001*x),
        #             xlabel='Frequency (1/d)',
        #             ylabel='Lomb-Scargle Power');
        #         plt.savefig(path+'LS/period_fit'+name_+'_primary.pdf')
        #         plt.close()
        #         if SB2==True:
        #             fig, ax = plt.subplots(figsize=(8, 6))
        #             fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        #             ax.plot(frequency2, power2, 'k-', alpha=0.5)
        #             ax.plot(frequency2, results2.init_fit, '--', c='grey')
        #             # ax.plot(frequency1[flim], results.init_fit, '--', c='grey')
        #             ax.plot(frequency2, results2.best_fit, 'r-', lw=2)
        #             # ax.plot(frequency1[flim], results.best_fit, 'r-', lw=2)
        #             ax.set(xlim=(best_freq1-0.001*x, best_freq1+0.001*x),
        #                 xlabel='Frequency (1/d)',
        #                 ylabel='Lomb-Scargle Power');
        #             plt.savefig(path+'LS/period_fit_'+name_+'_secondary.pdf')
        #             plt.close()

        # # if print_output==True:
        # #     t8 = time.time()
        # #     print('# P_err + plot time: ', t8-t7)
        # #####################################################################

 

        # '''
        # Plot RVs vs HJD
        # '''
        # #df_rv = df_rv.drop([11, 12, 13, 15]).reset_index()
        # a = min(df_rv['HJD'])
        # b = max(df_rv['HJD'])
        # y1 = min(df_rv['mean_rv'])
        # y2 = max(df_rv['mean_rv'])
        # delta_RV=y2-y1
        # delta_RV_er=np.sqrt( df_rv['mean_rv_er'][df_rv['mean_rv'].idxmax()]**2 + df_rv['mean_rv_er'][df_rv['mean_rv'].idxmin()]**2 )
        # if SB2==True:
        #     y1sec = min(df_rv2['mean_rv'])
        #     y2sec = max(df_rv2['mean_rv'])
        #     delta_RVsec=y2sec-y1sec
        #     delta_RVsec_er=np.sqrt( df_rv2['mean_rv_er'][df_rv2['mean_rv'].idxmax()]**2 + df_rv2['mean_rv_er'][df_rv2['mean_rv'].idxmin()]**2 )
        # if plots==True:
        #     fig, ax = plt.subplots(figsize=(8,6))
        #     fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        #     plt.errorbar(df_rv['HJD'], df_rv['mean_rv'], df_rv['mean_rv_er'], fmt='.', mfc='orange', mec='orange',ms=12, ecolor='orange', capsize=5, label='RV per epoch')
        #     if SB2==True:
        #         plt.errorbar(df_rv2['HJD'], df_rv2['mean_rv'], df_rv2['mean_rv_er'], fmt='.g',ms=12, ecolor='green', capsize=5, alpha=0.6, label='RV per epoch (sec)')
        #     plt.hlines(syst_RV, a-150, b+150, color='blue', linestyles='--',linewidth=1., label='systemic RV = {0:.2f}'.format(syst_RV))
        #     plt.title(name+' $-$ '+str(len(df_rv))+' epochs',fontsize=22)
        #     # plt.tick_params(which='both', width=0.6, labelsize=14)
        #     #plt.axis([a-150, b+150, y1-10, y2+20])
        #     ax.set(xlim=(a-100, b+100))
        #     plt.ylabel(r'Radial Velocity (km s$^{-1}$)',fontsize=22)
        #     plt.xlabel(r'HJD (days)',fontsize=22)
        #     #plt.legend(loc='best',numpoints=1,markerscale=0.8,fontsize=9,fancybox='true',shadow='true',handletextpad=1,borderaxespad=2)
        #     plt.legend(loc='best',fontsize=18, handletextpad=0.4, borderaxespad=1, edgecolor='k')
        #     #ax.ticklabel_format(usecont=False)
        #     plt.savefig(path+'LS/'+name_+'_epochs.pdf')
        #     plt.close()
        # # if print_output==True:
        # #     t9 = time.time()
        # #     print('# compute phases and plot time: ', t9-t8)
        # #####################################################################
