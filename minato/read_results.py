import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from . import myRC
import importlib
importlib.reload(myRC)
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/'
import sys
import math
import itertools
import traceback
import scipy.interpolate as scInterp
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from scipy.stats.distributions import chi2 as scipy_chi2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from lmfit import Model, Parameters, models
from lmfit.models import SkewedGaussianModel, PolynomialModel
from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

def compute_bestfit(df, cl=0.68, fit_type = 'poly', chi2max=10000, polydeg = [4, 9, 5, 4, 6, 6, 4, 4], save_to=None, chi2col='chi2', dof=None):
    """
    Compute best-fit values and perform statistical analysis on chi-squared values.

    This function computes the best-fit values and performs statistical analysis on the chi-squared values
    obtained from fitting synthetic spectra models to observations. It calculates the best-fit values and
    chi-squared minimum for each parameter, renormalizes the chi-squared values, plots chi-squared histograms,
    and determines confidence levels.

    :param df: DataFrame containing computed results including light ratio, temperatures, log surface gravities,
               rotational velocities, He/H ratios, chi-squared values, and related statistics.
               Type: pandas DataFrame
    :param cl: Confidence level for statistical analysis. Default is 0.68 (equivalent to 68% confidence interval).
               Type: float
    :param fit_type: Type of fit to be performed. Options are 'parab' for parabolic fit,
                     'skewedG' for skewed Gaussian fit, and 'poly' for polynomial fit.
                     Default is 'poly'.
                     Type: str ('parab', 'skewedG', or 'poly')
    :param chi2max: Maximum value of chi-square (χ²) to be displayed on the y-axis of the fit plots.
                    If None, the maximum χ² value from the dataset will be used.
                    Default is 10000.
                    Type: float or None
    :param polydeg: List of polynomial degrees for each parameter fit. Default is [4, 9, 5, 4, 6, 6, 4, 4].
                    Type: list of integers
    :param save_to: Path to save the generated plots. Default is None (plots are not saved).
                    Type: str or None

    """
    print(df.columns)
    if dof is None:
        dof = df.loc[0,'ndata']-len(df.columns)
    # dof = df.loc[0,'ndata']-4
    # read in unscaled chi2
    unscaled_chi2 = df['chi2_tot']
    # unscaled_chi2 = df[chi2col]
    print('min unscaled chi2 value =', unscaled_chi2.min())
    print('max unscaled chi2 value =', unscaled_chi2.max())

    # renormalize the chi2 such that the best fit corresponds to a chi2 = dof
    # which is similar to setting the reduced chi2 =1
    chi2 = unscaled_chi2 / unscaled_chi2.min() * dof
    print('min scaled chi2 value =', chi2.min())
    print('max scaled chi2 value =', chi2.max())

    # Plot to examine the chi2 values
    num_bins = int(np.sqrt(len(chi2)))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.hist(chi2, bins=np.logspace(np.log10(chi2.min()),np.log10(chi2.max()), num_bins))
    # plt.xlim(0.4,100)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Calculate the minimum and maximum powers of 10 that cover the range of the data
    min_power_of_10 = np.floor(np.log10(chi2.min()))
    max_power_of_10 = np.ceil(np.log10(chi2.max()))
    # Generate the xticks as powers of 10
    xticks = np.logspace(min_power_of_10, max_power_of_10, num=int(max_power_of_10-min_power_of_10+1))

    # ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(xticks)
    ax.set_xlim(chi2.min()*0.9, chi2.max()*1.1)
    ax.set_xlabel('$\chi^2$') 
    #
    ax2 = ax.twiny()
    # ax2.hist(unscaled_chi2, bins=np.logspace(np.log10(unscaled_chi2.min()),np.log10(unscaled_chi2.max()), 1000), alpha=0.3, color='darkorange')
    ax2.set_xlabel('Unscaled $\chi^2$') 
    ax2.set_xscale('log')
    # ax2.xaxis.set_ticks([1, 2, 5, 10, 20], labels=[1, 2, 5, 10, 20])
    # ax2.set_xlim(1500*unscaled_chi2.min()/dof, 90000*unscaled_chi2.min()/dof)
    plt.show()
    plt.close()

    # get confidence level
    print('dof =', dof)
    conf_level = scipy_chi2.ppf(cl, dof)
    sigma2 = scipy_chi2.ppf(0.95, dof)
    sigma3 = scipy_chi2.ppf(0.997, dof)
    print(str(cl*100)+'% confidence level =', conf_level)

    # teffA, loggA, micA, rotA = df['teffA'], df['loggA']/10, df['vmicA'], df['rotA']
    # teffB, loggB, micB, rotB = df['teffB'], df['loggB']/10, df['vmicB'], df['rotB']
    # lrat, he2h = df['lrat'], df['He2H']

    # pars = [lrat, he2h, teffA, loggA, teffB, loggB, rotA, rotB, micA, micB]

    # # get chi2 minimum
    # idx_min = chi2.idxmin()

    # pars_min = []
    # print('\nGetting min values and intercepts (no interpolation)')
    # for par in pars:
    #     # if par:
    #     if not par.isnull().any():
    #         print(par.name)
    #         par_i, par_chi, par_arr, par_interp = interp_models(par, chi2)
    #         # print(par_i) # values of the parameter (from the grid)
    #         # print(par_chi) # min chi2 value for each parameter value (to fit parabola/skew gaussian)
    #         # # sys.exit()
    #         pars_min.append([par_i, par_chi]) # chi2



    ###################################################################

    # List of columns to exclude
    exclude_columns = ['modelA', 'modelB', 'chisqr', 'ndata', 'chi2_tot', 'chi2A', 'chi2B', 'chi2r_tot', 'chi2redA', 'chi2redB']
    pars = []
    pars_min = []
    # Loop through the DataFrame columns
    for column in df.columns:
        # Skip non-physical parameter columns
        if column in exclude_columns  or df[column].isna().all():
            continue

        # If the column is not in the exclude list, it's a physical parameter
        par = df[column]
        pars.append(par)
        
        # Check if the column has any null values
        if not par.isnull().any():
            print(par.name)
            par_i, par_chi, par_arr, par_interp = interp_models(par, chi2)
            print(par_i) # values of the parameter (from the grid)
            print(par_chi) # min chi2 value for each parameter value (to fit parabola/skew gaussian)
            pars_min.append([par_i, par_chi])  # chi2

    print('pars_min = ', pars_min)
    ###################################################################
    # Performing the fit
    results = []
    fit_pars = []
    if fit_type == 'parab':
        print('\nFitting parabola to parameter:')
        out_file = open('parab_fitreport.txt', 'w')
        for par, pmin in zip(pars, pars_min):
            print('   '+par.name)
            try:
                h, k = pmin[0][np.argmin(pmin[1])], np.min(pmin[1])
                amp = 10000
                wid = (max(par_i) - min(par_i))*8
                fit_res = fit_parab(pmin[0], pmin[1], amp, h, k)
                results.append(fit_res)
                fit_pars.append(fit_res.best_values)
                out_file.write('\n'+par.name+'\n')
                out_file.write(fit_res.fit_report()+'\n')
            except ValueError:
                print('   # fit unsuccessful')
                fit_pars.append(np.nan)
                pass
    elif fit_type == 'skewedG':
        print('\nFitting skwed Gaussian to parameter:')
        gammas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        out_file = open('skewG_fitreport_noBalmer.txt', 'w')
        for par, pmin, gamm in zip(pars, pars_min, gammas):
            print('   '+par.name)
            try:
                amp = 100000
                cen = pmin[0][np.argmin(pmin[1])]
                wid = (max(pmin[0]) - min(pmin[0]))*8
                ymin = min(chi2)
                fit_res = fit_skewG(pmin[0], pmin[1], amp, cen, wid, gamm, ymin)
                results.append(fit_res)
                fit_pars.append(fit_res.best_values)
                out_file.write('\n'+par.name+'\n')
                out_file.write(fit_res.fit_report()+'\n')
            except Exception as e:
                print('   # fit unsuccessful. Error:', e)
                # print(traceback.format_exc())
                fit_pars.append(np.nan)
                pass
    if fit_type == 'poly':
        print('\nFitting polynomial to parameter:')
        out_file = open('polynom_fitreport.txt', 'w')
        for par, pmin in zip(pars, pars_min):
            print('   '+par.name)
            print('      ',pmin[0], pmin[1])
            # plt.plot(pmin[0], pmin[1], 'o')
            # if par.name == 'teffB':
            #     pmin[0] = np.delete(pmin[0], [3, 5])
            #     pmin[1] = np.delete(pmin[1], [3, 5])   
            try:
                coefs = fit_poly(pmin[0], pmin[1])
                results.append(coefs)
                if coefs is not None:
                    fit_pars.append(coefs)
                    out_file.write('\n'+par.name+'\n')
                    out_file.write('Coefficients: ' + ', '.join(map(str, coefs)) + '\n')
                else:
                    print("Failed to fit polynomial for parameter", par.name)
            except ValueError as e:
                print("   Error:", str(e))
                fit_pars.append(np.nan)
    out_file.close()

    ###################################################################
    # Making the plot
    print('\nMaking plot')
    # Constants for the plot
    tick_label_size = 20
    label_size = 24
    marker_size = 14
    line_width = 3

    labels= [ r'L_rat', r'He/H', r'$T_{\mathrm{eff}, A}$', r'$\log g_A$', 
            r'$T_{\mathrm{eff}, B}$', r'$\log g_B$', r'$\varv \sin i_A$', r'$\varv\sin i_B$', r'$\xi_A$', r'$\xi_B' ]

    x_labels = ['Light ratio', 'He/H', r'$T_{\mathrm{eff}, A}$ [kK]', r'$\log g_A$', r'$T_{\mathrm{eff}, B}$ [kK]', \
            r'$\log g_B$', r'$\varv \sin i_A$ [km/s]', r'$\varv \sin i_B$ [km/s]', r'$\xi_A$ [km/s]', r'$\xi_B$ [km/s]' ]
    props = dict(boxstyle='round', facecolor='papayawhip', alpha=0.9)
    panels_id = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
    yy = chi2 < chi2max
    
    
    def filter_unique_parameters(pars, pars_min, results, fit_pars, labels, x_labels):
        # Filter out parameters that have only one unique value
        unique_pars = [par for par in pars if len(np.unique(par.values)) > 1]
        unique_pars_min = [minval for par, minval in zip(pars, pars_min) if len(np.unique(par.values)) > 1]
        unique_results = [result for par, result in zip(pars, results) if len(np.unique(par.values)) > 1]
        unique_fit_pars = [fit_par for par, fit_par in zip(pars, fit_pars) if len(np.unique(par.values)) > 1]
        # Update labels and xlabels to match unique_pars
        labels = [label for par, label in zip(pars, labels) if len(np.unique(par.values)) > 1]
        x_labels = [xlabel for par, xlabel in zip(pars, x_labels) if len(np.unique(par.values)) > 1]
        return unique_pars, unique_pars_min, unique_results, unique_fit_pars, labels, x_labels

    def create_subplots(unique_pars):
        # Calculate the number of rows and columns needed
        nrows = len(unique_pars) // 2 + len(unique_pars) % 2
        ncols = 2 if len(unique_pars) > 1 else 1
        # Create the subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), sharey=True)
        fig.subplots_adjust(wspace=0., hspace=0.24)
        ax = axes.flatten()
        return fig, ax

    def plot_data(ax, chi2, yy, conf_level, unique_fit_pars, fit_type, unique_pars, unique_pars_min, labels, x_labels):
        for i, par, minval in zip(range(len(unique_pars)), unique_pars, unique_pars_min):
            ax[i].plot(par[yy].values, chi2[yy].values, ls='None', marker='.', ms=marker_size, c='grey', alpha=0.3, zorder=0)
            ax[i].axhline(conf_level, color='crimson', lw=line_width, alpha=0.7, zorder=1)
            if minval[1] is not np.nan:
                x_parab = np.linspace(minval[0][0], minval[0][-1], 1000)
                if fit_type == 'parab':
                    y_parab = parab(x_parab, unique_fit_pars[i]['a'], unique_fit_pars[i]['h'], unique_fit_pars[i]['k'])
                elif fit_type == 'skewedG':
                    y_parab = skewedG(x_parab, unique_fit_pars[i]['amp'], unique_fit_pars[i]['cen'], unique_fit_pars[i]['wid'], unique_fit_pars[i]['gam'], unique_fit_pars[i]['ymin'])
                elif fit_type == 'poly':
                    if unique_results[i] is not None:
                        y_parab = np.polyval(unique_results[i], x_parab)
                    else:
                        print(f"No polynomial fit for {par.name}")
                        # continue
                # ax[i].plot(minval[0], minval[1], 'crimson', marker='.', ms=marker_size, ls='none', alpha=1, zorder=2)
                ax[i].plot(x_parab, y_parab, lw=line_width, c='dodgerblue', zorder=3)
                ax[i].text(0.07, 0.84, panels_id[i], fontsize=26, horizontalalignment='center', transform = ax[i].transAxes)
                # print(x_parab, y_parab, conf_level)
                try:
                    par_val, par_ler, par_uer = get_interc(x_parab, y_parab, conf_level)
                    if i == 1:
                        label = labels[i]+' = '+f'{par_val:.3f}'+r'$^{+'+f'{par_uer:.3f}'+'}'+r'_{-'+f'{par_ler:.3f}'+'}$'
                    else:
                        label = labels[i]+' = '+f'{par_val:.2f}'+r'$^{+'+f'{par_uer:.2f}'+'}'+r'_{-'+f'{par_ler:.2f}'+'}$'
                    ax[i].text(0.5, 0.8, label, fontsize=tick_label_size, horizontalalignment='center', transform = ax[i].transAxes, bbox=props)
                except Exception as e:
                    print(par.name, 'computing interceptions failed')
                    print("Error:", str(e))
                # ax[i].plot(minval[0], minval[1], 'orange', lw=2, alpha=.75, zorder=3)
                # pass
            ax[i].set_xlabel(x_labels[i], fontsize=label_size)
            if i in [0, 2, 4, 6]:
                ax[i].set_ylabel(r'$\chi^2$', fontsize=label_size)
            xrange = minval[0][-1] - minval[0][0]
            ax[i].set_xlim(minval[0][0] - 0.2*xrange, minval[0][-1] + 0.2*xrange)
            ax[i].tick_params(axis='x', labelsize=tick_label_size)
            ax[i].tick_params(axis='y', labelsize=tick_label_size)
        # Set the y-axis limits
        y_min = min(chi2[yy].values.min(), y_parab.min())
        y_max = max(chi2[yy].values.max(), y_parab.max())
        # plt.ylim(y_parab.min() - 0.1 * y_parab.ptp(), y_parab.max() + 0.1 * y_parab.ptp())
        plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
        # plt.yscale('log')
        if save_to is not None:
            plt.savefig(save_to, bbox_inches='tight')
        plt.show()
        plt.close()

    unique_pars, unique_pars_min, unique_results, unique_fit_pars, labels, x_labels = filter_unique_parameters(pars, pars_min, results, fit_pars, labels, x_labels)
    fig, ax = create_subplots(unique_pars)
    plot_data(ax, chi2, yy, conf_level, unique_fit_pars, fit_type, unique_pars, unique_pars_min, labels, x_labels)


def interp_models(parameter, chi2_array):
    """
    Interpolate minimum chi-square values for different parameter values.

    This function takes an array of parameter values and an array of corresponding
    chi-square values. It calculates the minimum chi-square value for each unique
    parameter value, and then performs cubic interpolation to estimate the minimum
    chi-square values for a finer grid of parameter values.

    :param parameter: Array of parameter values.
                      Type: numpy array
    :param chi2_array: Array of chi-square values corresponding to parameter values.
                       Type: numpy array

    :return: Tuple containing:
               - param_u: Unique parameter values for which chi-square minima were found.
               - param_chi: Minimum chi-square values corresponding to unique parameter values.
               - param_arr: Finer grid of parameter values for interpolation.
               - param_interp: Interpolated minimum chi-square values for the finer parameter grid.
             If too few unique parameter values are available for interpolation, the tuple
             contains NaN values.
             Type: Tuple (numpy array, numpy array, numpy array, numpy array)
    """
    # make array with the minima of the parameter (unique values)
    param_u = list(np.unique(parameter))
    # print(param_u)
    param_chi = []
    # if the parameter was not explored:
    if len(param_u) < 3:
        print("Too few values to interpolate.")
        param_chi, param_arr, param_interp = np.nan, np.nan, np.nan
    # interpolate between the minimum chi2 of the different models
    else:
        # get chi2 for the minima
        for i in range(len(param_u)):
            param_chi.append(np.min(chi2_array[parameter == param_u[i]]))

        # interpolate chi2 minima
        interp = scInterp.interp1d(param_u, param_chi, kind='cubic',
                                   fill_value="extrapolate")
        # make finer grid of parameter to interpolate minima on
        min, max = np.min(param_u), np.max(param_u)  # + 0.01*np.min(param_u)
        step = (param_u[1]-param_u[0]) / 10
        param_arr = np.arange(min, max, step)

        # map minima to new array
        param_interp = interp(param_arr)
    return param_u, param_chi, param_arr, param_interp


def get_errs(param_arr, param_interp, conf_level, val_min):
    """
    Calculate asymmetric errors for parameter estimation.

    This function takes the finer grid of parameter values and the corresponding
    interpolated minimum chi-square values, along with a confidence level and
    the minimum value of the parameter. It determines the asymmetric errors
    for the parameter value at the minimum chi-square point based on the intersection
    of the interpolated chi-square curve with the confidence level.

    :param param_arr: Finer grid of parameter values used for interpolation.
                      Type: numpy array
    :param param_interp: Interpolated minimum chi-square values corresponding to
                         the finer parameter grid.
                         Type: numpy array
    :param conf_level: Confidence level for determining the parameter errors.
                       Type: float
    :param val_min: Minimum value of the parameter (corresponding to minimum chi-square).
                    Type: float

    :return: Tuple containing:
               - err_l: Asymmetric error on the lower side of the parameter.
               - err_u: Asymmetric error on the upper side of the parameter.
             If no errors could be determined, or if there is an issue in the process,
             the tuple contains NaN values.
             Type: Tuple (float, float)
    """
    # intersection between the curves
    try:
        idxs = np.argwhere(np.diff(np.sign(param_interp - conf_level))).flatten()
        if len(idxs) == 0:
            print("No errors could be determined.")
            err_l = np.nan
            err_u = np.nan

        elif len(idxs) == 1:
            print("Only one error determined")
            # intersection exactly the same as value => no error
            if param_arr[idxs] == val_min:
                err_u = np.nan
                err_l = np.nan
            if param_arr[idxs] < val_min:
                err_l = val_min - param_arr[idxs][0]
                err_u = np.nan
            elif param_arr[idxs] > val_min:
                err_u = param_arr[idxs][0] - val_min
                err_l = np.nan

        elif len(idxs) == 2:
            err_l = abs(val_min - param_arr[idxs[0]])
            err_u = abs(param_arr[idxs[1]] - val_min)

        elif len(idxs) > 2:
            # more that one interception with conf level => lowest and highest
            err_l = abs(val_min - param_arr[idxs[0]])
            err_u = abs(param_arr[idxs[-1]] - val_min)
        else:
            print("Error determination went somehow wrong")
            err_l = np.nan
            err_u = np.nan
    except ValueError:
        print('Interpolated minima is NaN')
        err_l = np.nan
        err_u = np.nan
    return(err_l, err_u)

# def skewedG0(xlist, amp, cen, sig, gam):
#     return [amp * np.exp(-(x-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(x-cen)/(sig*np.sqrt(2)))) / (sig*np.sqrt(2*np.pi)) for x in xlist]

#def skewedG(x, amp, cen, sig, gam, h):
#    return [h - amp * np.exp(-(t-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(t-cen)/(sig*np.sqrt(2))))  / (sig*np.sqrt(2*np.pi)) for t in x]

# def skewedG(x, amp, cen, sig, gam, h):
    # return [h - amp * np.exp(-(t-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(t-cen)/(sig*np.sqrt(2)))) for t in x]
def skewedG(x, amp, cen, wid, gam, ymin):
    """
    Calculate values of a skewed Gaussian function.

    This function computes the values of a skewed Gaussian function for given
    input values of `x`, amplitude `amp`, center `cen`, width `wid`, skewness
    parameter `gam`, and minimum value `ymin`.

    :param x: Input values at which to compute the function.
              Type: numpy array or scalar
    :param amp: Amplitude of the skewed Gaussian function.
                Type: float
    :param cen: Center of the skewed Gaussian function.
                Type: float
    :param wid: Width of the skewed Gaussian function.
                Type: float
    :param gam: Skewness parameter of the skewed Gaussian function.
                Type: float
    :param ymin: Minimum value of the skewed Gaussian function.
                 Type: float

    :return: Array of computed function values for each input value in `x`.
             Type: numpy array
    """
    h = ymin+amp
    y = [h - amp * np.exp(-2.355**2 * (t-cen)**2 / (2*wid**2)) * (1 + math.erf(2.355*gam*(t-cen)/(wid*np.sqrt(2)))) for t in x]
    return np.array(y) 

def fit_skewG(x, y, amp, cen, wid, gam, ymin):
    """
    Fit a skewed Gaussian model to data using nonlinear least squares.

    This function performs a fit of a skewed Gaussian model to given data `y`
    corresponding to input values `x`. The parameters of the skewed Gaussian model
    (`amp`, `cen`, `wid`, `gam`, `ymin`) are provided as initial guesses for the
    fit.

    :param x: Input values corresponding to the data `y`.
              Type: numpy array or list
    :param y: Data values to be fitted by the model.
              Type: numpy array or list
    :param amp: Initial guess for the amplitude of the skewed Gaussian model.
                Type: float
    :param cen: Initial guess for the center of the skewed Gaussian model.
                Type: float
    :param wid: Initial guess for the width of the skewed Gaussian model.
                Type: float
    :param gam: Initial guess for the skewness parameter of the skewed Gaussian model.
                Type: float
    :param ymin: Initial guess for the minimum value of the skewed Gaussian model.
                 Type: float

    :return: Results of the fit, including fitted parameter values, fit statistics,
             and other information.
             Type: lmfit.model.ModelResult
    """
    if len(x) < 5 or len(y) < 5:
        raise ValueError("The input data must have at least 5 points for a skewed Gaussian fit.")

    pars = Parameters()
    skG = Model(skewedG)
    pars.update(skG.make_params())
    
    #pars['amp'].set(amp, min=500)
    pars['cen'].set(cen, vary=True)
    #pars['sig'].set(cen*0.5, min=0)
    pars['gam'].set(gam)

    pars['amp'].set(amp, min=0)
    #pars['cen'].set(cen, min=cen*0.8, max=cen*1.2)
    #pars['cen'].set(cen, vary=False)
    pars['wid'].set(wid, min=0)
    #pars['gam'].set(gam, min=gam-np.abs(gam*2), max=gam+np.abs(gam*2))
    pars['ymin'].set(ymin, min=0)
    mod = skG
    results = mod.fit(y, pars, x=x)
    return results

def parab_interc(y, a, h, k):
    """
    Calculate the x-values for a given y using a quadratic equation.

    This function calculates the two possible x-values that correspond to a given
    y-value using the equation of a quadratic parabola: y = a*(x-h)^2 + k.

    :param y: The y-value for which the x-values need to be calculated.
              Type: float
    :param a: Coefficient of the quadratic term in the parabola equation.
              Type: float
    :param h: x-coordinate of the vertex (horizontal shift) in the parabola equation.
              Type: float
    :param k: y-coordinate of the vertex (vertical shift) in the parabola equation.
              Type: float

    :return: Two possible x-values that correspond to the given y-value.
             Type: tuple of floats
    """
    return h-np.sqrt((y-k)/a), h+np.sqrt((y-k)/a)

def parab(x, a, h, k):
    "Parabola, center = (h, k)"
    """
    Evaluate a quadratic parabola function for given x-values.

    This function evaluates a quadratic parabola function of the form:
    y = a*(x-h)**2 + k

    :param x: The x-values at which to evaluate the parabola.
              Type: float or array-like
    :param a: Coefficient of the quadratic term in the parabola equation.
              Type: float
    :param h: x-coordinate of the vertex (horizontal shift) in the parabola equation.
              Type: float
    :param k: y-coordinate of the vertex (vertical shift) in the parabola equation.
              Type: float

    :return: The calculated y-values corresponding to the given x-values.
             Type: float or array-like, same shape as input x
    """
    return a*(x-h)**2 + k

def fit_parab(x, y, a , h, k):
    """
    Fit a quadratic parabola to data using least-squares optimization.

    This function fits a quadratic parabola function of the form:
    y = a*(x-h)**2 + k

    :param x: The x-values of the data points.
              Type: array-like
    :param y: The y-values of the data points to be fitted.
              Type: array-like
    :param a: Initial guess for the coefficient of the quadratic term.
              Type: float
    :param h: Initial guess for the x-coordinate of the vertex (horizontal shift).
              Type: float
    :param k: Initial guess for the y-coordinate of the vertex (vertical shift).
              Type: float

    :return: The results of the fitting procedure, including parameters, statistics, and other information.
             Type: lmfit.model.ModelResult
    """
    pars = Parameters()
    pbol = Model(parab)
    pars.update(pbol.make_params())
    pars['a'].set(a, min=0)
    pars['h'].set(h, min=0)
    pars['k'].set(k, min=0)
    mod = pbol
    results = mod.fit(y, pars, x=x)
    return results

def polynfit(x, a, b, c, d, e, f, g, h, i):
    """
    Compute the value of a polynomial function at given x-values.

    This function evaluates a polynomial function of the form:
    y = a*x^0 + b*x^1 + c*x^2 + d*x^3 + e*x^4 + f*x^5 + g*x^6 + h*x^7 + i*x^8

    :param x: The x-values at which to compute the polynomial function.
              Type: array-like
    :param a, b, c, d, e, f, g, h, i: Coefficients of the polynomial terms. The function expects
                                      one coefficient for each corresponding power of x in the polynomial.
                                      For example, 'a' corresponds to x^0, 'b' corresponds to x^1, and so on.
                                      Type: float

    :return: The computed y-values corresponding to the given x-values using the polynomial function.
             Type: array-like
    """
    return a* x**0 + b* x**1 + c* x**2 + d* x**3 + e* x**4 + f* x**5 + g* x**6 + h* x**7 + i* x**8

# def fit_poly(x, y):
#     """
#     Fit a polynomial function to given data points using the least-squares method.

#     This function fits a polynomial function of the form:
#     y = a*x^0 + b*x^1 + c*x^2 + d*x^3 + e*x^4 + f*x^5 + g*x^6 + h*x^7 + i*x^8

#     to the provided data points (x, y) using the least-squares optimization technique.

#     :param x: The x-values of the data points.
#               Type: array-like
#     :param y: The corresponding y-values of the data points.
#               Type: array-like

#     :return: A lmfit Result object containing the fitting results and statistics.
#              It provides access to attributes like 'params', 'best_values', 'residual', etc.
#              Type: lmfit.Result
#     """
#     pmodel = Model(polynfit)
#     a, b, c, d, e, f, g, h, i = 0, 0, 0, 0, 0, 0, 0, 0, 0
#     coefs = [a, b, c, d, e, f, g, h, i]
#     coefs_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
#     for i,c in enumerate(coefs):
#         max_coef = len(x)
#         if i >= max_coef:
#             coefs[i] = 0
#         else:
#             coefs[i] = 1
#     params = pmodel.make_params(a=coefs[0], b=coefs[1], c=coefs[2], d=coefs[3], e=coefs[4], f=coefs[5], g=coefs[6], h=coefs[7], i=coefs[8])
#     for c,n in zip(coefs, coefs_names):
#         if c==0:
#             params[n].set(c, vary=False)
#     result = pmodel.fit(y, params, x=x)
#     return result

def fit_poly(x, y, max_degree=7, plots=False):
    # Handle cases with a small number of data points
    if len(x) < 4:
        print("Insufficient data points for fitting.")
        return None
    elif len(x) <= 5:
        max_degree = min(3, len(x)-1)

    # Ensure max_degree does not exceed 7
    max_degree = min(max_degree, 7)

    # Fit polynomials of degrees 1 to max_degree
    coefs = [np.polyfit(x, y, deg=i) for i in range(1, max_degree+1)]

    if plots==True:
        # Create subplots
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        axs = axs.ravel()

        # Generate more x-values for plotting
        x_plot = np.linspace(np.array(x).min(), np.array(x).max(), 100)

        for i, c in enumerate(coefs, start=1):
            y_fit = np.polyval(c, x_plot)
            axs[i-1].plot(x, y, 'o', label='Data')
            axs[i-1].plot(x_plot, y_fit, label=f'Degree {i}')
            axs[i-1].legend()

        plt.tight_layout()
        plt.show()

    # Find the degree with the smallest validation MSE
    mse = [np.mean((y - np.polyval(c, x))**2) for c in coefs]
    best_degree = np.argmin(mse) + 1

    # Return the best fit result
    return coefs[best_degree-1]


def get_interc(x_fit, y_fit, conf_level):
    """
    Estimate error intervals for a given curve based on a confidence level.

    This function estimates the error intervals for a curve described by the data points (x_fit, y_fit)
    at a specified confidence level. It identifies the minimum point of the curve and determines the x-values
    where the curve intersects the confidence level boundary.

    :param x_fit: The x-values of the curve.
                 Type: array-like
    :param y_fit: The corresponding y-values of the curve.
                 Type: array-like
    :param conf_level: The desired confidence level (between 0 and 1) to determine the error intervals.
                      Type: float

    :return: A tuple containing the x-value at the minimum point of the curve,
             the lower error estimate, and the upper error estimate.
             Type: (float, float, float)
    """
    y_min = min(y_fit)
    idmin = np.where(y_fit==y_min)
    try:
        x_min = x_fit[idmin].item()
        idxs = np.argwhere(np.diff(np.sign(y_fit - conf_level))).flatten()
        # print(idxs)
        if len(idxs)==2:
            err_l = abs(x_min - x_fit[idxs[0]])
            err_u = abs(x_fit[idxs[1]] - x_min)
            return x_fit[idmin].item(), err_l, err_u
        elif len(idxs)==1:
            print('Only one error determined')
            err_l = abs(x_min - x_fit[idxs[0]])
            err_u = np.nan
            return x_fit[idmin].item(), err_l, err_u
    except ValueError:
        print('errors could not be estimated')
        err_l = np.nan
        err_u = np.nan
        return np.nan, err_l, err_u

def fitplot(wA, fA, wM, fM, model, lr, dictionary, lines, figu='save', nrows=3, ncols=4, legend_ax=3,
            xlabel_ax=7, ylabel_ax=3, balmer_min_y=0.75):
    """
    Create a plot of observed and model spectra for specified spectral lines.

    This function generates a plot displaying observed and model spectra for specified spectral lines.
    It allows for customization of various plot parameters such as layout, legend position, and more.

    :param wA: Wavelength array of the observed spectrum.
              Type: array-like
    :param fA: Flux array of the observed spectrum.
              Type: array-like
    :param wM: List of wavelength arrays of the model spectra.
              Type: list of array-like
    :param fM: List of flux arrays of the model spectra corresponding to each model.
              Type: list of array-like
    :param model: List of names/identifiers for the models used for labeling in the legend.
                  Type: list of str
    :param lr: Light ratio contribution from the secondary star. Used in figure title and saved plot name.
               Type: int or float
    :param dictionary: Dictionary containing information about spectral lines, regions, and titles for subplots.
                      Type: dict
    :param lines: List of spectral lines to include in the plot.
                  Type: list
    :param figu: Default 'save'. Use 'show' to display the plot without saving it.
                 Type: str, optional
    :param nrows: Number of rows for subplots.
                 Type: int, optional
    :param ncols: Number of columns for subplots.
                 Type: int, optional
    :param legend_ax: Number of the preferred subplot to display the legend.
                      Type: int, optional
    :param xlabel_ax: Number of the subplot for x-axis label.
                      Type: int, optional
    :param ylabel_ax: Number of the subplot for y-axis label.
                      Type: int, optional
    :param balmer_min_y: Minimum y-value for Balmer lines' subplots.
                         Type: float, optional
    """
    colors=['dodgerblue', 'darkorange', 'forestgreen', 'tomato']
    # model=['disent. spec', '$T_{\\rm e}=13\,$kK, $\log g=2.4$\n$v\sin i=40$km/s', r'$T_{\rm e}=14\,$kK, $\log g=2.4$']
    # model=['disent. spec', r'$T_{\rm eff}=13\,$kK', r'$T_{\rm eff}=14\,$kK']
    mod_name=['disent. spec']
    he_regs = [[], [4025, 4028], [4091, 4111], [4120, 4122], [4142, 4146], [], [], [4330, 4350], [4386.5, 4389.5], [4471, 4473], []]
    print(len(dictionary))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6), sharey=False)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.11, wspace=0.4, hspace=0.5)
    if type(axes)==np.ndarray:
        ax = axes.flatten()
    else:
        ax = [axes]
    for i, line in enumerate(dictionary):
        if i >= legend_ax:
            j = i+1
        else:
            j = i
        reg = dictionary[line]['region']
        cond = (wA > reg[0]) & (wA < reg[1])
        hndl1, = ax[j].plot(wA[cond], fA[cond],c='k',ls='-', linewidth=2, label='disent. spec')
        # ax[i].plot(wA[cond], fA[cond], 'ko', ms=8,ls='none', label='disent. spec')
        handels = [hndl1]       
        # if he_regs[i]:
        #     ax[j].axvline(he_regs[i][0])
        #     ax[j].axvline(he_regs[i][1])
        #     if line==4026:
        #         ax[j].axvline(4008)
        #         ax[j].axvline(4010)
        for f, w, mod, col in zip(fM, wM, model, colors[1:]):
            cond = (w > reg[0]) & (w < reg[1])
            # if j==0:
            hndl2, = ax[j].plot(w[cond], f[cond],'--', c=col, linewidth=2, label=mod)
            handels.append(hndl2)
            mod_name.append(mod)
            # else:
                # hndl3, = ax[i].plot(w[cond], f[cond],'--', c=col, linewidth=2, label=mod)
            # ax[i].plot(w[cond], f[cond],'.', linewidth=2, label=mod)
        if line in [4102, 4340]:
            ax[j].set_ylim(balmer_min_y, 1.05)
        ax[j].set_title(dictionary[line]['title'], size=14)
        ax[j].tick_params(axis='both', which='major', labelsize=15)
    ax[3].axis('off')
    # ax[legend_ax].legend(frameon=False, handlelength=0.7, fontsize=10)
    # ax[legend_ax].legend([hndl1, hndl2, hndl3], (model[0], model[1], model[2]), loc='center', fontsize=13, frameon=True)
    ax[legend_ax].legend(handels, mod_name, loc='center', fontsize=13, frameon=True)
    ax[0].set_xticks([3990, 3995, 4000])
    ax[11].set_xticks([4540, 4550, 4560])
    fig.supxlabel(r'Wavelength (\AA)', size=20)
    fig.supylabel(r'Flux', x=0.01, size=20)
    # fig.suptitle('Secondary light contribution = '+str(int(lr))+'\%'+' - fitted lines: '+str(lines), y=1, fontsize=36)
    # plt.tight_layout()
    if figu=='save':
        # plt.savefig(model+'lr'+str(lr)+'_'+str(lines)+'.pdf')
        # plt.savefig(str(model[1])+'_lr'+str(int(lr))+'_2.pdf')#, bbox_inches='tight')
        plt.savefig('fitA_lr'+str(int(lr))+'_2.pdf')#, bbox_inches='tight')
    # else:
    plt.show()
    plt.close()


def combin(n, r):
    """
    Calculate the number of combinations (nCr) for given n and r.

    This function calculates the number of combinations (nCr) for a given total number of elements (n)
    and the desired size of the subset (r).

    :param n: Total number of elements.
             Type: int
    :param r: Desired size of the subset.
             Type: int

    :return: The number of combinations (nCr).
             Type: int
    """
    from math import factorial as fac
    ncomb = fac(n) / (fac( n - r )* fac(r))
    return int(ncomb)


def plot_corr(df, pars_dic, vmax=None, save=None, rot_labels=None, cmap='magma_r', interp='hanning', clabels='numeric'):
    '''
    Produce a corner plot of correlations between parameters.

    Parameters:
    - df (pd.DataFrame): A DataFrame with the results, parameter values, and chi2.
    - pars_dic (dict): A dictionary with the parameter name and values.
    - vmax (float): The maximum value for the color map. Default is 1.
    - save (str or None): The filename to save the plot. Default is None (no saving).
    - rot_labels (list or None): List of parameter labels to rotate for better readability. Default is None.
    - cmap (str): The colormap to use for the plot. Default is 'magma_r'.
    - interp (str): Interpolation method for the plot. Default is 'hanning'. (to be impolemented)
    - clabels (str): Label style for colorbar. Default is 'numeric'.

    Possible values for interpolation method :      None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                                                    'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                                                    'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'.


    '''
    # Calculate vmax if it's not provided
    if vmax is None:
        vmax = np.nanmax(df['chi2'])

    npars = len(pars_dic) # num of parameters
    ncomb = combin(npars,2) # num of possible param pairs
    pos0 = list(range(ncomb))
    positions = [pos0[int(i*(i+1)/2):int(i*(i+1)/2)+i+1] for i in range(npars-1)][::-1] 
    corner_position = [x for sublist in positions for x in sublist] # indexes of corner plot


    import copy
    labels = copy.deepcopy(list(pars_dic.keys()))

    for i,key in enumerate(pars_dic.keys()):
        if key=='lrat':
            labels[i] = 'Light ratio'
        elif key=='He2H':
            labels[i] = 'He/H'
        elif key=='teffA':
            labels[i] = r'$T_{\text{eff}, A}$ [kK]'
        elif key=='teffB':
            labels[i] = r'$T_{\text{eff}, B}$ [kK]'
        elif key=='loggA':
            labels[i] = r'$\log g_A$'
        elif key=='loggB':
            labels[i] = r'$\log g_B$'
        elif key=='rotA':
            labels[i] = r'$\varv \sin i_A$ [km/s]'
        elif key=='rotB':
            labels[i] = r'$\varv \sin i_B$ [km/s]'
        elif key=='vmicA':
            labels[i] = r'$\xi_A$ [km/s]'
        elif key=='vmicB':
            labels[i] = r'$\xi_B$ [km/s]'

    pair_pars = list(itertools.combinations(pars_dic.values(), 2))
    pair_names = list(itertools.combinations(pars_dic.keys(), 2))
    pair_labels = list(itertools.combinations(labels, 2))

    pair_pars  = [x for (y,x) in sorted(zip(corner_position,pair_pars), key=lambda pair: pair[0])]
    pair_names = [x for (y,x) in sorted(zip(corner_position,pair_names), key=lambda pair: pair[0])]
    pair_labels = [x for (y,x) in sorted(zip(corner_position,pair_labels), key=lambda pair: pair[0])]

    nrows=npars-1
    ncols=npars-1

    plot_idx = list(range(nrows**2))
    corner_idx = [plot_idx[i:i+nrows] for i in range(0,len(plot_idx),nrows)] # split list in <nrows> number of sublists
    corner_idx = [x[::-1] for x in corner_idx] # invert order of sublists
    corner_idx = [x[-i-1:] for i,x in enumerate(corner_idx)] # drop upper right indexes
    corner_idx = [x for sublist in corner_idx for x in sublist] # join sublists

    fig, axes = plt.subplots(nrows,ncols,figsize=(4*nrows, 4*ncols), sharey='row', sharex='col')
    plt.subplots_adjust(wspace=0.08,hspace=0.08)
    ax = axes.flatten()
    # if nrows*ncols==16:
    #     corner_idx = [0, 5, 4, 10, 9, 8, 15, 14, 13, 12]
    # elif nrows*ncols==49:
    #     # [21, 22, 23, 24, 25, 26, 27, 15, 16, 17, 18, 19, 20, 10, 11, 12, 13, 14,  6,  7,  8,  9,  3,  4,  5,  1,  2,  0]
    #     corner_idx = [0, 5, 4, 10, 9, 8, 15, 14, 13, 12,   ]
    # else:
    #     corner_idx = [0, 4, 3, 8, 7, 6]
    for k, (x, y), (u,v), (m,n) in zip(corner_idx, pair_pars, pair_names, pair_labels):
        corrgrid = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                corrgrid[i][j] = round(df['chi2'][(df[u]==x[i]) & (df[v]==y[j])].min(), 3)
        dfcorr = pd.DataFrame(corrgrid, columns=y, index=x)
        # extent = y.min(), y.max(), x.max(), x.min()
        extent = min(y), max(y), min(x), max(x)
        # vmax=np.nanmax(dfcorr)*vmaxfrac
        # if k in [0, 24]:
        #     heatmap = ax[k].imshow(dfcorr, cmap=cmap, interpolation='hanning', norm=LogNorm(vmin=np.nanmin(dfcorr), vmax=vmax), origin="lower", extent=extent, aspect='auto')
        # else:
        #     heatmap = ax[k].imshow(dfcorr, cmap=cmap, interpolation=interp, norm=LogNorm(vmin=np.nanmin(dfcorr), vmax=vmax), origin="lower", extent=extent, aspect='auto')
        # if v=='teffA':
        #     ax[k].axvline(13.18)
        # if u=='lrat':
        #     ax[k].axhline(0.38)
        # # Add contours to the plot
        from scipy.stats.distributions import chi2 as scipy_chi2
        dof = df.loc[0,'ndata']-len(pars_dic.keys())
        sig1 = scipy_chi2.ppf(0.68, dof)
        sig2 = scipy_chi2.ppf(0.95, dof)
        sig3 = scipy_chi2.ppf(0.99, dof)
        unscaled_sig1 = sig1 * df['chi2'].min() / dof
        unscaled_sig2 = sig2 * df['chi2'].min() / dof
        unscaled_sig3 = sig3 * df['chi2'].min() / dof
        X, Y = np.meshgrid(y, x)
        Z = corrgrid
        # clev = np.linspace(np.nanmin(Z),df['chi2'].max(),50)
        # clev = np.logspace(np.log10(np.nanmin(Z)),np.log10(0.977), 50)
        clev = np.logspace(np.log10(np.nanmin(Z)), df['chi2'].max(), 1500)

        # Create a new, finer grid
        xnew = np.linspace(X.min(), X.max(), 100)  # adjust as needed
        ynew = np.linspace(Y.min(), Y.max(), 100)  # adjust as needed
        Xnew, Ynew = np.meshgrid(xnew, ynew)
        # Interpolate Z onto this new grid
        # If Z contains nan values, replace them with the mean of the non-nan values
        if np.isnan(Z).any():
            Z = np.where(np.isnan(Z), np.nanmax(Z), Z)
        Znew = griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xnew, Ynew), method='cubic')
        clev_new = np.logspace(np.log10(np.nanmin(Znew)-0.3), df['chi2'].max(), 3000)
        sig1_new = np.percentile(Znew, 0.68)#+0.03
        sig2_new = np.percentile(Znew, 0.95)#+0.15
        sig3_new = np.percentile(Znew, 0.99)#+0.3

        ######### 
        if k==0:
            print('vmin:', np.nanmin(Z), 'vmax:', vmax)

        try:
            ax[k].contourf(Xnew, Ynew, Znew, clev_new, cmap=cmap, norm=LogNorm(vmin=np.nanmin(Znew), vmax=vmax))
            # ax[k].contourf(X, Y, Z, clev, cmap=cmap, norm=LogNorm(vmin=np.nanmin(Z), vmax=vmax) )
            # Plot the contours using the interpolated data
            contours = ax[k].contour(Xnew, Ynew, Znew, [sig1_new, sig2_new, sig3_new], colors='k')
            # contours = ax[k].contour(Xnew, Ynew, Znew, [unscaled_sig1, unscaled_sig2, unscaled_sig3], colors='k')
            # contours = ax[k].contour(X, Y, Z, [unscaled_sig1, unscaled_sig2, unscaled_sig3], colors='r')
            fmt = {}
            if clabels == 'sigma':
                strs = [ r'1$\sigma$', r'2$\sigma$', r'3$\sigma$' ]
            elif clabels == 'numeric':
                strs = [ r'68.3\%', r'95.4\%', r'99.7\%' ]
            for l,s in zip( contours.levels, strs ):
                fmt[l] = s
            ax[k].clabel(contours, [sig1_new, sig2_new, sig3_new], inline=True, fmt=fmt, fontsize=10)
            # ax[k].clabel(contours, [unscaled_sig1, unscaled_sig2, unscaled_sig3], inline=True, fmt=fmt, fontsize=10)
        except TypeError:
            XX = X[0]
            YY = Y[0]
            ZZ = Z[0]
            if len(np.unique(XX)) < 2:
                XX = [x[0] for x in X]
                YY = [x[0] for x in Y]
                ZZ = [x[0] for x in Z]
                yyy = np.linspace(YY[0], YY[-1], 100)
                zzz = np.interp(yyy, YY, ZZ)
                xxx = 100*[np.unique(XX)]
            elif len(np.unique(YY)) < 2:
                xxx = np.linspace(XX[0], XX[-1], 100)
                zzz = np.interp(xxx, XX, ZZ)
                yyy = 100*[np.unique(YY)]
            points = np.array([xxx, yyy], dtype=object).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(np.nanmin(zzz), vmax)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            # Set the values used for colormapping
            lc.set_array(zzz)
            lc.set_linewidth(10)
            line = ax[k].add_collection(lc)
            # pass
        # if v == 'teffA':
        #     ax[k].axvline(x=25.159, color='white', linestyle='-')
        #     ax[k].axvline(x=25.159-0.937, color='white', linestyle='--')
        #     ax[k].axvline(x=25.159+0.997, color='white', linestyle='--')
        # if u == 'teffA':
        #     ax[k].axhline(y=25.159, color='white', linestyle='-')
        #     ax[k].axhline(y=25.159-0.937, color='white', linestyle='--')
        #     ax[k].axhline(y=25.159+0.997, color='white', linestyle='--')
        # if v == 'vmicA':
        #     ax[k].axvline(x=11.97, color='white', linestyle='-')
        #     ax[k].axvline(x=11.97-1.52, color='white', linestyle='--')
        #     ax[k].axvline(x=11.97+1.48, color='white', linestyle='--')
        # if u == 'vmicA':
        #     ax[k].axhline(y=11.97, color='white', linestyle='-')
        #     ax[k].axhline(y=11.97-1.52, color='white', linestyle='--')
        #     ax[k].axhline(y=11.97+1.48, color='white', linestyle='--')
        ax[k].set_xticks(dfcorr.columns)
        if rot_labels=='All':
            ax[k].set_xticklabels(dfcorr.columns, rotation=45)
        elif type(rot_labels)==list and k in rot_labels:
            ax[k].set_xticklabels(dfcorr.columns, rotation=45)
        else:
            ax[k].set_xticklabels(dfcorr.columns)
        ax[k].set_yticks(dfcorr.index)
        ax[k].tick_params(axis='both')#, colors='grey')
        ax[k].set_yticklabels(dfcorr.index)
        if v=='rotB':
            ax[k].set_xticks(np.arange(0,700, 100))
            ax[k].set_xticklabels(np.arange(0,700, 100))
        if u=='rotB':
            ax[k].set_yticks(np.arange(0,700, 100))
            ax[k].set_yticklabels(np.arange(0,700, 100))
        if m==pair_labels[-1][0]:
            ax[k].set(xlabel=n)
        if n==pair_labels[0][1]:
            ax[k].set(ylabel=m)
        ax[k].tick_params(direction='out', top=False, right=False)
    for i,_  in enumerate(ax):
        if i not in corner_idx:
            ax[i].remove()
    # plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()