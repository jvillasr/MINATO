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
from pathlib import Path
import scipy.interpolate as scInterp
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import cm, colors
from scipy.stats.distributions import chi2 as scipy_chi2
from scipy.interpolate import griddata
from lmfit import Model, Parameters, models
from lmfit.models import SkewedGaussianModel, PolynomialModel
from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


def _profile_plot_rc(use_tex):
    if use_tex:
        return {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times'],
        }
    for font_name in ('Times', 'Times New Roman', 'STIXGeneral'):
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        font_name = 'DejaVu Serif'
    return {
        'text.usetex': False,
        'font.family': font_name,
        'mathtext.fontset': 'stix',
    }


def compute_bestfit(
    df,
    cl=0.682689,
    fit_type='pchip',
    chi2max=10000,
    polydeg=[4, 9, 5, 4, 6, 6, 4, 4],
    save_to=None,
    chi2col='chi2_tot',
    dof=None,
    show_histogram=True,
    report_to=None,
    use_tex=False,
    score_kind=None,
):
    """
    Compute best-fit values and perform statistical analysis on chi-squared values.

    This function computes the best-fit values and performs statistical analysis on the chi-squared values
    obtained from fitting synthetic spectra models to observations. It calculates the best-fit values and
    chi-squared minimum for each parameter, renormalizes the chi-squared values, plots chi-squared histograms,
    and determines confidence levels.

    :param df: DataFrame containing computed results including light ratio, temperatures, log surface gravities,
               rotational velocities, He/H ratios, chi-squared values, and related statistics.
               Type: pandas DataFrame
    :param cl: Confidence level for statistical analysis. Default is 0.682689
               (the central one-sigma probability for one parameter).
               Type: float
    :param fit_type: Profile interpolation or fit. ``"pchip"`` is the
                     shape-preserving default. Legacy options are ``"parab"``,
                     ``"skewedG"``, and ``"poly"``.
    :param chi2max: Maximum value of chi-square (χ²) to be displayed on the y-axis of the fit plots.
                    If None, the maximum χ² value from the dataset will be used.
                    Default is 10000.
                    Type: float or None
    :param polydeg: List of polynomial degrees for each parameter fit. Default is [4, 9, 5, 4, 6, 6, 4, 4].
                    Type: list of integers
    :param save_to: Path to save the generated plots. Default is None (plots are not saved).
                    Type: str or None
    :param chi2col: Name of the score column. Default is ``chi2_tot``.
                    Type: str
    :param show_histogram: Display the score-distribution histogram before the
                           parameter plot. Default is True.
                           Type: bool
    :param report_to: Optional path for the polynomial or model-fit report.
                      No report file is created when this is None.
                      Type: str or pathlib.Path or None
    :param use_tex: Use an external LaTeX installation for plot text. The
                    default uses Matplotlib's built-in maths renderer so the
                    plot works in a standard MINATO installation.
                    Type: bool
    :param score_kind: ``"chi2"`` for a weighted chi-square table or ``"rss"``
                    for an unweighted squared-residual table. Inferred from
                    ``df.attrs`` when omitted.

    """
    resolved_score_kind = _resolve_score_kind(df, score_kind)
    if not 0 < cl < 1:
        raise ValueError("cl must be between zero and one")
    if fit_type not in {'pchip', 'parab', 'skewedG', 'poly'}:
        raise ValueError(
            "fit_type must be 'pchip', 'parab', 'skewedG', or 'poly'"
        )
    df = df.copy()
    print(df.columns)
    if 'TA' in df.columns and df['TA'].max() > 1000:
        df['TA'] = df['TA'] / 1000
    if 'TB' in df.columns and df['TB'].max() > 1000:
        df['TB'] = df['TB'] / 1000
    if 'gA' in df.columns and df['gA'].max() > 100:
        df['gA'] = df['gA'] / 100
    if 'gB' in df.columns and df['gB'].max() > 100:
        df['gB'] = df['gB'] / 100


    if dof is None:
        dof = df.attrs.get('degrees_of_freedom')
    if dof is None and 'ndata' in df.columns:
        n_parameters = df.attrs.get('n_parameters')
        if n_parameters is None:
            excluded = {
                'modelA', 'modelB', 'chisqr', 'ndata', 'chi2_tot', 'chi2A',
                'chi2B', 'chi2r_tot', 'chi2redA', 'chi2redB',
            }
            n_parameters = sum(
                column not in excluded and df[column].nunique(dropna=True) > 1
                for column in df.columns
            )
        dof = int(df.iloc[0]['ndata'] - n_parameters)
    # dof = df.loc[0,'ndata']-4
    # read in unscaled chi2
    if chi2col not in df.columns:
        raise KeyError(f"score column is unavailable: {chi2col}")
    unscaled_chi2 = df[chi2col]
    print('min unscaled chi2 value =', unscaled_chi2.min())
    print('max unscaled chi2 value =', unscaled_chi2.max())

    chi2 = unscaled_chi2 - unscaled_chi2.min()
    print('min score above best fit =', chi2.min())
    print('max score above best fit =', chi2.max())

    plot_rc = _profile_plot_rc(use_tex)

    if show_histogram:
        with plt.rc_context(plot_rc):
            num_bins = int(np.sqrt(len(chi2)))
            histogram, ax = plt.subplots(figsize=(12,4))
            ax.hist(unscaled_chi2, bins=num_bins)
            ax.set_yscale('log')
            ax.set_xlabel(
                r'$\chi^2$'
                if resolved_score_kind == 'chi2'
                else 'Squared-residual score'
            )
            plt.show()
            plt.close(histogram)

    if dof is not None:
        print('dof =', dof)
    conf_level = (
        scipy_chi2.ppf(cl, 1)
        if resolved_score_kind == 'chi2'
        else None
    )
    if conf_level is not None:
        print(str(cl*100)+'% profile threshold =', conf_level)

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
    out_file = (
        Path(report_to).expanduser().open('w')
        if report_to is not None
        else None
    )
    if fit_type == 'parab':
        print('\nFitting parabola to parameter:')
        for par, pmin in zip(pars, pars_min):
            print('   '+par.name)
            try:
                h, k = pmin[0][np.argmin(pmin[1])], np.min(pmin[1])
                amp = 10000
                wid = (max(par_i) - min(par_i))*8
                fit_res = fit_parab(pmin[0], pmin[1], amp, h, k)
                results.append(fit_res)
                fit_pars.append(fit_res.best_values)
                if out_file is not None:
                    out_file.write('\n'+par.name+'\n')
                    out_file.write(fit_res.fit_report()+'\n')
            except ValueError:
                print('   # fit unsuccessful')
                fit_pars.append(np.nan)
                pass
    elif fit_type == 'skewedG':
        print('\nFitting skwed Gaussian to parameter:')
        gammas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
                if out_file is not None:
                    out_file.write('\n'+par.name+'\n')
                    out_file.write(fit_res.fit_report()+'\n')
            except Exception as e:
                print('   # fit unsuccessful. Error:', e)
                # print(traceback.format_exc())
                fit_pars.append(np.nan)
                pass
    if fit_type == 'poly':
        print('\nFitting polynomial to parameter:')
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
                    if out_file is not None:
                        out_file.write('\n'+par.name+'\n')
                        out_file.write(
                            'Coefficients: ' + ', '.join(map(str, coefs)) + '\n'
                        )
                else:
                    print("Failed to fit polynomial for parameter", par.name)
            except ValueError as e:
                print("   Error:", str(e))
                fit_pars.append(np.nan)
    elif fit_type == 'pchip':
        results = [None] * len(pars_min)
        fit_pars = [None] * len(pars_min)
    if out_file is not None:
        out_file.close()

    ###################################################################
    # Making the plot
    print('\nMaking plot')
    # Constants for the plot
    tick_label_size = 20
    label_size = 24
    marker_size = 14
    line_width = 3

    column_labels = {
        'lr': 'Light ratio',
        'TA': r'$T_{\mathrm{eff}, A}$ [kK]',
        'gA': r'$\log g_A$',
        'mA': r'$\xi_A$ [km/s]',
        'vA': r'$v \sin i_A$ [km/s]',
        'TB': r'$T_{\mathrm{eff}, B}$ [kK]',
        'gB': r'$\log g_B$',
        'mB': r'$\xi_B$ [km/s]',
        'vB': r'$v \sin i_B$ [km/s]',
    }

    # labels= [ r'L_rat', r'He/H', r'$T_{\mathrm{eff}, A}$', r'$\log g_A$', 
    #         r'$T_{\mathrm{eff}, B}$', r'$\log g_B$', r'$\varv \sin i_A$', r'$\varv\sin i_B$', r'$\xi_A$', r'$\xi_B' ]

    # x_labels = ['Light ratio', 'He/H', r'$T_{\mathrm{eff}, A}$ [kK]', r'$\log g_A$', r'$T_{\mathrm{eff}, B}$ [kK]', \
    #         r'$\log g_B$', r'$\varv \sin i_A$ [km/s]', r'$\varv \sin i_B$ [km/s]', r'$\xi_A$ [km/s]', r'$\xi_B$ [km/s]' ]
    props = dict(boxstyle='round', facecolor='papayawhip', alpha=0.9)
    panels_id = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
    yy = np.ones(len(chi2), dtype=bool) if chi2max is None else chi2 < chi2max
    if not np.any(yy):
        raise ValueError("chi2max excludes every result row")
    
    
    def filter_unique_parameters(pars, pars_min, results, fit_pars):#, labels, x_labels):
        # Filter out parameters that have only one unique value
        unique_pars = [par for par in pars if len(np.unique(par.values)) > 1]
        unique_pars_min = [minval for par, minval in zip(pars, pars_min) if len(np.unique(par.values)) > 1]
        unique_results = [result for par, result in zip(pars, results) if len(np.unique(par.values)) > 1]
        unique_fit_pars = [fit_par for par, fit_par in zip(pars, fit_pars) if len(np.unique(par.values)) > 1]
        # Update labels and xlabels to match unique_pars
        # labels = [label for par, label in zip(pars, labels) if len(np.unique(par.values)) > 1]
        # x_labels = [xlabel for par, xlabel in zip(pars, x_labels) if len(np.unique(par.values)) > 1]
        return unique_pars, unique_pars_min, unique_results, unique_fit_pars#, labels, x_labels

    def create_subplots(unique_pars):
        # Calculate the number of rows and columns needed
        nrows = len(unique_pars) // 2 + len(unique_pars) % 2
        ncols = 2 if len(unique_pars) > 1 else 1
        # Create the subplots
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5*ncols, 5*nrows),
            sharey=False,
        )
        fig.subplots_adjust(wspace=0.25, hspace=0.24)
        ax = np.atleast_1d(axes).ravel()
        return fig, ax

    def plot_data(ax, chi2, yy, conf_level, unique_fit_pars, fit_type, unique_pars, unique_pars_min):#, labels, x_labels):
        for i, par, minval in zip(range(len(unique_pars)), unique_pars, unique_pars_min):
            ax[i].plot(par[yy].values, chi2[yy].values, ls='None', marker='.', ms=marker_size, c='grey', alpha=0.3, zorder=0)
            if conf_level is not None:
                ax[i].axhline(
                    conf_level,
                    color='crimson',
                    lw=line_width,
                    alpha=0.7,
                    zorder=1,
                )
            label_base = column_labels.get(par.name, par.name)
            profile_x = np.asarray(minval[0], dtype=float)
            profile_y = np.asarray(minval[1], dtype=float)
            ax[i].plot(
                profile_x,
                profile_y,
                ls='None',
                marker='o',
                ms=marker_size * 0.65,
                color='black',
                label='Profile minima',
                zorder=4,
            )
            panel_y_values = [chi2[yy].values, profile_y]
            if conf_level is not None:
                panel_y_values.append(np.array([conf_level]))
            y_parab = None
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
                elif fit_type == 'pchip':
                    y_parab = scInterp.PchipInterpolator(
                        profile_x,
                        profile_y,
                        extrapolate=False,
                    )(x_parab)
                    y_parab = np.maximum(y_parab, 0.0)
                if y_parab is not None:
                    panel_y_values.append(np.asarray(y_parab, dtype=float))
                    ax[i].plot(x_parab, y_parab, lw=line_width, c='dodgerblue', zorder=3)
                ax[i].text(0.07, 0.84, panels_id[i], fontsize=26, horizontalalignment='center', transform = ax[i].transAxes)
                # print(x_parab, y_parab, conf_level)
                if conf_level is not None:
                    try:
                        par_val, par_ler, par_uer = get_interc(
                            x_parab,
                            y_parab,
                            conf_level,
                        )
                        if i == 1:
                            label = label_base+' = '+f'{par_val:.3f}'+r'$^{+'+f'{par_uer:.3f}'+'}'+r'_{-'+f'{par_ler:.3f}'+'}$'
                        else:
                            label = label_base+' = '+f'{par_val:.2f}'+r'$^{+'+f'{par_uer:.2f}'+'}'+r'_{-'+f'{par_ler:.2f}'+'}$'
                        ax[i].text(0.5, 0.8, label, fontsize=tick_label_size, horizontalalignment='center', transform = ax[i].transAxes, bbox=props)
                    except Exception as e:
                        print(par.name, 'computing interceptions failed')
                        print("Error:", str(e))
                # ax[i].plot(minval[0], minval[1], 'orange', lw=2, alpha=.75, zorder=3)
                # pass
            ax[i].set_xlabel(label_base, fontsize=label_size)
            if i in [0, 2, 4, 6]:
                ylabel = (
                    r'$\Delta\chi^2$'
                    if resolved_score_kind == 'chi2'
                    else 'Squared-residual score above best fit'
                )
                ax[i].set_ylabel(ylabel, fontsize=label_size)
            xrange = minval[0][-1] - minval[0][0]
            # xrange = minval_scaled[-1] - minval_scaled[0]
            ax[i].set_xlim(minval[0][0] - 0.2*xrange, minval[0][-1] + 0.2*xrange)
            ax[i].tick_params(axis='x', labelsize=tick_label_size)
            ax[i].tick_params(axis='y', labelsize=tick_label_size)
            finite_y = np.concatenate(panel_y_values)
            finite_y = finite_y[np.isfinite(finite_y)]
            y_min = finite_y.min()
            y_max = finite_y.max()
            y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
            ax[i].set_ylim(y_min - y_margin, y_max + y_margin)

        for unused_axis in ax[len(unique_pars):]:
            unused_axis.set_visible(False)

        if save_to is not None:
            plt.savefig(save_to, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # unique_pars, unique_pars_min, unique_results, unique_fit_pars, labels, x_labels = filter_unique_parameters(pars, pars_min, results, fit_pars, labels, x_labels)
    unique_pars, unique_pars_min, unique_results, unique_fit_pars = filter_unique_parameters(pars, pars_min, results, fit_pars)
    with plt.rc_context(plot_rc):
        fig, ax = create_subplots(unique_pars)
        plot_data(
            ax,
            chi2,
            yy,
            conf_level,
            unique_fit_pars,
            fit_type,
            unique_pars,
            unique_pars_min,
        )
    return fig


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Handle cases with a small number of data points
    if len(x) < 4:
        print("Insufficient data points for fitting.")
        return None
    elif len(x) <= 5:
        max_degree = min(3, len(x)-1)

    # A degree-N polynomial requires at least N+1 profile points.
    max_degree = min(max_degree, 7, len(x) - 1)

    # Polynomial.fit scales x before fitting, avoiding numerical problems for
    # quantities such as effective temperature. Convert back for np.polyval.
    coefs = [
        np.polynomial.Polynomial.fit(x, y, deg=degree).convert().coef[::-1]
        for degree in range(1, max_degree + 1)
    ]

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


def _plot_corr_legacy(df, pars_dic, vmax=None, save=None, rot_labels=None, cmap='magma_r', interp='hanning', clabels='numeric'):
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


_SPAN_PROFILE_LABELS = {
    'lr': r'$f_B$',
    'lrat': r'$f_B$',
    'He2H': 'He/H',
    'TA': r'$T_{\mathrm{eff},A}$ [kK]',
    'teffA': r'$T_{\mathrm{eff},A}$ [kK]',
    'gA': r'$\log g_A$',
    'loggA': r'$\log g_A$',
    'vA': r'$v \sin i_A$ [$\mathrm{km\,s^{-1}}$]',
    'rotA': r'$v \sin i_A$ [$\mathrm{km\,s^{-1}}$]',
    'TB': r'$T_{\mathrm{eff},B}$ [kK]',
    'teffB': r'$T_{\mathrm{eff},B}$ [kK]',
    'gB': r'$\log g_B$',
    'loggB': r'$\log g_B$',
    'vB': r'$v \sin i_B$ [$\mathrm{km\,s^{-1}}$]',
    'rotB': r'$v \sin i_B$ [$\mathrm{km\,s^{-1}}$]',
    'mA': r'$\xi_A$ [$\mathrm{km\,s^{-1}}$]',
    'vmicA': r'$\xi_A$ [$\mathrm{km\,s^{-1}}$]',
    'mB': r'$\xi_B$ [$\mathrm{km\,s^{-1}}$]',
    'vmicB': r'$\xi_B$ [$\mathrm{km\,s^{-1}}$]',
}

_SPAN_PROFILE_TITLE_SYMBOLS = {
    'lr': r'f_B',
    'lrat': r'f_B',
    'He2H': r'\mathrm{He/H}',
    'TA': r'T_{\mathrm{eff},A}',
    'teffA': r'T_{\mathrm{eff},A}',
    'gA': r'\log g_A',
    'loggA': r'\log g_A',
    'vA': r'v \sin i_A',
    'rotA': r'v \sin i_A',
    'TB': r'T_{\mathrm{eff},B}',
    'teffB': r'T_{\mathrm{eff},B}',
    'gB': r'\log g_B',
    'loggB': r'\log g_B',
    'vB': r'v \sin i_B',
    'rotB': r'v \sin i_B',
}

_SPAN_PROFILE_UNITS = {
    'TA': r'\mathrm{kK}',
    'teffA': r'\mathrm{kK}',
    'TB': r'\mathrm{kK}',
    'teffB': r'\mathrm{kK}',
    'vA': r'\mathrm{km\,s^{-1}}',
    'rotA': r'\mathrm{km\,s^{-1}}',
    'vB': r'\mathrm{km\,s^{-1}}',
    'rotB': r'\mathrm{km\,s^{-1}}',
}


def _resolve_score_kind(df, score_kind=None):
    """Resolve whether a result table contains chi-square or an RSS score."""

    if score_kind is None:
        score_kind = df.attrs.get('score_kind', 'rss')
    aliases = {
        'chi2': 'chi2',
        'weighted_chi2': 'chi2',
        'rss': 'rss',
        'unweighted': 'rss',
    }
    try:
        return aliases[str(score_kind).lower()]
    except KeyError as error:
        raise ValueError("score_kind must be 'chi2' or 'rss'") from error


def _resolve_contour_mode(contour_mode, score_kind):
    """Select confidence contours only for a genuine chi-square table."""

    if contour_mode == 'nominal':
        contour_mode = 'confidence'
    if contour_mode == 'auto':
        contour_mode = 'confidence' if score_kind == 'chi2' else 'rank'
    if contour_mode not in {'confidence', 'rank'}:
        raise ValueError(
            "contour_mode must be 'auto', 'confidence', or 'rank'"
        )
    if contour_mode == 'confidence' and score_kind != 'chi2':
        raise ValueError(
            "confidence contours require a weighted chi-square result table; "
            "use contour_mode='rank' for an unweighted score"
        )
    return contour_mode


def _prepare_profile_table(
    df,
    pars_dic,
    chi2col=None,
    dof=None,
    score_kind=None,
):
    if not isinstance(pars_dic, dict):
        raise TypeError("pars_dic must be an ordered mapping of parameter grids")

    work = df.copy()
    if chi2col is None:
        chi2col = next(
            (name for name in ('chi2_tot', 'chi2') if name in work.columns),
            None,
        )
    if chi2col is None or chi2col not in work.columns:
        raise KeyError("no SPAN score column is available")

    score = pd.to_numeric(work[chi2col], errors='coerce')
    finite_score = np.isfinite(score)
    if not finite_score.any():
        raise ValueError(f"score column contains no finite values: {chi2col}")
    score_min = score[finite_score].min()
    resolved_score_kind = _resolve_score_kind(df, score_kind)
    work['_span_delta_score'] = score - score_min
    parameter_values = {}
    for name, values in pars_dic.items():
        if name not in work.columns:
            raise KeyError(f"parameter column is unavailable: {name}")
        values = np.asarray(values, dtype=float)
        if name in {'TA', 'TB', 'teffA', 'teffB'} and np.nanmax(values) > 1000:
            work[name] = pd.to_numeric(work[name], errors='coerce') / 1000
            values = values / 1000
        values = np.unique(values[np.isfinite(values)])
        if len(values) > 1:
            parameter_values[name] = values

    parameter_names = list(parameter_values)
    if len(parameter_names) < 2:
        raise ValueError("at least two varying parameters are required")

    if dof is None:
        dof = df.attrs.get('degrees_of_freedom')
    if dof is None and 'ndata' in work.columns:
        n_parameters = df.attrs.get('n_parameters')
        if n_parameters is None:
            n_parameters = sum(
                len(np.unique(np.asarray(values))) > 1
                for values in pars_dic.values()
            )
        dof = int(work.iloc[0]['ndata'] - n_parameters)
    if dof is not None:
        dof = int(dof)
        if dof <= 0:
            raise ValueError("dof must be positive")

    return (
        work,
        score,
        dof,
        parameter_values,
        parameter_names,
        resolved_score_kind,
    )


def _profile_confidence_levels(dimensions):
    """Return nominal likelihood-ratio thresholds for a parameter region."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    probabilities = np.array([0.682689, 0.9545, 0.9973])
    return scipy_chi2.ppf(probabilities, df=dimensions)


def _profile_1d(work, parameter, values):
    return (
        work.groupby(parameter, sort=True)['_span_delta_score']
        .min()
        .reindex(values)
        .to_numpy(dtype=float)
    )


def _profile_2d(work, x_name, y_name, x_values, y_values):
    return (
        work.groupby([y_name, x_name], sort=True)['_span_delta_score']
        .min()
        .unstack(x_name)
        .reindex(index=y_values, columns=x_values)
        .to_numpy(dtype=float)
    )


def _pchip_profile_surface(
    x_values,
    y_values,
    profile,
    fine_x,
    fine_y,
):
    """Interpolate a regular profile grid without overshooting its nodes."""
    row_stage = np.full((len(y_values), len(fine_x)), np.nan)
    for row_index, row in enumerate(profile):
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            row_stage[row_index] = scInterp.PchipInterpolator(
                x_values[valid],
                row[valid],
                extrapolate=False,
            )(fine_x)

    x_then_y = np.full((len(fine_y), len(fine_x)), np.nan)
    for column_index in range(len(fine_x)):
        valid = np.isfinite(row_stage[:, column_index])
        if valid.sum() >= 2:
            x_then_y[:, column_index] = scInterp.PchipInterpolator(
                y_values[valid],
                row_stage[valid, column_index],
                extrapolate=False,
            )(fine_y)

    column_stage = np.full((len(fine_y), len(x_values)), np.nan)
    for column_index, column in enumerate(profile.T):
        valid = np.isfinite(column)
        if valid.sum() >= 2:
            column_stage[:, column_index] = scInterp.PchipInterpolator(
                y_values[valid],
                column[valid],
                extrapolate=False,
            )(fine_y)

    y_then_x = np.full((len(fine_y), len(fine_x)), np.nan)
    for row_index in range(len(fine_y)):
        valid = np.isfinite(column_stage[row_index])
        if valid.sum() >= 2:
            y_then_x[row_index] = scInterp.PchipInterpolator(
                x_values[valid],
                column_stage[row_index, valid],
                extrapolate=False,
            )(fine_x)

    count = np.isfinite(x_then_y).astype(int) + np.isfinite(y_then_x).astype(int)
    combined = np.nansum(np.stack([x_then_y, y_then_x]), axis=0)
    return np.divide(
        combined,
        count,
        out=np.full_like(combined, np.nan),
        where=count > 0,
    )


def _interpolate_profile_grid(x_values, y_values, profile, grid_size, interp):
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    valid = np.isfinite(profile)
    points = np.column_stack([x_grid[valid], y_grid[valid]])
    values = profile[valid]
    fine_x = np.unique(np.concatenate([
        np.linspace(x_values.min(), x_values.max(), grid_size),
        x_values,
    ]))
    fine_y = np.unique(np.concatenate([
        np.linspace(y_values.min(), y_values.max(), grid_size),
        y_values,
    ]))
    fine_x_grid, fine_y_grid = np.meshgrid(fine_x, fine_y)
    method = interp if interp in {'linear', 'nearest', 'cubic', 'pchip'} else 'linear'
    if len(points) < 4:
        method = 'nearest'
    if method == 'pchip':
        fine_profile = _pchip_profile_surface(
            x_values,
            y_values,
            profile,
            fine_x,
            fine_y,
        )
        support = griddata(
            points,
            np.ones(len(points)),
            (fine_x_grid, fine_y_grid),
            method='linear',
        )
        fine_profile = np.where(np.isfinite(support), fine_profile, np.nan)
    else:
        try:
            fine_profile = griddata(
                points,
                values,
                (fine_x_grid, fine_y_grid),
                method=method,
            )
        except Exception:
            fine_profile = None
    if fine_profile is None or not np.isfinite(fine_profile).any():
        fine_profile = griddata(
            points,
            values,
            (fine_x_grid, fine_y_grid),
            method='nearest',
        )
    # Retain the physical lower bound against numerical round-off or an
    # explicitly requested cubic interpolation's overshoot.
    fine_profile = np.where(
        np.isfinite(fine_profile),
        np.maximum(fine_profile, 0.0),
        np.nan,
    )
    return x_grid, y_grid, valid, fine_x_grid, fine_y_grid, fine_profile


def _profile_colormap(cmap):
    """Avoid the near-white and near-black extremes of a colour map."""
    base = plt.get_cmap(cmap)
    samples = base(np.linspace(0.10, 0.88, 256))
    return colors.LinearSegmentedColormap.from_list(
        f'{base.name}_profile',
        samples,
    )


def _profile_colour_max(panel_values, confidence_levels, vmax=None):
    """Choose a robust upper score limit while retaining all contours."""
    if vmax is None:
        panel_scales = [np.percentile(values, 75) for values in panel_values]
        vmax = float(np.median(panel_scales))
    return max(float(vmax), float(confidence_levels[-1]))


def _profile_interval(x_values, profile, threshold):
    minimum_index = int(np.nanargmin(profile))
    best_value = float(x_values[minimum_index])
    difference = profile - threshold
    crossings = []
    for index in range(len(x_values) - 1):
        x_left, x_right = x_values[index:index + 2]
        y_left, y_right = difference[index:index + 2]
        if not np.isfinite([y_left, y_right]).all():
            continue
        if y_left == 0:
            crossings.append(float(x_left))
        if y_left * y_right < 0:
            fraction = -y_left / (y_right - y_left)
            crossings.append(float(x_left + fraction * (x_right - x_left)))
    if difference[-1] == 0:
        crossings.append(float(x_values[-1]))

    lower = [value for value in crossings if value <= best_value]
    upper = [value for value in crossings if value >= best_value]
    lower_error = best_value - max(lower) if lower else np.nan
    upper_error = min(upper) - best_value if upper else np.nan
    return best_value, lower_error, upper_error


def _format_profile_title(name, value, lower_error, upper_error):
    if name in {'lr', 'lrat', 'He2H'}:
        precision = 3
    elif name in {'gA', 'gB', 'loggA', 'loggB'}:
        precision = 2
    elif name in {'vA', 'vB', 'rotA', 'rotB'}:
        precision = 1
    else:
        precision = 2

    symbol = _SPAN_PROFILE_TITLE_SYMBOLS.get(name, rf'\mathrm{{{name}}}')
    unit = _SPAN_PROFILE_UNITS.get(name)
    value_text = f"{value:.{precision}f}"
    if np.isfinite(lower_error) and np.isfinite(upper_error):
        value_text += (
            rf"^{{+{upper_error:.{precision}f}}}"
            rf"_{{-{lower_error:.{precision}f}}}"
        )
    elif np.isfinite(lower_error):
        value_text += rf"_{{-{lower_error:.{precision}f}}}"
    elif np.isfinite(upper_error):
        value_text += rf"^{{+{upper_error:.{precision}f}}}"
    if unit is not None:
        value_text += rf"\, {unit}"
    return rf"${symbol} = {value_text}$"


def plot_corr(
    df,
    pars_dic,
    vmax=None,
    save=None,
    rot_labels=None,
    cmap='magma_r',
    interp='linear',
    clabels='numeric',
    chi2col=None,
    dof=None,
    score_kind=None,
    use_tex=False,
    grid_size=100,
    contour_mode='auto',
    region_fractions=(0.10, 0.25, 0.50),
):
    """Plot two-dimensional profile scores for every parameter pair.

    Each panel shows the minimum score at each pair of parameter-grid values
    after profiling over all remaining parameters. The colour scale is the
    score above the global best fit. Weighted chi-square tables use common
    joint likelihood-ratio thresholds for two fitted parameters. Unweighted
    tables use explicitly descriptive panel ranks.

    Parameters
    ----------
    df : pandas.DataFrame
        SPAN result table.
    pars_dic : mapping
        Ordered mapping from result-column names to their grid values.
    vmax : float, optional
        Upper limit of the shared delta-score colour scale. The outer nominal
        contour sets the default so the constrained region remains legible.
    save : path-like, optional
        Save the figure only when a path is supplied.
    rot_labels : list or ``"All"``, optional
        Flat panel indexes whose x tick labels should be rotated.
    cmap : str
        Matplotlib colour map.
    interp : {``"linear"``, ``"nearest"``, ``"cubic"``, ``"pchip"``}
        Interpolation used only to draw the smooth contours.
    clabels : {``"numeric"``, ``"sigma"``, None}
        Confidence-contour label style.
    chi2col : str, optional
        Score column. Inferred from ``chi2_tot`` or ``chi2`` when omitted.
    dof : int, optional
        Degrees of freedom retained for result-table reporting. It does not
        alter the profile thresholds.
    score_kind : {``"chi2"``, ``"rss"``}, optional
        Score definition. Inferred from ``df.attrs`` when omitted. Set this
        explicitly after loading a result table from a format that discards
        DataFrame attributes.
    use_tex : bool
        Use an external LaTeX installation for plot text.
    grid_size : int
        Number of interpolation samples along each panel axis.

    Returns
    -------
    matplotlib.figure.Figure
        The generated corner figure.
    """
    if grid_size < 10:
        raise ValueError("grid_size must be at least 10")
    region_fractions = np.asarray(region_fractions, dtype=float)
    if (
        region_fractions.shape != (3,)
        or np.any(region_fractions <= 0)
        or np.any(region_fractions >= 1)
        or np.any(np.diff(region_fractions) <= 0)
    ):
        raise ValueError(
            "region_fractions must contain three increasing values in (0, 1)"
        )
    (
        work,
        score,
        dof,
        parameter_values,
        parameter_names,
        resolved_score_kind,
    ) = _prepare_profile_table(
        df,
        pars_dic,
        chi2col=chi2col,
        dof=dof,
        score_kind=score_kind,
    )
    contour_mode = _resolve_contour_mode(contour_mode, resolved_score_kind)
    labels = _SPAN_PROFILE_LABELS

    panels = []
    panel_values = []
    for row in range(1, len(parameter_names)):
        y_name = parameter_names[row]
        for column in range(row):
            x_name = parameter_names[column]
            x_values = parameter_values[x_name]
            y_values = parameter_values[y_name]
            profile = _profile_2d(
                work,
                x_name,
                y_name,
                x_values,
                y_values,
            )
            panels.append((row - 1, column, x_name, y_name, x_values, y_values, profile))
            finite = profile[np.isfinite(profile)]
            if finite.size:
                panel_values.append(finite)

    if not panel_values:
        raise ValueError("no finite two-dimensional profile scores are available")
    confidence_levels = _profile_confidence_levels(2)
    vmax = _profile_colour_max(panel_values, confidence_levels, vmax=vmax)

    interpolation = (
        interp
        if interp in {'linear', 'nearest', 'cubic', 'pchip'}
        else 'linear'
    )
    n_axes = len(parameter_names) - 1
    plot_rc = _profile_plot_rc(use_tex)

    with plt.rc_context(plot_rc):
        fig, axes = plt.subplots(
            n_axes,
            n_axes,
            figsize=(2.8 * n_axes, 2.8 * n_axes),
            squeeze=False,
        )
        fig.subplots_adjust(wspace=0.08, hspace=0.08, right=0.90)
        norm = colors.SymLogNorm(
            linthresh=confidence_levels[0],
            linscale=1.0,
            vmin=0,
            vmax=vmax,
        )
        profile_cmap = _profile_colormap(cmap)
        best_index = score.idxmin()
        visible_axes = []
        contour_set = None

        for row, column, x_name, y_name, x_values, y_values, profile in panels:
            axis = axes[row, column]
            visible_axes.append(axis)
            (
                x_grid,
                y_grid,
                valid,
                fine_x_grid,
                fine_y_grid,
                fine_profile,
            ) = _interpolate_profile_grid(
                x_values,
                y_values,
                profile,
                grid_size,
                interpolation,
            )

            contour_set = axis.pcolormesh(
                fine_x_grid,
                fine_y_grid,
                fine_profile,
                cmap=profile_cmap,
                norm=norm,
                shading='auto',
                rasterized=True,
            )
            candidate_levels = (
                confidence_levels
                if contour_mode == 'confidence'
                else np.quantile(profile[np.isfinite(profile)], region_fractions)
            )
            available_levels = np.unique(candidate_levels[
                (candidate_levels > np.nanmin(fine_profile))
                & (candidate_levels < np.nanmax(fine_profile))
            ])
            if available_levels.size:
                contours = axis.contour(
                    fine_x_grid,
                    fine_y_grid,
                    fine_profile,
                    levels=available_levels,
                    colors='black',
                    linewidths=1,
                )
                if clabels is not None:
                    if contour_mode == 'confidence':
                        level_labels = {
                            confidence_levels[0]: r'$1\sigma$' if clabels == 'sigma' else '68%',
                            confidence_levels[1]: r'$2\sigma$' if clabels == 'sigma' else '95%',
                            confidence_levels[2]: r'$3\sigma$' if clabels == 'sigma' else '99.7%',
                        }
                    else:
                        level_labels = {
                            level: f'best {fraction:.0%}'
                            for level, fraction in zip(
                                candidate_levels,
                                region_fractions,
                            )
                        }
                    axis.clabel(
                        contours,
                        fmt={level: level_labels[level] for level in available_levels},
                        fontsize=7,
                        inline=True,
                    )

            axis.scatter(
                x_grid[valid],
                y_grid[valid],
                s=7,
                color='black',
                alpha=0.25,
                zorder=3,
            )
            axis.scatter(
                work.loc[best_index, x_name],
                work.loc[best_index, y_name],
                marker='*',
                s=70,
                facecolor='white',
                edgecolor='black',
                linewidth=0.8,
                zorder=4,
            )
            axis.set_xlim(x_values.min(), x_values.max())
            axis.set_ylim(y_values.min(), y_values.max())
            axis.tick_params(direction='out', top=False, right=False, labelsize=8)

            panel_index = row * n_axes + column
            rotate = rot_labels == 'All' or (
                isinstance(rot_labels, list) and panel_index in rot_labels
            )
            if row == n_axes - 1:
                axis.set_xlabel(labels.get(x_name, x_name), fontsize=10)
                if rotate:
                    axis.tick_params(axis='x', labelrotation=45)
            else:
                axis.tick_params(axis='x', labelbottom=False)
            if column == 0:
                axis.set_ylabel(labels.get(y_name, y_name), fontsize=10)
            else:
                axis.tick_params(axis='y', labelleft=False)

        occupied = {(row, column) for row, column, *_ in panels}
        for row in range(n_axes):
            for column in range(n_axes):
                if (row, column) not in occupied:
                    axes[row, column].set_visible(False)

        colorbar = fig.colorbar(
            contour_set,
            ax=visible_axes,
            fraction=0.025,
            pad=0.02,
            extend='max',
        )
        colour_label = (
            r'$\Delta\chi^2$'
            if resolved_score_kind == 'chi2'
            else 'Squared-residual score above best fit'
        )
        colorbar.set_label(colour_label, fontsize=11)
        colorbar.ax.tick_params(labelsize=8)
        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)
    return fig


def plot_corner(
    df,
    pars_dic,
    vmax=None,
    save=None,
    rot_labels=None,
    cmap='Blues',
    interp='linear',
    clabels=None,
    chi2col=None,
    dof=None,
    score_kind=None,
    use_tex=False,
    grid_size=100,
    parameter_limits=None,
    reference_values=None,
    contour_mode='auto',
    region_fractions=(0.10, 0.25, 0.50),
):
    """Plot one- and two-dimensional SPAN profiles as a corner figure.

    One-dimensional score profiles and their shape-preserving interpolations occupy the
    diagonal. The lower triangle contains nested regions from the
    two-dimensional profiles; the upper triangle remains empty. This is a
    profile-likelihood diagnostic rather than a posterior-sample corner plot.
    Weighted chi-square tables use one-parameter likelihood-ratio thresholds
    on the diagonal and common two-parameter thresholds in every lower panel.
    Unweighted residual-sum-of-squares tables show no formal intervals and use
    explicitly non-statistical grid ranks in the lower triangle.

    Parameters are the same as :func:`plot_corr`. ``parameter_limits`` may map
    parameter names to display-only ``(minimum, maximum)`` limits.
    ``reference_values`` may map parameters to injected or independently known
    values, drawn as red dotted guides. ``cmap`` supplies the nested-region
    colours. ``vmax`` is retained for API compatibility but is unused because
    this figure has no continuous score scale. ``contour_mode='auto'`` selects
    confidence regions only for weighted chi-square results.
    ``contour_mode='confidence'`` requests the two-parameter likelihood-ratio
    thresholds, while ``contour_mode='rank'`` encloses the best panel-grid
    fractions given by ``region_fractions``. ``score_kind`` is inferred from
    ``df.attrs`` but may be supplied after loading a format that discarded
    those attributes.
    No file is created unless ``save`` is supplied.

    Returns
    -------
    matplotlib.figure.Figure
        The generated profile-score corner figure.
    """
    if grid_size < 10:
        raise ValueError("grid_size must be at least 10")
    region_fractions = np.asarray(region_fractions, dtype=float)
    if (
        region_fractions.shape != (3,)
        or np.any(region_fractions <= 0)
        or np.any(region_fractions >= 1)
        or np.any(np.diff(region_fractions) <= 0)
    ):
        raise ValueError(
            "region_fractions must contain three increasing values in (0, 1)"
        )
    parameter_limits = {} if parameter_limits is None else dict(parameter_limits)
    for name, limits in parameter_limits.items():
        if len(limits) != 2 or limits[0] >= limits[1]:
            raise ValueError(f"invalid display limits for {name}: {limits}")
    (
        work,
        score,
        dof,
        parameter_values,
        parameter_names,
        resolved_score_kind,
    ) = _prepare_profile_table(
        df,
        pars_dic,
        chi2col=chi2col,
        dof=dof,
        score_kind=score_kind,
    )
    contour_mode = _resolve_contour_mode(contour_mode, resolved_score_kind)
    reference_values = {} if reference_values is None else dict(reference_values)
    for name, value in list(reference_values.items()):
        if name in {'TA', 'TB', 'teffA', 'teffB'} and value > 1000:
            reference_values[name] = value / 1000
    interval_levels = _profile_confidence_levels(1)
    contour_levels = _profile_confidence_levels(2)
    labels = _SPAN_PROFILE_LABELS
    interpolation = (
        interp
        if interp in {'linear', 'nearest', 'cubic', 'pchip'}
        else 'linear'
    )
    best_index = score.idxmin()

    panels = []
    panel_values = []
    for row in range(1, len(parameter_names)):
        y_name = parameter_names[row]
        for column in range(row):
            x_name = parameter_names[column]
            x_values = parameter_values[x_name]
            y_values = parameter_values[y_name]
            profile = _profile_2d(
                work,
                x_name,
                y_name,
                x_values,
                y_values,
            )
            panels.append((row, column, x_name, y_name, x_values, y_values, profile))
            finite = profile[np.isfinite(profile)]
            if finite.size:
                panel_values.append(finite)
    if not panel_values:
        raise ValueError("no finite two-dimensional profile scores are available")
    n_parameters = len(parameter_names)
    plot_rc = _profile_plot_rc(use_tex)

    with plt.rc_context(plot_rc):
        fig, axes = plt.subplots(
            n_parameters,
            n_parameters,
            figsize=(3.2 * n_parameters, 3.2 * n_parameters),
            squeeze=False,
        )
        fig.subplots_adjust(wspace=0.08, hspace=0.08, right=0.98)
        base_cmap = plt.get_cmap(cmap)
        region_colours = [
            base_cmap(0.58),
            base_cmap(0.36),
            base_cmap(0.18),
        ]
        reference_colour = '#d84a3a'

        for index, name in enumerate(parameter_names):
            axis = axes[index, index]
            x_values = parameter_values[name]
            profile = _profile_1d(work, name, x_values)
            valid = np.isfinite(profile)
            x_profile = x_values[valid]
            y_profile = profile[valid]
            fine_x = np.linspace(x_profile.min(), x_profile.max(), grid_size * 5)
            if len(x_profile) >= 2:
                fine_profile = scInterp.PchipInterpolator(
                    x_profile,
                    y_profile,
                    extrapolate=False,
                )(fine_x)
            else:
                fine_profile = np.full_like(fine_x, y_profile[0])
            fine_profile = np.maximum(fine_profile, 0.0)

            axis.plot(
                x_profile,
                y_profile,
                ls='None',
                marker='o',
                ms=5,
                color='black',
                label='Profile minima',
                zorder=4,
            )
            axis.plot(fine_x, fine_profile, color='dodgerblue', lw=1.8, zorder=3)
            if resolved_score_kind == 'chi2':
                axis.axhline(
                    interval_levels[0],
                    color='crimson',
                    ls='--',
                    lw=1.2,
                    zorder=1,
                )
            best_x = float(work.loc[best_index, name])
            best_y = float(
                work.loc[work[name] == best_x, '_span_delta_score'].min()
            )
            axis.scatter(
                best_x,
                best_y,
                marker='*',
                s=55,
                facecolor='white',
                edgecolor='black',
                linewidth=0.8,
                zorder=5,
            )
            if resolved_score_kind == 'chi2':
                profile_value, lower_error, upper_error = _profile_interval(
                    fine_x,
                    fine_profile,
                    interval_levels[0],
                )
            else:
                profile_value = float(fine_x[np.nanargmin(fine_profile)])
                lower_error = np.nan
                upper_error = np.nan
            axis.set_title(
                _format_profile_title(
                    name,
                    profile_value,
                    lower_error,
                    upper_error,
                ),
                fontsize=22,
                pad=12,
            )
            x_limits = parameter_limits.get(
                name,
                (x_profile.min(), x_profile.max()),
            )
            visible_profile = (
                (x_profile >= x_limits[0]) & (x_profile <= x_limits[1])
            )
            visible_fine = (fine_x >= x_limits[0]) & (fine_x <= x_limits[1])
            displayed_parts = [
                y_profile[visible_profile],
                fine_profile[visible_fine],
            ]
            if resolved_score_kind == 'chi2':
                displayed_parts.append(np.array([interval_levels[0]]))
            displayed_scores = np.concatenate(displayed_parts)
            displayed_scores = displayed_scores[np.isfinite(displayed_scores)]
            y_min = displayed_scores.min()
            y_max = displayed_scores.max()
            y_margin = 0.08 * (y_max - y_min) if y_max > y_min else 1.0
            axis.set_xlim(x_limits)
            if name in parameter_limits:
                visible_ticks = x_profile[
                    (x_profile >= x_limits[0]) & (x_profile <= x_limits[1])
                ]
                if len(visible_ticks) <= 8:
                    axis.set_xticks(visible_ticks)
            axis.set_ylim(y_min - y_margin, y_max + y_margin)
            axis.tick_params(
                direction='in',
                top=True,
                right=True,
                labelsize=20,
                length=6,
                width=1.2,
            )
            if index != n_parameters - 1:
                axis.tick_params(axis='x', labelbottom=False)
            if index != 0:
                axis.tick_params(axis='y', labelleft=False)

        for row, column, x_name, y_name, x_values, y_values, profile in panels:
            axis = axes[row, column]
            (
                x_grid,
                y_grid,
                valid,
                fine_x_grid,
                fine_y_grid,
                fine_profile,
            ) = _interpolate_profile_grid(
                x_values,
                y_values,
                profile,
                grid_size,
                interpolation,
            )
            if contour_mode == 'rank':
                candidate_levels = np.quantile(
                    profile[np.isfinite(profile)],
                    region_fractions,
                )
            else:
                candidate_levels = contour_levels
            level_mask = (
                (candidate_levels > np.nanmin(fine_profile))
                & (candidate_levels < np.nanmax(fine_profile))
            )
            available_levels = np.unique(candidate_levels[level_mask])
            if available_levels.size:
                filled_levels = np.concatenate(([0.0], available_levels))
                axis.contourf(
                    fine_x_grid,
                    fine_y_grid,
                    fine_profile,
                    levels=filled_levels,
                    colors=region_colours[:len(filled_levels) - 1],
                    antialiased=True,
                    zorder=1,
                )
                contours = axis.contour(
                    fine_x_grid,
                    fine_y_grid,
                    fine_profile,
                    levels=available_levels,
                    colors='black',
                    linewidths=1.3,
                    linestyles='solid',
                    zorder=2,
                )
                if clabels is not None:
                    if contour_mode == 'rank':
                        level_labels = {
                            level: f'{fraction:.0%}'
                            for level, fraction in zip(
                                candidate_levels,
                                region_fractions,
                            )
                        }
                    else:
                        level_labels = {
                            contour_levels[0]: r'$1\sigma$' if clabels == 'sigma' else '68%',
                            contour_levels[1]: r'$2\sigma$' if clabels == 'sigma' else '95%',
                            contour_levels[2]: r'$3\sigma$' if clabels == 'sigma' else '99.7%',
                        }
                    axis.clabel(
                        contours,
                        fmt={level: level_labels[level] for level in available_levels},
                        fontsize=10,
                        inline=True,
                    )
            axis.scatter(
                x_grid[valid],
                y_grid[valid],
                s=5,
                color='black',
                alpha=0.24,
                zorder=3,
            )
            if x_name in reference_values:
                axis.axvline(
                    reference_values[x_name],
                    color=reference_colour,
                    ls=':',
                    lw=1.8,
                    zorder=3,
                )
            if y_name in reference_values:
                axis.axhline(
                    reference_values[y_name],
                    color=reference_colour,
                    ls=':',
                    lw=1.8,
                    zorder=3,
                )
            axis.scatter(
                work.loc[best_index, x_name],
                work.loc[best_index, y_name],
                marker='+',
                s=85,
                color=reference_colour,
                linewidth=2.0,
                zorder=4,
            )
            axis.set_xlim(
                parameter_limits.get(x_name, (x_values.min(), x_values.max()))
            )
            axis.set_ylim(
                parameter_limits.get(y_name, (y_values.min(), y_values.max()))
            )
            if x_name in parameter_limits:
                x_limits = parameter_limits[x_name]
                visible_ticks = x_values[
                    (x_values >= x_limits[0]) & (x_values <= x_limits[1])
                ]
                if len(visible_ticks) <= 8:
                    axis.set_xticks(visible_ticks)
            if y_name in parameter_limits:
                y_limits = parameter_limits[y_name]
                visible_ticks = y_values[
                    (y_values >= y_limits[0]) & (y_values <= y_limits[1])
                ]
                if len(visible_ticks) <= 8:
                    axis.set_yticks(visible_ticks)
            axis.tick_params(
                direction='in',
                top=True,
                right=True,
                labelsize=20,
                length=6,
                width=1.2,
            )
            if row != n_parameters - 1:
                axis.tick_params(axis='x', labelbottom=False)
            if column != 0:
                axis.tick_params(axis='y', labelleft=False)

        for row in range(n_parameters):
            for column in range(n_parameters):
                axis = axes[row, column]
                if column > row:
                    axis.set_visible(False)
                if row == n_parameters - 1 and column <= row:
                    name = parameter_names[column]
                    axis.set_xlabel(labels.get(name, name), fontsize=24)
                    panel_index = row * n_parameters + column
                    rotate = rot_labels == 'All' or (
                        isinstance(rot_labels, list) and panel_index in rot_labels
                    )
                    if rotate:
                        axis.tick_params(axis='x', labelrotation=45)
                if column == 0 and row > 0:
                    name = parameter_names[row]
                    axis.set_ylabel(labels.get(name, name), fontsize=24)

        score_label = (
            r'$\Delta\chi^2$'
            if resolved_score_kind == 'chi2'
            else 'Squared-residual score above best fit'
        )
        axes[0, 0].set_ylabel(score_label, fontsize=24)
        score_offset = axes[0, 0].yaxis.get_offset_text()
        score_offset.set_fontsize(20)
        score_offset.set_position((-0.02, 1.14))
        score_offset.set_ha('left')
        score_offset.set_va('bottom')

        if contour_mode == 'rank':
            region_labels = [
                f'Best {fraction:.0%} of panel grid'
                for fraction in region_fractions
            ]
        else:
            region_labels = [
                '68.3% joint region',
                '95.4% joint region',
                '99.7% joint region',
            ]
        legend_handles = [
            Patch(
                facecolor=colour,
                edgecolor='black',
                label=label,
            )
            for colour, label in zip(region_colours, region_labels)
        ] + [
            Line2D(
                [],
                [],
                marker='+',
                ls='None',
                color=reference_colour,
                label='Best grid model',
            ),
        ]
        if reference_values:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=reference_colour,
                    ls=':',
                    lw=2,
                    label='Injected value',
                )
            )
        fig.legend(
            handles=legend_handles,
            loc='upper right',
            bbox_to_anchor=(0.96, 0.92),
            frameon=False,
            fontsize=19,
        )
        if save is not None:
            fig.savefig(save, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)
    return fig
