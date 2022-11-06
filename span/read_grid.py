import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import myRC
import sys
import math
import scipy.interpolate as scInterp
from scipy.stats.distributions import chi2 as scipy_chi2
from lmfit import Model, Parameters, models
from lmfit.models import SkewedGaussianModel
from datetime import datetime


def interp_models(parameter, chi2_array):
    # make array with the minima of the parameter (unique values)
    param_u = np.unique(parameter)
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
    # intersection between the curves
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

    return(err_l, err_u)

def parab(x, a, h, k):
    "Parabola, center = (h, k)"
    return a*(x-h)**2 + k

def skewedG0(xlist, amp, cen, sig, gam):
    return [amp * np.exp(-(x-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(x-cen)/(sig*np.sqrt(2)))) / (sig*np.sqrt(2*np.pi)) for x in xlist]

#def skewedG(x, amp, cen, sig, gam, h):
#    return [h - amp * np.exp(-(t-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(t-cen)/(sig*np.sqrt(2))))  / (sig*np.sqrt(2*np.pi)) for t in x]

# def skewedG(x, amp, cen, sig, gam, h):
    # return [h - amp * np.exp(-(t-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(t-cen)/(sig*np.sqrt(2)))) for t in x]
def skewedG(x, amp, cen, wid, gam, ymin):
    h = ymin+amp
    return [h - amp * np.exp(-2.355**2 * (t-cen)**2 / (2*wid**2)) * (1 + math.erf(2.355*gam*(t-cen)/(wid*np.sqrt(2)))) for t in x]

def fit_skewG(x, y, amp, cen, wid, gam, ymin):
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
    "return x-value for certain y"
    return h-np.sqrt((y-k)/a), h+np.sqrt((y-k)/a)

def fit_parab(x, y, a , h, k):
    pars = Parameters()
    pbol = Model(parab)
    pars.update(pbol.make_params())
    pars['a'].set(a, min=0)
    pars['h'].set(h, min=0)
    pars['k'].set(k, min=0)
    mod = pbol
    results = mod.fit(y, pars, x=x)
    return results

def get_interc(x_fit, y_fit, conf_level):
    y_min = min(y_fit)
    idmin = np.where(y_fit==y_min)
    x_min = x_fit[idmin].item()
    idxs = np.argwhere(np.diff(np.sign(y_fit - conf_level))).flatten()
    err_l = abs(x_min - x_fit[idxs[0]])
    err_u = abs(x_fit[idxs[1]] - x_min)
    return x_fit[idmin].item(), err_l, err_u

def fitplot(wA, fA, wM, fM, model, lr, dictionary, lines, figu='save', nrows=3, ncols=3, legend_ax=3,
            xlabel_ax=7, ylabel_ax=3, balmer_min_y=0.75):
    '''
    wA, fA : wavelength and flux of the observed spectrum.
    wM, fM : wavelength and flux of the mdoels (list).
    model  : name/identifier of the models. Used for labels in legend (list).
    lr     : light ratio contribution from the secondary. Used in figure title and name of the saved plot.
    dictionary : Python dictionary with name/identifier on the spectral lines, the region and title for each subplot.
    lines  : lines used in the dictionary (list).
    #savefig : default False. True will save the figure (bool).
    figu : default 'save'. Use 'show' to show the plot without saving it.
    nrows, ncols : number of rows and columns for subplots (int).
    legend_ax : number of the preferred subplot to display the legend (int).
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows), sharey=False)
    if type(axes)==np.ndarray:
        ax = axes.flatten()
    else:
        ax = [axes]
    for i, line in enumerate(dictionary):
        reg = dictionary[line]['region']
        cond = (wA > reg[0]) & (wA < reg[1])
        ax[i].plot(wA[cond], fA[cond],c='k',ls='-', linewidth=4, label='disent. spec')
        # ax[i].plot(wA[cond], fA[cond], 'ko', ms=8,ls='none', label='disent. spec')
        for f, w, mod in zip(fM, wM, model):
            cond = (w > reg[0]) & (w < reg[1])
            ax[i].plot(w[cond], f[cond],'--', c='orange', linewidth=4, label=mod)
            # ax[i].plot(w[cond], f[cond],'.', linewidth=2, label=mod)
        if line in [4102, 4340]:
            ax[i].set_ylim(balmer_min_y, 1.05)
        ax[i].set_title(dictionary[line]['title'], size=36)
        ax[i].tick_params(axis='both', which='major', labelsize=32)
    ax[legend_ax].legend(frameon=False, fontsize=20)
    fig.supxlabel(r'Wavelength (\AA)', size=48)
    fig.supylabel(r'Flux', x=0.01, size=48)
    fig.suptitle('Secondary light contribution = '+str(int(lr))+'\%'+' - fitted lines: '+str(lines), y=1, fontsize=36)
    plt.tight_layout()
    if figu=='save':
    # plt.savefig(model+'lr'+str(lr)+'_'+str(lines)+'.pdf')
        plt.savefig(str(model[0])+'_lr'+str(int(lr))+'.pdf')
    else:
        plt.show()
    plt.close()