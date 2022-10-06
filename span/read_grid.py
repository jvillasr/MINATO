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

def skewedG(x, amp, cen, sig, gam, h):
    return [h - amp * np.exp(-(t-cen)**2 / (2*sig**2)) * (1 + math.erf(gam*(t-cen)/(sig*np.sqrt(2)))) for t in x]


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

def fit_skewG(x, y, amp, cen, gam, h):
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
    pars['sig'].set(cen*0.5, min=0)
    #pars['gam'].set(gam, min=gam-np.abs(gam*2), max=gam+np.abs(gam*2))
    pars['h'].set(h, min=0)
    mod = skG
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

    