# RIM, 2013-06-14.
# Last change: 2014-12-30.

import numpy as np

# -------------------------------------------------------------------------------

# Write parameters file.
def writeparamfile(filename, npar, param, low, high, chis, nobs1, nobs2, dt):
    uni = open(filename, 'w')
    for g in range(npar):
        uni.write("{0:.6e} {1:.6e} {2:.6e} \n".format(param[g], low[g], high[g]))
    uni.write("{0} \n".format(chis))
    uni.write("{0} \n".format(nobs1))
    uni.write("{0} \n".format(nobs2))
    uni.write("{0} \n".format(dt))
    uni.close()
    print(f"Out file: {filename}")

# -------------------------------------------------------------------------------

# Read parameters file.
def readparamfile(filename):
    # Read the parameters.
    all = np.loadtxt(filename, dtype=float)
    nlines = len(all)
    if nlines == 11:
        pars = all[0:6]
        chs = all[7]
        nn1 = all[8]
        nn2 = all[9]
        dt = all[10]

        # Read the uncertainties.
        lo = all[11::2]
        hi = all[12::2]

        return pars, lo, hi, chs, nn1, nn2, dt

    else:
        print("You must use a valid parameters file:")


# -------------------------------------------------------------------------------

# Solve the Kepler equation.
# M must be in radians.
def keplerec2(M,e):
    M = np.double(M)
    e = np.double(e)

    # Valor aproximado inicial para reducir ó mínimo as iteracións.
    # Calcúlase usando un desenrolo en serie de MacLaurin de potencias
    # de e usando E como parámetro, ver Apuntes de Astronomía, U.
    # de Barcelona, ecuación 47.3, ou Heintz, p35.
    #EE = M + e * np.sin(M) + e ** 2 / 2. * np.sin(2. * M)
    EE = M + e * np.sin(M) + e ** 2 / 2. * np.sin(2. * M) + e ** 3 / 8. * (3 * np.sin(3 * M) - np.sin(M))

    eps = 1e-10   # Precission.
    j = 0         # Iteration counter.
    while True:
        # Newton method (Heintz, p 35).
        EE0 = EE
        EE = EE0 + (M + e * np.sin(EE0) - EE0) / (1. - e * np.cos(EE0))

        # Kepler method.
        # Ver Apuntes de Astronomía, U. de Barcelona, apartado 3.6.2
        #EE0=EE
        #EE=M+e*sin(EE)

        #print('EE=',EE,' j=',j)
        j = j + 1
        if max(np.abs(EE0 - EE)) <= eps:
            break

    #print('EE=',EE)
    return EE

# -------------------------------------------------------------------------------


def keplerec2(M, e):
    # Solve Kepler's equation using iterative procedure
    # M = E - e*sin(E)
    # Input:
    # M - Mean anomaly in radians
    # e - Eccentricity
    # Output:
    # E - Eccentric anomaly in radians
    
    # Initial guess for E
    E = M
    
    # Set tolerance and maximum number of iterations
    tol = 1e-12
    maxiter = 1000
    
    # Iterate until convergence
    for i in range(maxiter):
        dE = (M + e*np.sin(E) - E) / (1. - e*np.cos(E))
        E = E + dE
        if abs(dE) < tol:
            break
    
    return E

def radvel2(t, param):
    # Compute RV from the orbital parameters
    # Input:
    # t - Time in days
    # param - List of orbital parameters in the following order:
    #         P - Period in days
    #         Tp - Time of periastron passage in Heliocentric Julian Days
    #         e - Eccentricity
    #         omega - Longitude of periastron in degrees
    #         gamma - Systemic velocity in km/s
    #         K - Radial velocity semi-amplitude in km/s
    # Output:
    # rv - Radial velocity in km/s
    
    P = param[0]
    Tp = param[1]
    e = param[2]
    omega = param[3] * np.pi / 180.  # Convert to radians
    gamma = param[4]
    K = param[5]
    
    # Mean anomaly
    M = 2. * np.pi * ((t - Tp) / P % 1.)
    
    # Eccentric anomaly
    EE = keplerec2(M, e)
    
    # True anomaly
    theta = 2. * np.arctan(np.sqrt((1. + e) / (1. - e)) * np.tan(EE / 2.))
    
    # Radial velocity
    rv = gamma + K * (np.cos(theta + omega) + e * np.cos(omega))
    
    return rv

# -------------------------------------------------------------------------------

# This is the chi^2 function for a single-lined binary.
def chisqSL(param):
    global t1, rv1, srv1
    rvcalc1 = radvel2(t1, param)
    return np.sum(((rvcalc1 - rv1) / srv1) ** 2)


# This is the chi^2 function for a double-lined binary.
def chisqDL(param):
    global t1, rv1, srv1, t2, rv2, srv2
    param1 = param[[0, 1, 2, 3, 4, 5]]
    rvcalc1 = radvel2(t1, param1)

    param2 = param[[0, 1, 2, 3, 4, 6]]
    param2[3] = (param2[3] + 180.) % 360.  # omega+180º for the secondary.
    rvcalc2 = radvel2(t2, param2)

    return np.sum(((rvcalc1 - rv1) / srv1) ** 2) + np.sum(((rvcalc2 - rv2) / srv2) ** 2)


# -------------------------------------------------------------------------------

def ConfIntShort(h, theta, C, theta_max, maxhisto=False):
    # Function to compute the shortest confidence interval from the histogram.
    # h = histogram.
    # theta = position of each bin in the histogram.
    # C = confidence interval in percentage.
    # theta_max = histogram maximum or expected value.
    # /maxhisto = locate the bin which contains the maximum of the histogram.

    nbins = len(h)

    # Locate the maximum of the histogram.
    if maxhisto:
        ind = np.argmax(h)
        theta_max = theta[ind]

    # CDF from the histogram.
    cdf = np.cumsum(h) / np.sum(h)

    # Compute the shortest interval.
    eps = 0.01 * np.abs(theta[1] - theta[0])  # a piece of binsize.
    dmin = np.abs(theta[0] - theta[nbins - 1])  # initial separation = domain of the histogram.
    theta_inf_test = theta[0]

    kk = 0
    while True:
        # Add one step.
        theta_inf_test = theta_inf_test + eps

        # Compute the lower cdf of the confidence interval.
        cdf_inf_test = np.interp(theta_inf_test, theta, cdf)

        # The difference between the upper cdf and the lower cdf always is C.
        cdf_sup_test = cdf_inf_test + C

        if cdf_sup_test >= 1.0:
            # When the upper computed cdf is greater than 1 then end the loop.
            break
        else:
            # Compute the theta associated to the upper cdf and the width of the confidence interval.
            theta_sup_test = np.interp(cdf_sup_test, cdf, theta)
            d = np.abs(theta_sup_test - theta_inf_test)  # Compute the difference.

            # When the difference is smaller, then save the data.
            if d < dmin:
                dmin = d
                theta_inf = theta_inf_test
                theta_sup = theta_sup_test
                cdf_inf = cdf_inf_test
                cdf_sup = cdf_sup_test

    porcent = np.abs(cdf_inf - cdf_sup)
    return [theta_inf, theta_max, theta_sup, porcent]
