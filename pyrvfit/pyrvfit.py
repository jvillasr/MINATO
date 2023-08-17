# NAME:
#	rvfit
#
# PURPOSE:
#	Fits the parameters for a keplerian radial velocity function using ASA.
#
# EXPLANATION:
#	rvfit fits a keplerian radial velocity function to get the values for the parameters
#	P, Tp, e, omega, gamma, K1, K2. If the system is double-lined all the parameters are
#	fitted. If the system is single-lined K2 is not fitted. This code uses ASA, Adaptive
#	Simulated Annealing to fit simultaneously all the parameters.
#
# CALLING SEQUENCE:
#	rvfit,configfile=configfile[,outfile=outfile][,evolfile=evolfile][,/physics][,/autodom]
#	rvfit,rvfile1,rvfile2,fitparam,valparam,L,U[,outfile=outfile][,evolfile=evolfile][,/physics][,/autodom]
#
# INPUTS:
#	configfile = name of the configuration file (string). It must contain the values for
#		the variables rvfile1, rvfile2, fitparam, valparam, LL, and UU, one per line.
#	outputfile = name of the output file (string).
#	evolfile = name of the evolution file (string). It contains the parameter values as
#		the algorithm evolves in the parameter space.
#	rvfile1 = name of the file containing the primary radial velocities (string).
#	rvfile2 = name of the file containing the secondary radial velocities (string).
#	fitparam = seven elements vector with 0 or 1 depending wether the param is fitted.
#		The parameters must be ordered following [P, Tp, e, omega, gamma, K1, K2].
#	valparam = seven elements vector with the values for the non fitted parameters, with
#		a 0 in fitparam.
#	L = seven elements vector with the lower limit of the paramters.
#	U = seven elements vector with the upper limit of the paramters.
#
# OPTIONAL INPUT PARAMETERS:
#
# KEYWORD PARAMETERS:
#	/physics = compute the physical parameters of the system and propagate their
#		uncertainties.
#	/autodom = compute the domain limits, i.e. the L and U vectors, based in the data.
#
# OUTPUTS
#	Fitted values for the parameters [P, Tp, e, omega, gamma, K1, K2] and their uncertainties.
#	Plot with the RV curve and the residuals. Values for the physical parameters.
#
# EXAMPLES
#   To fit all the keplerian parameters of LV Her and compute the physical parameters:
#   rvfit('LVHer_RV1.dat','LVHer_RV2.dat', [1,1,1,1,1,1,1], 
#          [18.4359535,2453652.19147,0.61273,352.20,-10.278,67.24,68.59], 
#          [10.,2448407.8794,0.,0.,-100.,0.,0.], 
#          [25.,2452425.8862,0.999,360.,100.,100.,150.],physics=True)
#
#   It is easier to put all the parameters in a config file and run rvfit with it. In this
#   case we also want to see how the parameters evolves:
#   rvfit(configfile='LVHer.conf',outfile='LVHer_param.out',evolfile='LVHer_ASA.out',physics=True)
#
#   A typical config file would contain this lines (whithout the ;):
#
# rvfile1  = LVHer_RV1.dat
# rvfile2  = LVHer_RV2.dat
# fitparam = [0,          1,              1,       1,       1,      1,      1]
# valparam = [18.4359535, 2448414.90436,  0.61273, 352.20, -10.278, 67.24,  68.59]
# L        = [10.,        2448407.8794,   0.,      0.,     -100.,   0.,     0.]
# U        = [25.,        2452425.8862,   0.999,   360.,    100,    100.,   150.]
#
# REFERENCES:
#   S. Chen, B.L. Luk, "Adaptive simulated annealing for optimization in
#       signal processing applications", Signal Processing, 1999, 79, p117-128
#   L. Ingber, "Very fast simulated re-annealing", Mathl. Comput. Modelling, 1989,
#       12, p967-973.
#   L. Ingber, "Simulated annealing: Practice versus theory", Mathl. Comput. Modelling,
#       1993, 18, Nº11, p29-57.
#   L. Ingber, "Adaptive simulated annealing (ASA): Lessons learned", Control and
#       Cybernetics, 1996, Vol 25, Nº1, p33-54.
#   Dreo, Petrowski, Siarry, Taillard, "Metaheuristics For Hard Optimization", Springer.
#
# MODIFICATION HISTORY:
#       Written by Ramon Iglesias Marzoa, Dep. Astrofisica (ULL), 2013-03-09.
#   Last change: 2014-12-30.

# Append the necessary functions and procedures.
from RVlib import *
from computeRVparams import *
import sys

# Generate a new vector xnew from x using the generation temperatures.
def genera(x, fitparam, Tgen):
    global limits, L, U
    nx = len(x)
    xnew = np.array(x)

    for g in range(nx): # Loop through the dimensions
        if fitparam[g] == 1: # If this parameter is fixed then continue with the next one.
            while True:
                rand = np.random.rand() # rand between 0 and 1.
                sgn = 1 if (rand - 0.5) >= 0 else -1 # sign of rand-1/2
                q = sgn * Tgen[g] * ((1.0 + 1.0 / Tgen[g]) ** (abs(2 * rand - 1.)) - 1.)
                xnew[g] = x[g] + q * (U[g] - L[g])
                if xnew[g] >= L[g] and xnew[g] <= U[g]:
                    break
        xnew[0] = xnew[0] - np.floor((xnew[0] - L[0]) / xnew[0]) * xnew[0]
    
    return xnew

def rvfit(rvfile1, rvfile2, fitparam, valparam, LL, UU, 
          configfile=None, outfile=None, autodom=False, evolfile=None, physics=False):
    
    # To measure the execution time.
    import time
    tprocini = time.time()
    
    # Global variables.
    global datos, t1, rv1, srv1, t2, rv2, srv2
    global limits, L, U
    
    usageconfigfile = 'rvfit,configfile=configfile[,outfile=outfile][,evolfile=evolfile][,/physics][,/autodom]'
    usageparameters = 'rvfit,rvfile1,rvfile2,fitparam,valparam,L,U[,outfile=outfile][,evolfile=evolfile][,/physics][,/autodom]'

    # Distinguish between the two input modes and format the input.
    if len(sys.argv) == 1:
        # Check the configfile.
        if not configfile:
            print('You must use a valid configuration file:')
            print(usageconfigfile)
            sys.exit()

        # Open the configfile and data formatting.
        with open(configfile, 'r') as f:
            data = f.readlines()
        rvfile1 = data[0].split('=')[1].strip()[1:-1]
        rvfile2 = data[1].split('=')[1].strip()[1:-1]
        fitparam = list(map(int, data[2].split('=')[1].strip()[1:-1].split(',')))
        valparam = list(map(float, data[3].split('=')[1].strip()[1:-1].split(',')))
        L = list(map(float, data[4].split('=')[1].strip()[1:-1].split(',')))
        U = list(map(float, data[5].split('=')[1].strip()[1:-1].split(',')))
    elif len(sys.argv) == 7:
        L = LL
        U = UU
    else:
        print('Usage:')
        print(usageconfigfile)
        print(usageparameters)
        sys.exit()

    # Name of the parameters file.
    outfile = 'rvfit.out' if 'outfile' not in locals() else outfile

    # Flag to distinguish between double and single-line binaries.
    flagDL = 1 if rvfile2 else 0
    npar = 7 # total number of parameters.

    # Check the number of elements.
    if (len(fitparam) != npar or len(valparam) != npar or len(U) != npar or len(L) != npar):
        print(f'fitparam, valparam, L and U must have {npar} elements.')
        print('Nothing is done.')
        sys.exit()

    # Check the domain.
    ind = np.where(U-L <= 0)[0]
    if len(ind) > 0:
        print('All the elements of U must be greater than of L.')
        print('Nothing is done.')
        sys.exit()

    # Check the fitting flags.
    ind = np.where(fitparam == 1)[0]
    if len(ind) < 2:
        print('The number of fitted parameters must be greater or equal to 2.')
        print('Nothing is done.')
        sys.exit()

    # Data reading.
    data = np.genfromtxt(rvfile1, comments='#', dtype=float, usecols=(0, 1, 2)).T
    t1, rv1, srv1 = data[0], data[1], data[2]
    nobs1 = len(rv1)

    # Check the uncertainties.
    ind = np.where(srv1 <= 0)[0]
    if len(ind) > 0:
        print('All the elements of srv1 must be greater than 0.')
        print('Nothing is done.')
        sys.exit()

    # For double-line binaries.
    if flagDL == 1:
        
        # Read the secondary RVs.
        t2, rv2, srv2 = np.loadtxt(rvfile2, usecols=(0, 1, 2), unpack=True)
        nobs2 = len(rv2)
        
        # Check the uncertainties.
        kk = np.where(srv2 <= 0)[0]
        if len(kk) > 0:
            print('All the elements of srv2 must be greater than 0.')
            print('Nothing is done.')
            return None
        
        tt = np.concatenate((t1, t2))
        rrvv = np.concatenate((rv1, rv2))
        ssrrvv = np.concatenate((srv1, srv2))
        
        # Name of the function for single-line binaries.
        funcname = 'chisqDL'
        
    #     chislimit = np.sum(srv1) + np.sum(srv2)
    
    else:
        # For single-line binaries and exoplanets.
        tt = t1
        rrvv = rv1
        ssrrvv = srv1
        
        # This ensures that K2 is fixed to 0.
        fitparam[5] = 0
        valparam[5] = 0
        L[5] = 0.   # This is to maintain coherence.
        U[5] = 1.
        nobs2 = 0
        
        # Name of the function for single-line binaries.
        funcname = 'chisqSL'
        
    #     chislimit = np.sum(srv1)

    # This limit is to stop the annealing loop.
    meansrv = np.mean(ssrrvv)
    chislimit = np.sum((meansrv/ssrrvv)**2)
    # print(chislimit)
    # print(nobs1+nobs2)

    # If e=0.0 and fixed then set omega=90. and fixed.
    if fitparam[1] == 0 and valparam[1] == 0:
        fitparam[2] = 0
        valparam[2] = 90.

    # ------------------Function domain------------------

    if autodom:

        # Compute maximum and minimum period.
        tprov = sorted(tt)
        dt = np.diff(tprov)  # differences among neighbour times.
        Pmin = 2.0 * np.min(np.abs(dt))  # Pmin is the inverse of the Nyquist frequency.
        Pmax = (np.max(tt) - np.min(tt)) / 2.0

        # P, Tp, e, omega, gamma, K1, K2
        L = [Pmin, np.min(tt), 0.0, 0.0, np.min(rrvv), 0.0, 0.0]  # lower limit
        U = [Pmax, np.min(tt) + Pmax, 0.999, 360.0, np.max(rrvv), np.max(rv1) - np.min(rv1), 1.0]  # upper limit

        # Update the K2 limits in case of double-line binary.
        if flagDL:
            L[6] = 0.0
            U[6] = np.max(rv2) - np.min(rv2)

    # ------------------Initial parameter values-----------------

    # Starting parameters.
    x = (U + L) / 2.
    # print(np.array([L, U, x]).T)

    # ------------------Initialization---------------------------

    # Freeze the parameters set as fixed.
    indfixed = np.where(fitparam == 0)
    nfixed = len(indfixed)
    indfitted = np.where(fitparam == 1)
    nfitted = len(indfitted)
    if nfixed > 0:
        x[indfixed] = valparam[indfixed]

    # First values for f, xbest and fbest.
    if flagDL == 1:
        f = chisqDL(x)
    else:
        f = chisqSL(x)
    xbest = x
    fbest = f

    # Stopping parameters.
    eps = 1e-5     # Allowed tolerance in the function minimum value.
    Neps = 5       # Number of times that tolerance eps is achieved before termination.
    Nterm = 20     # npar   # Number of consecutive re-annealings to stop.
    fbestlist = np.full(Neps, 1.)
    nrean = 0      # Re-annealing counter.

    # Acceptance temperature, it depends on ka.
    ka = 0
    Ta0 = f
    Ta = Ta0
    nacep = 0      # Acceptance counter.

    # Generating temperature for each adjusted parameter,
    # it depends on kgen (also for each parameter).
    kgen = np.zeros(npar, dtype=np.uint64)
    Tgen0 = np.ones(npar, dtype=np.float64)
    Tgen = Tgen0

    # Adjustable parameters of the algorithm.
    # WARNING: changing these parameters can make the algorithm
    # doesn't work or becomes too slow.
    c = 20.
    Na = 1000
    Ngen = 10000
    delta = np.abs(U - L) * 1e-8    # to compute the sensibilities.

    # ------------------Initial acceptance temperature-----------------------

    # This follows the prescription of Dreo_Petrowski_Siarry_Taillard
    # "Metaheuristics For Hard Optimization-Springer", p44.
    print('Setting the initial Ta...')
    acep=0.25
    ntest=100
    ftest=np.zeros(ntest)
    for j in range(ntest):
        xnew=genera(x,fitparam,Tgen)
        if flagDL == 1:
            ftest[j]=chisqDL(xnew)
        else:
            ftest[j]=chisqSL(xnew)
        # Save the best values.
        if ftest[j] < fbest:
            fbest=ftest[j]
            xbest=xnew

    dftest=ftest[1:]-ftest[:-1]
    avdftest=np.mean(np.abs(dftest))
    Ta0=avdftest/np.log(1./acep-1.)
    print('Initial Ta = '+str(round(Ta0,2)))

    # ------------------Simulated annealing algorithm-----------------------

    # Save the Markov Chain in a file.
    if evolfile is not None:
        # Delete previous results.
        # os.system('rm -f ' + evolfile)
        with open(evolfile, 'w') as f:
            f.write('# Ta F(X) X(7 elements)\n')

    # Annealing loop. It reduces the temperature.
    while True:

        for j in range(Ngen):  # Loop of generated points for each temperature.

            flag_aceptancia = 0

            # Generate a new value, xnew.
            xnew = genera(x, fitparam, Tgen)

            # Metrópolis criterium.
            # -----------------------------------------
            fnew = call_function(funcname, xnew)
            if fnew <= f:
                flag_aceptancia = 1
            else:
                # This is used to prevent that the exponential cause overflow
                # with (fnew-f)/Ta ~ +20. Actually, it can be used until 50 without
                # problems but it doesn't make sense because 1./(1+exp(+20)) ~ 0.
                test = (fnew - f) / Ta  # como fnew > f => test > 0 sempre.
                Pa = 0 if test > 20 else 1 / (1 + np.exp(test))
                # print('test=', test, '  Pa=', Pa)

                # It is accepted con prob Punif.
                Punif = np.random.rand()
                if Punif <= Pa:
                    flag_aceptancia = 1

            # -----------------------------------------

            # If there is an acceptance save the data in a file.
            if flag_aceptancia == 1:

                # Se é o mellor f garda os parámetros.
                if fnew < fbest:
                    fbest = fnew
                    xbest = xnew
                    nrean = 0

                nacep = nacep + 1
                ka = ka + 1
                x = xnew
                f = fnew
                # print('ka='+str(ka)+' Ta='+str(Ta)+ \
                #       ' fnew='+str(fnew)+' fbest='+str(fbest))

                if evolfile is not None:
                    with open(evolfile, 'a') as f:
                        f.write(str(Ta)+' '+ \
                                str(f)+' '+ \
                                str(x[0])+' '+ \
                                '{:.6f}'.format(x[1])+' '+ \
                                str(x[2])+' '+ \
                                str(x[3])+' '+ \
                                str(x[4])+' '+ \
                                str(x[5])+' '+ \
                                str(x[6])+'\n')

            # Following Na acceptances do a reannealing.
            if nacep >= Na:

                print('Re-annealing...')

                # Compute the sensibilities s.
                s = np.zeros(npar)
                for g in range(npar):

                    # Compute only for the fitted parameters.
                    if fitparam[g] == 1:

                        ee = np.zeros(npar)
                        ee[g] = delta[g]
                        fbestdelta = call_function(funcname, xbest+ee)
                        s[g] = abs((fbestdelta - fbest) / delta[g])

                # This is to avoid s=0 in denominator.
                ind0 = np.where(s == 0.0)[0]
                if len(ind0) > 0:
                    s[ind0] = np.min(s[np.where(s != 0.0)])

                smax = np.max(s[indfitted])

                # Change the generating temperature and set kgen.
                Tgen[indfitted] = Tgen[indfitted] * (smax / s[indfitted])
                kgen[indfitted] = (np.log(Tgen0[indfitted] / Tgen[indfitted]) / c) ** np.double(nfitted)
                kgen[indfitted] = np.abs(kgen[indfitted])

                # Change the acceptance temperature and set ka.
                Ta0 = f
                Ta = fbest
                ka = (np.log(Ta0 / Ta) / c) ** np.double(nfitted)

                # --------------------CHECK
                # print('smax/s=', smax/s)
                # print('kgen=', kgen)
                # print('ka=', ka, '  kgen=', kgen)
                kk = np.where(np.isfinite(Tgen[indfitted]) == 0, count)
                if count > 0:
                    raise ValueError('Invalid value in Tgen[indfitted].')
                # --------------------CHECK

                # Reset counters.
                nacep = 0UL
                nrean = nrean + 1
                # print('nrean=', nrean)

        # Print the best values found for this acceptance temperature.
        print("Ta=" + "{:.4E}".format(Ta) + " param=[" + 
            str(xbest[0]) + "," + "{:.4f}".format(xbest[1]) + "," +
            "{:.6f}".format(xbest[2]) + "," + "{:.3f}".format(xbest[3]) + "," +
            "{:.3f}".format(xbest[4]) + "," + "{:.3f}".format(xbest[5]) + "," +
            "{:.3f}".format(xbest[6]) + "]" + " chi^2=" + "{:.2f}".format(fbest))

        # Reduction of generation temperature.
        kgen[indfitted] = kgen[indfitted] + 1
        Tgen[indfitted] = Tgen0[indfitted] * np.exp(-c * kgen[indfitted]**(1./nfitted))

        # Reduction of acceptance temperature.
        ka = ka + 1
        Ta = Ta0 * np.exp(-c * ka**(1./nfitted))

        # Place fbest at the end of fbestlist.
        fbestlist = np.concatenate((fbestlist[1:Neps-1], [fbest]))

        # Check that the last Neps values of fbestlist are less than eps.
        diff = np.abs(fbestlist - np.roll(fbestlist, 1))
        ind = np.where(diff < eps)[0]
        if len(ind) == Neps - 1:
            if fbest < chislimit:
                break  # Termination.
            else:
                Ta = Ta0

        # Ends if the number of reannealings with no improvemenent in fbest is equal to Nterm.
        if nrean >= Nterm:
            print("Maximum number of reannealings reached.")
            break

        # Close the evolution file if it was opened.
        if evolfile:
            uni.close()
            print("Parameters evolution file: " + evolfile)

    # ------------------Compute the uncertainties-----------------------

    # This is done by computing the Fisher matrix in the best point found.
    # Usually this method underestimates the uncertainties, but it is only for
    # a preliminary guess. If a detailed computation is needed another method
    # (such as the MCMC) must be applied. Also, this guess can be used
    # to run the MCMC near the solution.

    # Fisher matrix.
    FF = np.zeros((nfitted, nfitted))

    n1 = 0	# Counter
    for g1 in range(npar):

        if fitparam[g1] == 1:
        
            ee1 = np.zeros(npar)
            ee1[g1] = delta[g1]
            
            n2 = 0
            for g2 in range(npar):
            
                if fitparam[g2] == 1:
                    
                    if g1 == g2:
                    
                        fm = call_function(funcname, xbest-ee1)
                        fp = call_function(funcname, xbest+ee1)
                        ddf = (fp - 2.*fbest + fm) / delta[g1]**2.
                    
                    else:
                    
                        ee2 = np.zeros(npar)
                        ee2[g2] = delta[g2]
                        fpp = call_function(funcname, xbest+ee1+ee2)
                        fpm = call_function(funcname, xbest+ee1-ee2)
                        fmp = call_function(funcname, xbest-ee1+ee2)
                        fmm = call_function(funcname, xbest-ee1-ee2)
                        ddf = (fpp - fpm - fmp + fmm) / (4.*delta[g1]*delta[g2])
                        
                    FF[n1, n2] = 0.5*ddf
                    
                    n2 += 1
            
            n1 += 1

    # detF = np.linalg.det(FF)

    # Covariance matrix.
    cov = np.linalg.inv(FF)

    # Computation of the variances (elements in the diagonal).
    # The abs inside the sqrt() is to avoid negative variances.
    diag = np.arange(nfitted)
    sxbest = np.zeros(npar)
    sxbest[indfitted] = np.sqrt(np.abs(cov[diag,diag]))

    # Time vector for the fitted curve.
    tini = np.min(tt)
    tfin = np.max(tt)

    # Save the parameters and their uncertainties in a file.
    writeparamfile(outfile, npar, xbest, sxbest, np.zeros(npar), fbest, nobs1, nobs2, tfin-tini)

    # This part was extracted to an external program to re-run it separately if needed.
    if 'physics' in locals():
        computeRVparams(readparfile=outfile, rvdatafile1=rvfile1, rvdatafile2=rvfile2, fase=True)

    print('Processed in', SYSTIME(1)-tprocini, 'seconds.')
    print('FIN.')
