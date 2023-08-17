# +
# NAME:
#     computeRVparams
#
# PURPOSE:
#     Compute the physical parameters for a spectroscopic binary.
#
# EXPLANATION:
#     This procedure can do several things: it computes the parameters and their
#     uncertainties from the keplerian parameters fitted, it plots the RV curve
#     with the residuals, and it can build a latex table with the parameters fitted.
#     It needs the RVlib.pro file with the procedures to compute the radial velocities.
#     Also it needs the pxperfect.pro file to produce pretty ps files.
#
# CALLING SEQUENCE:
#     computeRVparams(readparfile=file, rvdatafile1=rvdatafile1, rvdatafile2=rvdatafile2
#                     [, latextable=latextable][, fase=fase][, ps=ps])
#
# INPUTS:
#     readparfile - name of the parameters file as a result of rvfit (string).
#     rvdatafile1 - name of the file containing the primary radial velocities (string).
#     rvdatafile2 - name of the file containing the secondary radial velocities (string).
#     latextable - if set, name of the output .tex file with the results table (string).
#
# OPTIONAL INPUT PARAMETERS:
#
# KEYWORD PARAMETERS:
#     fase - plot the RV data in phase with the period and Tp.
#     ps - save a postscript file with the RV plot.
#
# OUTPUTS:
#     Values for physical parameters of the system, for a single-line or double-line
#     binary.
#     Plot with the RV curve and the residuals for the computed curve given by
#     the parameters in readparfile.
#     If set, .tex file with the parameters for publication.
#
# EXAMPLES:
#     Compute the physical parameters for the eclipsing binary LV Her and plot
#     the RV curve in phase with the period:
#     computeRVparams(readparfile='LVHer.out', rvdatafile1='LVHer_RV1.dat', rvdatafile2='LVHer_RV2.dat', fase=True)
#
#     Compute the physical parameters for the exoplanet HD 37605, plot the RV curve
#     folded in phase and save the latex table with the results:
#     computeRVparams(readparfile='HD37605.out', rvdatafile1='HD37605_RV.dat', fase=True, latextable='output.tex')
#
# REFERENCES:
#
# MODIFICATION HISTORY:
#       Written by Ramon Iglesias Marzoa, Dep. Astrofisica (ULL), 2013-03-09.
#     Last modified: 2015-02-20.
# -

from RVlib import readparamfile, radvel2
from math import sqrt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def computeRVparams(readparfile, rvdatafile1, rvdatafile2, fase=False, latextable=None, ps=False):
    
    # Number of parameters
    nx = 7
    
    # Read the parameters
    if readparfile is None:
        print('You must set a valid parameters file.')
        return
    
    params, sinf, ssup, chis, nobs1, nobs2, dt = readparamfile(readparfile, nx)
    P, Tp, e, omega, gamma, K1, K2 = params
    
    # Check whether the uncertainties are symmetric
    sel = np.where(ssup == 0.)[0]
    if len(sel) == nx:
        
        # If all the upper uncert. are null then the actual uncert. are the lower
        sP = sinf[0]
        sTp = sinf[1]
        se = sinf[2]
        somega = sinf[3]
        sgamma = sinf[4]
        sK1 = sinf[5]
        sK2 = sinf[6]
        
    else:
        
        # If the upper uncert. are not null then symmetrize them
        sP = (sinf[0] + ssup[0]) / 2.
        sTp = (sinf[1] + ssup[1]) / 2.
        se = (sinf[2] + ssup[2]) / 2.
        somega = (sinf[3] + ssup[3]) / 2.
        sgamma = (sinf[4] + ssup[4]) / 2.
        sK1 = (sinf[5] + ssup[5]) / 2.
        sK2 = (sinf[6] + ssup[6]) / 2.
    

    tbase=2450000.0
    # Save a postscript file with the RV plot if requested
    if ps:
        # Simbolo = círculo.
        A = np.arange(17) * (2 * np.pi / 16)
        A = np.append(A, A[0])
        plt.fill(np.cos(A), np.sin(A))
        tsimbol = 1.2 # tamaño

        # Save original graphic variables and create new ones.
        p_old = plt.rcParams['lines.linewidth']
        x_old = plt.rcParams['axes.linewidth']
        y_old = plt.rcParams['ytick.major.width']
    
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['ytick.major.width'] = 2
    
        # Open postscript file
        psfile = readparfile + '.ps'
        plt.savefig(psfile, dpi=300, bbox_inches='tight')
    else:
        # wysize=500
        # wxsize=wysize*1.618; golden rectangle.
        wxsize = 640
        wysize = 700
        plt.figure(figsize=(wxsize/100, wysize/100))
        plt.axes([0, 0, 1, 1])
    
    plt.cm.get_cmap('gist_heat', lut=13)
    plt.rcParams['font.size'] = 17


    if nobs2 > 0:
        # This is for double-line binaries.
        if not rvdatafile1 or not rvdatafile2:
            print('You need to use valid data files.')
            # retall()
            
        # Data reading.
        t1, rv1, srv1 = np.loadtxt(rvdatafile1, usecols=(0,1,2), unpack=True)
        t2, rv2, srv2 = np.loadtxt(rvdatafile2, usecols=(0,1,2), unpack=True)
        
        print()
        print('Result:')
        print('------------------------------------------------')
        print('              Fitted parameters')
        print('------------------------------------------------')
        print('P (d)              = {:.2f} +/- {:.2f}'.format(P, sP))
        print('Tp (HJD/BJD)       = {:.2f} +/- {:.2f}'.format(Tp, sTp))
        print('e                  = {:.2f} +/- {:.2f}'.format(e, se))
        print('omega (deg)        = {:.2f} +/- {:.2f}'.format(omega, somega))
        print('gamma (km/s)       = {:.2f} +/- {:.2f}'.format(gamma, sgamma))
        print('K1 (km/s)          = {:.2f} +/- {:.2f}'.format(K1, sK1))
        print('K2 (km/s)          = {:.2f} +/- {:.2f}'.format(K2, sK2))

        M1sini3 = 1.0361e-7*(1.-e**2)**(1.5)*(K1+K2)**2*K2*P # Msun
        M2sini3 = 1.0361e-7*(1.-e**2)**(1.5)*(K1+K2)**2*K1*P # Msun
        a1sini = 13751.*np.sqrt(1.-e**2)*K1*P # km
        a2sini = 13751.*np.sqrt(1.-e**2)*K2*P # km
        asini = a1sini + a2sini # km
        q = M2sini3/M1sini3

        sM1sini3 = 1.0361e-7*np.sqrt((3*(K1+K2)**2*K2*P*e)**2*(1-e**2)*se**2 + \
                                    (2*P*(K1+K2)*K2)**2*(1-e**2)**3*sK1**2 + \
                                    (2*(K1+K2)*K2+(K1+K2)**2)**2*P**2*(1-e**2)**3*sK2**2 + \
                                    (K1+K2)**4*K2**2*(1-e**2)**3*sP**2)
        sM2sini3 = 1.0361e-7*sqrt((3*(K1+K2)**2*K1*P*e)**2*(1-e**2)*se**2 + \
                                 (2*(K1+K2)*K1+(K1+K2)**2)**2*P**2*(1-e**2)**3*sK1**2 + \
                                 (2*P*(K1+K2)*K1)**2*(1-e**2)**3*sK2**2 + \
                                 (K1+K2)**4*K1**2*(1-e**2)**3*sP**2)
        sa1sini=13751.*sqrt((1-e**2)*( (K1*P*e/(1-e**2))**2*se**2 + P**2*sK1**2 + K1**2*sP**2))
        sa2sini=13751.*sqrt((1-e**2)*( (K2*P*e/(1-e**2))**2*se**2 + P**2*sK2**2 + K1**2*sP**2))
        sasini=sqrt(sa1sini**2+sa2sini**2)
        sq=sqrt((sK1/K2)**2+(K1*sK2/K2**2)**2)

        print('------------------------------------------------')
        print('              Derived quantities')
        print('------------------------------------------------')
        print('M1sin(i)^3 (Msun)  = {:.2f} +/- {:.2f}'.format(M1sini3, sM1sini3))
        print('M2sin(i)^3 (Msun)  = {:.2f} +/- {:.2f}'.format(M2sini3, sM2sini3))
        print('q = M2/M1          = {:.2f} +/- {:.2f}'.format(q, sq))
        print('a1sin(i) (10^6 km) = {:.2f} +/- {:.2f}'.format(a1sini/1e6, sa1sini/1e6))
        print('         (Rsun)    = {:.2f} +/- {:.2f}'.format(a1sini*0.019758/13751., sa1sini*0.019758/13751.))
        print('a2sin(i) (10^6 km) = {:.2f} +/- {:.2f}'.format(a2sini/1e6, sa2sini/1e6))
        print('         (Rsun)    = {:.2f} +/- {:.2f}'.format(a2sini*0.019758/13751., sa2sini*0.019758/13751.))
        print('asin(i) (10^6 km)  = {:.2f} +/- {:.2f}'.format(asini/1e6, sasini/1e6))
        print('        (Rsun)     = {:.2f} +/- {:.2f}'.format(asini*0.019758/13751., sasini*0.019758/13751.))

        # Time array for the fitted curve.
        tt=[t1,t2]
        rrvv=[rv1,rv2]
        tini=min(tt)
        tfin=max(tt)
        np=int((tfin-tini)/P*100.)	# 100 points in each observed period.
        tfit=tini+(tfin-tini)*np.arange(np)/(np-1)
        #print(tfit)

        # Array with the fitted radial velocities.
        vfit1 = radvel2(tfit,[P,Tp,e,omega,gamma,K1])
        vfit2 = radvel2(tfit,[P,Tp,e,omega+180.,gamma,K2])

        # Compute the residuals.
        vinterpol1 = interp1d(tfit, vfit1)(t1)
        OC1 = rv1 - vinterpol1
        vinterpol2 = interp1d(tfit, vfit2)(t2)
        OC2 = rv2 - vinterpol2

        rms1 = np.sqrt(np.sum(OC1**2)/nobs1)
        rms2 = np.sqrt(np.sum(OC2**2)/nobs2)

        print('------------------------------------------------')
        print('              Other quantities')
        print('------------------------------------------------')
        print(f'chi^2              = {chis:.2f}')
        print(f'Nobs (primary)     = {nobs1:.2f}')
        print(f'Nobs (secondary)   = {nobs2:.2f}')
        print(f'Time span (days)   = {dt:.2f}')
        print(f'rms1 (km/s)        = {rms1:.2f}')
        print(f'rms2 (km/s)        = {rms2:.2f}')

    #-----------------Latex table--------------------
    # Build latex table with the parameters for publication if requested
    if latextable:
        latexfile = readparfile + '.tex'
        with open(latexfile, 'w') as uni:
            uni.write('\\begin{center}\n')
            uni.write('{\\scriptsize\n')
            uni.write('\\begin{table}[h]\n')
            uni.write('\\begin{tabular}{lr}\n')
            uni.write('\\multicolumn{2}{c}{TITLE OF THE TABLE}\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\hline\n')
            uni.write('Parameter & Value\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\multicolumn{2}{c}{Adjusted Quantities}\\\\\n')
            uni.write('\\hline\n')
            uni.write('$P$ (d) & ' + str(P).strip() + ' $\\pm$ ' + str(sP).strip() + '\\\\\n')
            uni.write('$T_p$ (HJD) & ' + str(Tp).strip() + ' $\\pm$ ' + str(sTp).strip() + '\\\\\n')
            uni.write('$e$ & ' + str(e).strip() + ' $\\pm$ ' + str(se).strip() + '\\\\\n')
            uni.write('$\\omega$ (deg) & ' + str(omega).strip() + ' $\\pm$ ' + str(somega).strip() + '\\\\\n')
            uni.write('$\\gamma$ (km/s) & ' + str(gamma).strip() + ' $\\pm$ ' + str(sgamma).strip() + '\\\\\n')
            uni.write('$K_1$ (km/s) & ' + str(K1).strip() + ' $\\pm$ ' + str(sK1).strip() + '\\\\\n')
            uni.write('$K_2$ (km/s) & ' + str(K2).strip() + ' $\\pm$ ' + str(sK2).strip() + '\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\multicolumn{2}{c}{Derived Quantities}\\\\\n')
            uni.write('\\hline\n')
            uni.write('$M_1\\sin ^3i$ ($M_\\odot$) & ' + str(M1sini3).strip() + ' $\\pm$ ' + str(sM1sini3).strip() + '\\\\\n')
            uni.write('$M_2\\sin ^3i$ ($M_\\odot$) & ' + str(M2sini3).strip() + ' $\\pm$ ' + str(sM2sini3).strip() + '\\\\\n')
            uni.write('$q = M_2/M_1$ & ' + str(q).strip() + ' $\\pm$ ' + str(sq).strip() + '\\\\\n')
            uni.write('$a_1\\sin i$ ($10^6$ km) & ' + str(a1sini / 1e6).strip() + ' $\\pm$ ' + str(sa1sini / 1e6).strip() + '\\\\\n')
            uni.write('$a_2\\sin i$ ($10^6$ km) & ' + str(a2sini / 1e6).strip() + ' $\\pm$ ' + str(sa2sini / 1e6).strip() + '\\\\\n')
            uni.write('$a  \\sin i$ ($10^6$ km)\t&'+str(asini/1e6)[:2]+' $\\pm$ '+str(sasini/1e6)[:2]+'\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\multicolumn{2}{c}{Other Quantities}\\\\\n')
            uni.write('\\hline\n')
            uni.write('$\\chi^2$\t\t\t&'+str(chis)[:2]+'\\\\\n')
            uni.write('$N_{obs}$ (primary)\t\t&'+str(nobs1)[:2]+'\\\\\n')
            uni.write('$N_{obs}$ (secondary)\t&'+str(nobs2)[:2]+'\\\\\n')
            uni.write('Time span (days)\t\t&'+str(dt)[:2]+'\\\\\n')
            uni.write('$rms_1$ (km/s)\t\t&'+str(rms1)[:2]+'\\\\\n')
            uni.write('$rms_2$ (km/s)\t\t&'+str(rms2)[:2]+'\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\end{tabular}\n')
            uni.write('\\caption{\\footnotesize $^a$ Parameter fixed beforehand.}\n')
            uni.write('\\label{table:test2}\n')
            uni.write('\\end{table}\n')
            uni.write('}\n')
            uni.write('\\end{center}\n')
        print('LaTeX file: ' + latexfile)


    #-----------------Plots--------------------
    # Plot RV data in phase with the period and Tp if requested

    rangot=np.array([tini,tfin])-tbase
    rangorv=np.array([min([rrvv,vfit1,vfit2]),max([rrvv,vfit1,vfit2])])
    rangoOC=np.array([min([OC1,OC2]),max([OC1,OC2])])

    pos1=[0.15, 0.535, 0.96, 0.95]
    pos2=[0.15, 0.11+0.2025+0.01, 0.96, 0.535-0.01]
    pos3=[0.15, 0.11, 0.96, 0.11+0.2025]

    if fase is not None:

        # Fases.
        ciclo = (t1 - Tp) / P	# ciclo e o instante da observacion medido en unidades de periodo
        fase1 = ciclo - np.floor(ciclo)	# floor quedase coa parte enteira do ciclo
        ordenobs1 = np.argsort(fase1)
        fase1 = fase1[ordenobs1]

        ciclo=(t2-Tp)/P
        fase2=ciclo-np.floor(ciclo)
        ordenobs2=np.argsort(fase2)
        fase2=fase2[ordenobs2]

        ciclo=(tfit-Tp)/P
        fase=ciclo-np.floor(ciclo)
        ordenfit=np.argsort(fase)
        fasefit=fase[ordenfit]

        wtitle='RV fit and residuals'
        xx1=fase1
        xx2=fase2
        yy1=rv1[ordenobs1]
        yy2=rv2[ordenobs2]
        rr1=OC1[ordenobs1]
        rr2=OC2[ordenobs2]
        syy1=srv1[ordenobs1]
        syy2=srv2[ordenobs2]
        xxfit=fasefit
        yyfit1=vfit1[ordenfit]
        yyfit2=vfit2[ordenfit]
        xxrange=[0.,1.]
        yyrange=[min(rrvv),max(rrvv)]
        yyrange=rangorv
        #rrrange=[min([OC1,OC2]),max([OC1,OC2])]
        rrrange=np.max(np.abs([OC1,OC2]))*[-1.,1.]
        xxtitle='Phase'
        yytitle='RV (km/s)'
        rrtitle1='(O-C)!D1!N (km/s)'
        rrtitle2='(O-C)!D2!N (km/s)'

    else:

        wtitle='RV fit and residuals'
        xx1=t1-tbase
        xx2=t2-tbase
        yy1=rv1
        yy2=rv2
        rr1=OC1
        rr2=OC2
        syy1=srv1
        syy2=srv2
        xxfit=tfit-tbase
        yyfit1=vfit1
        yyfit2=vfit2
        xxrange=np.array([tini,tfin])-tbase
        #yyrange=[min(rrvv),max(rrvv)]
        yyrange=rangorv
        #rrrange=[min([OC1,OC2]),max([OC1,OC2])]
        rrrange=np.max(np.abs([OC1,OC2]))*[-1.,1.]
        xxtitle='HJD-'+str(tbase).strip()
        rrtitle1='(O-C)!D1!N (km/s)'
        rrtitle2='(O-C)!D2!N (km/s)'


    # To plot curves with only 1 RV observed point in one component
    # (usually the secondary).
    if len(xx1) == 1:
        xx1 = [xx1, xx1]
        yy1 = [yy1, yy1]
        syy1 = [syy1, syy1]
        rr1 = [rr1, rr1]
    if len(xx2) == 1:
        xx2 = [xx2, xx2]
        yy2 = [yy2, yy2]
        syy2 = [syy2, syy2]
        rr2 = [rr2, rr2]

    # Plots with the measurements and the fit.
    plt.errorbar(xx1, yy1, yerr=syy1, fmt='o', mec='black', mfc='white', ms=5)
    plt.plot(xxfit, yyfit1, 'k-', linewidth=1)
    plt.errorbar(xx2, yy2, yerr=syy2, fmt='o', mec='black', mfc='white', ms=5)
    plt.plot(xxfit, yyfit2, 'k--', linewidth=1)
    plt.plot([min(xxrange), max(xxrange)], [gamma, gamma], 'k-', linewidth=1)

    # Plots with residuals.
    plt.errorbar(xx1, rr1, yerr=syy1, fmt='o', mec='black', mfc='white', ms=5)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.subplots_adjust(left=0.15, bottom=0.11, right=0.96, top=0.95, wspace=0, hspace=0.01)
    plt.xlabel('Phase')
    plt.ylabel('(O-C) (km/s)')
    plt.ylim(rrrange)
    plt.yticks([rr1[0]])
    plt.subplot(313)
    plt.errorbar(xx2, rr2, yerr=syy2, fmt='o', mec='black', mfc='white', ms=5)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.subplots_adjust(left=0.15, bottom=0.11, right=0.96, top=0.535, wspace=0, hspace=0.01)
    plt.xlabel('Phase')
    plt.ylabel('(O-C) (km/s)')
    plt.ylim(rrrange)
    plt.yticks([rr2[0]])
    plt.gca().invert_yaxis()


    # This is for single-line binaries.

    if "rvdatafile1" not in locals():
        print("You need to use valid data files.")
        # Assuming "retall" is a function defined somewhere else
        # retall()

    # Data reading.
    with open(rvdatafile1, 'r') as f:
        t1, rv1, srv1 = np.loadtxt(f, usecols=(0, 1, 2), comments='#', unpack=True)
        t1 = t1.astype(np.int32)

    print()
    print('Result:')
    print('------------------------------------------------')
    print('              Fitted parameters')
    print('------------------------------------------------')
    print('P (d)              = {:.2f} +/- {:.2f}'.format(P, sP))
    print('Tp (HJD/BJD)       = {:.5f} +/- {:.2f}'.format(Tp, sTp))
    print('e                  = {:.2f} +/- {:.2f}'.format(e, se))
    print('omega (deg)        = {:.2f} +/- {:.2f}'.format(omega, somega))
    print('gamma (km/s)       = {:.2f} +/- {:.2f}'.format(gamma, sgamma))
    print('K1 (km/s)          = {:.2f} +/- {:.2f}'.format(K1, sK1))

    a1sini = 13751. * np.sqrt(1. - e ** 2) * K1 * P  # Km 
    f = 1.0361e-7 * (1 - e ** 2) ** (3. / 2.) * K1 ** 3 * P

    sa1sini = 13751. * np.sqrt((1 - e ** 2) * ((K1 * P * e / (1 - e ** 2)) ** 2 * se ** 2 + P ** 2 * sK1 ** 2 + K1 ** 2 * sP ** 2))
    sf = 1.0361e-7 * np.sqrt((9 * K1 ** 6 * P ** 2 * (1 - e ** 2)) * se ** 2 + (9 * (1 - e ** 2) ** 3 * K1 ** 4 * P ** 2) * sK1 ** 2 + ((1 - e ** 2) ** 3 * K1 ** 6) * sP ** 2)

    print('------------------------------------------------')
    print('              Derived quantities')
    print('------------------------------------------------')
    print('a1sin(i) (10^6 km) = ' + str(round(a1sini/1e6, 2)) + ' +/- ' + str(round(sa1sini/1e6, 2)))
    print('         (Rsun)    = ' + str(round(a1sini*0.019758/13751., 2)) + ' +/- ' + str(round(sa1sini*0.019758/13751., 2)))
    print('f(m1,m2) (Msun)    = ' + str(round(f, 2)) + ' +/- ' + str(round(sf, 2)))

    # Time array for the fitted curve.
    tini = min(t1)
    tfin = max(t1)
    np = int((tfin - tini) / P * 100)  # 100 points in each observed period.
    tfit = np.linspace(tini, tfin, np)

    # Array with the fitted radial velocities.
    # if e < 1e-4:
    #     e = 0.
    #     omega = 0.
    vfit1 = radvel2(tfit, [P, Tp, e, omega, gamma, K1])

    # Compute the residuals.
    vinterpol1 = np.interp(t1, tfit, vfit1)
    OC1 = rv1 - vinterpol1
    rms1 = np.sqrt(np.sum(OC1**2) / nobs1)

    print('------------------------------------------------')
    print('              Other quantities')
    print('------------------------------------------------')
    print('chi^2              = '+str(round(chis, 2)))
    print('Nobs (primary)     = '+str(nobs1))
    print('Time span (days)   = '+str(round(dt, 2)))
    print('rms1 (km/s)        = '+str(round(rms1, 2)))

#-----------------LaTeX table--------------------

    if latextable:

        latexfile = readparfile + '.tex'
        with open(latexfile, 'w') as uni:

            uni.write('\\begin{center}\n')
            uni.write('{\\scriptsize\n')
            uni.write('\\begin{table}[h]\n')
            uni.write('\\begin{tabular}{lr}\n')
            uni.write('\\multicolumn{2}{c}{TITLE OF THE TABLE}\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\hline\n')
            uni.write('Parameter             &Value\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\multicolumn{2}{c}{Adjusted Quantities}\\\\\n')
            uni.write('\\hline\n')
            uni.write('$P$ (d)           &' + str(P) + ' $\\pm$ ' + str(sP) + '\\\\\n')
            uni.write('$T_p$ (HJD)           &' + format(Tp, '.5f') + ' $\\pm$ ' + str(sTp) + '\\\\\n')
            uni.write('$e$                &' + str(e) + ' $\\pm$ ' + str(se) + '\\\\\n')
            uni.write('$\\omega$ (deg)           &' + str(omega) + ' $\\pm$ ' + str(somega) + '\\\\\n')
            uni.write('$\\gamma$ (km/s)     &' + str(gamma) + ' $\\pm$ ' + str(sgamma) + '\\\\\n')
            uni.write('$K_1$ (km/s)           &' + str(K1) + ' $\\pm$ ' + str(sK1) + '\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\multicolumn{2}{c}{Derived Quantities}\\\\\n')
            uni.write('\\hline\n')
            uni.write('$a_1\\sin i$ ($10^6$ km)     &' + str(a1sini/1e6) + ' $\\pm$ ' + str(sa1sini/1e6) + '\\\\\n')
            uni.write('$f(m_1,m_2)$ ($M_\\odot$)           &' + str(f) + ' $\\pm$ ' + str(sf) + '\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\multicolumn{2}{c}{Other Quantities}\\\\\n')
            uni.write('\\hline\n')
            uni.write('$\\chi^2$           &' + str(chis) + '\\\\\n')
            uni.write('$N_{obs}$ (primary)     &' + str(nobs1) + '\\\\\n')
            uni.write('Time span (days)     &' + str(dt) + '\\\\\n')
            uni.write('$rms_1$ (km/s)    &' + str(rms1) + '\\\\\n')
            uni.write('\\hline\n')
            uni.write('\\end{tabular}\n')
            uni.write('\\caption{\\footnotesize $^a$ Parameter fixed beforehand.}\n')
            uni.write('\\label{table:test4}\n')
            uni.write('\\end{table}\n')
            uni.write('}\n')
            uni.write('\\end{center}\n')

        print('LaTeX file: ' + latexfile)


    #-----------------Plots--------------------

    pos1=[0.15, 0.11+0.2025+0.01, 0.96, 0.95]
    pos2=[0.15, 0.11, 0.96, 0.11+0.2025]

    if fase:
        # Fases.
        ciclo=(t1-Tp)/P    # ciclo e o instante da observacion medido en unidades de periodo
        fase1=ciclo-np.floor(ciclo)    # floor quedase coa parte enteira do ciclo
        ordenobs1=np.argsort(fase1)
        fase1=fase1[ordenobs1]

        ciclo=(tfit-Tp)/P    # ciclo e o instante da observacion medido en unidades de periodo
        fasefit=ciclo-np.floor(ciclo)    # floor quedase coa parte enteira do ciclo
        ordenfit=np.argsort(fasefit)
        fasefit=fasefit[ordenfit]

        wtitle='RV fit and residuals (phased)'
        xx1=fase1
        yy1=rv1[ordenobs1]
        rr1=OC1[ordenobs1]
        syy1=srv1[ordenobs1]
        xxfit=fasefit
        yyfit=vfit1[ordenfit]
        xrange=[0.,1.]
        yrange=[min([rv1,vfit1]),max([rv1,vfit1])]
        rrange=max(abs(OC1))*[-1.,1.]
        xtitle='Phase'
        ytitle1='RV (km/s)'
        rrtitle='(O-C)!D!N (km/s)'
    else:
        wtitle='RV fit and residuals'
        xx1=t1-tbase
        yy1=rv1
        rr1=OC1
        syy1=srv1
        xxfit=tfit-tbase
        yyfit=vfit1
        xrange=[min(t1),max(t1)]-tbase
        yrange=[min([rv1,vfit1]),max([rv1,vfit1])]
        rrange=max(abs(OC1))*[-1.,1.]
        xtitle='HJD-'+str(tbase).strip()
        ytitle1='RV (km/s)'
        rrtitle='(O-C)!D!N (km/s)'

    # Plots with the measurements and the fit.
    plt.errorbar(xx1, yy1, yerr=syy1, fmt='o', color='black', capsize=5)
    plt.plot(xxfit, yyfit, linestyle='-', color='blue')
    plt.axhline(y=gamma, color='red', linestyle='--')

    # Plots with residuals.
    residuals = yy1 - yyfit
    plt.figure()
    plt.errorbar(xx1, residuals, yerr=syy1, fmt='o', color='black', capsize=5)
    plt.plot(xx1, [0]*len(xx1), linestyle='-', color='blue')
    plt.ylim(rrange)

    # Set the x and y axis labels
    plt.xlabel(xtitle)
    plt.ylabel(ytitle1)
    plt.show()


    # if 'ps' in locals():
    #     device.close()
    #     print(f"Output: {psfile}")
    #     set_plot('x')
    #     p = p_old
    #     x = x_old
    #     y = y_old






