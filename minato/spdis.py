import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.special as sc
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from astropy.io import ascii
from astropy.table import Table
from astropy.constants import c
from datetime import date
import myRC

class SpecDisent:
    def __init__(self, lines, orbparams, epochs, spectra, extension='.fits'):
        '''
        lines       : Spectral lines on which the disentangling will be performed.
        orbparams   : file with orbital parameters. Coulmns must be: star_ID, P, T0, e, Omega, Gamma, K1, K2.
                    K1, K2 can be derived, but initial values need to be provided to set the search grids.
        epochs      : File with Julian dates. First column must give epoch of observations and second column the Julian dates.
        spectra     : Path to the spectra. Format can be 'fits' or other text extensions.
        '''
        self.lines = lines
        self.orbparams = orbparams
        self.epochs = epochs


        lines_dic = {
                        3798: { 'region':[3785., 3815.], 'title':'H$\theta$'},
                        3889: { 'region':[3875., 3905.], 'title':'H$\zeta$'},
                        3970: { 'region':[3960., 3984.], 'title':'H$\epsilon$'},
                        4009: { 'region':[4003., 4018.], 'title':'He I $\lambda$4009'},
                        4026: { 'region':[4016., 4036.], 'title':'He I $\lambda$4026'},
                        4102: { 'region':[4087., 4130.], 'title':'H$\delta$'},
                        4121: { 'region':[4105., 4136.], 'title':'He I $\lambda$4121, Si II $\lambda$4128/32'},
                        4144: { 'region':[4133., 4154.], 'title':'He I $\lambda$4144'},
                        4233: { 'region':[4225, 4241], 'title':'Fe II $\lambda$4233'},
                        4267: { 'region':[4260, 4275], 'title':'C II $\lambda$4267'},
                        4340: { 'region':[4320., 4360.], 'title':'H$\gamma$'},
                        4388: { 'region':[4378., 4398.], 'title':'He I $\lambda$4388'},
                        4471: { 'region':[4459., 4488.], 'title':'He I $\lambda$4471, Mg II $\lambda$4481'},
                        # 4553: { 'region':[4536, 4560], 'title':'Fe II $\lambda$4550/56, Si III $\lambda$4553'} }
                        4553: { 'region':[4546., 4561.], 'title':'He II $\lambda$4542, Si III $\lambda$4553'}, 
                        4713: { 'region':[4690., 4725.], 'title':'He I $\lambda$4713'}, 
                        4861: { 'region':[4840., 4880.], 'title':'H$\beta$'}, 
                        4922: { 'region':[4910., 4933.], 'title':'He I $\lambda$4922'} 
                    }

        # Using the lines provided by the user to obtain the wavelength ranges
        Ranges = []
        for line in lines:
            Ranges.append(lines_dic[line]['region'])
        if len(lines)==1:
            Rangestr = str(lines[0])
        else:
            Rangestr = 'spLines'
        
        self.Rangestr = Rangestr
        self.Ranges = Ranges


        # Reading the file with orbital parameters
        df = pd.read_csv(orbparams, skipinitialspace=True, header=0, comment='#')
        # display(df.columns)
        StarName, P, T0, ecc, Gamma, Omega, K1, K2 = df['star_ID'].values[0], df['P'].values, df['T0'].values, df['e'].values, df['Gamma'].values, df['Omega'].values, df['K1'].values[0], df['K2'].values[0]
        self.StarName, self.P, self.T0, self.ecc, self.Gamma, self.Omega, self.K1, self.K2 = StarName, P, T0, ecc, Gamma, Omega, K1, K2

        # Reading Julian dates of observations
        PhaseFiles = ascii.read(epochs)
        # print(PhaseFiles)
        MJDs = PhaseFiles['col2']
        # MJDs = np.array(MJDs)
        self.MJDs = MJDs
        specnames = [spectra + name + extension for name in PhaseFiles['col1']]
        # specnames = [spectra + 'c' + name + extension for name in PhaseFiles['col1']]
        self.specnames = specnames
        print(specnames)
        
        # Compute phase
        phis = (MJDs-T0)/P - ((MJDs-T0)/P).astype(int)
        self.phis = phis

        # Set working path
        working_path = spectra.replace('data/', '')+'disentangling_'+str(date.today())+'/'
        if not os.path.exists(working_path):
            os.makedirs(working_path)
        self.working_path = working_path

    def get_orbparams(self):
        print('These are the orbital parameters:\n', 
                'starid =', self.StarName, '\n', 
                'P      =', self.P, '\n', 
                'T0     =', self.T0, '\n', 
                'e      =', self.ecc, '\n', 
                'Gamma  =', self.Gamma, '\n', 
                'Omega  =', self.Omega, '\n', 
                'K1     =', self.K1, '\n', 
                'K2     =', self.K2 )


    def get_disspec(self, lguess1=0.7, S2Nblue=4200, S2Nred=4215, GridDis=True, oneD=False, Triple=False, CleanCos=False, 
                    PLOTCONV=False, itrnumlim=30, NumItrFinal=500, PLOTITR=False, N_Itr_Plot=5, PLOTFITS=False, PLOTEXTREMES=True, 
                    NebOff=False, rmNebChi2=False, StrictNegA=True, StrictNegB=True, ContSig=3, Renormalise=False, NormPoints=None, 
                    N_K1=15, N_K2=15, minK1=0.6, maxK1=1.4, minK2=0.01, maxK2=1.99):
        '''
        lguess1     : Primary's light contribution = f1/(f1 + f2) [Only important for scaling of final spectra].
        S2Nblue     : Initial value of wavelength range to compute S/N (for defining continuum and weighting of spectra when co-adding).
        S2Nred      : Final value of wavelength range to compute S/N.
        GridDis     : Perform grid disentangling. Dafault 'True', if False, K1, K2 are adopted from the user.
        oneD        : Perform 1D disentangling. Default 'False', if True, K1 is adopted from the user.
        Triple      : Include a third static star. Default 'False'.
        CleanCos    : Clean cosmic rays
        PLOTCONV    : Plot convergence grid, diff vs. iterations 
        itrnumlim   : Maximum number of shift-and-add iterations
        NumItrFinal : Maximum number of shift-and-add iterations before computing the final spectra
        PLOTITR     : Plot disentangled spectra after each N_Iteration_Plot iterations
        N_Itr_Plot  : Number of iteration for PLOTITR
        PLOTFITS    : Plot fits between dis1+dis2 and observations for all observations
        PLOTEXTREMES: Plot two panels showing the fits at extremes 
        NebOff      : Switch off nebular emission. Default 'False', if True, nebular emission is not disentangled
        rmNebChi2   : Remove nebular lines from chi2 fitting
        StrictNegA  : Force disentangled spectra of the primary to remain below 1 in flux units. Set to False if emission features or select range in PosLimCondA
        StrictNegB  : Force disentangled spectra of the secondary to remain below 1 in flux units. Set to False if emission features or select range
        ContSig     : how many sigmas above continuum can spectra lie if forced to be below continuum (approximate).
        Renormalise : Renormalise spectra at pre-specified points.
        NormPoints  : Wavelength points for renormalisation
        N_K1        : Number of points to be used in the K1 grid
        N_K2        : Number of points to be used in the K2 grid
        minK1       : Fraction of K1 used as initial point in the K1 grid. Default minK1 = 0.6*K1
        maxK1       : Fraction of K1 used as final point in the K1 grid. de hecho Default maxK1 = 1.4*K1
        minK2       : Fraction of K2 used as initial point in the K2 grid. Default minK2 = 0.05*K2
        maxK2       : Fraction of K2 used as final point in the K2 grid. Default maxK2 = 1.95*K2
                    In general, use is as follows: Karr = np.arange(minK*K, maxK*K, N_K)
        '''

        self.oneD = oneD
        self.itrnumlim = itrnumlim
        self.StrictNegA = StrictNegA
        self.StrictNegB = StrictNegB
        self.NebOff = NebOff
        self.Triple = Triple
        self.PLOTCONV = PLOTCONV
        self.PLOTITR = PLOTITR
        self.rmNebChi2 = rmNebChi2
        self.PLOTEXTREMES = PLOTEXTREMES
        self.PLOTFITS = PLOTFITS
        self.N_Itr_Plot = N_Itr_Plot

        # Secondary's light contribution
        # print(lguess1)
        lguess2 = 1-lguess1
        print('Light ratio 1 guess =', lguess1)
        print('Light ratio 2 guess =', lguess2)

        # for PLOTFITS
        Velo_plot_usrK2 = self.K2
        Velo_plot_usrK1 = self.K1

        # for PLOTEXTREMES
        Velo_plot_usrK2_ext = self.K2
        Velo_plot_usrK1_ext = self.K1

        #Only relevant if StrictNegA=True
        PosLimCondA = np.array([ [3968., 3969.]
                                ])
        self.PosLimCondA = PosLimCondA

        # Dis. spectra of secondary must be <1 in all regions but those specified below
        PosLimCondB = np.array([ 
                    [3968., 3969.]
                    ])
        self.PosLimCondB = PosLimCondB

        # For nebular emission
        PosLimCondC = np.array([ 
                [3970.075-1.5, 3970.075+1.5], [4104.4-3, 4106.5-4], [4343-4, 4345.-4], [4861.297-0.6, 4861.297+0.6]
                ]) 
        # 4471: [4466, 4476]
        # 4861: [4864.9-4, 4865.8-4] OR [4856.0-4, 4874.7-4] 
        # 3970: [3970.075-1.5, 3970.075+1.5], 
        self.PosLimCondC = PosLimCondC

        # For renormalisation
        if Renormalise==True and NormPoints==False:
            NormPoints = [3961., 4006., 4016., 4038., 4088., 4116., 4129., 4138., 4154., 4195., 4210., 4328., 4362., 4386., 4400., 4462., 4490., 4494., 4530., 4557., 4560]
            self.NormPoints = NormPoints

        clight = 2.9979E5
        self.clight = clight
        kcount = 0
        self.kcount = kcount

        if PLOTITR and PLOTCONV:
            print("Both options PLOTCONV, PLOTITR cannot be true...")
            sys.exit()

        if Triple:
            lguess3 = 0.1

        specnamesFin =[]
        # Remove observations manually if needed
        for el in self.specnames:
            specnamesFin.append(el)
        self.specnamesFin = specnamesFin
        print("Total number of observations: ", len(self.specnames))    
        print("final number of observations: ", len(specnamesFin))        



        # phis[phis<0] += 1
        # Ms = 2 * np.pi * phis
        # Es =  Kepler(1., Ms, ecc)
        # eccfac = np.sqrt((1 + ecc) / (1 - ecc))
        # nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
        # print(nusdata)

        vrads1, vrads2 = self.v1andv2(self.K1, self.K2)

        vrads3 = vrads1*0 + self.Gamma
        self.vrads3 = vrads3



        S2Ns = []

        minFluxHgArr = []
        ObsSpecs = []
        for i, filepath in enumerate(specnamesFin):
            spec = self.read_file(filepath)
            if CleanCos:
                SpecClean = self.Cosclean(np.copy(spec))
            else:
                SpecClean = np.copy(spec)
            if Renormalise and GridDis:
                SpecNorm = self.Normalise(np.copy(SpecClean), points=self.NormPoints)
            else:
                SpecNorm= np.copy(SpecClean)
            ObsSpecs.append(SpecNorm)
        #measure S2N    
            waves = SpecNorm[:,0]
            fluxes = SpecNorm[:,1]
            S2Nrange  = (waves > S2Nblue) * (waves < S2Nred)  
            S2Ns.append(1./np.std(spec[:,1][S2Nrange]))
        # print('wave =', waves)
        # print('flux =', fluxes)
        for i, spec in enumerate(ObsSpecs):
            print(f"Shape of array at index {i}: {spec.shape}")
        # ObsSpecs = np.array(ObsSpecs, dtype=object)
        ObsSpecs = np.array(ObsSpecs)
        self.ObsSpecs = ObsSpecs
        # print(ObsSpecs)

        S2Ns = np.array(S2Ns)    

        # Determines by how much "negative" spectra can be above 1.
        Poslim = 1./S2Ns
        self.Poslim = Poslim

        #print Poslim

        #dsaad

        S2Nsmean = np.mean(S2Ns) *np.sqrt(len(S2Ns))
        PoslimallA =ContSig/S2Nsmean *0.5
        PoslimallB =ContSig/S2Nsmean *0.5
        Poslimall =ContSig/S2Nsmean *0.5
        self.PoslimallA = PoslimallA
        self.PoslimallB = PoslimallB
        self.Poslimall = Poslimall
 
        # Michael weighting!
        weights = S2Ns**2 / np.sum(S2Ns**2)
        self.weights = weights

        # Grid on which the spectra are calculated on (taken here as wavelength grid of first spectrum)
        wavegridall = ObsSpecs[0][:,0]
        # print(' ### self.Ranges = ', self.Ranges)
        wavegridDiffCondall = np.array([(wavegridall > Range[0])*(wavegridall < Range[1]) for Range in self.Ranges])
        wavegrid = wavegridall[np.sum(wavegridDiffCondall,axis=0).astype(bool)]
        waveRanges = [wavegridall[el.astype(bool)] for el in wavegridDiffCondall]
        # print(' ### waveRanges = ', waveRanges)

        A = interp1d(wavegrid, np.zeros(len(wavegrid)),bounds_error=False, fill_value=0.)  
        B = interp1d(wavegrid, np.zeros(len(wavegrid)),bounds_error=False, fill_value=0.)  

        if GridDis:
            if oneD:
                K2s = np.linspace(minK2*self.K2, maxK2*self.K2, N_K2)
                self.K2s = K2s
                # print(K2s)
                # sys.exit()
                vrads1, vrads2 = self.v1andv2(self.K1, 100.)
                RVExtmaxInd, RVExtminInd = np.argmax(vrads1), np.argmin(vrads1)        
                self.RVExtmaxInd, self.RVExtminInd = RVExtmaxInd, RVExtminInd
                kcount_extremeplot = np.argmin(np.abs(K2s - Velo_plot_usrK2_ext))
                self.kcount_extremeplot = kcount_extremeplot
                kcount_usr = np.argmin(np.abs(K2s - Velo_plot_usrK2))
                K2 = self.Grid_disentangling(B, vrads1, self.K1, K2s, waveRanges, Ini='B', ShowItr=False)
                self.K2 = K2
                self.kcount_usr = kcount_usr
            else:
                K1s = np.linspace(minK1*self.K1, maxK1*self.K1, N_K1)
                K2s = np.linspace(minK2*self.K2, maxK2*self.K2, N_K2)
                self.K1s = K1s
                self.K2s = K2s
                vrads1, vrads2 = self.v1andv2(100., 100.)
                RVExtmaxInd, RVExtminInd = np.argmax(vrads1), np.argmin(vrads1)
                self.RVExtmaxInd, self.RVExtminInd = RVExtmaxInd, RVExtminInd
                kcount_extremeplot = np.argmin(np.abs(K1s - Velo_plot_usrK1_ext)) * len(K2s) + np.argmin(np.abs(K2s - Velo_plot_usrK2_ext))
                self.kcount_extremeplot = kcount_extremeplot
                kcount_usr = np.argmin(np.abs(K1s - Velo_plot_usrK1)) * len(K2s) + np.argmin(np.abs(K2s - Velo_plot_usrK2))
                K1, K2 = self.Grid_disentangling2D(B, K1s, K2s, waveRanges, Ini='B', ShowItr=False)
                self.K1 = K1
                self.K2 = K2
                self.kcount_usr = kcount_usr
            print("K2 found:", K2)
            # if self.K2 < 0:
            #     print('setting K2=2*K1')
            #     K2 = 2*K1
            # print("disentangling...., K1, K2:", K1, K2)
            # vrads1, vrads2 = self.v1andv2(K1, K2)
        else:
            K1s = np.linspace(minK1*self.K1, maxK1*self.K1, N_K1)
            K2s = np.linspace(minK2*self.K2, maxK2*self.K2, N_K2)
            self.K1s = K1s
            self.K2s = K2s
            PLOTEXTREMES = False
            PLOTFITS = False
            #K2 = Velo_plot_usrK2_ext
            #K1 = Velo_plot_usrK1_ext
            #K2=2*K1
            self.PLOTEXTREMES = PLOTEXTREMES
            self.PLOTFITS = PLOTFITS
            print("K2 defined by user:", self.K2)
        if self.K2 < 0:
            print('setting K2=2*K1')
            K2 = 2*self.K1
            self.K2 = K2
        print("disentangling...., K1, K2:", self.K1, self.K2)
        vrads1, vrads2 = self.v1andv2(self.K1, self.K2)

        itrnumlim=NumItrFinal
        self.itrnumlim = itrnumlim
        A, B, C, redchi2 = self.disentangle(np.zeros(len(wavegridall)), vrads1, vrads2, wavegridall, Resid=False, Reduce=True, ShowItr=True, Once=True)


        # These are the final, scaled spectra:
        if Triple:
            B = (B-1)/lguess2 + 1.
            A = (A-1)/ (1-lguess2-lguess3) + 1.
            C = (C-1)/lguess3 + 1.
        else:
            B = (B-1)/lguess2 + 1. 
            A = (A-1)/lguess1 + 1.


        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(wavegridall, A, label='dis', zorder=1)
        plt.plot(wavegridall, B, label='dis', zorder=0)
        if not NebOff:
            plt.plot(wavegridall, C, label='dis')

        np.savetxt(self.working_path+'ADIS_lguess2_K1K2_[line]=' + str(np.round(lguess2,2)) + '_' + str(np.round(self.K1, 2)) + '_' +  str(np.round(self.K2, 2))+ '_'  + str(self.lines) + '.txt', np.c_[wavegridall, A])
        np.savetxt(self.working_path+'BDIS_lguess2_K1K2_[line]=' + str(np.round(lguess2,2)) + '_' + str(np.round(self.K1, 2)) + '_' + str(np.round(self.K2, 2))+ '_'  + str(self.lines) + '.txt', np.c_[wavegridall, B])
        if not NebOff:
            np.savetxt(self.working_path+'CDIS_lguess2_K1K2_[line]=' + str(np.round(lguess2,2)) + '_' + str(np.round(self.K1, 2)) + '_' + str(np.round(self.K2, 2))+ '_'  + str(self.lines) + '.txt', np.c_[wavegridall, C])
        plt.title('Disentangled spectra')
        ymin, ymax = ax.get_ylim()
        if ymax > 1.5:
            plt.gca().set_ylim(top=1.4)
        if ymin < 0:
            plt.gca().set_ylim(bottom=-1)
        plt.savefig(self.working_path+self.StarName+'_disentangled_'+self.Rangestr+'_lguess1_'+str(lguess1)+'.pdf')
        # plt.show()
        plt.close()



    # Documentation: 
    # "B" = array, initial guess for flux of "secondary"
    # vrads1, vrads2 = RVs of primary, secondary
    # waves: wavelength grid on which disentanglement should take place
    # Note: If initial guess for primary is preferred, roles of primary should change, i.e., one should call:
    # disentangle(Aini, vrads2, vrads1, waves)
    # Resid --> returns array of residual spectra between obs and dis1+dis2
    # Reduce --> Returns reduced chi2
    def disentangle(self, B, vrads1, vrads2, waves, Resid=False, Reduce=False, ShowItr=False, Once=False, NebFac=1., InterKind='linear'):    
        global kcount, k1, k2, DoFs, kcount, Nchi2, K1now, K2now
        ScalingNeb=np.ones(len(waves))    
        C = waves*0.
        if not self.oneD:
            if (not Once) and (not self.oneD):
                try:
                    K1now = self.K1s[k1]
                    K2now = self.K2s[k2]   
                    print("Disentangeling..... K1, K2=", K1now, K2now)
                except:
                    
                    pass
            else:
                k1, k2 = 0, 0
        else:
            print("Disentangeling.....")
        Facshift1 = np.sqrt( (1 + vrads1/self.clight) / (1 - vrads1/self.clight))
        Facshift2 = np.sqrt( (1 + vrads2/self.clight) / (1 - vrads2/self.clight))
        Facshift3 = np.sqrt( (1 + self.vrads3/self.clight) / (1 - self.vrads3/self.clight))    
        Ss1 = [interp1d(self.ObsSpecs[i][:, 0] /Facshift1[i], self.ObsSpecs[i][:, 1]-1.,
                        bounds_error=False, fill_value=0., kind=InterKind)(waves) for i in np.arange(len(self.ObsSpecs))] 
        Ss2 = [interp1d(self.ObsSpecs[i][:, 0] /Facshift2[i], self.ObsSpecs[i][:, 1]-1.,
                        bounds_error=False, fill_value=0., kind=InterKind)(waves) for i in np.arange(len(self.ObsSpecs))]  
        Ss3 = [interp1d(self.ObsSpecs[i][:, 0] / Facshift3[i], self.ObsSpecs[i][:, 1]-1.,
                        bounds_error=False, fill_value=0., kind=InterKind)(waves) for i in np.arange(len(self.ObsSpecs))]      
    # Constant component for nebular lines    
    #Frame of Refernce star 1:    

        WavesBA = np.array([waves * Facshift2[i] /Facshift1[i] for i in np.arange(len(vrads1))])  
        WavesCA = np.array([waves *Facshift3[i]/Facshift1[i] for i in np.arange(len(vrads1))])  
    #Frame of Refernce star 2:        
        WavesAB = np.array([waves * Facshift1[i] /Facshift2[i] for i in np.arange(len(vrads1))])  
        WavesCB = np.array([waves * Facshift3[i] /Facshift2[i] for i in np.arange(len(vrads1))])  
    #Frame of Refernce star 3:        
        WavesAC = np.array([waves *Facshift1[i] /Facshift3[i] for i in np.arange(len(vrads1))])  
        WavesBC = np.array([waves * Facshift2[i] /Facshift3[i] for i in np.arange(len(vrads1))])   
        itr = 0
        #plt.plot(waves, C)
        #plt.show()
        #dasd
        while itr<self.itrnumlim:
            itr+=1
            BAshifts = [interp1d(WavesBA[i], B,bounds_error=False, fill_value=0., kind=InterKind)
                            for i in np.arange(len(self.ObsSpecs))]    
            CAshifts = [interp1d(WavesCA[i], C,bounds_error=False, fill_value=0., kind=InterKind) for i in np.arange(len(self.ObsSpecs))]    
            #SpecMean = np.sum(np.array([weights[i]*Limit(waves, Ss1[i] - BAshifts[i](waves) - ScalingNeb[i]*CAshifts[i](waves), Poslim[i], PosLimCond) for i in np.arange(len(Ss1))]), axis=0) 
            SpecMean = np.sum(np.array([self.weights[i]*(Ss1[i] - BAshifts[i](waves) - NebFac*ScalingNeb[i]*CAshifts[i](waves)) for i in np.arange(len(Ss1))]), axis=0)  
            Anew = interp1d(waves, SpecMean, bounds_error=False, fill_value=0., kind=InterKind)(waves)  
            if self.StrictNegA:
                Anew = self.Limit(waves, Anew, self.PoslimallA, self.PosLimCondA)        
            if 'A' in locals():
                Epsnew = np.amax(np.sum(A - Anew))        
            else:
                Epsnew = 0.
            A = np.copy(Anew)     
            ABshifts = [interp1d(WavesAB[i], A,bounds_error=False, fill_value=0., kind=InterKind)
                            for i in np.arange(len(self.ObsSpecs))]     
            CBshifts = [interp1d(WavesCB[i], C,bounds_error=False, fill_value=0., kind=InterKind)
                            for i in np.arange(len(self.ObsSpecs))]             
            #SpecMean = np.sum(np.array([weights[i]*self.Limit(waves, Ss2[i] - ABshifts[i](waves) - ScalingNeb[i]*CBshifts[i](waves), Poslim[i], PosLimCond) for i in np.arange(len(Ss1))]), axis=0)    s
            SpecMean = np.sum(np.array([self.weights[i]*(Ss2[i] - ABshifts[i](waves) - NebFac*ScalingNeb[i]*CBshifts[i](waves)) for i in np.arange(len(Ss1))]), axis=0)            
            Bnew = interp1d(waves, SpecMean, bounds_error=False, fill_value=0., kind=InterKind)(waves) 
            #Bnew = interp1d(waves,SpecMean,PoslimallB, PosLimCond), bounds_error=False, fill_value=0.)(waves)  
            if self.StrictNegB:
                Bnew = self.Limit(waves, Bnew, self.PoslimallB, self.PosLimCondB)            
            Epsnew = max(Epsnew, np.sum(np.abs(B - Bnew)))
            B = Bnew    
            #B[B<-1] = -1          
            ACshifts = [interp1d(WavesAC[i], A,bounds_error=False, fill_value=0., kind=InterKind)
                            for i in np.arange(len(self.ObsSpecs))]     
            BCshifts = [interp1d(WavesBC[i], B,bounds_error=False, fill_value=0., kind=InterKind)
                            for i in np.arange(len(self.ObsSpecs))]             
            #SpecMean = np.sum(np.array([weights[i]*self.Limit(waves, Ss3[i] - ACshifts[i](waves) - BCshifts[i](waves), Poslim[i], PosLimCondC) for i in np.arange(len(Ss1))]), axis=0)   
            SpecMean = np.sum(np.array([self.weights[i]*self.Limit(waves, Ss3[i] - ACshifts[i](waves) - BCshifts[i](waves), self.Poslim[i], self.PosLimCondC) for i in np.arange(len(Ss1))]), axis=0)        
            # SpecMean = np.sum(np.array([weights[i]*(Ss3[i] - ACshifts[i](waves) - BCshifts[i](waves)) for i in np.arange(len(Ss1))]), axis=0)                  
            Cnew = interp1d(waves, SpecMean, bounds_error=False, fill_value=0., kind=InterKind)(waves)
            #if Triple:
                #Cnew = self.Limit(waves, Cnew, PoslimallC, PosLimCond)
            #Bnew = self.Limit(waves, Bnew, Poslimall, PosLimCond) + 1.
            Epsnew = max(Epsnew, np.sum(np.abs(C - Cnew)))
            C = Cnew   
            if self.NebOff == True:
                C*=0        
            #plt.plot(waves, A, label='A')
            #plt.plot(waves, B, label='B')
            #plt.plot(waves, C, label='C')
            #plt.legend()
            #plt.show()
            #dasds        
            ##plt.plot(waves, 
    # Enforce nebular solution to be strictly positive:
            if not self.Triple:
                C[C<self.Poslimall] = 0.
                #else:
                    #C[~CCondWaves]=0
            #if itr==450:
                    #plt.plot(waves, A, label=itr)
                    #plt.plot(waves, B, label=itr)
                    #plt.plot(waves, C, label=itr)
                    #plt.show()
            if ShowItr:
                print(itr, Epsnew)
            if self.PLOTCONV:
                plt.scatter(itr, Epsnew, color='blue')
            if self.PLOTITR:
                if itr%self.N_Itr_Plot==0:
                    plt.plot(waves, A, label=itr)
                    plt.plot(waves, B, label=itr)
                    plt.plot(waves, C, label=itr)
                    # plt.title('what is this?')
        print("Finished after ", itr, " iterations")
        if self.PLOTCONV or self.PLOTITR:
            plt.legend()
            plt.show()  
        #plt.plot(waves, A, label='A')
        #plt.plot(waves, B, label='B')
        #plt.plot(waves, C, label='C')
        #plt.legend()
        #plt.show()
        #dasds               
        return A+1., B+1., C+1., self.CalcDiffs(A, B, C, vrads1,  vrads2, waves, Resid=Resid, Reduce=Reduce, ShowItr=ShowItr, NebFac=NebFac)    
        

    # Assuming spectrum for secondary (Bini) and vrads1, gamma, and K1, explore K2s array for best-fitting K2
    # Ini = determines initial assumption for 0'th iteration
    # ShowItr = determines whether 
    def Grid_disentangling(self, Bini, vrads1, K1, K2s, waveRanges, Ini=None, ShowItr=False):
        global kcount, Nchi2
        Nchi2 = 0
        N = 0
        Diffs=K2s * 0.       
        for waves in waveRanges:
            kcount = 0
            print("Performing grid disentanglement... ")
            vrads2 = np.array([self.Gamma*(1 + K2s[k]/K1) - K2s[k]/K1*vrads1 for k in np.arange(len(K2s))])     
            if Ini=='A':
                print("Initial guess provided for component " + Ini)      
                Diffs += np.array([self.disentangle(Bini(waves), vrads2[k], vrads1, waves, ShowItr=ShowItr)[3] for k in np.arange(len(vrads2))])            
            elif Ini=='B':
                print("Initial guess provided for component " + Ini)        
                Diffs += np.array([self.disentangle(Bini(waves), vrads1, vrads2[k], waves, ShowItr=ShowItr)[3] for k in np.arange(len(vrads2))])            
            else:
                print("No initial approximation given, assuming flat spectrum for secondary...")
                Bini = interp1d(waves, np.ones(len(waves)),bounds_error=False, fill_value=1.)  
                Diffs += np.array([self.disentangle(Bini(waves), vrads2[k], vrads1, waves, ShowItr=ShowItr)[3] for k in np.arange(len(vrads2))])         
        Diffs /= (Nchi2 * len(self.ObsSpecs) - 1)
        print("Nchi2 = ", Nchi2)
        #plt.scatter(K2s, Diffs)
        fig, ax = plt.subplots(figsize=(8,6))
        chi2P, K2, K2err = self.Chi2con(Diffs, Nchi2 * len(self.ObsSpecs) - 1)
        Nchi2 = 0
        plt.plot([K2s[0], K2s[-1]], [chi2P, chi2P], color='red', label=r'1$\sigma$ contour')
        plt.legend()
        plt.xlabel(r'$K_2$ [km/s]')
        plt.ylabel(r'Normalised reduced $\chi^2$')    
        plt.savefig(self.working_path+self.Rangestr +  '_Grid_disentangling.pdf', bbox_inches='tight')
        # plt.show()
        plt.close()
        print("K2, K2 min error, K2 plus error:", K2, K2err)
        #print("K2err:", K2- K2minerr)    
        return K2
        #dasd
        
    # Assuming spectrum for secondary (Bini) and vrads1, gamma, and K1, explore K2s array for best-fitting K2
    # Ini = determines initial assumption for 0'th iteration
    # ShowItr = determines whether 
    def Grid_disentangling2D(self, Bini, K1s, K2s, waveRanges, Ini=None, ShowItr=False):
        global kcount, k1, k2, DoFs
        N = 0
        Diffs=np.zeros(len(K1s)*len(K2s)).reshape(len(K1s), len(K2s))  
        DoFs=0 
        for waves in waveRanges:       
            kcount = 0
            for k1, K1 in enumerate(K1s):
                for k2, K2 in enumerate(K2s):
                    vrads1, vrads2 = self.v1andv2(K1, K2)  
                    if Ini=='A':
                        print("Initial guess provided for component " + Ini)      
                        #print Bini(waves)
                        Diffs[k1,k2] += self.disentangle(Bini(waves), vrads2, vrads1, waves)[3]      
                    elif Ini=='B':
                        print("Initial guess provided for component " + Ini)        
                        Diffs[k1,k2] += self.disentangle(Bini(waves), vrads1, vrads2, waves)[3]              
                    else:
                        print("No initial approximation given, assuming flat spectrum for secondary...")
                        Bini = interp1d(waves, np.ones(len(waves)),bounds_error=False, fill_value=1.)  
                        Diffs[k1,k2] += self.disentangle(Bini(waves), vrads2, vrads1, waves)[3]
        Diffs /= (DoFs)  
        np.savetxt(self.working_path+self.Rangestr + '_' + 'grid_dis_K1K2.txt', np.array(Diffs), header='#K1min, K2min, stepK1, stepK2 = ' + str(K1s[0]) + ', ' + str(K2s[0]) + ', ' + str(K1s[1] - K1s[0]) + ', ' + str(K2s[1]-K2s[0]))    
        k1min, k2min = np.argwhere(Diffs == np.min(Diffs))[0]
        #print Diffs
        print("True velocities: ", k1min, k2min, K1s[k1min], K2s[k2min])
    # Start with uncertainty on K1:    
        #plt.scatter(K2s, Diffs[k1min,:])
        #plt.show()
        #dsss
        fig, ax = plt.subplots(figsize=(8,6))
        chi2P, K2, K2err = self.Chi2con(Diffs[k1min,:], DoFs, comp='secondary')
        plt.plot([K2s[0], K2s[-1]], [chi2P, chi2P], color='red', label=r'1$\sigma$ contour')
        plt.legend()
        plt.xlabel(r'$K_2$ [km/s]')
        plt.ylabel(r'Normalised reduced $\chi^2$')    
        np.savetxt(self.working_path+self.Rangestr + '_' + '_grid_dis_K2.txt', np.c_[K2s, Diffs[k1min,:]], header='#1sigma = ' + str(chi2P))
        plt.savefig(self.working_path+self.Rangestr +  '_Grid_disentangling_K2.pdf', bbox_inches='tight')
        # plt.savefig(working_path+self.Rangestr +  '_Grid_disentangling_K2.png', bbox_inches='tight')
        # plt.show()
        plt.close()
        print("K2, K2 min error:", K2, K2err) 
    # continue with uncertainty on K2:    
        #plt.scatter(K1s, Diffs[:,k2min])
        fig, ax = plt.subplots(figsize=(8,6))
        chi2P, K1, K1err = self.Chi2con(Diffs[:,k2min], DoFs, comp='primary')
        plt.plot([K1s[0], K1s[-1]], [chi2P, chi2P], color='red', label=r'1$\sigma$ contour')
        plt.legend()
        plt.xlabel(r'$K_1$ [km/s]')
        plt.ylabel(r'Normalised reduced $\chi^2$')    
        np.savetxt(self.working_path+self.Rangestr + '_grid_dis_K1.txt', np.c_[K1s, Diffs[:,k2min]], header='#1sigma = ' + str(chi2P))
        plt.savefig(self.working_path+self.Rangestr +  '_Grid_disentangling_K1.pdf', bbox_inches='tight')
        # plt.savefig(working_path+self.Rangestr +  '_Grid_disentangling_K1.png', bbox_inches='tight')
        # plt.show()
        plt.close()
        print("K1, K1 min error:", K1, K1err) 
        with open(self.working_path+'K_velocities.dat', 'a+') as file:
            file.write(str(self.Rangestr)+' '+str(K1)+' '+str(K1err)+' '+str(K2)+' '+str(K2err))
            file.write('\n')
        return K1, K2

    # Calculate difference 
    def CalcDiffs(self, A, B, C, vrA, vrB, waves, Resid=False, Reduce=False, ShowItr=False, NebFac=1., linewidExt=3, legsize=13, locleg='lower left', alphaleg = 0.):
        global kcount, Nchi2, k1, k2, DoFs, K1now, K2now
        ScalingNeb = np.ones(len(waves))
        if Resid:
            Residuals = []    
        if 'DoFs' not in globals():
            DoFs=1
        WaveCalcCond = self.Reduce_Waves(waves)
        #WaveCalcCond = waves > 0  
        Sum = 0     
        if self.kcount<=1 or Nchi2==0 or 'Nchi2' not in locals():
            if 'Nchi2' not in locals():
                Nchi2 = 0
            if Nchi2 ==0 or 'Nchi2' not in locals():
                Nchi2= len(waves[WaveCalcCond])
            else:
                Nchi2+= len(waves[WaveCalcCond])
        if self.PLOTEXTREMES:
            plotminyarr=[]
            plotmaxyarr=[]
            for ind in np.arange(len(self.ObsSpecs)):
                plotminyarr.append(np.amin(interp1d(self.ObsSpecs[ind][:,0], self.ObsSpecs[ind][:,1]-1,bounds_error=False, fill_value=0.)(waves[WaveCalcCond])))
                plotmaxyarr.append(np.amax(interp1d(self.ObsSpecs[ind][:,0], self.ObsSpecs[ind][:,1]-1,bounds_error=False, fill_value=0.)(waves[WaveCalcCond])))
            pltExtyMin = min(plotminyarr)*0.9
            pltExtyMax = max(plotmaxyarr)*1.1
            # print('y-axis min and max: ', pltExtyMin, pltExtyMax)
        for ind in np.arange(len(self.ObsSpecs)):
            # print('ind =', ind)
            vA = vrA[ind]/self.clight
            vB = vrB[ind]/self.clight
            vC = self.vrads3[ind]/self.clight        
            Facshift1 = np.sqrt( (1 + vA) / (1 - vA))
            Facshift2 = np.sqrt( (1 + vB) / (1 - vB))
            Facshift3 = np.sqrt( (1 + vC) / (1 - vC))         
            Ashift = interp1d(waves*Facshift1, A,bounds_error=False, fill_value=0.)(waves[WaveCalcCond])   
            Bshift = interp1d(waves*Facshift2, B,bounds_error=False, fill_value=0.)(waves[WaveCalcCond]) 
            Cshift = NebFac*ScalingNeb[ind]*interp1d(waves*Facshift3, C,bounds_error=False, fill_value=0.)(waves[WaveCalcCond])  
            ObsSpec = interp1d(self.ObsSpecs[ind][:,0], self.ObsSpecs[ind][:,1]-1,bounds_error=False, fill_value=0.)(waves[WaveCalcCond])   
            specsum = Ashift + Bshift + Cshift
            # print('   ObsSpec =', ObsSpec)
            sigma =  (np.std(ObsSpec[:8]) +  np.std(ObsSpec[-8:]))/2.
            # print('   sigma, 1/sigma, np.sum( (ObsSpec - specsum)**2)', sigma, 1/sigma, np.sum( (ObsSpec - specsum)**2))
            # sys.exit()
            if Resid:
                Residuals.append(ObsSpec - specsum)        
            Sum +=np.sum( (ObsSpec - specsum)**2/sigma**2)            
            if self.PLOTFITS:
            # User needs to change kcount==1 condition if a specific K2 is desired for plotting.             
                if kcount==self.kcount_usr:      
                    fig, ax = plt.subplots(figsize=(8,6))
                    plt.plot(waves[WaveCalcCond], 1+specsum)
                    plt.plot(waves[WaveCalcCond], 1+Ashift)
                    plt.plot(waves[WaveCalcCond], 1+Bshift)   
                    plt.plot(waves[WaveCalcCond], 1+Cshift)                  
                    plt.plot(waves[WaveCalcCond], 1+ObsSpec, label=self.specnamesFin[ind].split('/')[-1]  + r', $\varphi=$' + str(round(self.phis[ind], 2)))
                    plt.title('PLOTFITS')
                    plt.legend()
                    plt.savefig(self.working_path+'PLOTFITS.pdf', bbox_inches='tight')                
                    plt.show()   
            if self.PLOTEXTREMES:
                if self.kcount==self.kcount_extremeplot:
                    if ind==min(self.RVExtminInd, self.RVExtmaxInd):
                        rv=250.75
                        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,7), sharex=True, sharey=True)
                        axes[0].plot(waves[WaveCalcCond], 1+Ashift, label='Prim Dis.', color='red', linestyle = 'dotted', linewidth=linewidExt)
                        axes[0].plot(waves[WaveCalcCond], 1+Bshift, label='Sec. Dis.', color='green')   
                        axes[0].plot(waves[WaveCalcCond], 1+Cshift, label='Nebular', color='purple') 
                        axes[0].plot(waves[WaveCalcCond], 1+ObsSpec, color='blue', label=str(round(self.MJDs[ind],0)) + r', $\varphi=$' + str(round(self.phis[ind], 2)))
                        axes[0].plot(waves[WaveCalcCond], 1+specsum, label='Sum Dis.', color='black', linestyle = '--', linewidth=linewidExt)
                        # if self.Rangestr == '3970':  
                        #     axes[0].axvline(self.rv_shift(3970.075, rv), color='red', linestyle='--')
                        #     axes[0].axvline(self.rv_shift(3970.075, rv)-1.5, color='grey', linestyle='--')
                        #     axes[0].axvline(self.rv_shift(3970.075, rv)+1.5, color='grey', linestyle='--')                        
                        # if self.Rangestr == '4102':  
                        #     axes[0].axvline(4104.3, color='grey', linestyle='--')
                        #     axes[0].axvline(4106.5, color='grey', linestyle='--')
                        # if self.Rangestr == '4340':
                        #     axes[0].axvline(self.rv_shift(4340.472, rv), color='red', linestyle='--')
                        #     axes[0].axvline(4343, color='grey', linestyle='--')
                        #     axes[0].axvline(4345.0, color='grey', linestyle='--')
                        # if self.Rangestr == '4861':
                        #     axes[0].axvline(self.rv_shift(4861.297, rv), color='red', linestyle='--')
                        #     axes[0].axvline(4864.9, color='grey', linestyle='--')
                        #     axes[0].axvline(4865.8, color='grey', linestyle='--')
                        axes[0].set_title(self.specnamesFin[ind].split('/')[-1]+' - line $\lambda$'+self.Rangestr)
                        axes[0].legend(prop={'size': legsize}, loc=locleg, framealpha=alphaleg)
                        axes[0].set_ylabel('Normalised flux')
                        # axes[0].set_xlabel(r'Wavelength $[\AA]$')     
                        # axes[0].set_ylim(pltExtyMin, pltExtyMax  )
                        DiffMajor = int((waves[WaveCalcCond][-1] - waves[WaveCalcCond][0])/3)
                        # axes[0].xaxis.set_major_locator(MultipleLocator(DiffMajor))
                        # axes[0].xaxis.set_minor_locator(MultipleLocator(DiffMajor/5.))
                        axes[0].yaxis.set_minor_locator(MultipleLocator(.05))
                    elif ind==max(self.RVExtminInd, self.RVExtmaxInd):
                        axes[1].plot(waves[WaveCalcCond], 1+Ashift, color='red', linestyle = 'dotted', linewidth=linewidExt)
                        axes[1].plot(waves[WaveCalcCond], 1+Bshift, color='green')   
                        axes[1].plot(waves[WaveCalcCond], 1+Cshift,  color='purple')
                        axes[1].plot(waves[WaveCalcCond], 1+ObsSpec, color='blue', label=str(round(self.MJDs[ind],0)) + r', $\varphi=$' + str(round(self.phis[ind], 2)))                    
                        axes[1].plot(waves[WaveCalcCond], 1+specsum, color='black', linestyle = '--', linewidth=linewidExt)
                        # if self.Rangestr == '3970':
                        #     axes[1].axvline(self.rv_shift(3970.075, rv), color='red', linestyle='--')
                        #     axes[1].axvline(self.rv_shift(3970.075, rv)-1.5, color='grey', linestyle='--')
                        #     axes[1].axvline(self.rv_shift(3970.075, rv)+1.5, color='grey', linestyle='--')  
                        # if self.Rangestr == '4102':  
                        #     axes[1].axvline(4104.3, color='grey', linestyle='--')
                        #     axes[1].axvline(4106.5, color='grey', linestyle='--')
                        # if self.Rangestr == '4340':
                        #     axes[1].axvline(self.rv_shift(4340.472, rv), color='red', linestyle='--')
                        #     axes[1].axvline(4343, color='grey', linestyle='--')
                        #     axes[1].axvline(4345.0, color='grey', linestyle='--')
                        # if self.Rangestr == '4861':
                        #     axes[1].axvline(self.rv_shift(4861.297, rv), color='red', linestyle='--')
                        #     axes[1].axvline(4864.8, color='grey', linestyle='--')
                        #     axes[1].axvline(4865.9, color='grey', linestyle='--')
                        #     axes[1].axvline(self.rv_shift(4861.297, rv)-0.6, color='yellow', linestyle='--')
                        #     axes[1].axvline(self.rv_shift(4861.297, rv)+1.7, color='yellow', linestyle='--')
                        axes[1].set_title(self.specnamesFin[ind].split('/')[-1]+' - line $\lambda$'+self.Rangestr) 
                        axes[1].legend(prop={'size': legsize}, loc=locleg, framealpha=alphaleg)
                        axes[1].set_ylabel('Normalised flux')
                        axes[1].set_xlabel(r'Wavelength $[\AA]$')                      
                        
                        # axes[1].set_ylim(pltExtyMin, pltExtyMax)
                        # axes[1].tick_params(labelleft=False)                   
                        # axes[1].tick_params(axis='x', which='minor', bottom=True)
                        DiffMajor = int((waves[WaveCalcCond][-1] - waves[WaveCalcCond][0])/3)
                        # axes[1].xaxis.set_major_locator(MultipleLocator(DiffMajor))
                        # axes[1].xaxis.set_minor_locator(MultipleLocator(DiffMajor/5.))   
                        # axes[1].yaxis.set_minor_locator(MultipleLocator(.05))                       
                        # plt.tight_layout()    
                        #plt.subplots_adjust(wspace=0, hspace=1)
                        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)                    
                        try:
                            print(self.StarName, self.Rangestr)
                            NameFile =  self.StarName + '_' + self.Rangestr + '_Extremes_' + str(np.round(K1now)) + '_' + str(np.round(K2now)) + '.pdf'   
                            plt.savefig(self.working_path+NameFile, bbox_inches='tight')                
                            # plt.show()       
                            plt.close()                 
                        except:
                            print('exception found')
                            NameFile =  self.StarName + '_' + self.Rangestr + '_Extremes_' + str(np.round(self.K1)) + '_' + str(np.round(self.K2s[kcount])) + '.pdf'   
                            plt.savefig(self.working_path+NameFile, bbox_inches='tight')              
                            # plt.show()
                            plt.close()      

        print("kcount:", self.kcount)   
        if self.kcount==0:
            try:
                DoFs += len(waves[WaveCalcCond]) * (len(self.ObsSpecs)-2)
            except:
                pass
        self.kcount+=1
        if not ShowItr:
            if self.oneD:
                try:
                    print("K2=" + str(self.K2s[self.kcount-1]) + ": red. chi2=", Sum/ (len(waves[WaveCalcCond])* len(self.ObsSpecs) - 1))
                except:
                    pass
            else:
                print("chi2:",  Sum/ (len(waves[WaveCalcCond])* len(self.ObsSpecs) - 2), '\n')
        if Resid:
            return Residuals
        if Reduce:
            return Sum/ (len(waves[WaveCalcCond]) * len(self.ObsSpecs) - 1)
        else:
            return Sum

    # Shrinks the wavelength domain on which obs-mod is calculated to avoid edge issues
    def Reduce_Waves(self, waves):
        # print('Reduce_Waves is using these K1 and K2s[-1] :', K1, K2s[-1])
        vrA, vrB = self.v1andv2(self.K1, self.K2s[-1])
        Inds = np.where(np.diff(waves) > 1.)
        WaveCalcCond = waves < 0
        if len(Inds[0]) == 0:
            LamMin = waves[0]*(1. +  max(max(vrA), max(vrB))/self.clight)
            LamMax = waves[-1]*(1. +  min(min(vrA), min(vrB))/self.clight)
            WaveCalcCond = (waves > LamMin) * (waves < LamMax)
        else:
            Inds = np.append(0, Inds[0])
            Inds = np.append(Inds, len(waves)-1)
            for j in np.arange(len(Inds)-1):
                LamMin = waves[Inds[j]+1]*(1. +  max(max(vrA), max(vrB))/self.clight)
                LamMax = waves[Inds[j+1]]*(1. +  min(min(vrA), min(vrB))/self.clight)
                WaveCalcCond = WaveCalcCond + (waves > LamMin) * (waves < LamMax)
            
    #Reduce nebular lines regions:
        if self.rmNebChi2 == True:
            for wrange in self.PosLimCondC:
                if 'NebCond' in locals():
                    NebCond += (waves > wrange[0]*(1+self.Gamma/self.clight) ) * (waves < wrange[1]*(1+self.Gamma/self.clight)  )
                else:
                    NebCond = (waves > wrange[0]*(1+self.Gamma/self.clight)) * (waves < wrange[1]*(1+self.Gamma/self.clight))
            WaveCalcCond *= ~NebCond
        return WaveCalcCond

    #Calculate K2 corresponding to chi2 minimum by fitting parabola to minimum region + confidence interval (default: 1sig=68%)    
    def Chi2con(self, redchi2, nu, P1=0.68, comp='secondary', fitrange_fraction = 0.2):    
        if comp == 'secondary':
            Kscomp = np.copy(self.K2s)
        else:
            Kscomp = np.copy(self.K1s)
        # Calculate fitrange as a fraction of the number of K2 values
        fitrange = int(fitrange_fraction * len(Kscomp))
        #First fit parabola to chi2 distribution to find minimum:
        indmin = np.argmin(redchi2)    
        i1 = max(0, indmin-fitrange)
        i2 = min(len(Kscomp)-1, indmin+fitrange)
        a,b,c = np.polyfit(Kscomp[i1:i2], redchi2[i1:i2], 2)        
        while a<0:
            i2 -=1
            a,b,c = np.polyfit(Kscomp[i1:i2], redchi2[i1:i2], 2) 
            if i1==i2:
                sys.exit()
        ParbMin = c - b**2/4./a
        # Now compute non-reduced, normalised chi2 distribution  
        chi2 = redchi2 * nu /  ParbMin
        # The probability distribution of (non-reduced) chi2 peaks at nu. Compute array around this value    
        xarr = np.arange(nu/10., nu*10, 1)
        # The cumulative probability distribution of chi^2 (i.e., Prob(chi2)<x) with nu DoFs is the regularised incomplete Gamma function Gamma(nu/2, x/2)
        # Hence, we look for the index and x value at which Prob(chi2) < P1
        ys1 = sc.gammainc(nu/2., xarr/2) - P1
        minarg1 = np.argmin(np.abs(ys1))
        # This is the chi2 value corresponding to P1 (typically 1-sigma)    
        chi2P1 = xarr[minarg1]/nu   
        print('\n chi2P1 =', chi2P1)
        # Now fit parabola to reduced chi2, after normalisation:
        a,b,c = np.polyfit(Kscomp[i1:i2], chi2[i1:i2]/nu, 2)        
        while a<0:
            i2 -=1
            a,b,c = np.polyfit(Kscomp[i1:i2], chi2[i1:i2]/nu, 2) 
            if i1==i2:
                sys.exit()
        chi2fine = np.arange(Kscomp[i1], Kscomp[i2], 0.01)
        parb = a*chi2fine**2 + b*chi2fine  + c
        plt.scatter(Kscomp, chi2/nu, color='teal', alpha=1)
        plt.plot(chi2fine, parb, color='darkorange', lw=2)    
        K2min = -b/2./a
        K2err = K2min - (-b - np.sqrt(b**2 - 4*a*(c-chi2P1))) / 2./a
        return chi2P1, K2min, K2err

    # Solves the Kepler equation
    def Kepler(self, E, M, ecc):
        E2 = (M - ecc*(E*np.cos(E) - np.sin(E))) / (1. - ecc*np.cos(E))
        eps = np.abs(E2 - E) 
        if np.all(eps < 1E-10):
                return E2
        else:
                return self.Kepler(E2, M, ecc)

    def rv_shift(self, lambda_rest, rv):
        self.lambda_rest = lambda_rest
        self.rv = rv
        c_kms = c.to('km/s').value
        # self.c_kms = c_kms
        # return lambda_rest*np.sqrt( (1-(rv/c_kms)) / (1+(rv/c_kms)))
        return lambda_rest* (1+(rv/c_kms))

    # Given true anomaly nu and parameters, 
    def v1andv2(self, K1, K2):

        self.phis[self.phis<0] += 1
        Ms = 2 * np.pi * self.phis
        Es =  self.Kepler(1., Ms, self.ecc)
        eccfac = np.sqrt((1 + self.ecc) / (1 - self.ecc))
        nu = 2. * np.arctan(eccfac * np.tan(0.5 * Es))

        Omegapi = self.Omega/180. * np.pi      
        v1 = self.Gamma + K1*(np.cos(Omegapi + nu) + self.ecc* np.cos(Omegapi))   
        v2 = self.Gamma - K2*(np.cos(Omegapi + nu) + self.ecc* np.cos(Omegapi))         
        #return np.column_stack((v1,v2))
        return v1, v2

    #Ensures that arr values where arr > lim  are set to 0 in domains specified in Poslim array
    def Limit(self, waves, arr, lim, Poslim):
        for i, Range in enumerate(Poslim):
            if 'PosCond' not in locals():
                PosCond = (Poslim[i][0] < waves)  * (waves < Poslim[i][1])
            else:
                PosCond += (Poslim[i][0] < waves)  * (waves < Poslim[i][1])
        NegCond = np.logical_not(PosCond)    
        arr[(arr > lim) * NegCond] = 0.
        return arr

    def Normalise(self, spec, points):
        ContIndices = np.array([np.argmin(np.abs(spec[:,0] - points[i])) for i in np.arange(len(points))])  
        Contfluxes = np.array([np.average(spec[:,1][max(ContIndices[i] - 7, 0): min(ContIndices[i] + 7, len(spec[:,0]))]) for i in np.arange(len(ContIndices))])
        #print spec[:,0][ContIndices]
        #print Contfluxes
        #dasd
        ContSpline = interp1d(spec[:,0][ContIndices], Contfluxes, bounds_error = False, fill_value = 'extrapolate')(spec[:,0])
        return np.array([spec[:,0], spec[:,1] / ContSpline]).T 


        print(("Data written to %s" % outfilename))

    def Cosclean(self, Spec, thold=6, cosize=10, ForbiddenRanges = [[3970, 3975], [4020, 4030], [4103., 4108.], [4342, 4347], [4365, 4369],  [4391, 4393], [4470, 4477.]]):
        for itr in range(10):
            waves = np.copy(Spec[:,0])
            fluxes = np.copy(Spec[:,1])
            for wrange in ForbiddenRanges:
                if 'WaveCond' not in locals():
                    WaveCond = (waves > wrange[0]) * (waves < wrange[1])
                else:
                    WaveCond += (waves > wrange[0]) * (waves < wrange[1])
            fluxes[WaveCond] = 1.        
            fluxdiff = np.append(0, np.diff(fluxes))
            sigma =  thold*np.mean(np.absolute(fluxdiff))
    #Find points whose gradients are larger than thold*average
            gradient_condition = (np.absolute(fluxdiff) > sigma)     
    #Weak point are of no interest
            flux_condition = fluxes > 1.0
            posgrad = fluxdiff>0
            flagged_args = np.where(gradient_condition & flux_condition)[0]
            if (not len(flagged_args)):
                print("There are no cosmics detected with the given threshhold and size")
                return Spec
            blimit = 0
            N = len(waves)
            for i in flagged_args:
                if waves[i] < blimit or i<cosize or i>N-cosize:
                    continue
                cosmic = fluxes[i-cosize:i+cosize+1]    
                if posgrad[i]:
                    if np.any(cosmic[:cosize] - fluxes[i] > 0):
                        continue
                else:
                    if np.any(cosmic[cosize+1:] - fluxes[i] > 0):
                        continue    
                ipeak = i - cosize + np.argmax(fluxes[i-cosize:i+cosize+1])
                fpeak = fluxes[ipeak]
                cosmic = fluxes[ipeak - cosize : ipeak + cosize + 1]
                cosb = cosmic[:cosize + 1]
                cosr = cosmic[cosize:]
                fmeadb = np.mean(cosb)
                fmeadr = np.mean(cosr)
                cosbdiff = np.append(np.diff(cosb), 0)
                cosrdiff = np.append(0, np.diff(cosr))
                sigmab = np.mean(np.absolute(cosbdiff))
                sigmar = np.mean(np.absolute(cosrdiff))       
                condsmallb = cosb - fmeadb < 0.1*(fpeak - fmeadb)
                condsmallr = cosr - fmeadr < 0.1*(fpeak - fmeadr)
                argb = np.where((np.roll(cosbdiff,-1) > sigmab) & (condsmallb) & (cosb > 0.5))[0]
                argr = np.where((cosrdiff < -sigmar) & (condsmallr) & (cosr > 0.5))[0]
                if len(argb) == 0  or len(argr) == 0:
                    continue
                argb = ipeak - cosize + argb[-1]
                argr = ipeak + argr[0]
                if abs(fluxes[argb] - fpeak) < sigmab or abs(fluxes[argr] - fpeak) < sigmar:
                    continue
                Spec[argb:argr+1,1] = np.interp(waves[argb:argr+1], [waves[argb], waves[argr]], [fluxes[argb], fluxes[argr]]) 
                blimit = waves[argr]
        return Spec

    def read_FEROS(self, infile):
        print(("%s: Input file is a FEROS file." % infile))
        header = fits.getheader(infile)
        flux = fits.getdata(infile)
        crval = header['CRVAL1']
        crpix = header['CRPIX1']
        cdelt = header['CDELT1']

        wave = crval - (cdelt * crpix - cdelt) + np.arange(flux.shape[0]) * cdelt
        return np.array([wave, flux]).T

    def read_GIRAFFE(self, infile):
        # print(("%s: Input file is a GIRAFFE file." % infile))
        header = fits.getheader(infile)
        data = fits.getdata(infile)
        wl0 = header['CRVAL1']  # Starting wl at CRPIX1
        delt = header['CDELT1']  # Stepwidth of wl
        pix = header['CRPIX1']  # Reference Pixel
        wave = wl0 - (delt * pix - delt) + np.arange(data.shape[0]) * delt
        table = Table.read(infile, hdu=1)
        flux = table['NORM_SKY_SUB_CR']
        return wave, flux

    def read_file(self, infile):
        ext = str(infile.split('.')[-1])

        # Check type of input file (fits or ascii) to read in data correctly
        if (ext == 'fits') or (ext ==  'fit'):
            wave, flux = self.read_fits(infile)

        elif (ext == 'gz'):
            wave, flux = self.read_tlusty(infile)

        elif (ext == 'dat' or ext == 'ascii' or ext == 'txt' or ext == 'nspec'):
            wave, flux = self.read_xytable(infile)

        elif (ext == 'tfits'):
            wave, flux = self.read_uvespop(infile)

        elif (ext == 'hfits'):
            wave, flux = self.read_hermes_normalized(infile)

        else:
            wave, flux = self.read_xytable(infile)

        return np.array([wave, flux]).T

    def read_fits(self, infile):
        # print(("%s: Input file is a fits file." % infile))

        header = fits.getheader(infile)

        if 'HIERARCH SPECTRUM EXTRACTION' in header:
            wave, flux = self.read_psfSpec(infile)

        elif 'INSTRUME' in header:
            ins = header['INSTRUME']
            if (ins == 'MUSE'):
                wave, flux = self.read_pampelMUSE(infile)

            elif (ins == 'HERMES'):
                wave, flux = self.read_HERMES(infile)

            elif (ins == 'FEROS'):
                wave, flux = self.read_FEROS(infile)
            elif (ins == 'XSHOOTER'):
                wave, flux = self.read_XSHOOTER(infile)

            elif (ins == 'UVES'):
                wave, flux = self.read_UVES(infile)
            elif (ins == 'GIRAFFE' and 'nLR' in infile):
                wave, flux = self.read_GIRAFFE(infile)     
            elif (ins == 'GIRAFFE'):
                wave, flux = self.read_GIRAFFE2(infile)               
            elif (ins == 'ESPCOUDE'):
                wave, flux = self.read_NLA(infile)   
            elif (ins == 'COS'):
                wave, flux = self.read_COS(infile)               
            elif (ins == 'STIS'):
                wave, flux = self.read_STIS(infile)              
            else:
                print('File type unkown, trying HERMES')
                wave, flux = self.read_HERMES(infile)                        
        else:
            wave, flux = self.read_HERMES(infile)
        return wave, flux

    def read_xytable(self, infile):
        # print(("%s: Input file is an xytable file." % infile))    
        spec = pd.read_csv(infile, sep="\s+", header=None)
        wave = spec[0]
        flux = spec[1]    
        return wave, flux