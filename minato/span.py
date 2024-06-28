import time
import sys
import csv
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from glob import glob
from datetime import timedelta, date, datetime
from math import prod
# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
current_date = str(date.today())

class atmfit:
    def __init__(self, spectrumA, spectrumB, grid=None, lrat0=None, modelsA_path=None, modelsB_path=None, binary=False, crop_nebular=False, He2H=False, He_ini=0.1):
        """
        Initialize the atmosphere fitting class.

        This constructor initializes the `atmfit` class with specified parameters.
        It sets up the initial configuration for the fitting process.

        :param grid: List of parameter grids for the fitting process.
                     Format: [light_ratio_grid, He_H_ratio_grid, Teff_A_grid, logg_A_grid,
                              rot_A_grid, Teff_B_grid, logg_B_grid, rot_B_grid]
                     Type: list of lists (floats)
        :param spectrumA: Path to the spectrum file for star A.
                         Type: str
        :param spectrumB: Path to the spectrum file for star B.
                         Type: str
        :param lrat0: Initial light ratio. If not provided, defaults to None.
                     Type: float or None
        :param modelsA_path: Path to the folder containing models for star A.
                     Type: str
        :param modelsB_path: Path to the folder containing models for star B.
                        Type: str
        :param binary: Flag to indicate if the system is a binary or single star.
                        Default: False
                        Type: bool
        :param He2H: Flag to indicate if the He/H ratio should be modified.
                        Default: False
                        Type: bool
        :param crop_nebular: Flag to indicate if nebular emission should be cropped from the spectrum.
                        Default: False
                        Type: bool
        :param He_ini: Initial He/H ratio to be used when modifying the He/H ratio.
                        Default: 0.1
                        Type: float
        """
        self.grid = grid
        
        self.spectrumA = spectrumA
        self.spectrumB = spectrumB
        self.lrat0 = lrat0
        self.modelsA_path = modelsA_path
        self.modelsB_path = modelsB_path
        self.binary = binary
        self.He2H = He2H
        self.crop_nebular = crop_nebular
        self.He_ini = He_ini
        self.missing_models = False
        self.warning_printed = False
        
    lines_dic = {
                    3995: { 'region':[3990, 4000],  'HeH_region':[], 'title':'N II $\lambda$3995'},
                    4026: { 'region':[4005, 4033],  'HeH_region':[4005, 4033], 'title':'He I $\lambda$4009/26'},
                    4102: { 'region':[4084-20, 4117],  'HeH_region':[4091, 4111], 'title':'H$\delta$'},
                    4121: { 'region':[4117, 4135],  'HeH_region':[4120, 4122], 'title':'He I $\lambda$4121, Si II $\lambda$4128/32'},
                    4144: { 'region':[4137, 4151],  'HeH_region':[4142, 4146], 'title':'He I $\lambda$4144'},
                    4233: { 'region':[4225, 4241],  'HeH_region':[], 'title':'Fe II $\lambda$4233'},
                    4267: { 'region':[4260, 4275],  'HeH_region':[], 'title':'C II $\lambda$4267'},
                    4340: { 'region':[4320, 4362],  'HeH_region':[4330, 4350], 'title':'H$\gamma$'},
                    4388: { 'region':[4380, 4396],  'HeH_region':[4386.5, 4389.5], 'title':'He I $\lambda$4388'},
                    4471: { 'region':[4465, 4485],  'HeH_region':[4471, 4473], 'title':'He I $\lambda$4471, Mg II $\lambda$4481'},
                    # 4553: { 'region':[4536, 4560],  'HeH_region':[], 'title':'Fe II $\lambda$4550/56, Si III $\lambda$4553'} }
                    4553: { 'region':[4536, 4560],  'HeH_region':[], 'title':'He II $\lambda$4542, Si III $\lambda$4553'} }

    def user_dic(self, lines):
        """
        Create a dictionary containing user-defined spectral lines and their regions.

        This function takes a list of spectral line identifiers and returns a dictionary
        containing information about the spectral lines and their associated regions. The
        dictionary is created based on the provided lines and information stored in self.lines_dic.

        :param lines: List of spectral line identifiers.
                    Type: list of integers

        :return: Dictionary containing spectral lines and their regions.
                Type: dict
        """
        self.lines = lines
        dic = { line: self.lines_dic[line] for line in self.lines }
        return dic

    def compute_single_set(self, params):
        # interpolate models to the wavelength of the sliced disentangled spectrum
        # modA_f_interp = np.interp(dst_A_w_slc, modA_w, modA_f)
        # models should already be interpolated to the wavelength of the disentangled spectrum

        # Get parameters from the grid and make them accessible as attributes of self, e.g. self.lr, self.TA, self.gA, etc.
        for key, value in zip(self.grid.keys(), params):
            setattr(self, key, value)

        # Get models
        if self.binary:
            modelA_params = {key.replace('A', ''): getattr(self, key) for key in self.grid.keys() if 'A' in key}
            modelB_params = {key.replace('B', ''): getattr(self, key) for key in self.grid.keys() if 'B' in key}
        else:
            modelA_params = {key: getattr(self, key) for key in self.grid.keys()}
            modelB_params = {}

        try:
            modA_w, modA_f, modelA = self.get_model(modelA_params, models_path=self.modelsA_path)
            if self.binary:
                modB_w, modB_f, modelB = self.get_model(modelB_params, models_path=self.modelsB_path) 
            else:
                modB_w, modB_f, modelB = None, None, None
        except Exception as e:
            # print('model not found: ', modelA_params)
            # print(f'Exception in get_model(): {type(e).__name__}: {e}')
            # print('there was a typerror 0', [TA,gA,rA, micA])
            pass   
        
        else:
            # print('MODEL FOUND (2): ', modelA, modA_w, modA_f)
            # Rescale flux of the disentangled specrta to new light ratio 
            fluA, fluB = self.rescale_flux(self.lr)

            # slice data to regions for chi^2 computation
            dst_A_w_slc, dst_A_f_slc = self.slicedata(self.wavA, fluA, self.user_dicA)
            dst_B_w_slc, dst_B_f_slc = self.slicedata(self.wavB, fluB, self.user_dicB) if self.binary else (None, None)
            mod_A_w_slc, mod_A_f_slc = self.slicedata(modA_w, modA_f, self.user_dicA)
            mod_B_w_slc, mod_B_f_slc = self.slicedata(modB_w, modB_f, self.user_dicB) if self.binary else (None, None)

            # apply He/H ratio to the sliced model of star A
            if self.He2H:
                mod_A_f_slc = self.He2H_ratio(mod_A_w_slc, mod_A_f_slc, self.He_ini, self.He, self.user_dicA, join=True, plot=False, model=modelA.replace(self.modelsA_path, ''))

            # crop nebular emission from disentangled spectrum and model of star B
            if self.crop_nebular:
                dst_B_w_slc, dst_B_f_slc = self.crop_data(dst_B_w_slc, dst_B_f_slc, [[4100, 4104], [4338, 4346]])
                mod_B_w_slc, mod_B_f_slc = self.crop_data(mod_B_w_slc, mod_B_f_slc, [[4100, 4104], [4338, 4346]])

            # compute the chi2 values
            # if modA_f.size and modB_f.size:
            ndataA = len(mod_A_f_slc)
            chi2A = self.chi2(dst_A_f_slc, mod_A_f_slc)
            chi2redA = chi2A/(ndataA-self.nparams)

            if self.binary:
                ndataB = len(mod_B_f_slc)    
                chi2B = self.chi2(dst_B_f_slc, mod_B_f_slc)
                chi2_tot = chi2A + chi2B
                ndata = ndataA + ndataB
                chi2redB = chi2B/(ndataB-self.nparams)
                chi2r_tot = chi2redA + chi2redB
            else:
                chi2_tot = chi2A
                ndata = ndataA
                chi2r_tot = chi2redA

            if chi2_tot < 0:
                raise ValueError("\nWarning: chi2 < O")

            # Create row by getting the values of the parameters from self
            row = [getattr(self, key) for key in self.cols if hasattr(self, key)]
            if self.binary:
                row.extend([chi2_tot, chi2A, chi2B, chi2r_tot, chi2redA, chi2redB, ndata])
            else:
                row.extend([chi2_tot, chi2r_tot, ndata])
            
            return row

    def compute_chi2(self, dic_lines_A, dic_lines_B):
        """
        Perform parameter grid search and compute chi-squared values for different model combinations.

        This function iterates over a grid of parameter values to compare synthetic spectra models to
        observations and compute the chi-squared values for the fits. The chi-squared values are calculated
        for both star A and star B using the provided dictionaries of spectral lines.

        :param dic_lines_A: Dictionary defining the spectral lines for star A with 'region' and 'HeH_region' information.
                            Type: dict
        :param dic_lines_B: Dictionary defining the spectral lines for star B with 'region' and 'HeH_region' information.
                            Type: dict

        :return: DataFrame containing the computed results including light ratio, temperatures, log surface gravities,
                rotational velocities, He/H ratios, chi-squared values, and related statistics.
                Type: pandas DataFrame
        """
        if self.grid is None:
            raise ValueError("Grid is required for compute_chi2")
        
        self.dic_lines_A = dic_lines_A
        self.dic_lines_B = dic_lines_B
        nparams = len(self.grid)
        self.nparams = nparams

        # compute length of the grid
        gridlen = prod([len(x) for x in self.grid])
        
        # retrieve wavelength from the disentangled spectra
        wavA, wavB = self.get_wave()
        self.wavA = wavA
        self.wavB = wavB
        # setting the dictionaries with the spectral lines selected for the fit
        usr_dicA = self.user_dic(dic_lines_A)
        usr_dicB = self.user_dic(dic_lines_B)
        self.user_dicA = usr_dicA
        self.user_dicB = usr_dicB

        # creating dictionary to store results
        result_dic = {key: [] for key in self.grid.keys()}
        if self.binary:
            result_dic.update({'chi2_tot': [], 'chi2A': [], 'chi2B': [], 'chi2r_tot': [], 'chi2redA': [], 'chi2redB': [], 'ndata': []})
        else:
            result_dic.update({'chi2_tot': [], 'chi2r_tot': [], 'ndata': []})
        # self.result_dic = result_dic
        cols = list(result_dic.keys())
        self.cols = cols
        t0 = time.time()

        # Get all possible combinations of parameters
        parameters = list(itertools.product(*self.grid.values()))
        # print('parameters:', parameters)
        
        # Compute chi2 values for each set of parameters
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.compute_single_set, parameters), total=len(parameters)))
        if self.missing_models:
            print('WARNING: Some models were not found')

        # Convert list of rows into a dictionary
        for row in results:
            if row is not None:
                for key, val in zip(cols, row):
                    result_dic[key].append(val)
        
        tf = time.time()
        print('Computation completed in: ' + str(timedelta(seconds=tf-t0)) + ' [s] \n')
        output = pd.DataFrame.from_dict(result_dic)
        # print('total number of points used in the fit:', ndata)
        return output

    def rescale_flux(self, lrat, lrat0=0.3):
        """
        Rescales the flux of two stars based on a desired light ratio.

        This function rescales the flux of two stars (A and B) to achieve a desired
        light ratio while considering an initial light ratio. The new flux values are
        calculated using the given light ratio and initial light ratio.

        :param lrat:  Desired light ratio for rescaling.
                      Type: float
        :param lrat0: Initial light ratio with which the input spectra have been scaled.
                      Default: 0.3
                      Type: float, optional

        :return: Rescaled flux values for star A and star B.
                 Type: tuple of floats
        """
        self.lrat = lrat
        if self.lrat0:
            ratio0 = self.lrat0
        else:
            ratio0 = lrat0

        # Print warning only if it hasn't been printed before
        # if not atmfit.warning_printed:    
        #     print('\n#     Warning: you are using an initial light ratio of', ratio0, '\n')
        #     atmfit.warning_printed = True  # Update the class variable
        ratio1 = lrat
        fluxA, fluxB = self.get_flux()
        flux_new_A = (fluxA -1)*((1-ratio0)/(1-ratio1)) + 1
        flux_new_B = (fluxB -1)*(ratio0/ratio1) + 1
        return flux_new_A, flux_new_B

    def slicedata(self, x_data, y_data, dictionary):
        """
        Slice data based on specified spectral lines and wavelength ranges.

        This function slices the provided x_data and y_data arrays based on the specified
        dictionary containing spectral lines and their corresponding wavelength ranges.

        :param x_data: Original x-axis data (wavelength).
                       Type: numpy array or list of floats
        :param y_data: Original y-axis data (flux/intensity).
                       Type: numpy array or list of floats
        :param dictionary: Dictionary containing spectral lines and their wavelength ranges.
                           The dictionary should be in the format:
                           {'line_name': {'region': (min_wavelength, max_wavelength)},
                            ...}
                           Type: dict

        :return: Sliced x-axis data and corresponding y-axis data after applying the specified
                 wavelength range conditions.
                 Type: tuple of numpy arrays
        """
        self.x_data = x_data
        self.y_data = y_data
        self.dictionary = dictionary
        x_data_sliced = []
        y_data_sliced = []
        for line in dictionary:
            reg = dictionary[line]['region']
            cond = (x_data > reg[0]) & (x_data < reg[1])
            x_data_sliced.extend(x_data[cond])
            y_data_sliced.extend(y_data[cond])
        return np.array(x_data_sliced), np.array(y_data_sliced)

    def get_model(self, pars, models_path=None, source=None):
        """
        Obtain a precomputed TLUSTY or ATLAS9 model based on temperature, logg, and rotational velocity.

        This function retrieves a model spectrum from either the TLUSTY or ATLAS9 model grids, based on the
        provided temperature (T), log surface gravity (logg), and rotational velocity (vrot) parameters.

        :param pars: Temperature (T), log surface gravity (logg), and rotational velocity (vrot) parameters.
                     Type: tuple of three floats
        :param source: Source of the models. Options are 'tlusty' and 'atlas'.
                       Default: 'tlusty'
                       Type: str

        :return: Wavelength array, flux array, and name of the retrieved model.
                 Type: tuple of numpy arrays (floats), str
        """
        if models_path:
            model = models_path 
            for key, value in pars.items():
                model += key + str(int(value))
            # print('   get_model: getting model: ', model)
            try:
                model_found = glob(model+'*')
                # print('   get_model: model_found: ', model_found)
                df = pd.read_csv(model_found[0], header=None, sep='\s+')
                return df[0].to_numpy(), df[1].to_numpy(), model
            except FileNotFoundError:
                # print('WARNING: No model named '+model+' was found')
                self.missing_models = True
                return None, None, None
        else:
            T, g, rot = pars
            lowT_models_path = '~/Science/github/jvillasr/MINATO/minato/models/ATLAS9/'             # Users will have to add the path to the models
            tlustyB_path =     '~/Science/github/jvillasr/MINATO/minato/models/TLUSTY/BLMC_v2/'
            tlustyO_path =     '~/Science/github/jvillasr/MINATO/minato/models/TLUSTY/OLMC_v10/'
        
            lowT_models_list = sorted(glob(lowT_models_path+'*fw05'))
            lowT_models_list = [x.replace(lowT_models_path, '') for x in lowT_models_list]

            tlustyB_list = sorted(glob(tlustyB_path+'*fw05'))
            tlustyB_list = [x.replace(tlustyB_path, '') for x in tlustyB_list]

            tlustyO_list = sorted(glob(tlustyO_path+'*fw05'))
            tlustyO_list = [x.replace(tlustyO_path, '') for x in tlustyO_list]
            tlustyOB_list = tlustyB_list + tlustyO_list

            if source=='tlusty':
                try:
                    if T>30:                        
                        model = 'T'+str(int(T*10))+'g'+str(int(g*10))+'v10r'+str(int(rot))+'fw05'
                        df = pd.read_csv(tlustyO_path+model,header=None, sep='\s+')
                    else:
                        model = 'T'+str(int(T))+'g'+str(int(g*10))+'v2r'+str(int(rot))+'fw05'
                        df = pd.read_csv(tlustyB_path+model,header=None, sep='\s+')
                    # return df[0].array, df[1].array, model
                    return df[0].to_numpy(), df[1].to_numpy(), model
                except FileNotFoundError:
                    # print('WARNING: No model named '+model+' was found')
                    # raise ValueError('   WARNING: No model available for '+model)
                    pass
            elif source=='atlas':
                model = 'T'+str(int(T))+'g'+str(int(g))+'v2r'+str(int(rot))+'fw05'
                try:
                    df = pd.read_csv(lowT_models_path+model,header=None, sep='\s+')
                    # return df[0].array, df[1].array, model  # pandas array are not accepted by slicedata
                    return df[0].to_numpy(), df[1].to_numpy(), model
                except FileNotFoundError:
                    # print('WARNING: No model named '+model+' was found')
                    # raise ValueError('   WARNING: No model available for '+model)
                    pass

    def He2H_ratio(self, wave, flux, ratio0, ratio1, dictionary, join=False, plot=False, model=None):
        """
        Modify the Helium-to-Hydrogen (He/H) ratio in a given spectrum and optionally plot the modifications.

        This function modifies the He/H ratio in a given spectrum based on the provided wavelength
        range and ratio values. The spectrum is modified for specific lines defined in the dictionary.
        If the plot parameter is set to True, a plot of the original and modified spectrum is generated
        for each line and saved to a file.

        :param wave: Wavelength array of the spectrum.
                    Type: numpy array or list of floats
        :param flux: Flux array of the spectrum.
                    Type: numpy array or list of floats
        :param ratio0: Initial He/H ratio.
                    Type: float
        :param ratio1: Desired He/H ratio after modification.
                    Type: float
        :param dictionary: Dictionary containing line information with 'region' and 'HeH_region'.
                    Type: dict
        :param join: If True, the modified spectrum is joined and returned as a single array. If False,
                    a list of modified segments is returned.
                    Default: False
                    Type: bool
        :param plot: If True, a plot of the original and modified spectrum is generated for each line
                    and saved to a file. The regions where the He/H ratio is modified are highlighted.
                    Default: False
                    Type: bool

        :return: Modified spectrum segments or a joined modified spectrum.
                Type: list of numpy arrays (floats) or numpy array (floats)
        """
        self.wave = wave
        self.flux = flux
        self.ratio0 = ratio0
        self.ratio1 = ratio1
        self.dictionary = dictionary
        self.join = join
        # print('length of wave:', len(wave), 'length of flux:', len(flux))
        # Iterate over dictionary to modify spectrum segments
        new_spectrum = []
        original_flux = np.copy(flux)
        for i,line in enumerate(dictionary):
            reg = dictionary[line]['region']
            he_regs = dictionary[line]['HeH_region']
            cond = (wave > reg[0]) & (wave < reg[1])
            if line in [4026, 4121, 4144, 4388, 4471]:
                reg_heline = []
                # Handle line-specific modifications
                if line==4026:
                    cond1 = wave[cond] < 4007
                    cond2 = (wave[cond] >= 4007) & (wave[cond] < 4012)
                    cond3 = (wave[cond] >= 4012) & (wave[cond] < 4022)
                    cond4 = (wave[cond] >= 4022) & (wave[cond] < 4030)
                    cond5 = wave[cond] > 4030
                    reg_heline.append( flux[cond][cond1] )
                    reg_heline.append( (flux[cond][cond2] -1)*(ratio1/ratio0) + 1 )
                    reg_heline.append( flux[cond][cond3] )
                    reg_heline.append( (flux[cond][cond4] -1)*(ratio1/ratio0) + 1 )
                    reg_heline.append( flux[cond][cond5] )
                    temp_spec = np.array(list(itertools.chain.from_iterable(reg_heline)))
                    new_spectrum.append(temp_spec)
                    # flux[cond][cond1] = flux[cond][cond1]
                    # flux[cond][cond2] = (flux[cond][cond2] -1)*(ratio1/ratio0) + 1
                    # flux[cond][cond3] = flux[cond][cond3]
                    # flux[cond][cond4] = (flux[cond][cond4] -1)*(ratio1/ratio0) + 1
                    # flux[cond][cond5] = flux[cond][cond5]
                    # new_spectrum.append(flux[cond])
                else:
                    cond1 = wave[cond] < he_regs[0]
                    cond2 = (wave[cond] > he_regs[0]) & (wave[cond] < he_regs[1])
                    cond3 = wave[cond] > he_regs[1]
                    reg_heline.append( flux[cond][cond1] )
                    reg_heline.append( (flux[cond][cond2] -1)*(ratio1/ratio0) + 1 )
                    reg_heline.append( flux[cond][cond3] )
                    temp_spec = np.array(list(itertools.chain.from_iterable(reg_heline)))
                    new_spectrum.append(temp_spec)
                    # flux[cond][cond1] = flux[cond][cond1]
                    # print("Number of points where cond2 is true:", np.sum(cond2))
                    # print("Original flux where cond2 is true:", flux[cond][cond2])
                    # flux[cond][cond2] = (flux[cond][cond2] -1)*(ratio1/ratio0) + 1
                    # print("Modified flux where cond2 is true:", flux[cond][cond2])
                    # flux[cond][cond3] = flux[cond][cond3]
                    # new_spectrum.append(flux[cond])
            elif line in [4102, 4340]:
                temp_spec = (flux[cond] -1)*((1-ratio1)/(1-ratio0)) + 1
                new_spectrum.append( temp_spec  )
            else:
                new_spectrum.append(flux[cond])
            # Plot the region where the He/H ratio is being modified
            if plot and line in [4026, 4102, 4121, 4144, 4340, 4388, 4471]:
                plt.figure(figsize=(6,4))
                plt.plot(wave[cond], original_flux[cond], label='Original')
                plt.plot(wave[cond], temp_spec, alpha=0.5, label='Modified')
                if line==4026:
                    # for lin in [4007, 4012, 4022, 4030]:
                    #     plt.axvline(x=lin, color='r', linestyle='--', alpha=0.5)
                    for lin1, lin2 in [(4007, 4012), (4022, 4030)]:
                        # plt.fill_between(wave, original_flux, where=(wave > lin1) & (wave < lin2), color='red', alpha=0.5)
                        plt.fill_between(wave[cond], min(original_flux[cond]), max(original_flux[cond]), where=(wave[cond] > lin1) & (wave[cond] < lin2), color='orange', alpha=0.3)
                else:
                    # for lin in he_regs:
                    #     plt.axvline(x=lin, color='r', linestyle='--', alpha=0.5)
                    # plt.fill_between(wave, original_flux, where=(wave > he_regs[0]) & (wave < he_regs[1]), color='red', alpha=0.5)
                    plt.fill_between(wave[cond], min(original_flux[cond]), max(original_flux[cond]), where=(wave[cond] > he_regs[0]) & (wave[cond] < he_regs[1]), color='orange', alpha=0.3)
                plt.title(f'Line {line}')
                plt.xlabel('Wavelength')
                plt.ylabel('Flux')
                plt.legend()
                # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                # print('line:', line, 'model:', model)
                plt.savefig('line'+str(line)+'_'+model+'he'+str(ratio1)+'.png', dpi=100)
                plt.close()
        if join==True:
            new_spectrum = np.array(list(itertools.chain.from_iterable(new_spectrum)))
            # print('join==True. Length of new spectrum:', len(new_spectrum))
            return new_spectrum
        else:
            # print('join==False. Length of new spectrum:', len(new_spectrum))
            return new_spectrum

    def chi2(self, obs, exp):
        """
        Compute the chi-squared (χ²) statistic for comparing observed and expected data.

        This function calculates the chi-squared statistic for assessing the goodness-of-fit between
        observed and expected data, based on their differences.

        :param obs: Observed data array.
                    Type: numpy array or list of floats
        :param exp: Expected data array.
                    Type: numpy array or list of floats

        :return: Calculated chi-squared statistic.
                Type: float
        """
        self.obs = obs
        self.exp = exp
        return np.sum(((obs-exp)**2)/exp)

    def crop_data(self, x_data, y_data, wavelength_ranges):
        """
        Crop data within specified wavelength ranges.

        This function crops the provided x_data and y_data arrays by removing data points that fall
        inside the specified wavelength ranges.

        :param x_data: Original x-axis data (wavelength).
                    Type: numpy array or list of floats
        :param y_data: Original y-axis data (flux).
                    Type: numpy array or list of floats
        :param wavelength_ranges: List of wavelength ranges to be cropped.
                                Each range should be specified as [min_wavelength, max_wavelength].
                                Type: list of lists, each containing two floats

        :return: Cropped x-axis data and corresponding y-axis data.
                Type: numpy arrays (floats)
        """
        self.x_data = x_data
        self.y_data = y_data
        self.wavelength_ranges = wavelength_ranges
        for wav in wavelength_ranges:
            cond = (x_data < wav[0]) | (x_data > wav[1])
            x_data = x_data[cond]
            y_data = y_data[cond]
        return x_data, y_data

    def interpolate_models(self, models_path, models_extension, wavelength, output_path=None):
        """
        Interpolate models to the wavelength of the disentangled spectrum.

        This function interpolates the models to the wavelength of the disentangled spectrum
        based on the provided models_path and wavelength data.

        :param models_path: Path to the folder containing models.
                        Type: str
        :param models_extension: Extension of the model files.
                        Type: str
        :param wavelength: Wavelength array of the disentangled spectrum.
                        Type: numpy array of floats
        :param output_path: Path to save the interpolated models.
                        Default: None
                        Type: str, optional

        :return: Interpolated wavelength array, Interpolated flux array.
                Type: numpy arrays (floats)
        """
        self.models_path = models_path
        self.wavelength = wavelength
        models_list = sorted(glob(models_path+'*'+models_extension))
        for model in models_list:
            mod = pd.read_csv(model, header=None, sep='\s+')
            mod_w = mod[0]
            mod_f = mod[1]
            mod_f_interp = np.interp(wavelength, mod_w, mod_f)
            mod_interp = pd.DataFrame({'wavelength': wavelength, 'flux': mod_f_interp})
            # output_filename = os.path.splitext(model)[0] + models_extension
            output_filename = os.path.basename(model)
            mod_interp.to_csv(output_path+output_filename, header=False, index=False, sep=' ')

# class Spectra(atmfit):
    def read_spec(self):
        """
        Read and load spectrum data from two files.

        This function reads and loads the spectrum data from two files specified by
        self.spectrumA and self.spectrumB. The data is read using pandas and returned as
        two separate dataframes.

        :return: Dataframe containing spectrum data for star A, Dataframe containing spectrum data for star B.
                Type: pandas DataFrames
        """
        dsnt_A = pd.read_csv(self.spectrumA, header=None, sep='\s+')        
        dsnt_B = pd.read_csv(self.spectrumB, header=None, sep='\s+')
        return dsnt_A, dsnt_B

    def get_wave(self, shift=-0.2):
        """
        Get wavelength data for star A and star B after applying a specified shift.

        This function reads and loads the spectrum data for star A and star B using the
        read_spec function. It then adds the specified shift value to the wavelength data
        and returns the shifted wavelength arrays for both stars.

        :param shift: Wavelength shift to be applied to the data.
                    Default: -0.2
                    Type: float

        :return: Wavelength array for star A after applying the shift,
                Wavelength array for star B after applying the shift.
                Type: numpy arrays (floats)
        """
        self.shift = shift
        specA, specB = self.read_spec() 
        waveA = specA[0]+shift
        waveB = specB[0]+shift
        return waveA, waveB

    def get_flux(self):
        """
        Get flux data for star A and star B.

        This function reads and loads the spectrum data for star A and star B using the
        read_spec function. It extracts and returns the flux arrays for both stars.

        :return: Flux array for star A,
                Flux array for star B.
                Type: numpy arrays (floats)
        """
        specA, specB = self.read_spec() 
        fluxA = specA[1]
        fluxB = specB[1]
        return fluxA, fluxB

