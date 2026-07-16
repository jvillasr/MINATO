"""Fast, grid-based stellar-atmosphere fitting for binary-star spectra."""

import time
import sys
import csv
import os
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from glob import glob
from datetime import timedelta, date, datetime
# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
current_date = str(date.today())


def _format_model_value(value):
    """Return a stable filename representation for a numerical grid value."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("model-grid values must be finite")
    if numeric.is_integer():
        return str(int(numeric))
    return np.format_float_positional(numeric, trim="-")


def _model_stem(parameters):
    """Build the filename stem used by SPAN's precomputed model grids."""
    return "".join(f"{key}{_format_model_value(value)}" for key, value in parameters.items())


class AtmFit:
    """Fit one or two stellar spectra through an explicit atmosphere grid."""

    def __init__(
        self,
        spectrumA,
        spectrumB,
        grid=None,
        lrat0=None,
        modelsA_path=None,
        modelsB_path=None,
        modelsA_grid=None,
        modelsB_grid=None,
        binary=False,
        crop_nebular=False,
        He2H=False,
        He_ini=0.1,
        wavelength_shift=-0.2,
        max_workers=None,
        chunksize=1,
        flux_errorA=None,
        flux_errorB=None,
        inverse_varianceA=None,
        inverse_varianceB=None,
    ):
        """
        Initialize the atmosphere fitting class.

        This constructor initializes the `atmfit` class with specified parameters.
        It sets up the initial configuration for the fitting process.

        :param spectrumA: Path to the spectrum file for star A.
                         Type: str
        :param spectrumB: Path to the spectrum file for star B.
                         Type: str
        :param grid: Dictionary of parameter for the fitting process.
                        Format: grid = {'lr': np.arange(5, 30, 5)/100, 'TA': [15000, 20000, 25000, 30000], etc.}
                        Type: dict
        :param lrat0: Initial light ratio. If not provided, defaults to None.
                        Type: float or None
        :param modelsA_path: Path to the folder containing models for star A.
                        Type: str
        :param modelsB_path: Path to the folder containing models for star B.
                        Type: str
        :param modelsA_grid: In-memory
                        :class:`minato.synthetic.RenderedAtmosphereGrid` for
                        star A, used instead of ``modelsA_path``.
        :param modelsB_grid: In-memory
                        :class:`minato.synthetic.RenderedAtmosphereGrid` for
                        star B, used instead of ``modelsB_path``. Binary fits
                        may pass the same object for both components.
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
        :param wavelength_shift: Offset in Angstrom applied to both input spectra.
                        The legacy default is -0.2; use 0.0 for spectra already
                        on their intended rest-wavelength scale.
                        Type: float
        :param max_workers: Maximum number of SPAN worker processes. ``None``
                        keeps the ProcessPoolExecutor default.
                        Type: int or None
        :param chunksize: Number of parameter combinations submitted in each
                        multiprocessing chunk.
                        Type: int
        :param flux_errorA: One-sigma flux uncertainty for spectrum A. A
                        positive scalar applies to every pixel; otherwise pass
                        an array matching the complete input spectrum.
                        Mutually exclusive with ``inverse_varianceA``.
        :param flux_errorB: One-sigma flux uncertainty for spectrum B. Binary
                        fits must provide uncertainties for both components or
                        for neither component.
        :param inverse_varianceA: Inverse flux variance for spectrum A. A
                        scalar applies to every pixel and zero-valued pixels
                        are excluded from the fit.
        :param inverse_varianceB: Inverse flux variance for spectrum B.
        """
        self.grid = grid
        
        self.spectrumA = spectrumA
        self.spectrumB = spectrumB
        self.lrat0 = lrat0
        self.modelsA_path = modelsA_path
        self.modelsB_path = modelsB_path
        self.modelsA_grid = modelsA_grid
        self.modelsB_grid = modelsB_grid
        if modelsA_grid is not None and modelsA_path is not None:
            raise ValueError("use either modelsA_grid or modelsA_path, not both")
        if modelsB_grid is not None and modelsB_path is not None:
            raise ValueError("use either modelsB_grid or modelsB_path, not both")
        for name, model_grid in (
            ('modelsA_grid', modelsA_grid),
            ('modelsB_grid', modelsB_grid),
        ):
            if model_grid is not None and not hasattr(model_grid, 'get_model'):
                raise TypeError(f"{name} must define get_model(teff, logg, vsini)")
        self.binary = binary
        if binary and (modelsA_grid is None) != (modelsB_grid is None):
            raise ValueError("binary in-memory fitting requires modelsA_grid and modelsB_grid")
        if not binary and modelsB_grid is not None:
            raise ValueError("modelsB_grid is only valid for binary fitting")
        self.He2H = He2H
        self.crop_nebular = crop_nebular
        self.He_ini = He_ini
        self.wavelength_shift = float(wavelength_shift)
        if max_workers is not None and int(max_workers) < 1:
            raise ValueError("max_workers must be positive or None")
        if int(chunksize) < 1:
            raise ValueError("chunksize must be positive")
        self.max_workers = None if max_workers is None else int(max_workers)
        self.chunksize = int(chunksize)
        self.flux_errorA = flux_errorA
        self.flux_errorB = flux_errorB
        self.inverse_varianceA = inverse_varianceA
        self.inverse_varianceB = inverse_varianceB
        self._validate_uncertainty_configuration()
        self.ivarA = None
        self.ivarB = None
        self.missing_models = False
        self.warning_printed = False

    def _validate_uncertainty_configuration(self):
        """Validate whether the fit has a complete statistical noise model."""

        for component in ('A', 'B'):
            flux_error = getattr(self, f'flux_error{component}')
            inverse_variance = getattr(self, f'inverse_variance{component}')
            if flux_error is not None and inverse_variance is not None:
                raise ValueError(
                    f"use either flux_error{component} or "
                    f"inverse_variance{component}, not both"
                )

        weighted_a = (
            self.flux_errorA is not None or self.inverse_varianceA is not None
        )
        weighted_b = (
            self.flux_errorB is not None or self.inverse_varianceB is not None
        )
        if self.binary and weighted_a != weighted_b:
            raise ValueError(
                "binary weighted fitting requires uncertainties for both "
                "spectrum A and spectrum B"
            )
        if not self.binary and weighted_b:
            raise ValueError(
                "spectrum B uncertainties are only valid for binary fitting"
            )
        self.score_kind = 'chi2' if weighted_a else 'rss'

    @staticmethod
    def _as_pixel_array(values, size, name):
        """Broadcast a scalar or validate a complete per-pixel array."""

        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return np.full(size, float(array), dtype=float)
        if array.ndim != 1 or len(array) != size:
            raise ValueError(
                f"{name} must be a scalar or a one-dimensional array with "
                f"{size} values"
            )
        return array.copy()

    def _component_inverse_variance(self, component, size):
        """Return validated inverse variance for a complete input spectrum."""

        flux_error = getattr(self, f'flux_error{component}')
        inverse_variance = getattr(self, f'inverse_variance{component}')
        if flux_error is None and inverse_variance is None:
            return None
        if flux_error is not None:
            errors = self._as_pixel_array(
                flux_error,
                size,
                f'flux_error{component}',
            )
            if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
                raise ValueError(
                    f"flux_error{component} must contain finite positive values"
                )
            return 1.0 / errors**2

        weights = self._as_pixel_array(
            inverse_variance,
            size,
            f'inverse_variance{component}',
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(
                f"inverse_variance{component} must contain finite non-negative "
                "values"
            )
        if not np.any(weights > 0):
            raise ValueError(
                f"inverse_variance{component} must contain at least one "
                "positive value"
            )
        return weights

    def _prepare_fit_weights(self):
        """Prepare per-pixel weights after the spectrum lengths are known."""

        self.ivarA = self._component_inverse_variance('A', len(self.wavA))
        self.ivarB = (
            self._component_inverse_variance('B', len(self.wavB))
            if self.binary
            else None
        )

    def _component_parameter_count(self, component):
        """Count parameters entering one component score."""

        suffix = component.upper()
        component_parameters = sum(
            key.endswith(suffix) and self._grid_parameter_varies(key)
            for key in self.grid
        )
        if self.binary and any(
            key in self.grid and self._grid_parameter_varies(key)
            for key in ('lr', 'lrat')
        ):
            component_parameters += 1
        return component_parameters

    def _grid_parameter_varies(self, name):
        """Return whether a grid parameter contains multiple distinct values."""

        return len(np.unique(np.asarray(self.grid[name]))) > 1

    def _attach_result_metadata(self, output, ndata):
        """Record the score definition without adding repeated table columns."""

        degrees_of_freedom = int(ndata - self.nparams)
        output.attrs.update(
            {
                'score_kind': self.score_kind,
                'score_column': 'chi2_tot',
                'n_parameters': int(self.nparams),
                'degrees_of_freedom': degrees_of_freedom,
                'uncertainty_model': (
                    'independent Gaussian per-pixel uncertainties'
                    if self.score_kind == 'chi2'
                    else 'none; unweighted residual sum of squares'
                ),
            }
        )
        return output
        
    lines_dic = {
                    3995: { 'region':[3990, 4000],  'HeH_region':[], 'title':r'N II $\lambda$3995'},
                    4026: { 'region':[4005, 4033],  'HeH_region':[4005, 4033], 'title':r'He I $\lambda$4009/26'},
                    4102: { 'region':[4087, 4130],  'HeH_region':[4091, 4111], 'title':r'H$\delta$'},
                    4121: { 'region':[4117, 4135],  'HeH_region':[4120, 4122], 'title':r'He I $\lambda$4121, Si II $\lambda$4128/32'},
                    4144: { 'region':[4137, 4151],  'HeH_region':[4142, 4146], 'title':r'He I $\lambda$4144'},
                    4233: { 'region':[4225, 4241],  'HeH_region':[], 'title':r'Fe II $\lambda$4233'},
                    4267: { 'region':[4260, 4275],  'HeH_region':[], 'title':r'C II $\lambda$4267'},
                    4340: { 'region':[4320, 4362],  'HeH_region':[4330, 4350], 'title':r'H$\gamma$'},
                    4388: { 'region':[4380, 4396],  'HeH_region':[4386.5, 4389.5], 'title':r'He I $\lambda$4388'},
                    4471: { 'region':[4465, 4485],  'HeH_region':[4471, 4473], 'title':r'He I $\lambda$4471, Mg II $\lambda$4481'},
                    # 4553: { 'region':[4536, 4560],  'HeH_region':[], 'title':'Fe II $\lambda$4550/56, Si III $\lambda$4553'} }
                    4553: { 'region':[4536, 4560],  'HeH_region':[], 'title':r'He II $\lambda$4542, Si III $\lambda$4553'} }

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
        """
        Compute fit scores for a single set of parameters.

        This method takes a set of parameters, from which it retrieves the model(s), and computes 
        the chi-squared values for the fit between the model and the data. It handles both binary 
        and single star cases.

        Parameters
        ----------
        params : list
            A list of parameter values corresponding to the keys in self.grid. 

        Returns
        -------
        row : list
            A list containing the input parameters and the computed chi-squared values. 
            The order of values in this list is determined by self.cols, followed by the chi-squared values. 
        """

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
            # print(f'Exception in get_model(): {type(e).__name__}: {e}')
            pass   
        
        else:
            # Keep observations in their original disentangling scale for
            # likelihood ranking.  Rescaling observations also rescales their
            # noise, so model spectra are instead diluted to the same scale.
            fluA, fluB = self.get_flux()
            lrat = getattr(self, 'lr', None)
            if lrat is not None:
                self.lrat = lrat

            modA_f = self.model_on_observed_grid(self.wavA, modA_w, modA_f)
            modA_w = self.wavA
            if self.binary:
                modB_f = self.model_on_observed_grid(self.wavB, modB_w, modB_f)
                modB_w = self.wavB

            # slice data to regions for chi^2 computation
            dst_A_w_slc, dst_A_f_slc = self.slicedata(self.wavA, fluA, self.user_dicA)
            dst_B_w_slc, dst_B_f_slc = self.slicedata(self.wavB, fluB, self.user_dicB) if self.binary else (None, None)
            ivar_A_slc = (
                self.slicedata(self.wavA, self.ivarA, self.user_dicA)[1]
                if self.ivarA is not None
                else None
            )
            ivar_B_slc = (
                self.slicedata(self.wavB, self.ivarB, self.user_dicB)[1]
                if self.binary and self.ivarB is not None
                else None
            )
            mod_A_w_slc, mod_A_f_slc = self.slicedata(modA_w, modA_f, self.user_dicA)
            mod_B_w_slc, mod_B_f_slc = self.slicedata(modB_w, modB_f, self.user_dicB) if self.binary else (None, None)

            # apply He/H ratio to the sliced model of star A
            if self.He2H:
                mod_A_f_slc = self.He2H_ratio(mod_A_w_slc, mod_A_f_slc, self.He_ini, self.He, self.user_dicA, join=True, plot=False, model=modelA.replace(self.modelsA_path, ''))

            if lrat is not None:
                mod_A_f_slc = self.model_flux_to_initial_light_ratio(mod_A_f_slc, 'A', lrat)
                if self.binary:
                    mod_B_f_slc = self.model_flux_to_initial_light_ratio(mod_B_f_slc, 'B', lrat)

            # crop nebular emission from disentangled spectrum and model of star B
            if self.crop_nebular:
                if ivar_B_slc is not None:
                    _, ivar_B_slc = self.crop_data(
                        dst_B_w_slc,
                        ivar_B_slc,
                        [[4100, 4104], [4338, 4346]],
                    )
                dst_B_w_slc, dst_B_f_slc = self.crop_data(dst_B_w_slc, dst_B_f_slc, [[4100, 4104], [4338, 4346]])
                mod_B_w_slc, mod_B_f_slc = self.crop_data(mod_B_w_slc, mod_B_f_slc, [[4100, 4104], [4338, 4346]])

            # compute the chi2 values
            ndataA = self._fitted_sample_count(dst_A_f_slc, ivar_A_slc)
            chi2A = self.chi2(
                dst_A_f_slc,
                mod_A_f_slc,
                inverse_variance=ivar_A_slc,
            )
            dofA = ndataA - self._component_parameter_count('A')
            if dofA <= 0:
                raise ValueError("spectrum A has insufficient fitted samples")
            chi2redA = chi2A / dofA

            if self.binary:
                ndataB = self._fitted_sample_count(dst_B_f_slc, ivar_B_slc)
                chi2B = self.chi2(
                    dst_B_f_slc,
                    mod_B_f_slc,
                    inverse_variance=ivar_B_slc,
                )
                chi2_tot = chi2A + chi2B
                ndata = ndataA + ndataB
                dofB = ndataB - self._component_parameter_count('B')
                if dofB <= 0:
                    raise ValueError("spectrum B has insufficient fitted samples")
                chi2redB = chi2B / dofB
                total_dof = ndata - self.nparams
                if total_dof <= 0:
                    raise ValueError("fit has insufficient fitted samples")
                chi2r_tot = chi2_tot / total_dof
            else:
                chi2_tot = chi2A
                ndata = ndataA
                chi2r_tot = chi2A / (ndataA - self.nparams)

            if chi2_tot < 0:
                raise ValueError("\nWarning: chi2 < O")

            # Create row by getting the values of the parameters from self
            row = [getattr(self, key) for key in self.cols if hasattr(self, key)]
            if self.binary:
                row.extend([chi2_tot, chi2A, chi2B, chi2r_tot, chi2redA, chi2redB, ndata])
            else:
                row.extend([chi2_tot, chi2r_tot, ndata])
            
            return row

    def model_on_observed_grid(self, observed_wavelength, model_wavelength, model_flux):
        """Interpolate one model onto an observed wavelength grid."""
        observed_wavelength = np.asarray(observed_wavelength, dtype=float)
        model_wavelength = np.asarray(model_wavelength, dtype=float)
        model_flux = np.asarray(model_flux, dtype=float)
        if model_wavelength.size != model_flux.size:
            raise ValueError("model wavelength and flux arrays must have the same length")
        if observed_wavelength[0] < model_wavelength[0] or observed_wavelength[-1] > model_wavelength[-1]:
            raise ValueError("model wavelength range does not cover the observed spectrum")
        return np.interp(observed_wavelength, model_wavelength, model_flux)

    def compute_chi2(self, dic_lines_A, dic_lines_B):
        """
        Perform a parameter-grid search and compute model-comparison scores.

        With flux errors or inverse variances supplied at construction, the
        scores are weighted chi-square values. Otherwise they are unweighted
        residual sums of squares. The score definition is recorded in the
        returned DataFrame attributes.

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
        nparams = sum(
            self._grid_parameter_varies(name) for name in self.grid
        )
        self.nparams = nparams

        # retrieve wavelength from the disentangled spectra
        wavA, wavB = self.get_wave(shift=self.wavelength_shift)
        self.wavA = wavA
        self.wavB = wavB
        self._prepare_fit_weights()
        # setting the dictionaries with the spectral lines selected for the fit
        usr_dicA = self.user_dic(dic_lines_A)
        usr_dicB = self.user_dic(dic_lines_B) if self.binary else {}
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

        if self.modelsA_grid is not None:
            output = self._compute_chi2_rendered_grid()
            tf = time.time()
            print('Computation completed in: ' + str(timedelta(seconds=tf-t0)) + ' [s] \n')
            return output

        # Get all possible combinations of parameters
        parameters = list(itertools.product(*self.grid.values()))
        # print('parameters:', parameters)
        
        # Compute chi2 values for each set of parameters
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self.compute_single_set,
                        parameters,
                        chunksize=self.chunksize,
                    ),
                    total=len(parameters),
                )
            )
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
        if not output.empty:
            output = self._attach_result_metadata(output, output.iloc[0]['ndata'])
        # print('total number of points used in the fit:', ndata)
        return output

    def _compute_chi2_rendered_grid(self):
        """Score an in-memory synthetic grid without writing intermediate files."""

        if self.He2H:
            raise NotImplementedError(
                "He2H fitting is not yet supported with in-memory model grids"
            )

        if self.binary:
            expected_keys = {'lr', 'TA', 'gA', 'vA', 'TB', 'gB', 'vB'}
            if set(self.grid) != expected_keys:
                raise ValueError(
                    "rendered binary grids require lr, TA, gA, vA, TB, gB, and vB"
                )

            light_ratios = [float(value) for value in self.grid['lr']]
            if any(not 0 < value < 1 for value in light_ratios):
                raise ValueError("binary light ratios must be strictly between 0 and 1")
            if len(set(light_ratios)) != len(light_ratios):
                raise ValueError("binary light ratios must not contain duplicates")
            initial_light_ratio = self.lrat0 if self.lrat0 is not None else 0.3
            if not 0 < float(initial_light_ratio) < 1:
                raise ValueError("lrat0 must be strictly between 0 and 1")

            observed_flux_a, observed_flux_b = self.get_flux()
            scores_a, ndata_a = self._score_rendered_component(
                component='A',
                parameter_keys=('TA', 'gA', 'vA'),
                light_ratios=light_ratios,
                observed_flux=observed_flux_a,
            )
            scores_b, ndata_b = self._score_rendered_component(
                component='B',
                parameter_keys=('TB', 'gB', 'vB'),
                light_ratios=light_ratios,
                observed_flux=observed_flux_b,
            )
            output = scores_a.merge(
                scores_b,
                on='lr',
                how='inner',
                validate='many_to_many',
            )
            output['chi2_tot'] = output['chi2A'] + output['chi2B']
            output['ndata'] = ndata_a + ndata_b
            total_dof = ndata_a + ndata_b - self.nparams
            if total_dof <= 0:
                raise ValueError("fit has insufficient fitted samples")
            output['chi2r_tot'] = output['chi2_tot'] / total_dof
            columns = list(self.grid) + [
                'chi2_tot',
                'chi2A',
                'chi2B',
                'chi2r_tot',
                'chi2redA',
                'chi2redB',
                'ndata',
            ]
            output = output.loc[:, columns]
            return self._attach_result_metadata(output, ndata_a + ndata_b)

        expected_keys = {'T', 'g', 'v'}
        if set(self.grid) != expected_keys:
            raise ValueError("rendered single-star grids require T, g, and v")
        observed_flux, _ = self.get_flux()
        output = self._score_rendered_single_star(observed_flux)
        return self._attach_result_metadata(output, output.iloc[0]['ndata'])

    def _score_rendered_component(
        self,
        component,
        parameter_keys,
        light_ratios,
        observed_flux,
    ):
        """Score one binary component before the component score tables are joined."""

        if component == 'A':
            observed_wavelength = np.asarray(self.wavA, dtype=float)
            line_dictionary = self.user_dicA
            chi2_column = 'chi2A'
            reduced_column = 'chi2redA'
        else:
            observed_wavelength = np.asarray(self.wavB, dtype=float)
            line_dictionary = self.user_dicB
            chi2_column = 'chi2B'
            reduced_column = 'chi2redB'
        inverse_variance = self.ivarA if component == 'A' else self.ivarB
        observed_flux = np.asarray(observed_flux, dtype=float)

        observed_slice_wavelength, observed_slice_flux = self.slicedata(
            observed_wavelength,
            observed_flux,
            line_dictionary,
        )
        observed_slice_ivar = (
            self.slicedata(
                observed_wavelength,
                inverse_variance,
                line_dictionary,
            )[1]
            if inverse_variance is not None
            else None
        )
        if component == 'B' and self.crop_nebular:
            if observed_slice_ivar is not None:
                _, observed_slice_ivar = self.crop_data(
                    observed_slice_wavelength,
                    observed_slice_ivar,
                    [[4100, 4104], [4338, 4346]],
                )
            observed_slice_wavelength, observed_slice_flux = self.crop_data(
                observed_slice_wavelength,
                observed_slice_flux,
                [[4100, 4104], [4338, 4346]],
            )

        ndata = self._fitted_sample_count(
            observed_slice_flux,
            observed_slice_ivar,
        )
        degrees_of_freedom = (
            ndata - self._component_parameter_count(component)
        )
        if degrees_of_freedom <= 0:
            raise ValueError(
                f"component {component} has {ndata} fitted samples but "
                f"{self.nparams} free parameters"
            )

        rows = []
        model_grid = self.modelsA_grid if component == 'A' else self.modelsB_grid
        model_nodes = self._matching_rendered_nodes(
            model_grid,
            parameter_keys,
            component,
        )
        for values in model_nodes:
            model = model_grid.get_model(*values)
            model_flux = self.model_on_observed_grid(
                observed_wavelength,
                model.wavelength,
                model.flux,
            )
            model_slice_wavelength, model_slice_flux = self.slicedata(
                observed_wavelength,
                model_flux,
                line_dictionary,
            )
            if component == 'B' and self.crop_nebular:
                model_slice_wavelength, model_slice_flux = self.crop_data(
                    model_slice_wavelength,
                    model_slice_flux,
                    [[4100, 4104], [4338, 4346]],
                )

            for light_ratio in light_ratios:
                scaled_model_flux = self.model_flux_to_initial_light_ratio(
                    model_slice_flux,
                    component,
                    light_ratio,
                )
                chi2_value = self.chi2(
                    observed_slice_flux,
                    scaled_model_flux,
                    inverse_variance=observed_slice_ivar,
                )
                row = dict(zip(parameter_keys, values))
                row.update(
                    {
                        'lr': light_ratio,
                        chi2_column: chi2_value,
                        reduced_column: chi2_value / degrees_of_freedom,
                    }
                )
                rows.append(row)

        columns = list(parameter_keys) + ['lr', chi2_column, reduced_column]
        return pd.DataFrame(rows, columns=columns), ndata

    def _matching_rendered_nodes(self, model_grid, parameter_keys, component):
        """Return rendered nodes matching the requested, possibly irregular grid."""

        requested = {
            key: tuple(dict.fromkeys(self.grid[key]))
            for key in parameter_keys
        }
        if not hasattr(model_grid, 'parameter_nodes'):
            return list(itertools.product(*(requested[key] for key in parameter_keys)))

        def matches(value, choices):
            return any(
                np.isclose(value, choice, rtol=0.0, atol=1e-10)
                for choice in choices
            )

        nodes = [
            tuple(node)
            for node in model_grid.parameter_nodes
            if all(
                matches(node[index], requested[key])
                for index, key in enumerate(parameter_keys)
            )
        ]
        missing = {}
        for index, key in enumerate(parameter_keys):
            represented = [node[index] for node in nodes]
            missing_values = [
                value
                for value in requested[key]
                if not matches(value, represented)
            ]
            if missing_values:
                missing[key] = missing_values
        if missing:
            details = '; '.join(
                f"{key}={values}" for key, values in missing.items()
            )
            raise LookupError(
                f"rendered component {component} has no valid models for {details}"
            )
        return [
            tuple(
                next(
                    choice
                    for choice in requested[key]
                    if np.isclose(node[index], choice, rtol=0.0, atol=1e-10)
                )
                for index, key in enumerate(parameter_keys)
            )
            for node in nodes
        ]

    def _score_rendered_single_star(self, observed_flux):
        """Score a standard single-star ``T``, ``g``, and ``v`` grid."""

        observed_wavelength = np.asarray(self.wavA, dtype=float)
        observed_flux = np.asarray(observed_flux, dtype=float)
        _, observed_slice_flux = self.slicedata(
            observed_wavelength,
            observed_flux,
            self.user_dicA,
        )
        observed_slice_ivar = (
            self.slicedata(
                observed_wavelength,
                self.ivarA,
                self.user_dicA,
            )[1]
            if self.ivarA is not None
            else None
        )
        ndata = self._fitted_sample_count(
            observed_slice_flux,
            observed_slice_ivar,
        )
        degrees_of_freedom = ndata - self.nparams
        if degrees_of_freedom <= 0:
            raise ValueError(
                f"single-star fit has {ndata} fitted samples but "
                f"{self.nparams} free parameters"
            )

        rows = []
        for teff, logg, vsini in itertools.product(
            self.grid['T'],
            self.grid['g'],
            self.grid['v'],
        ):
            model = self.modelsA_grid.get_model(teff, logg, vsini)
            model_flux = self.model_on_observed_grid(
                observed_wavelength,
                model.wavelength,
                model.flux,
            )
            _, model_slice_flux = self.slicedata(
                observed_wavelength,
                model_flux,
                self.user_dicA,
            )
            chi2_value = self.chi2(
                observed_slice_flux,
                model_slice_flux,
                inverse_variance=observed_slice_ivar,
            )
            rows.append(
                {
                    'T': teff,
                    'g': logg,
                    'v': vsini,
                    'chi2_tot': chi2_value,
                    'chi2r_tot': chi2_value / degrees_of_freedom,
                    'ndata': ndata,
                }
            )

        return pd.DataFrame(rows, columns=self.cols)

    def rescale_flux(self, lrat, lrat0=0.3):
        """
        Rescales the flux of two stars based on a desired light ratio.

        This function rescales the flux of two stars (A and B) to achieve a desired
        light ratio while considering an initial light ratio. The new flux values are
        calculated using the given light ratio and initial light ratio.

        :param lrat:  Desired secondary light fraction for rescaling.
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

    def model_flux_to_initial_light_ratio(self, model_flux, component, lrat, lrat0=None):
        """
        Dilute an intrinsic model spectrum to the original disentangling light ratio.

        ``lrat`` is the secondary light fraction used by ``rescale_flux`` and by the
        fitted grid.  When fitting across ``lrat`` values, the likelihood must compare
        all candidates in a common flux/noise scale: the observed disentangled spectra
        stay at their initial light ratio, and the intrinsic model fluxes are scaled
        into that same reference frame.

        :param model_flux: Intrinsic normalised model flux.
                           Type: numpy array or list of floats
        :param component: Binary component identifier, either ``'A'`` or ``'B'``.
                          Type: str
        :param lrat: Secondary light fraction of the model candidate.
                     Type: float
        :param lrat0: Initial secondary light fraction of the disentangled spectra.
                      Defaults to ``self.lrat0`` when available, otherwise 0.3.
                      Type: float, optional

        :return: Model flux diluted to the initial light-ratio scale.
                 Type: numpy array of floats
        """
        if lrat0 is None:
            ratio0 = self.lrat0 if self.lrat0 is not None else 0.3
        else:
            ratio0 = lrat0

        component = component.upper()
        model_flux = np.asarray(model_flux)
        if component == 'A':
            return 1 + (model_flux - 1)*((1-lrat)/(1-ratio0))
        if component == 'B':
            return 1 + (model_flux - 1)*(lrat/ratio0)
        raise ValueError("component must be 'A' or 'B'")

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
            model_directory = Path(models_path).expanduser()
            model_stem = _model_stem(pars)
            exact_model = model_directory / f"{model_stem}.txt"
            if exact_model.exists():
                model_found = exact_model
            else:
                matches = sorted(model_directory.glob(f"{model_stem}*"))
                if not matches:
                    self.missing_models = True
                    raise FileNotFoundError(
                        f"no model matching {model_stem!r} in {model_directory}"
                    )
                model_found = matches[0]
            df = pd.read_csv(model_found, header=None, sep=r'\s+', comment='#')
            return df[0].to_numpy(), df[1].to_numpy(), str(model_found)
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
                        df = pd.read_csv(tlustyO_path+model,header=None, sep=r'\s+')
                    else:
                        model = 'T'+str(int(T))+'g'+str(int(g*10))+'v2r'+str(int(rot))+'fw05'
                        df = pd.read_csv(tlustyB_path+model,header=None, sep=r'\s+')
                    # return df[0].array, df[1].array, model
                    return df[0].to_numpy(), df[1].to_numpy(), model
                except FileNotFoundError:
                    # print('WARNING: No model named '+model+' was found')
                    # raise ValueError('   WARNING: No model available for '+model)
                    pass
            elif source=='atlas':
                model = 'T'+str(int(T))+'g'+str(int(g))+'v2r'+str(int(rot))+'fw05'
                try:
                    df = pd.read_csv(lowT_models_path+model,header=None, sep=r'\s+')
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

    @staticmethod
    def _fitted_sample_count(observed, inverse_variance=None):
        """Count samples contributing to one score."""

        observed = np.asarray(observed, dtype=float)
        if inverse_variance is None:
            return int(observed.size)
        inverse_variance = np.asarray(inverse_variance, dtype=float)
        if inverse_variance.shape != observed.shape:
            raise ValueError(
                "inverse variance and observed flux arrays must have the same shape"
            )
        return int(np.count_nonzero(inverse_variance > 0))

    def chi2(self, obs, exp, flux_error=None, inverse_variance=None):
        """
        Compute a weighted chi-square or legacy squared-residual score.

        Supplying one-sigma errors or inverse variances produces a statistical
        chi-square under independent Gaussian pixel errors. Without either,
        the return value is an unweighted residual sum of squares and must not
        be interpreted as chi-square.

        :param obs: Observed data array.
                    Type: numpy array or list of floats
        :param exp: Expected data array.
                    Type: numpy array or list of floats
        :param flux_error: Positive one-sigma errors, either a scalar or an
                    array matching ``obs``.
        :param inverse_variance: Non-negative inverse variances, either a
                    scalar or an array matching ``obs``. Zero-valued pixels
                    are excluded.

        :return: Calculated chi-squared statistic.
                Type: float
        """
        self.obs = obs
        self.exp = exp
        obs = np.asarray(obs, dtype=float)
        exp = np.asarray(exp, dtype=float)
        if obs.shape != exp.shape:
            raise ValueError("observed and expected flux arrays must have the same shape")
        if flux_error is not None and inverse_variance is not None:
            raise ValueError(
                "use either flux_error or inverse_variance, not both"
            )
        if flux_error is not None:
            errors = self._as_pixel_array(flux_error, obs.size, 'flux_error')
            errors = errors.reshape(obs.shape)
            if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
                raise ValueError("flux_error must contain finite positive values")
            inverse_variance = 1.0 / errors**2
        elif inverse_variance is not None:
            inverse_variance = self._as_pixel_array(
                inverse_variance,
                obs.size,
                'inverse_variance',
            ).reshape(obs.shape)
            if (
                not np.all(np.isfinite(inverse_variance))
                or np.any(inverse_variance < 0)
            ):
                raise ValueError(
                    "inverse_variance must contain finite non-negative values"
                )
            if not np.any(inverse_variance > 0):
                raise ValueError(
                    "inverse_variance must contain at least one positive value"
                )

        if inverse_variance is None:
            if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(exp)):
                raise ValueError("observed and expected flux must be finite")
            return float(np.sum((obs - exp) ** 2))

        fitted = inverse_variance > 0
        if not np.all(np.isfinite(obs[fitted])) or not np.all(
            np.isfinite(exp[fitted])
        ):
            raise ValueError(
                "observed and expected flux must be finite where weight is positive"
            )
        residual = obs[fitted] - exp[fitted]
        return float(np.sum(residual**2 * inverse_variance[fitted]))

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
            mod = pd.read_csv(model, header=None, sep=r'\s+')
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
        dsnt_A = pd.read_csv(self.spectrumA, header=None, sep=r'\s+', comment='#')
        dsnt_B = (
            pd.read_csv(self.spectrumB, header=None, sep=r'\s+', comment='#')
            if self.spectrumB is not None
            else None
        )
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
        waveB = specB[0]+shift if specB is not None else None
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
        fluxB = specB[1] if specB is not None else None
        return fluxA, fluxB
