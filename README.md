# MINATO: Massive bINaries Analysis TOols

Python tools for the comprehensive analysis of massive binary stars, focusing on precise radial velocity measurement, spectral line profile fitting, and robust time series analysis.

## Version 0.1.0
- **Spectral analysis (`span`)**: 
  - Simultaneous spectral fitting of binary stars using synthetic models. 
  - Computes effective temperatures, log surface gravities, rotational velocities, He/H ratios, and the light ratio of the binary.

## Version 0.2.0

### Added Features:

- **Radial Velocity Determination and Time Series Analysis (`ravel`)**

  - **Spectral Line Profile Fitting (`SLfit`)**
    - Supports single-lined (SB1) and double-lined (SB2) spectroscopic binaries.
    - Automated Gaussian/Lorentzian line profile fitting with customizable priors.
    - For SB2s:
      - Probabilistic modeling using Bayesian inference with Numpyro. 
      - Direct radial velocity computation.

  - **Radial Velocity Analysis for SB1s (`GetRVs`)**
    - Automated computation of radial velocities (RVs) from fitted spectral lines.
    - Weighted mean RV calculations with built-in outlier rejection based on median absolute deviation (MAD).
    - Comprehensive statistical summaries and error propagation for reliable velocity measurements.

  - **Time Series and Period Analysis**
    - Implementation of Lomb-Scargle periodograms with false alarm probability (FAP) estimation.
    - Automatic peak detection with adjustable significance thresholds.
    - Probabilistic sinusoidal model fitting to phased radial velocity curves for orbital characterisation.


### Planned Future Features

- Simulations of binary populations
- Spectral Energy Distribution (SED) fitting
- Automated spectral classification of massive stars

## Installation

Clone the repository:
```bash
git clone https://github.com/jvillasr/MINATO/
```
Use your favourite dependency manager:
```bash
cd MINATO
mamba env create -f minato_env.yml
```

## Usage

Detailed examples are provided in the `minato/tutorials` directory.

## Dependencies

- tested under Python 3.10
- astropy
- jax
- lmfit
- matplotlib
- numpy
- numpyro
- pandas
- scipy
- tqdm

## Contributing and Issues
Contributions and bug reports are welcome! Please submit an issue on GitHub or open a pull request.

## Citation

If you use MINATO in your research, please cite:

- For `span`:
> [Villaseñor et al., 2023, MNRAS, 525, 5121, 10.1093/mnras/stad2533](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.5121V/abstract)
- For `ravel`:
> [Villaseñor et al., 2025, A&A accepted, 10.48550/arXiv.2503.21936](https://ui.adsabs.harvard.edu/abs/2025arXiv250321936V/abstract)

## License

This project is licensed under the MIT License. See `LICENSE` for more information.