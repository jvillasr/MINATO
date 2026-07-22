# Development-only shift-and-add adaptation

This directory contains local adaptations that wrap spectral-disentangling
code as a Python class for use during MINATO development. It is not a MINATO
module or product, and it is excluded from MINATO packages and releases.

The original implementation is the
[Disentangling_Shift_And_Add repository](https://github.com/TomerShenar/Disentangling_Shift_And_Add),
written by Tomer Shenar with contributions from Matthias Fabry and Julia
Bodensteiner. Credit for the method and code belongs to that project.

Follow the upstream repository for current usage instructions, permissions,
and citation guidance. Its README asks users to cite:

- Gonzalez & Levato (2006), A&A, 448, 283, for the shift-and-add algorithm;
- Shenar et al. (2020), A&A, 639, A6;
- Shenar et al. (2022), A&A, 665, A148.

The upstream repository does not currently publish a software licence file.
Keeping a copy on a public development branch may itself require permission
from the upstream owner. Record that permission or a suitable upstream licence
before treating this copy as redistributable. It must not be distributed under
MINATO's MIT licence or included in a MINATO source archive, wheel, or release
branch without a separately reviewed licence decision.

Within a development checkout, the adapted class is available as:

```python
from minato.contrib.spdis import SpecDisent
```

Do not document or import it as `minato.spdis`; the `minato.contrib` namespace
marks it as external, development-only code.

The development script `scripts/generate_span_tutorial_data.py` uses this
adaptation to reproduce the synthetic disentangled inputs for the public SPAN
tutorial. The released tutorial consumes only the resulting synthetic spectra;
it does not expose `minato.contrib.spdis` as a supported MINATO API. The
upstream credit and citation requirements listed above still apply to that
generation step.

## Development-only flux-error propagation

`SpecDisent.get_disspec` can estimate marginal one-sigma flux errors by
repeating the final disentangling calculation after drawing independent
Gaussian noise from supplied epoch-level errors:

```python
disentangler.get_disspec(
    lguess1=0.8,
    flux_errors=epoch_flux_errors,
    uncertainty_samples=100,
    uncertainty_seed=1234,
)
```

`epoch_flux_errors` may be a scalar, one pixel array shared by every epoch, or
an array shaped `(number_of_epochs, number_of_pixels)`. The orbit, epoch
weights, preprocessing, and reference light ratio remain fixed. When enabled,
the saved component files contain wavelength, normalised flux, and marginal
one-sigma flux error. The same reference light ratio must be supplied to SPAN
as `lrat0`; it is a flux scale, not an uncertainty contribution or a fitted
light ratio.

Set `keep_uncertainty_samples=True` to retain the realisations as
`flux_samplesA` and `flux_samplesB`. They are needed to estimate the
wavelength-to-wavelength and cross-component covariance introduced by
disentangling. The one-sigma columns alone are a diagonal approximation when
passed to SPAN through `flux_errorA` and `flux_errorB`.
