# MINATO: Massive bINaries Analysis TOols

MINATO is a Python toolkit for analysing massive stars and binary-star systems.
It provides spectral fitting, radial-velocity measurements, time-series tools,
binary-population inference, and synthetic-spectrum utilities.

The latest stable release is `v0.2.0`. The `develop` branch contains the
candidate work for `v0.3.0` and may include experimental or development-only
modules.

## Upcoming `v0.3.0`

The next planned release is `v0.3.0`. Its official release date will be the
date when the approved `develop` release is merged into `main`.

Planned highlights already available on `develop` include:

- installation as the `minato-astro` Python distribution while retaining
  `import minato`;
- Python 3.12-3.13 support and reproducible mamba, uv, and Pixi environments;
- new `binary_population`, `synthetic`, and `observing` modules;
- expanded scalable binary-population inference and tutorials;
- repaired RAVEL SB2 probabilistic fitting, configurable two-stage sampling,
  and automated SB1/SB2 smoke tests;
- automated tests and package builds for the supported Python versions.

Before release, the candidate still requires full tutorial validation, a final
decision on the experimental `spdis` module, approval of the public
`observing` API, successful CI on the pushed branch, and installation testing
through TestPyPI. Until those checks pass, `v0.2.0` remains the stable release.

## Modules

| Module | Purpose | Status |
| --- | --- | --- |
| `minato.span` | Simultaneous atmosphere-model fitting for disentangled binary spectra | Stable |
| `minato.ravel` | SB1/SB2 line-profile fitting, radial velocities, and period analysis | Stable |
| `minato.binary_population` | Intrinsic binary populations, survey simulation, and scalable population inference | Release candidate |
| `minato.synthetic` | Synthetic single-star and binary spectra with configurable atmosphere-grid backends | Release candidate |
| `minato.observing` | Orbital-phase scheduling, observability checks, and night-visibility plots | Release candidate |
| `minato.spdis` | Shift-and-add spectral disentangling | Experimental |

## Installation

### pip

The existing `v0.2.0` tag predates Python packaging metadata, so it still
requires the historical clone-and-environment workflow. During `0.3.0`
preparation, the installable development snapshot can be installed directly
from GitHub without a manual clone:

```bash
python -m pip install "git+https://github.com/jvillasr/MINATO.git@develop"
```

This follows a moving development branch. Use a tagged release once `0.3.0` is
published.

The planned PyPI distribution name is `minato-astro`, while the import name
remains `minato`:

```python
import minato
print(minato.__version__)
```

Do not run `pip install minato`: that PyPI name belongs to an unrelated file-I/O
library. `pip install minato-astro` will be documented after the first PyPI
publication.

### Development environments

Clone the repository when developing MINATO or running repository tutorials:

```bash
git clone https://github.com/jvillasr/MINATO.git
cd MINATO
git switch develop
```

Choose one environment manager.

With mamba:

```bash
mamba env create -f minato_env.yml
mamba activate minato
```

For the exact locked environment, install `conda-lock` once, then run:

```bash
conda-lock install --name minato conda-lock.yml
mamba activate minato
python -m pip install --no-deps -e .
```

The final command installs the current checkout; editable local projects are
deliberately not embedded in `conda-lock.yml`.

With uv:

```bash
uv sync --frozen --extra dev
```

With Pixi:

```bash
pixi install --locked
```

`pyproject.toml` is the package dependency definition. `uv.lock`, `pixi.lock`,
and `conda-lock.yml` provide exact resolved environments. The current locks
cover Linux x86-64, Intel macOS, and Apple Silicon macOS where supported by the
respective manager.

MINATO uses Python 3.13 for development and supports Python 3.12-3.13. The
orbital solver is implemented with SciPy, avoiding the Python 3.10-only wheels
from the former `kepler.py` dependency.

## Quick start

```python
from minato.binary_population import BinaryPopulation

population = BinaryPopulation()
sample = population.generate_intrinsic_sample_vectorized(N=1_000, f_bin=0.7)
print(sample.head())
```

See the [tutorial index](minato/README.md) for module-specific walkthroughs.

## Documentation

Module documentation currently lives beside the code and in clean tutorial
notebooks. A versioned Sphinx/MyST-NB site hosted on Read the Docs is planned
after the `0.3.0` tutorial set is cleaned and validated, with the aim of having
versioned documentation in place before `1.0.0`.

## Contributing and issues

Bug reports and contributions are welcome through
[GitHub Issues](https://github.com/jvillasr/MINATO/issues) and pull requests.
Ongoing development targets `develop`; `main` contains release-ready code.

## Citation

If you use MINATO in your research, please cite the relevant method:

- `span`: [Villaseñor et al. (2023), MNRAS, 525, 5121](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.5121V/abstract)
- `ravel`: [Villaseñor et al. (2025), A&A](https://ui.adsabs.harvard.edu/abs/2025arXiv250321936V/abstract)

## Licence

MINATO is distributed under the MIT Licence. See [LICENSE.txt](LICENSE.txt).
