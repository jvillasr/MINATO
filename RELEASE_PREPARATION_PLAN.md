# MINATO release preparation plan

## Goal

Prepare the next MINATO release without exposing development logs, unfinished
plans, or experimental archive recovery on `main`.

## Approved release decisions

- Release `0.3.0`, not `1.0.0`.
- Publish the distribution as `minato-astro` while preserving `import minato`.
- Use the date of the approved `develop`-to-`main` merge commit as the official
  release date.
- Keep `main` release-only by removing development records and unfinished
  archive recovery from the pending merge tree before committing the merge.
- Retain `mamba`, `uv`, and `pixi` workflows. Treat `pyproject.toml` as the
  package dependency definition and commit each manager's generated lock file.
- Distribute no atmosphere grids, trained models, observed spectra, or fitted
  results. Users supply their own scientific models and data. The GitHub
  release tree may retain the two approved, provenance-tracked synthetic SPAN
  fixtures; package-index archives continue to exclude the tutorial tree.
- Keep the adapted shift-and-add code as a development-only external
  contribution, with all credit and citations directed to
  `TomerShenar/Disentangling_Shift_And_Add`. Do not release it as a MINATO
  module or under the MINATO licence.

## Remaining recommendation

- Build a Sphinx/MyST-NB documentation site after the release tutorial set is
  clean and validated.

The version metadata and dated changelog section must still wait until the
release checks pass and the merge is ready to be committed.

## Why `0.3.0`

`0.3.0` accurately signals a substantial feature release whose public surface
is still settling. `binary_population`, `synthetic`, and `observing` are new,
the package has not yet been published under its final PyPI name, and the
release tutorial set plus versioned documentation are incomplete.

Use `1.0.0` when MINATO has:

- an agreed stable public API and deprecation policy;
- installation from a package index under the final distribution name;
- locked and validated environments on supported platforms;
- automated tests for the supported Python versions;
- clean, executed tutorials and versioned user/API documentation;
- a validated policy for user-supplied atmosphere grids and other data.

## Distribution and environments

The environment options are compatible because they serve different users:

| Workflow | Manifest | Exact lock | Intended use |
| --- | --- | --- | --- |
| pip | `pyproject.toml` | Package-index artefact hashes | End users |
| uv | `pyproject.toml` | `uv.lock` | Fast Python development and CI |
| Pixi | `pyproject.toml` Pixi sections | `pixi.lock` | Cross-platform conda/PyPI development |
| mamba | `minato_env.yml` | `conda-lock.yml` | Existing conda-based development |

`minato` is already occupied on PyPI by an unrelated project. The proposed
distribution name is `minato-astro`; Python code continues to use the `minato`
package namespace.

Remaining environment work:

1. Publish the validated wheel to TestPyPI before the real package index.

Fresh Python 3.13 environments created from the mamba, uv, and Pixi locks pass
the unit suite. The mamba check also verifies that the resolved NumPy 2.2.6
installation has no broken package requirements.

The mamba lock covers third-party dependencies. Install the current checkout
with `python -m pip install --no-deps -e .` after `conda-lock install`; editable
local packages are intentionally excluded from conda lock files.

## Branch contents

Keep the following on `develop` and release-preparation branches, but out of
`main` unless their role changes:

- `AGENTS.md`
- `ROADMAP.md`
- `NOTES.md`
- `CHANGELOG.md`
- `*_PLAN.md` and `*_PLANS.md`
- benchmark logs and local run notes
- `minato/contrib/`, including the external shift-and-add adaptation
- `minato/models/`
- observed tutorial spectra, fitted results, plots, tables, and unapproved
  generated assets

The cleaned tutorial notebooks and their Markdown documentation may remain on
`main`, but their outputs must be cleared and their examples must generate
small synthetic inputs, use an explicitly approved synthetic fixture, or
require explicit user paths. The two SPAN disentangled fixtures and their
provenance are approved for the GitHub release tree. `MANIFEST.in` excludes the
entire tutorial tree from package-index artefacts, and
`scripts/check_release_artifacts.py` independently verifies that wheels and
source archives contain no package data, models, tests, external contributions,
or development records.

Approved release flow:

1. Complete and approve the release checks on `develop`.
2. Check out `main` and run `git merge --no-commit --no-ff develop`.
3. Remove development-only records and unfinished archive recovery from the
   pending merge tree without changing `develop`. Also remove tracked models,
   contributed external code, observed tutorial datasets, and unapproved
   generated tutorial outputs. Retain the approved synthetic SPAN fixtures and
   their provenance.
4. Update release-facing version metadata, the README, and GitHub release
   notes, then validate the exact pending release tree.
5. Commit the merge. The merge commit date is the official release date.
6. Tag the merge commit as `v0.3.0`, publish `minato-astro`, and create the
   GitHub release.
7. Record the release outcome in the development log files on `develop`.

This direct merge policy may require resolving modify/delete conflicts for
development-only files in later releases. Resolve them by keeping those files
on `develop` and absent from the release tree.

## Archive recovery

- The adapted shift-and-add class is retained only at
  `minato/contrib/spdis.py` on `develop`, where it is imported as
  `minato.contrib.spdis`. It is derived from
  `TomerShenar/Disentangling_Shift_And_Add`, whose upstream README owns the
  credit and citation guidance. The adaptation is not `minato.spdis` and is
  excluded from packages and release branches.
- Useful archived observing logic has been redesigned as the installable
  `minato.observing` package. It has synthetic unit tests, a clean executable
  tutorial, explicit output-file protection, and no archived notebook outputs
  or figures.

## Tutorial input policy

- RAVEL and disentangling examples must create small analytic or synthetic
  spectra during execution instead of bundling observed or pre-rendered
  spectra.
- SPAN examples may demonstrate public TLUSTY or PoWR grids, but users must
  obtain those grids separately, provide their local paths, and follow the
  original licence and citation instructions. The SPAN fitting tutorial may
  consume the two approved synthetic disentangled fixtures; their generation
  remains reproducible on `develop` and credits the upstream disentangling
  implementation.
- Atmosphere-fit result tutorials must generate a small result in an earlier
  cell or accept an explicit user-supplied result path.
- Notebooks committed to the release tree must have no outputs, local absolute
  paths, or dependencies on files removed from `main`.

## Documentation strategy

Recommended stack: Sphinx with MyST-NB on Read the Docs.

Advantages:

- versioned documentation for each release;
- notebook and Markdown content in one navigation tree;
- API reference generated from docstrings;
- pull-request build previews and link checking;
- a stable home for installation and upgrade guidance.

Costs:

- another build configuration and dependency set to maintain;
- current large/output-heavy notebooks must be cleaned first;
- slow or data-dependent notebooks need mocked, cached, or non-executing docs
  variants;
- API imports during docs builds may expose optional-dependency problems.

Start the site after the targeted tutorial clean-up, but before `1.0.0`. Keep
the first site small: installation, module overview, four clean tutorials, and
generated API pages for `binary_population` and `synthetic`.

## Release gates

- Keep the approved `0.3.0`, merge-date, and `minato-astro` decisions recorded.
- Keep the validated mamba, uv, and Pixi locks unchanged unless dependency
  updates are intentionally reviewed and retested.
- Confirm the new GitHub Actions jobs pass after pushing `develop`.
- Complete the full multi-epoch SB1/SB2 tutorial validation; reduced synthetic
  probabilistic fits now run in CI.
- Complete the remaining spectral-analysis and RAVEL tutorial rewrites against
  the approved external-model and synthetic-input policy. The SPAN fitting
  tutorial itself is rewritten and its complete real-PoWR workflow is
  validated.
- Validate the binary-population tutorial in the final release environment.
- Build wheel and source archives and pass
  `scripts/check_release_artifacts.py` before TestPyPI publication.
- Review and approve `minato.observing` as part of the `0.3.0` public surface.
- Publish the release candidate to TestPyPI and verify installation from the
  package index before publishing to PyPI.
