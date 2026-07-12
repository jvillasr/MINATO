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

## Remaining recommendation

- Build a Sphinx/MyST-NB documentation site after the release tutorial set is
  clean and validated.

The version metadata and dated changelog section must still wait until the
release checks pass and the merge is ready to be committed.

## Why `0.3.0`

`0.3.0` accurately signals a substantial feature release whose public surface
is still settling. `binary_population`, `synthetic`, and `observing` are new,
`spdis` is still experimental, the package has not yet been published under
its final PyPI name, and CI plus versioned documentation are incomplete.

Use `1.0.0` when MINATO has:

- an agreed stable public API and deprecation policy;
- installation from a package index under the final distribution name;
- locked and validated environments on supported platforms;
- automated tests for the supported Python versions;
- clean, executed tutorials and versioned user/API documentation;
- a documented policy for external atmosphere grids and other large data.

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

Approved release flow:

1. Complete and approve the release checks on `develop`.
2. Check out `main` and run `git merge --no-commit --no-ff develop`.
3. Remove development-only records and unfinished archive recovery from the
   pending merge tree without changing `develop`.
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

- `minato/spdis.py` is already identical to the copy on `archive-develop`. Its
  package-relative import has been repaired, but it still needs focused tests,
  API documentation, and a small synthetic tutorial before release.
- Useful archived observing logic has been redesigned as the installable
  `minato.observing` package. It has synthetic unit tests, a clean executable
  tutorial, explicit output-file protection, and no archived notebook outputs
  or figures.

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

- Approve `0.3.0`, the release date, and `minato-astro`.
- Keep the validated mamba, uv, and Pixi locks unchanged unless dependency
  updates are intentionally reviewed and retested.
- Confirm the new GitHub Actions jobs pass after pushing `develop`.
- Complete the full multi-epoch SB1/SB2 tutorial validation; reduced synthetic
  probabilistic fits now run in CI.
- Validate the binary-population tutorial in the final release environment.
- Decide whether `spdis` is documented experimental API or excluded from the
  release notes.
- Review and approve `minato.observing` as part of the `0.3.0` public surface.
- Publish the release candidate to TestPyPI and verify installation from the
  package index before publishing to PyPI.
