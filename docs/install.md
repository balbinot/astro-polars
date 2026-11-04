# Install

```bash
pip install astro-polars
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `viz` | matplotlib, skyproj | plotting (`df.viz`, `viz.skymap`, `viz.plot_heatmap`) |
| `fits` | astropy | FITS tables (`io.fits.read_fits_table`, `surveys.euclid.read_mer`) |
| `all` | all of the above | |

```bash
pip install "astro-polars[viz]"
```

The development version installs straight from GitHub:

```bash
pip install "git+https://github.com/balbinot/astro-polars"
```

## Requirements

Python 3.9 or newer. Core dependencies: polars (>= 1.30), numpy, numba,
scipy, scikit-learn and hpgeom.

!!! note "Python 3.9"
    Recent releases of Polars, numba and astropy need Python 3.10 or 3.11, so
    on Python 3.9 pip installs older versions of them (Polars 1.36 at most).
    astro-polars is tested on 3.9 with those versions. The `fits` extra
    installs `numpy<2` on 3.9, since no astropy release for 3.9 supports
    NumPy 2.

## From a clone, for development

```bash
git clone https://github.com/balbinot/astro-polars
cd astro-polars
pip install -e ".[dev]"
pytest
```
