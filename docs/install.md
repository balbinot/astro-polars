# Install

```bash
pip install astro-polars
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `viz` | matplotlib, skyproj | plotting (`df.viz`, `viz.skymap`, `viz.plot_heatmap`) |
| `fits` | astropy | FITS readers (planned) |
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
scikit-learn and hpgeom.

!!! note "Python 3.9"
    Recent releases of Polars, numba and astropy need Python 3.10 or 3.11, so
    on Python 3.9 pip installs older versions of them (Polars 1.36 at most).
    astro-polars is tested on 3.9 with those versions. If you also use astropy
    on 3.9, keep `numpy<2`, since astropy releases for 3.9 predate NumPy 2.

## From a clone, for development

```bash
git clone https://github.com/balbinot/astro-polars
cd astro-polars
pip install -e ".[dev]"
pytest
```
