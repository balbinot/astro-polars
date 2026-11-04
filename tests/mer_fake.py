"""Synthetic Euclid MER final-catalogue tiles for tests.

Every column of ``euclid.KEEPCOLS`` is present except the JPCAM ones, so the
"missing columns are skipped" path is exercised too. The values are only
plausible, not realistic: what matters is that every step of the pipeline has
something to do (NaN and infinite values to clean, sources outside the horn,
sources fainter than the magnitude limit, negative fluxes, duplicates).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from astro_polars.surveys import euclid

DROPPED = [c for c in euclid.KEEPCOLS if "JPCAM" in c]


def mer_columns(n: int, rng: np.random.Generator, ra0: float, dec0: float, tile: int) -> dict:
    """Column name -> numpy array for one fake tile of ``n`` rows."""
    cols = {}
    mag = rng.uniform(18.0, 29.0, n)
    total = 10 ** (-0.4 * (mag - 23.9))
    for c in euclid.KEEPCOLS:
        if c in DROPPED:
            continue
        if c == "OBJECT_ID":
            cols[c] = tile * 10_000_000 + np.arange(n, dtype=np.int64)
        elif c == "RIGHT_ASCENSION":
            cols[c] = ra0 + rng.uniform(-1.5, 1.5, n)
        elif c == "DECLINATION":
            cols[c] = dec0 + rng.uniform(-1.5, 1.5, n)
        elif c in ("TILE_INDEX", "GAIA_ID"):
            cols[c] = np.full(n, tile, dtype=np.int64)
        elif c.startswith("FLAG") or c.endswith("_FLAG") or c == "VIS_DET":
            cols[c] = rng.integers(0, 4, n).astype(np.int32)
        elif c == "FLUX_DETECTION_TOTAL":
            cols[c] = total.astype(np.float32)
        elif c.startswith("FLUX_VIS_") and c.endswith("FWHM_APER"):
            k = int(c[len("FLUX_VIS_")])
            cols[c] = (total * (0.25 * k) * rng.normal(1, 0.02, n)).astype(np.float32)
        elif c.startswith("FLUXERR"):
            cols[c] = (total * rng.uniform(0.01, 0.2, n)).astype(np.float32)
        elif c.startswith("FLUX"):
            f = total * rng.lognormal(0.0, 0.5, n)
            f[rng.random(n) < 0.02] *= -1  # some negative fluxes -> NaN magnitudes
            cols[c] = f.astype(np.float32)
        elif c == "MUMAX_MINUS_MAG":
            m = -3.0 + rng.normal(0, 0.6, n)
            m[rng.random(n) < 0.01] = np.inf
            m[rng.random(n) < 0.01] = np.nan
            cols[c] = m.astype(np.float32)
        elif c == "POINT_LIKE_PROB":
            p = rng.random(n)
            p[rng.random(n) < 0.01] = np.nan
            cols[c] = p.astype(np.float32)
        else:
            cols[c] = rng.normal(1.0, 0.1, n).astype(np.float32)
    # exact duplicate rows, as when a source is listed twice
    dup = rng.choice(n, n // 50, replace=False)
    return {c: np.concatenate([a, a[dup]]) for c, a in cols.items()}


def astropy_fits():
    """``astropy.io.fits``, or skip the test if astropy is missing or broken.

    Old astropy versions import but fail later with NumPy 2, so import the
    table machinery too.
    """
    try:
        import astropy.table  # noqa: F401
        from astropy.io import fits
    except Exception as err:  # ImportError, or AttributeError with NumPy 2
        pytest.skip(f"astropy not usable: {err}")
    return fits


def write_mer_tile(
    root, tile: int, n: int = 2000, ra0: float = 150.0, dec0: float = 2.0, seed: int = 0
) -> str:
    """Write ``root/<tile>/EUC_MER_FINAL-CAT_TILE<tile>-...fits``; returns the path."""
    fits = astropy_fits()
    rng = np.random.default_rng(seed)
    cols = mer_columns(n, rng, ra0, dec0, tile)
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=c, array=a, format=_fits_format(a)) for c, a in cols.items()]
    )
    d = os.path.join(os.fspath(root), str(tile))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(
        d, f"EUC_MER_FINAL-CAT_TILE{tile}-ABC123_20250101T000000.000000Z_00.00.fits"
    )
    hdu.writeto(path, overwrite=True)
    return path


def _fits_format(a: np.ndarray) -> str:
    return {"i4": "J", "i8": "K", "f4": "E", "f8": "D"}[a.dtype.str[1:]]
