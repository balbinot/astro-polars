"""FITS binary tables to Polars. Needs the ``fits`` extra (astropy)."""

from __future__ import annotations

import os
import warnings
from typing import Sequence

import numpy as np
import polars as pl

from .._utils import require

__all__ = ["read_fits_table"]


def _native(a: np.ndarray) -> np.ndarray:
    """FITS data are big-endian; Polars needs native byte order."""
    a = np.array(a)  # copy out of the memory map
    if a.dtype.byteorder not in ("=", "|"):
        a = a.astype(a.dtype.newbyteorder("="))
    return a


def read_fits_table(
    path: str | os.PathLike,
    columns: Sequence[str] | None = None,
    hdu: int | str = 1,
    warn_missing: bool = True,
) -> pl.DataFrame:
    """Read a FITS binary table into a DataFrame.

    Parameters
    ----------
    path
        FITS file, optionally compressed (``.fits.gz``).
    columns
        Columns to read, in this order. Names not in the table are skipped,
        with a warning if ``warn_missing``. ``None`` reads all columns.
    hdu
        HDU index or name holding the table.
    """
    fits = require("astropy.io.fits", "fits")
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        names = list(data.columns.names)
        if columns is None:
            use = names
        else:
            use = [c for c in columns if c in names]
            missing = [c for c in columns if c not in names]
            if missing and warn_missing:
                warnings.warn(
                    f"{os.path.basename(os.fspath(path))}: {len(missing)} requested columns "
                    f"missing (e.g. {missing[:3]})",
                    stacklevel=2,
                )
        series = []
        for c in use:
            a = _native(data[c])
            if a.ndim > 1:
                series.append(pl.Series(c, list(a)))
            else:
                series.append(pl.Series(c, a))
    return pl.DataFrame(series)
