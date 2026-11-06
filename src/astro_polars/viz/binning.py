"""Streaming binned statistics on a regular 2D grid or on HEALPix pixels.

These functions only need Polars and NumPy. They aggregate with a Polars
``group_by`` on the streaming engine, so they work on LazyFrames larger
than memory.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from .._utils import collect as _collect
from ..healpix import degrade, healpix_index, nside_to_order
from ..robust import K_NMAD

__all__ = ["AGGREGATIONS", "heatmap", "healpix_map"]

Frame = Union[pl.DataFrame, pl.LazyFrame]
Lims = Optional[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]]

#: Aggregations accepted by ``what``. Each maps the ``z`` column name to an expression.
AGGREGATIONS: dict[str, Callable[[Optional[str]], pl.Expr]] = {
    "len": lambda z: pl.len(),
    "count": lambda z: pl.len(),
    "mean": lambda z: pl.col(z).mean(),
    "median": lambda z: pl.col(z).median(),
    "sum": lambda z: pl.col(z).sum(),
    "min": lambda z: pl.col(z).min(),
    "max": lambda z: pl.col(z).max(),
    "std": lambda z: pl.col(z).std(),
    "nmad": lambda z: pl.col(z).robust.nmad(K_NMAD),
}


def _agg_expr(what: str, z: str | None) -> pl.Expr:
    if what not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {what!r}; use one of {sorted(AGGREGATIONS)}")
    if what not in ("len", "count") and z is None:
        raise ValueError(f"what={what!r} requires a z column")
    return AGGREGATIONS[what](z)


def _finite_rows(d: pl.LazyFrame, cols: Sequence[str]) -> pl.LazyFrame:
    """Drop rows with null, NaN or inf in any of ``cols``."""
    schema = d.collect_schema()
    conds = []
    for c in cols:
        e = pl.col(c).is_not_null()
        if schema[c] in (pl.Float32, pl.Float64):
            e = e & pl.col(c).is_finite()
        conds.append(e)
    return d.filter(pl.all_horizontal(conds))


def heatmap(
    d: Frame,
    x: str,
    y: str,
    what: str = "len",
    z: str | None = None,
    lims: Lims = None,
    shape: int | tuple[int, int] = 256,
    fill: float = 0.0,
    *,
    nside: int | None = None,
    nest: bool = True,
    ipix: str | None = None,
    ipix_nside: int | None = None,
):
    """Streaming 2D binned statistic (a "heatmap").

    Parameters
    ----------
    d
        DataFrame or LazyFrame.
    x, y
        Column names. In HEALPix mode these are longitude and latitude in degrees.
    what
        Aggregation: ``len`` (counts), ``mean``, ``median``, ``sum``, ``min``,
        ``max``, ``std`` or ``nmad``.
    z
        Column to aggregate; required unless ``what="len"``.
    lims
        ``((x0, x1), (y0, y1))``. Either axis, or the whole argument, may be
        ``None`` to infer limits from the finite data. Grid mode only.
    shape
        Number of bins, an int or ``(nx, ny)``. Grid mode only.
    fill
        Value for empty bins. Use ``np.nan`` for mean or median maps.
    nside, nest, ipix, ipix_nside
        Give ``nside`` to bin on HEALPix pixels instead of a grid; see
        ``healpix_map``.

    Returns
    -------
    (H, extent)
        Grid mode: ``H`` has shape ``(nx, ny)`` (x first, like
        ``numpy.histogram2d``) and ``extent = [x0, x1, y0, y1]``. Show it with
        ``plt.imshow(H.T, origin="lower", extent=extent)`` or
        ``astro_polars.viz.plot_heatmap``.
    numpy.ndarray
        HEALPix mode: a full-sky map of length ``12 * nside**2``.
    """
    if nside is not None or ipix is not None:
        if nside is None:
            raise ValueError("nside is required in HEALPix mode")
        return healpix_map(
            d,
            lon=x,
            lat=y,
            nside=nside,
            what=what,
            z=z,
            fill=fill,
            nest=nest,
            ipix=ipix,
            ipix_nside=ipix_nside,
        )

    if isinstance(shape, int):
        nbinsx = nbinsy = shape
    else:
        nbinsx, nbinsy = shape

    aggfunc = _agg_expr(what, z)
    d = d.lazy()

    # 1. Drop rows with null/NaN/inf in any relevant column
    cols = [x, y] + ([z] if what not in ("len", "count") else [])
    d = _finite_rows(d, cols)

    # 2. Infer missing limits from the (now finite) data in one pass
    xlims, ylims = (None, None) if lims is None else lims
    stats_exprs = []
    if xlims is None:
        stats_exprs += [pl.col(x).min().alias("_x0"), pl.col(x).max().alias("_x1")]
    if ylims is None:
        stats_exprs += [pl.col(y).min().alias("_y0"), pl.col(y).max().alias("_y1")]
    if stats_exprs:
        s = _collect(d.select(stats_exprs)).row(0, named=True)
        if any(v is None for v in s.values()):
            raise ValueError("No finite rows available to infer limits from")
        if xlims is None:
            xlims = (s["_x0"], s["_x1"])
        if ylims is None:
            ylims = (s["_y0"], s["_y1"])

    def _pad(lo, hi):
        lo, hi = float(lo), float(hi)
        if lo == hi:  # degenerate range (e.g. constant column)
            eps = 0.5 if lo == 0 else 0.5 * abs(lo)
            return lo - eps, hi + eps
        return lo, hi

    (x0, x1), (y0, y1) = _pad(*xlims), _pad(*ylims)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Limits must be increasing: (lo, hi)")

    xbsize = (x1 - x0) / nbinsx
    ybsize = (y1 - y0) / nbinsy

    _d = _collect(
        d.filter(pl.col(x).is_between(x0, x1) & pl.col(y).is_between(y0, y1))
        .with_columns(
            ((pl.col(x) - x0) / xbsize).floor().clip(0, nbinsx - 1).cast(pl.Int32).alias("_j"),
            ((pl.col(y) - y0) / ybsize).floor().clip(0, nbinsy - 1).cast(pl.Int32).alias("_i"),
        )
        .group_by("_i", "_j")
        .agg(aggfunc.alias("_v"))
    )

    H = np.full((nbinsy, nbinsx), fill, dtype=np.float64)
    H[_d["_i"].to_numpy(), _d["_j"].to_numpy()] = _d["_v"].to_numpy()
    extent = np.array([x0, x1, y0, y1])
    return H.T, extent


def healpix_map(
    d: Frame,
    lon: str = "ra",
    lat: str = "dec",
    nside: int = 64,
    what: str = "len",
    z: str | None = None,
    fill: float = 0.0,
    *,
    nest: bool = True,
    ipix: str | None = None,
    ipix_nside: int | None = None,
    sparse: bool = False,
):
    """Streaming binned statistic on HEALPix pixels.

    Parameters
    ----------
    d
        DataFrame or LazyFrame.
    lon, lat
        Longitude and latitude columns in degrees.
    nside
        HEALPix resolution of the output map (power of 2).
    what, z
        Aggregation and the column it applies to, as in ``heatmap``.
    fill
        Value for empty pixels in the dense map. ``np.nan`` hides them in plots.
    nest
        Nested (default) or ring pixel ordering for the output.
    ipix, ipix_nside
        Fastest path: name of a column that already holds *nested* pixel
        indices at ``ipix_nside`` (for example ``hpx4096`` written by the
        Euclid pipeline). Pixels are degraded by integer division, with no
        trigonometry and without reading ``lon``/``lat``.
    sparse
        Return ``(pixels, values)`` for non-empty pixels only, instead of a
        full-sky array. Use it for large ``nside``.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
    """
    nside_to_order(nside)
    aggfunc = _agg_expr(what, z)
    zcols = [z] if what not in ("len", "count") else []
    d = d.lazy()

    if ipix is not None:
        if not nest:
            raise ValueError("ipix degrading needs nested ordering (nest=True)")
        if ipix_nside is None:
            raise ValueError("ipix_nside is required with ipix")
        d = _finite_rows(d, [ipix] + zcols)
        pix = degrade(ipix, ipix_nside, nside)
    else:
        d = _finite_rows(d, [lon, lat] + zcols)
        pix = healpix_index(lon, lat, nside, nest=nest)

    res = _collect(d.group_by(pix.alias("_pix")).agg(aggfunc.alias("_v")))
    pixels = res["_pix"].to_numpy()
    values = res["_v"].cast(pl.Float64).to_numpy()
    if sparse:
        order_idx = np.argsort(pixels)
        return pixels[order_idx], values[order_idx]
    m = np.full(12 * nside * nside, fill, dtype=np.float64)
    m[pixels] = values
    return m
