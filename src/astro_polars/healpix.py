"""HEALPix pixel indices as Polars expressions.

Pixel computations use `hpgeom <https://github.com/LSSTDESC/hpgeom>`_ (C,
vectorised). Nested ordering is the default everywhere, because it makes
changing resolution an integer division: the parent of nested pixel ``p`` at
``nside_in`` is ``p // (nside_in // nside_out)**2`` at ``nside_out``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._utils import to_expr

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "nside_to_order",
    "order_to_nside",
    "healpix_index",
    "degrade",
    "gaia_source_id_to_healpix",
    "cone_pixels",
    "pixel_area",
]

#: Gaia source_id encodes the nested HEALPix index at order 12 (nside 4096)
#: in its top bits: ``source_id // 2**35``.
GAIA_SOURCE_ID_SHIFT = 35
GAIA_SOURCE_ID_ORDER = 12


def nside_to_order(nside: int) -> int:
    """HEALPix order (depth) for a power-of-two ``nside``."""
    nside = int(nside)
    order = nside.bit_length() - 1
    if nside < 1 or (1 << order) != nside or order > 29:
        raise ValueError(f"nside must be a power of 2 between 1 and 2**29, got {nside}")
    return order


def order_to_nside(order: int) -> int:
    """``nside = 2**order``."""
    if not 0 <= int(order) <= 29:
        raise ValueError(f"order must be between 0 and 29, got {order}")
    return 1 << int(order)


def pixel_area(nside: int) -> float:
    """Area of one pixel in square degrees."""
    return 4 * np.pi * (180 / np.pi) ** 2 / (12 * nside * nside)


def healpix_index(lon: IntoExpr, lat: IntoExpr, nside: int, nest: bool = True) -> pl.Expr:
    """Pixel index of (lon, lat) in degrees, as an Int64 expression.

    Rows with a null coordinate give null. Latitudes must lie in [-90, 90].
    ::

        lf.with_columns(hpx4096=healpix_index("ra", "dec", 4096))
    """
    import hpgeom

    if nest:
        nside_to_order(nside)

    def _batch(s: pl.Series) -> pl.Series:
        lo = s.struct.field("lon")
        la = s.struct.field("lat")
        ok = (lo.is_not_null() & la.is_not_null()).to_numpy()
        out = np.zeros(len(s), dtype=np.int64)
        if ok.any():
            out[ok] = hpgeom.angle_to_pixel(
                nside,
                lo.to_numpy()[ok].astype(np.float64),
                la.to_numpy()[ok].astype(np.float64),
                nest=nest,
                lonlat=True,
                degrees=True,
            )
        res = pl.Series(out, dtype=pl.Int64)
        return res if ok.all() else res.scatter(np.flatnonzero(~ok), None)

    return pl.struct(
        to_expr(lon).cast(pl.Float64).alias("lon"), to_expr(lat).cast(pl.Float64).alias("lat")
    ).map_batches(_batch, return_dtype=pl.Int64, is_elementwise=True)


def degrade(ipix: IntoExpr, nside_in: int, nside_out: int) -> pl.Expr:
    """Parent nested pixel at ``nside_out`` of nested pixel ``ipix`` at ``nside_in``.

    Exact and fast (integer division, no trigonometry). Only valid for nested
    ordering.
    """
    dk = nside_to_order(nside_in) - nside_to_order(nside_out)
    if dk < 0:
        raise ValueError("nside_out must be <= nside_in")
    e = to_expr(ipix).cast(pl.Int64)
    return e // (4**dk) if dk else e


def gaia_source_id_to_healpix(source_id: IntoExpr = "source_id", nside: int = 4096) -> pl.Expr:
    """Nested HEALPix index at ``nside`` (<= 4096) encoded in a Gaia ``source_id``.

    Gaia source ids carry the order-12 nested pixel in their top bits, so this
    needs no coordinates at all (Gaia DR2/DR3 documentation, ``source_id``).
    """
    order = nside_to_order(nside)
    if order > GAIA_SOURCE_ID_ORDER:
        raise ValueError("Gaia source_id only encodes pixels up to nside 4096")
    shift = 2 ** (GAIA_SOURCE_ID_SHIFT + 2 * (GAIA_SOURCE_ID_ORDER - order))
    return to_expr(source_id).cast(pl.Int64) // shift


def cone_pixels(
    lon: float, lat: float, radius: float, nside: int, nest: bool = True, inclusive: bool = True
) -> np.ndarray:
    """Pixels overlapping a cone of ``radius`` degrees around (lon, lat).

    Use it to prune HEALPix-partitioned datasets before reading them.
    """
    import hpgeom

    pix = hpgeom.query_circle(
        nside, lon, lat, radius, inclusive=inclusive, nest=nest, lonlat=True, degrees=True
    )
    return np.unique(np.asarray(pix, dtype=np.int64))
