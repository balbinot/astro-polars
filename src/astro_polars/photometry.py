"""Fluxes, magnitudes, colours and their errors as Polars expressions.

All magnitudes are AB unless you pass your own zero point. For a flux ``f``
in some unit, ``mag = -2.5 log10(f) + zp`` with ``zp = -2.5 log10(unit / 3631 Jy)``.
Common zero points are in ``AB_ZEROPOINTS``, e.g. 23.9 for microjansky
(Euclid MER) and 22.5 for nanomaggies (Legacy Surveys, SDSS).

Non-positive fluxes give NaN or infinite magnitudes, as ``log10`` does; filter
them with ``is_finite()`` when needed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Sequence, Tuple, Union

import polars as pl

from ._utils import to_expr

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "AB_ZEROPOINTS",
    "MAGERR_FACTOR",
    "ab_zeropoint",
    "flux_to_mag",
    "mag_to_flux",
    "fluxerr_to_magerr",
    "magerr_to_fluxerr",
    "flux_color",
    "aperture_correction",
    "deredden",
    "Photometry",
]

#: Flux of a zero-magnitude AB source, in Jy.
AB_FLUX_JY = 3631.0

#: ``2.5 / ln(10)``: converts a relative flux error into a magnitude error.
MAGERR_FACTOR = 2.5 / math.log(10.0)

_UNITS_JY = {
    "Jy": 1.0,
    "mJy": 1e-3,
    "uJy": 1e-6,
    "µJy": 1e-6,
    "nJy": 1e-9,
    "nanomaggy": 3631.0e-9,
    "nmgy": 3631.0e-9,
    "maggy": 3631.0,
}


def ab_zeropoint(unit: str | float) -> float:
    """AB zero point for fluxes in ``unit``: ``-2.5 log10(unit_in_Jy / 3631)``.

    ``unit`` is a name (``"Jy"``, ``"mJy"``, ``"uJy"``, ``"nJy"``,
    ``"nanomaggy"``) or the size of the unit in Jy.
    """
    jy = _UNITS_JY[unit] if isinstance(unit, str) else float(unit)
    return -2.5 * math.log10(jy / AB_FLUX_JY)


#: AB zero points by flux unit.
AB_ZEROPOINTS = {u: ab_zeropoint(u) for u in _UNITS_JY}

#: A zero point: a number, a unit name (see ``ab_zeropoint``) or an
#: expression, e.g. ``pl.col("zp")`` for a per-row zero point.
ZeroPoint = Union[float, str, pl.Expr]


def _zp(zeropoint: ZeroPoint) -> float | pl.Expr:
    if isinstance(zeropoint, pl.Expr):
        return zeropoint
    return ab_zeropoint(zeropoint) if isinstance(zeropoint, str) else float(zeropoint)


def flux_to_mag(flux: IntoExpr, zeropoint: ZeroPoint = "uJy") -> pl.Expr:
    """``-2.5 log10(flux) + zeropoint``.

    ``zeropoint`` is a number, a unit name such as ``"uJy"`` or
    ``"nanomaggy"``, or an expression (``pl.col("zp")``). The result has the
    dtype of ``flux`` (Float32 stays Float32) unless the zero point is a
    wider expression, e.g. ``pl.lit(zp, dtype=pl.Float64)``.
    """
    return -2.5 * to_expr(flux).log10() + _zp(zeropoint)


def mag_to_flux(mag: IntoExpr, zeropoint: ZeroPoint = "uJy") -> pl.Expr:
    """Inverse of ``flux_to_mag``: ``10**(-0.4 (mag - zeropoint))``."""
    return pl.lit(10.0).pow(-0.4 * (to_expr(mag) - _zp(zeropoint)))


def fluxerr_to_magerr(fluxerr: IntoExpr, flux: IntoExpr) -> pl.Expr:
    """First-order magnitude error, ``2.5 / ln(10) * fluxerr / flux``."""
    return to_expr(fluxerr) / to_expr(flux) * MAGERR_FACTOR


def magerr_to_fluxerr(magerr: IntoExpr, flux: IntoExpr) -> pl.Expr:
    """Inverse of ``fluxerr_to_magerr``: ``magerr * flux / (2.5 / ln(10))``."""
    return to_expr(magerr) * to_expr(flux) / MAGERR_FACTOR


def flux_color(flux_a: IntoExpr, flux_b: IntoExpr) -> pl.Expr:
    """Colour ``mag_a - mag_b = -2.5 log10(flux_a / flux_b)``; zero points cancel."""
    return -2.5 * (to_expr(flux_a) / to_expr(flux_b)).log10()


def aperture_correction(total_flux: IntoExpr, aperture_flux: IntoExpr) -> pl.Expr:
    """Multiplicative aperture-to-total correction, ``total_flux / aperture_flux``."""
    return to_expr(total_flux) / to_expr(aperture_flux)


def deredden(mag: IntoExpr, ebv: IntoExpr, coefficient: float) -> pl.Expr:
    """Extinction-corrected magnitude, ``mag - coefficient * E(B-V)``.

    ``coefficient`` is ``A_band / E(B-V)`` for your band and extinction law.
    """
    return to_expr(mag) - coefficient * to_expr(ebv)


def _present(df, cols: Sequence[str]) -> bool:
    names = set(df.collect_schema().names())
    return all(c in names for c in cols)


@pl.api.register_dataframe_namespace("phot")
@pl.api.register_lazyframe_namespace("phot")
class Photometry:
    """``df.phot``: add many magnitudes, errors and colours at once.

    Each method takes a mapping from output name to input column(s). Entries
    whose input columns are missing are skipped (``strict=False``, default)
    or raise (``strict=True``), so one mapping can serve catalogues with
    slightly different columns.
    """

    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        self._df = df

    def _keep(self, needed: Sequence[str], strict: bool) -> bool:
        ok = _present(self._df, needed)
        if not ok and strict:
            missing = sorted(set(needed) - set(self._df.collect_schema().names()))
            raise pl.exceptions.ColumnNotFoundError(f"missing columns: {missing}")
        return ok

    def add_mags(
        self, mapping: Mapping[str, str], zeropoint: ZeroPoint = "uJy", strict: bool = False
    ):
        """Add ``name = flux_to_mag(flux, zeropoint)`` for each ``name: flux``."""
        exprs = [
            flux_to_mag(flux, zeropoint).alias(name)
            for name, flux in mapping.items()
            if self._keep([flux], strict)
        ]
        return self._df.with_columns(exprs)

    def add_magerrs(self, mapping: Mapping[str, Tuple[str, str]], strict: bool = False):
        """Add ``name = fluxerr_to_magerr(fluxerr, flux)`` for each ``name: (fluxerr, flux)``."""
        exprs = [
            fluxerr_to_magerr(err, flux).alias(name)
            for name, (err, flux) in mapping.items()
            if self._keep([err, flux], strict)
        ]
        return self._df.with_columns(exprs)

    def add_colors(self, mapping: Mapping[str, Tuple[str, str]], strict: bool = False):
        """Add ``name = flux_color(flux_a, flux_b)`` for each ``name: (flux_a, flux_b)``."""
        exprs = [
            flux_color(a, b).alias(name)
            for name, (a, b) in mapping.items()
            if self._keep([a, b], strict)
        ]
        return self._df.with_columns(exprs)

    def deredden(
        self,
        coefficients: Mapping[str, float],
        ebv: str = "ebv",
        suffix: str = "_0",
        strict: bool = False,
    ):
        """Add ``{mag}{suffix} = mag - coeff * ebv`` for each ``mag: coeff``."""
        exprs = [
            deredden(mag, ebv, coeff).alias(f"{mag}{suffix}")
            for mag, coeff in coefficients.items()
            if self._keep([mag, ebv], strict)
        ]
        return self._df.with_columns(exprs)
