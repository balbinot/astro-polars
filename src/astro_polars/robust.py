"""Robust statistics as Polars expressions.

``pl.col(x).robust.nmad()`` and ``pl.col(x).robust.zscore()`` are available
after ``import astro_polars``.
"""

from __future__ import annotations

import polars as pl

__all__ = ["K_NMAD", "nmad", "zscore", "Robust"]

#: Scale factor making the MAD a consistent estimator of a Gaussian sigma.
K_NMAD = 1.4826022185056018


@pl.api.register_expr_namespace("robust")
class Robust:
    """``expr.robust``: median-based statistics, insensitive to outliers."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def nmad(self, k: float = K_NMAD) -> pl.Expr:
        """Normalised median absolute deviation, ``k * median(|x - median(x)|)``."""
        x = self._expr
        return k * (x - x.median()).abs().median()

    def zscore(self, k: float = K_NMAD) -> pl.Expr:
        """Robust z-score, ``(x - median(x)) / nmad(x)``."""
        x = self._expr
        return (x - x.median()) / (k * (x - x.median()).abs().median())


def nmad(col: str, k: float = K_NMAD) -> pl.Expr:
    """NMAD of column ``col``, aliased ``"{col}_nmad"``."""
    return pl.col(col).robust.nmad(k).alias(f"{col}_nmad")


def zscore(col: str, k: float = K_NMAD) -> pl.Expr:
    """Robust z-score of column ``col``, aliased ``"{col}_zscore"``."""
    return pl.col(col).robust.zscore(k).alias(f"{col}_zscore")
