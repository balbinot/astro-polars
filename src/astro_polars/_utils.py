"""Internal helpers: expression parsing, optional imports, streaming collect."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars._typing import IntoExpr

# pip extra that provides each optional module
_EXTRAS = {
    "matplotlib": "viz",
    "cycler": "viz",
    "skyproj": "viz",
    "astropy": "fits",
    "agama": "dynamics",
    "zero_point": "gaia",
}


def require(module: str, extra: str | None = None):
    """Import an optional dependency, or raise an ImportError that says how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        extra = extra or _EXTRAS.get(module.split(".")[0])
        hint = f'pip install "astro-polars[{extra}]"' if extra else f"pip install {module}"
        raise ImportError(f"{module} is needed for this feature: {hint}") from err


def to_expr(value: IntoExpr) -> pl.Expr:
    """Column name -> ``pl.col``, expression -> itself, anything else -> ``pl.lit``."""
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str):
        return pl.col(value)
    return pl.lit(value)


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect with the streaming engine (works across Polars versions)."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # very old polars
        return lf.collect(streaming=True)
