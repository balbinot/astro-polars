"""``df.sky`` / ``lf.sky``: the main entry point on DataFrames and LazyFrames."""

from __future__ import annotations

from typing import Sequence

import polars as pl

from . import coords
from .crossmatch import crossmatch, remove_duplicates
from .healpix import healpix_index
from .polygon import in_polygon
from .sphere import angsep, in_cone


@pl.api.register_dataframe_namespace("sky")
@pl.api.register_lazyframe_namespace("sky")
class Sky:
    """Astronomy helpers on a DataFrame or LazyFrame. Angles are in degrees."""

    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        self._df = df

    # -- selections ---------------------------------------------------------
    def cone(self, lon0: float, lat0: float, radius: float, lon: str = "ra", lat: str = "dec"):
        """Rows within ``radius`` degrees of ``(lon0, lat0)``."""
        return self._df.filter(in_cone(lon, lat, lon0, lat0, radius))

    def polygon(self, polygon, x: str = "x", y: str = "y"):
        """Rows inside ``polygon`` (an ``(N, 2)`` array) in the ``x``-``y`` plane."""
        return self._df.filter(in_polygon(polygon, x, y))

    # -- new columns ----------------------------------------------------------
    def separation(
        self, lon0: float, lat0: float, lon: str = "ra", lat: str = "dec", name: str = "sep"
    ):
        """Add the angular separation (degrees) from ``(lon0, lat0)`` as ``name``."""
        return self._df.with_columns(angsep(lon, lat, lon0, lat0).alias(name))

    def add_healpix(
        self,
        nside: int,
        lon: str = "ra",
        lat: str = "dec",
        nest: bool = True,
        name: str | None = None,
    ):
        """Add the HEALPix index at ``nside`` as ``name`` (default ``hpx{nside}``)."""
        return self._df.with_columns(
            healpix_index(lon, lat, nside, nest=nest).alias(name or f"hpx{nside}")
        )

    def to_galactic(
        self,
        ra: str = "ra",
        dec: str = "dec",
        pmra: str | None = "pmra",
        pmdec: str | None = "pmdec",
        out: Sequence[str] = ("l", "b", "pml", "pmb"),
        wrap180: bool = False,
    ):
        """Add Galactic ``l, b`` (and ``pml, pmb`` if the proper-motion columns exist)."""
        return coords._add_rotated(
            self._df, coords.ICRS_TO_GALACTIC, ra, dec, out, pmra, pmdec, wrap180
        )

    def to_equatorial(
        self,
        l: str = "l",  # noqa: E741
        b: str = "b",
        pml: str | None = "pml",
        pmb: str | None = "pmb",
        out: Sequence[str] = ("ra", "dec", "pmra", "pmdec"),
        wrap180: bool = False,
    ):
        """Add ICRS ``ra, dec`` (and ``pmra, pmdec``) from Galactic coordinates."""
        return coords._add_rotated(self._df, coords.GALACTIC_TO_ICRS, l, b, out, pml, pmb, wrap180)

    def to_frame(
        self,
        matrix,
        lon: str = "ra",
        lat: str = "dec",
        pmlon: str | None = "pmra",
        pmlat: str | None = "pmdec",
        out: Sequence[str] = ("phi1", "phi2", "pm_phi1", "pm_phi2"),
        wrap180: bool = True,
    ):
        """Rotate into any frame given by a 3x3 matrix, e.g. a stream frame.

        See :func:`astro_polars.coords.frame_from_pole` and
        :data:`astro_polars.coords.GD1_KOPOSOV10`. Longitudes are wrapped to
        (-180, 180] by default, as is usual for stream coordinates.
        """
        return coords._add_rotated(self._df, matrix, lon, lat, out, pmlon, pmlat, wrap180)

    # -- catalogue operations -----------------------------------------------
    def crossmatch(self, other: pl.DataFrame | pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """Cross-match against ``other``; see :func:`astro_polars.crossmatch`."""
        return crossmatch(self._df, other, **kwargs)

    def remove_duplicates(
        self, x: str = "ra", y: str = "dec", radius_arcsec: float = 1.0
    ) -> pl.DataFrame:
        """Drop sources with a neighbour closer than ``radius_arcsec``; see
        :func:`astro_polars.remove_duplicates`."""
        return remove_duplicates(self._df, x=x, y=y, radius_arcsec=radius_arcsec)
