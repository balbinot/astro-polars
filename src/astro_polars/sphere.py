"""Angular separations and cone selections on the sphere.

All angles are in degrees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numba import njit

from ._utils import to_expr

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = ["angular_separation", "angsep", "in_cone"]


@njit(cache=True)
def angular_separation(lon1, lat1, lon2, lat2):
    """Angular separation between points on a sphere (Vincenty formula).

    Works on scalars and NumPy arrays (broadcasting as NumPy does). The
    Vincenty formula is stable at all distances, including the poles and
    antipodes ([great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance)).

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float or numpy.ndarray
        Longitudes and latitudes of the two sets of points, in degrees.

    Returns
    -------
    float or numpy.ndarray
        Separation in degrees.
    """
    d2r = np.deg2rad(1)
    r2d = np.rad2deg(1)

    sdlon = np.sin(d2r * lon2 - d2r * lon1)
    cdlon = np.cos(d2r * lon2 - d2r * lon1)
    slat1 = np.sin(d2r * lat1)
    slat2 = np.sin(d2r * lat2)
    clat1 = np.cos(d2r * lat1)
    clat2 = np.cos(d2r * lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return r2d * np.arctan2(np.sqrt(num1**2 + num2**2), denominator)


def angsep(lon1: IntoExpr, lat1: IntoExpr, lon2: IntoExpr, lat2: IntoExpr) -> pl.Expr:
    """Polars expression for the angular separation in degrees (Vincenty).

    Same formula as ``angular_separation``, evaluated natively by Polars
    so it is lazy, parallel and streams. Arguments are column names,
    expressions or numbers (for example a fixed position):

        df.with_columns(sep=angsep("ra", "dec", 150.1, 2.2))
    """
    lon1, lat1, lon2, lat2 = (to_expr(e) for e in (lon1, lat1, lon2, lat2))
    dlon = (lon2 - lon1).radians()
    phi1, phi2 = lat1.radians(), lat2.radians()
    sdlon, cdlon = dlon.sin(), dlon.cos()
    slat1, slat2 = phi1.sin(), phi2.sin()
    clat1, clat2 = phi1.cos(), phi2.cos()

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    den = slat1 * slat2 + clat1 * clat2 * cdlon
    return pl.arctan2((num1**2 + num2**2).sqrt(), den).degrees()


def in_cone(lon: IntoExpr, lat: IntoExpr, lon0: float, lat0: float, radius: float) -> pl.Expr:
    """Boolean expression: is the point within ``radius`` degrees of (lon0, lat0)?

    A cheap latitude pre-cut is applied first so most rows skip the
    trigonometry:

        lf.filter(in_cone("ra", "dec", 10.68, 41.27, radius=0.5))
    """
    lat_e = to_expr(lat)
    return lat_e.is_between(lat0 - radius, lat0 + radius) & (angsep(lon, lat, lon0, lat0) <= radius)
