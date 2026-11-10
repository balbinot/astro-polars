"""Sky rotations between frames (equatorial, Galactic, stream frames), with proper motions.

Everything is a native Polars expression, so it is lazy, parallel and works
on the streaming engine. Angles are in degrees, proper motions in any unit
(mas/yr typically) with the longitude component including ``cos(lat)``
(Gaia's ``pmra`` convention).

A frame is a 3x3 rotation matrix ``M`` taking unit vectors from the input
frame to the output frame: ``r_out = M @ r_in``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import polars as pl

from ._utils import to_expr

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "ICRS_TO_GALACTIC",
    "GALACTIC_TO_ICRS",
    "GD1_KOPOSOV10",
    "frame_from_pole",
    "rotate",
    "rotate_pm",
    "rotate_pm_errors",
    "lonlat_to_xyz",
]

#: ICRS -> Galactic rotation (Hipparcos catalogue, ESA 1997, Vol. 1, eq. 1.5.11).
#: This is the definition Gaia uses for its ``l, b``; astropy's ``Galactic``
#: frame goes through FK5 and differs from it by about 25 mas.
ICRS_TO_GALACTIC = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
    ]
)
#: Galactic -> ICRS (transpose of ``ICRS_TO_GALACTIC``).
GALACTIC_TO_ICRS = ICRS_TO_GALACTIC.T.copy()

#: ICRS -> GD-1 stream frame (phi1, phi2) of Koposov et al. (2010, ApJ 712, 260).
GD1_KOPOSOV10 = np.array(
    [
        [-0.4776303088, -0.1738432154, +0.8611897727],
        [+0.510844589, -0.8524449229, +0.111245042],
        [+0.7147776536, +0.4930681392, +0.4959603976],
    ]
)


def _unit(lon_deg: float, lat_deg: float) -> np.ndarray:
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def frame_from_pole(
    lon_pole: float, lat_pole: float, lon_origin: float, lat_origin: float
) -> np.ndarray:
    """Rotation matrix to a frame with a given pole and longitude origin.

    In the new frame, latitude is 90 deg at ``(lon_pole, lat_pole)`` and
    longitude 0 lies on the great circle through the pole and
    ``(lon_origin, lat_origin)``. This is how stream frames (phi1, phi2) are
    usually defined: the pole of the stream's great circle and its centre.

    Returns ``M`` with ``r_new = M @ r_in``.
    """
    z = _unit(lon_pole, lat_pole)
    o = _unit(lon_origin, lat_origin)
    x = o - (o @ z) * z
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise ValueError("origin coincides with the pole")
    x = x / norm
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _as_matrix(matrix) -> list[list[float]]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError("matrix must be 3x3")
    return m.tolist()


def lonlat_to_xyz(lon: IntoExpr, lat: IntoExpr, distance: IntoExpr = 1.0) -> pl.Expr:
    """Cartesian ``struct{x, y, z}`` from lon, lat (deg) and distance."""
    lo, la, d = to_expr(lon).radians(), to_expr(lat).radians(), to_expr(distance)
    return pl.struct(
        (d * la.cos() * lo.cos()).alias("x"),
        (d * la.cos() * lo.sin()).alias("y"),
        (d * la.sin()).alias("z"),
    )


def _rotate_vec(m, x, y, z):
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


def _lonlat_from_xyz(x, y, z, wrap180: bool):
    lon = pl.arctan2(y, x).degrees()
    lon = lon if wrap180 else lon % 360.0
    lat = pl.arctan2(z, (x * x + y * y).sqrt()).degrees()
    return lon, lat


def rotate(lon: IntoExpr, lat: IntoExpr, matrix, wrap180: bool = False) -> pl.Expr:
    """Rotate positions. Returns ``struct{lon, lat}`` in degrees.

    Longitude is in [0, 360), or (-180, 180] with ``wrap180=True``.

        lf.with_columns(rotate("ra", "dec", ICRS_TO_GALACTIC).struct.unnest())
    """
    m = _as_matrix(matrix)
    lo, la = to_expr(lon).radians(), to_expr(lat).radians()
    cl = la.cos()
    x, y, z = cl * lo.cos(), cl * lo.sin(), la.sin()
    xr, yr, zr = _rotate_vec(m, x, y, z)
    lon_o, lat_o = _lonlat_from_xyz(xr, yr, zr, wrap180)
    return pl.struct(lon_o.alias("lon"), lat_o.alias("lat"))


def _pm_coefficients(lon: IntoExpr, lat: IntoExpr, matrix, wrap180: bool):
    """Expressions for the output position and the 2x2 proper-motion transform.

    ``(pmlon_out, pmlat_out) = [[a, b], [c, d]] @ (pmlon_in, pmlat_in)``.
    """
    m = _as_matrix(matrix)
    lo, la = to_expr(lon).radians(), to_expr(lat).radians()
    sa, ca, sd, cd = lo.sin(), lo.cos(), la.sin(), la.cos()

    # position and local tangent basis (east, north) in the input frame
    x, y, z = cd * ca, cd * sa, sd
    e_lon = (-sa, ca, pl.lit(0.0))
    e_lat = (-sd * ca, -sd * sa, cd)

    xr, yr, zr = _rotate_vec(m, x, y, z)
    lon_o, lat_o = _lonlat_from_xyz(xr, yr, zr, wrap180)

    # tangent basis at the output position, in the output frame
    lo2, la2 = pl.arctan2(yr, xr), pl.arctan2(zr, (xr * xr + yr * yr).sqrt())
    s2, c2, sb, cb = lo2.sin(), lo2.cos(), la2.sin(), la2.cos()
    f_lon = (-s2, c2, pl.lit(0.0))
    f_lat = (-sb * c2, -sb * s2, cb)

    ge_lon = _rotate_vec(m, *e_lon)
    ge_lat = _rotate_vec(m, *e_lat)

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    a, b = dot(f_lon, ge_lon), dot(f_lon, ge_lat)
    c, d = dot(f_lat, ge_lon), dot(f_lat, ge_lat)
    return lon_o, lat_o, a, b, c, d


def rotate_pm(
    lon: IntoExpr,
    lat: IntoExpr,
    pmlon: IntoExpr,
    pmlat: IntoExpr,
    matrix,
    wrap180: bool = False,
) -> pl.Expr:
    """Rotate positions and proper motions. Returns ``struct{lon, lat, pmlon, pmlat}``.

    ``pmlon`` must include the ``cos(lat)`` factor (like Gaia's ``pmra``); the
    output ``pmlon`` does too.
    """
    lon_o, lat_o, a, b, c, d = _pm_coefficients(lon, lat, matrix, wrap180)
    mu_l, mu_b = to_expr(pmlon), to_expr(pmlat)
    return pl.struct(
        lon_o.alias("lon"),
        lat_o.alias("lat"),
        (a * mu_l + b * mu_b).alias("pmlon"),
        (c * mu_l + d * mu_b).alias("pmlat"),
    )


def rotate_pm_errors(
    lon: IntoExpr,
    lat: IntoExpr,
    pmlon_error: IntoExpr,
    pmlat_error: IntoExpr,
    pm_corr: IntoExpr | None,
    matrix,
) -> pl.Expr:
    """Propagate proper-motion uncertainties through a rotation.

    The 2x2 covariance is transformed exactly (``C' = J C J^T``). Returns
    ``struct{pmlon_error, pmlat_error, pm_corr}``. Pass ``pm_corr=None`` for
    uncorrelated input errors.
    """
    _, _, a, b, c, d = _pm_coefficients(lon, lat, matrix, False)
    s1, s2 = to_expr(pmlon_error), to_expr(pmlat_error)
    rho = pl.lit(0.0) if pm_corr is None else to_expr(pm_corr)
    c11, c22, c12 = s1 * s1, s2 * s2, rho * s1 * s2
    v1 = a * a * c11 + b * b * c22 + 2 * a * b * c12
    v2 = c * c * c11 + d * d * c22 + 2 * c * d * c12
    cov = a * c * c11 + b * d * c22 + (a * d + b * c) * c12
    return pl.struct(
        v1.sqrt().alias("pmlon_error"),
        v2.sqrt().alias("pmlat_error"),
        (cov / (v1 * v2).sqrt()).alias("pm_corr"),
    )


def _add_rotated(
    df,
    matrix,
    lon: str,
    lat: str,
    out: Sequence[str],
    pmlon: str | None,
    pmlat: str | None,
    wrap180: bool,
):
    names = df.collect_schema().names()
    have_pm = pmlon is not None and pmlat is not None and pmlon in names and pmlat in names
    if have_pm:
        s = rotate_pm(lon, lat, pmlon, pmlat, matrix, wrap180)
        fields = ["lon", "lat", "pmlon", "pmlat"]
    else:
        s = rotate(lon, lat, matrix, wrap180)
        fields = ["lon", "lat"]
    tmp = "__ap_rot"
    return (
        df.with_columns(s.alias(tmp))
        .with_columns(pl.col(tmp).struct.field(f).alias(o) for f, o in zip(fields, out))
        .drop(tmp)
    )
