"""Point-in-polygon selection.

Eduardo Balbinot's ray-casting test, compiled with numba and run in parallel
(https://gist.github.com/balbinot/b8c6b18000363b16825c04d2d84146df). Points
exactly on an edge or a vertex count as inside.

Polygons are ``(N, 2)`` arrays of ``(x, y)`` vertices, given open or closed
(first vertex repeated at the end).
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl
from numba import njit

__all__ = ["inpoly", "inpoly_parallel", "in_polygon", "pinside", "close_polygon"]


@njit(cache=True)
def inpoly(polygon, x, y):
    """Ray-casting test for one point. Returns 0 (outside), 1 (inside) or 2 (on edge).

    ``polygon`` must be closed (first vertex repeated at the end); use
    :func:`close_polygon` if unsure.
    """
    length = len(polygon) - 1
    dy2 = y - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1
    while ii < length:
        dy = dy2
        dy2 = y - polygon[jj][1]
        # consider only lines which are not completely above/below/right from the point
        if dy * dy2 <= 0.0 and (x >= polygon[ii][0] or x >= polygon[jj][0]):
            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]
                if x > F:  # line is left of the point: a ray moving left intersects it
                    intersections += 1
                elif x == F:  # point on line
                    return 2
            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                x == polygon[jj][0]
                or (dy == 0 and (x - polygon[ii][0]) * (x - polygon[jj][0]) <= 0)
            ):
                return 2
        ii = jj
        jj += 1
    return intersections & 1


@njit(parallel=True, cache=True)
def inpoly_parallel(x, y, polygon):
    """Vectorised :func:`inpoly` over arrays ``x`` and ``y``. Returns a bool array."""
    ln = len(x)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = inpoly(polygon, x[i], y[i])
    return D


def close_polygon(polygon) -> np.ndarray:
    """Return ``polygon`` as a float64 ``(N, 2)`` array whose last vertex equals the first."""
    poly = np.ascontiguousarray(np.asarray(polygon, dtype=np.float64))
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("polygon must be an (N, 2) array with N >= 3")
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[:1]])
    return poly


def in_polygon(polygon, x: str = "x", y: str = "y") -> pl.Expr:
    """Boolean expression: is ``(x, y)`` inside ``polygon``? (numba backend)

    ``x`` and ``y`` are column names. Works on DataFrames and LazyFrames,
    including streaming::

        lf.filter(in_polygon(poly, "g_r", "g"))
    """
    poly = close_polygon(polygon)

    def _batch(s: pl.Series) -> pl.Series:
        xs = s.struct.field(x).cast(pl.Float64).to_numpy()
        ys = s.struct.field(y).cast(pl.Float64).to_numpy()
        return pl.Series(inpoly_parallel(xs, ys, poly), dtype=pl.Boolean)

    return pl.struct(x, y).map_batches(_batch, return_dtype=pl.Boolean, is_elementwise=True)


def pinside(df, poly, x: str = "x", y: str = "y", name: str = "inside") -> pl.Expr:
    """Eduardo's original interface: expression named ``name`` flagging points in ``poly``.

    ``df`` is accepted for backwards compatibility and is not used. Prefer
    :func:`in_polygon`.
    """
    return in_polygon(poly, x, y).alias(name)
