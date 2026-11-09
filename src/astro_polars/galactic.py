"""Galactocentric positions and velocities, solar reflex motion and kinematics.

Everything here is a native Polars expression (no optional dependencies), so
it runs lazily, in parallel and on the streaming engine.

Units: degrees for angles, kpc for distances and positions, mas/yr for proper
motions (longitude component including ``cos(lat)``, like Gaia's ``pmra``),
km/s for velocities.

Galactocentric frame (the same as astropy's `Galactocentric`): right-handed,
the Galactic centre at the origin, the Sun at ``X ~ -R0`` (and ``Z = z0``),
``Y`` positive towards ``l = 90 deg`` (the direction of Galactic rotation at
the Sun) and ``Z`` towards the North Galactic Pole. In this frame the disc
rotates clockwise seen from the North Galactic Pole.

Sign conventions for derived quantities (see `kinematics`):

- ``vphi`` (alias ``vT``) is positive in the direction of Galactic rotation,
  so the Sun has ``vphi ~ +245 km/s``.
- Angular momenta are ``L = -(r x v)``, so that ``Lz = R * vphi`` is positive
  for disc (prograde) orbits. The same sign is used for the action ``Jphi``
  in `astro_polars.dynamics`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._utils import to_expr
from .coords import GALACTIC_TO_ICRS, ICRS_TO_GALACTIC

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "K_PM",
    "SolarParameters",
    "DEFAULT_SOLAR",
    "set_default_solar",
    "get_default_solar",
    "heliocentric_cartesian",
    "galactocentric",
    "from_galactocentric",
    "reflex_correct",
    "vgsr",
]

#: km/s per (kpc mas/yr): ``v_tan = K_PM * distance * mu``. Equal to 1 au/yr in km/s.
K_PM = 4.740470463533348


@dataclass(frozen=True)
class SolarParameters:
    """Position and velocity of the Sun relative to the Galactic centre.

    Attributes
    ----------
    R0 : float
        Sun to Galactic centre distance, kpc.
    z0 : float
        Height of the Sun above the Galactic mid-plane, kpc. With ``z0 != 0``
        the frame is tilted so that the Galactic centre is at the origin and
        the Sun at ``Z = z0`` (as in astropy).
    U, V, W : float
        Solar peculiar motion relative to the local standard of rest, km/s
        (``U`` towards the Galactic centre, ``V`` towards rotation, ``W``
        towards the North Galactic Pole).
    vcirc : float
        Circular velocity at the Sun (velocity of the local standard of rest), km/s.

    The defaults are R0 = 8.2 kpc and vcirc = 232.8 km/s from McMillan (2017,
    MNRAS 465, 76) and (U, V, W) = (11.1, 12.24, 7.25) km/s from Schoenrich,
    Binney & Dehnen (2010, MNRAS 403, 1829).
    """

    R0: float = 8.2
    z0: float = 0.0
    U: float = 11.1
    V: float = 12.24
    W: float = 7.25
    vcirc: float = 232.8

    @property
    def velocity(self) -> tuple[float, float, float]:
        """Velocity of the Sun in the Galactocentric frame, ``(vX, vY, vZ)`` in km/s."""
        return (self.U, self.vcirc + self.V, self.W)

    def replace(self, **changes) -> SolarParameters:
        """Copy with some parameters changed, e.g. ``DEFAULT_SOLAR.replace(R0=8.122)``."""
        return replace(self, **changes)

    # -- frame algebra ------------------------------------------------------
    def _tilt(self) -> np.ndarray:
        """Rotation about Y that puts the Sun at ``Z = z0`` (identity for ``z0 = 0``)."""
        s = self.z0 / self.R0
        c = np.sqrt(1.0 - s * s)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def _icrs_to_gc(self) -> np.ndarray:
        """Rotation from ICRS-aligned heliocentric axes to Galactocentric axes."""
        return self._tilt() @ ICRS_TO_GALACTIC

    def _sun_position(self) -> np.ndarray:
        return self._tilt() @ np.array([-self.R0, 0.0, 0.0])


#: Default solar parameters (see `SolarParameters`).
DEFAULT_SOLAR = SolarParameters()
_default = [DEFAULT_SOLAR]


def set_default_solar(solar: SolarParameters | None = None, **changes) -> SolarParameters:
    """Set the solar parameters used when a function gets ``solar=None``.

    ``set_default_solar(R0=8.122, vcirc=229.0)`` changes some values;
    ``set_default_solar(SolarParameters())`` restores the package defaults.
    Returns the new default.
    """
    base = solar if solar is not None else _default[0]
    _default[0] = base.replace(**changes) if changes else base
    return _default[0]


def get_default_solar() -> SolarParameters:
    """The solar parameters currently used by default."""
    return _default[0]


def _solar(solar: SolarParameters | None) -> SolarParameters:
    return _default[0] if solar is None else solar


# -- small vector helpers on expressions -------------------------------------
def _matvec(m, x, y, z):
    m = np.asarray(m, dtype=np.float64).tolist()
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


def _basis(lon: IntoExpr, lat: IntoExpr):
    """Unit vectors (radial, east, north) at ``(lon, lat)`` in degrees."""
    lo, la = to_expr(lon).radians(), to_expr(lat).radians()
    sa, ca, sd, cd = lo.sin(), lo.cos(), la.sin(), la.cos()
    r = (cd * ca, cd * sa, sd)
    e = (-sa, ca, pl.lit(0.0))
    n = (-sd * ca, -sd * sa, cd)
    return r, e, n


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


# -- public expressions --------------------------------------------------------
def heliocentric_cartesian(
    lon: IntoExpr,
    lat: IntoExpr,
    distance: IntoExpr,
    pmlon: IntoExpr | None = None,
    pmlat: IntoExpr | None = None,
    radial_velocity: IntoExpr | None = None,
) -> pl.Expr:
    """Heliocentric Cartesian position (and velocity) in the frame of ``lon, lat``.

    With Galactic ``l, b`` this gives ``x`` towards the Galactic centre, ``y``
    towards rotation and ``z`` to the North Galactic Pole. Returns
    ``struct{x, y, z}``, plus ``vx, vy, vz`` (km/s, relative to the Sun) when
    the proper motions and radial velocity are given.
    """
    r, e, n = _basis(lon, lat)
    d = to_expr(distance)
    fields = [(d * r[i]).alias(c) for i, c in enumerate("xyz")]
    if pmlon is not None and pmlat is not None and radial_velocity is not None:
        vr = to_expr(radial_velocity)
        vl, vb = K_PM * d * to_expr(pmlon), K_PM * d * to_expr(pmlat)
        fields += [(vr * r[i] + vl * e[i] + vb * n[i]).alias(f"v{c}") for i, c in enumerate("xyz")]
    return pl.struct(fields)


def galactocentric(
    ra: IntoExpr = "ra",
    dec: IntoExpr = "dec",
    distance: IntoExpr = "distance",
    pmra: IntoExpr | None = "pmra",
    pmdec: IntoExpr | None = "pmdec",
    radial_velocity: IntoExpr | None = "radial_velocity",
    solar: SolarParameters | None = None,
) -> pl.Expr:
    """Galactocentric Cartesian coordinates from ICRS observables.

    Returns ``struct{X, Y, Z, vX, vY, vZ}`` (kpc, km/s); only ``X, Y, Z``
    when ``pmra``, ``pmdec`` or ``radial_velocity`` is None. The distance is
    in kpc (``1 / parallax`` for a parallax in mas).

        lf.with_columns(galactocentric(distance=1 / pl.col("parallax")).struct.unnest())

    Matches astropy's ``Galactocentric`` with the same solar parameters and
    the Galactic centre at ``l = b = 0`` to better than 1e-6 relative.
    """
    sol = _solar(solar)
    m = sol._icrs_to_gc()
    r, e, n = _basis(ra, dec)
    d = to_expr(distance)
    pos = _matvec(m, *(d * r[i] for i in range(3)))
    sun = sol._sun_position()
    fields = [(pos[i] + float(sun[i])).alias(c) for i, c in enumerate("XYZ")]
    if pmra is not None and pmdec is not None and radial_velocity is not None:
        vr = to_expr(radial_velocity)
        va, vd = K_PM * d * to_expr(pmra), K_PM * d * to_expr(pmdec)
        vel = _matvec(m, *(vr * r[i] + va * e[i] + vd * n[i] for i in range(3)))
        vsun = sol.velocity
        fields += [(vel[i] + vsun[i]).alias(f"v{c}") for i, c in enumerate("XYZ")]
    return pl.struct(fields)


def from_galactocentric(
    X: IntoExpr = "X",
    Y: IntoExpr = "Y",
    Z: IntoExpr = "Z",
    vX: IntoExpr | None = "vX",
    vY: IntoExpr | None = "vY",
    vZ: IntoExpr | None = "vZ",
    solar: SolarParameters | None = None,
) -> pl.Expr:
    """ICRS observables from Galactocentric coordinates (inverse of `galactocentric`).

    Returns ``struct{ra, dec, distance, pmra, pmdec, radial_velocity}``
    (deg, kpc, mas/yr, km/s); only ``ra, dec, distance`` when a velocity is None.
    Useful for mock catalogues and orbit predictions.
    """
    sol = _solar(solar)
    minv = sol._icrs_to_gc().T
    sun = sol._sun_position()
    p = _matvec(minv, *(to_expr(c) - float(sun[i]) for i, c in enumerate((X, Y, Z))))
    d = (p[0] ** 2 + p[1] ** 2 + p[2] ** 2).sqrt()
    ra = pl.arctan2(p[1], p[0]).degrees() % 360.0
    dec = pl.arctan2(p[2], (p[0] ** 2 + p[1] ** 2).sqrt()).degrees()
    fields = [ra.alias("ra"), dec.alias("dec"), d.alias("distance")]
    if vX is not None and vY is not None and vZ is not None:
        vsun = sol.velocity
        v = _matvec(minv, *(to_expr(c) - vsun[i] for i, c in enumerate((vX, vY, vZ))))
        r, e, n = _basis(ra, dec)
        fields += [
            (_dot(e, v) / (K_PM * d)).alias("pmra"),
            (_dot(n, v) / (K_PM * d)).alias("pmdec"),
            _dot(r, v).alias("radial_velocity"),
        ]
    return pl.struct(fields)


def _sun_velocity_in(frame: str, sol: SolarParameters) -> np.ndarray:
    """Solar velocity (km/s) along the heliocentric axes of ``frame``."""
    v_gc = np.asarray(sol.velocity)
    v_gal = sol._tilt().T @ v_gc
    if frame == "galactic":
        return v_gal
    if frame == "icrs":
        return GALACTIC_TO_ICRS @ v_gal
    raise ValueError("frame must be 'icrs' or 'galactic'")


def reflex_correct(
    lon: IntoExpr = "ra",
    lat: IntoExpr = "dec",
    distance: IntoExpr = "distance",
    pmlon: IntoExpr = "pmra",
    pmlat: IntoExpr = "pmdec",
    solar: SolarParameters | None = None,
    frame: str = "icrs",
) -> pl.Expr:
    """Proper motions corrected for the solar reflex motion (Galactic standard of rest).

    Adds back the Sun's velocity projected on the sky, so a star at rest in
    the Galactocentric frame has zero corrected proper motion. ``frame`` says
    whether the inputs are ICRS (``ra, dec, pmra, pmdec``) or Galactic
    (``l, b, pml, pmb``). Returns ``struct{pmlon, pmlat}``; null where the
    distance is not positive and finite.

        lf.with_columns(reflex_correct().struct.rename_fields(["pmra_gsr", "pmdec_gsr"]).struct.unnest())

    Pass a constant ``distance`` (e.g. ``10.0``) for stream work where a
    single distance is assumed.
    """
    v = _sun_velocity_in(frame, _solar(solar)).tolist()
    _, e, n = _basis(lon, lat)
    d = to_expr(distance)
    ok = d.is_finite() & (d > 0)
    kd = K_PM * d
    return pl.struct(
        pl.when(ok).then(to_expr(pmlon) + _dot(e, v) / kd).alias("pmlon"),
        pl.when(ok).then(to_expr(pmlat) + _dot(n, v) / kd).alias("pmlat"),
    )


def vgsr(
    lon: IntoExpr = "ra",
    lat: IntoExpr = "dec",
    radial_velocity: IntoExpr = "radial_velocity",
    solar: SolarParameters | None = None,
    frame: str = "icrs",
) -> pl.Expr:
    """Line-of-sight velocity in the Galactic standard of rest (km/s).

    ``v_gsr = v_los + v_sun . r_hat``, with ``frame`` as in `reflex_correct`.
    """
    v = _sun_velocity_in(frame, _solar(solar)).tolist()
    r, _, _ = _basis(lon, lat)
    return to_expr(radial_velocity) + _dot(r, v)
