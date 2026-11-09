"""Galactic dynamics on Polars frames: kinematics, energies, actions, orbits' extent.

The entry point is the ``df.galactic`` / ``lf.galactic`` namespace:

    import astro_polars  # registers the namespaces

    lf = pl.scan_parquet("gaia_rvs.parquet")
    lf = lf.with_columns(distance=1 / pl.col("parallax"))
    out = lf.galactic.calc(["R", "vphi", "Lz", "E", "JR", "Jz", "Jphi", "ecc"]).collect()

`calc` works out which intermediate columns it needs (Galactocentric
positions and velocities from ``ra, dec, distance, pmra, pmdec,
radial_velocity``, then energies, actions, ...), adds them in dependency order
and drops the internal helper columns. ``astro_polars.dynamics.available()``
lists every quantity.

Purely kinematic quantities are plain Polars expressions. Quantities that
need a gravitational potential (energies, actions, angles, frequencies,
pericentre and apocentre, circular orbits) use
[AGAMA](https://github.com/GalacticDynamics-Oxford/Agama), an optional
dependency (``pip install agama``). AGAMA runs in units of kpc, km/s and
solar masses, so energies are in (km/s)^2 and actions in kpc km/s.

Sign conventions are those of `astro_polars.galactic`: ``vphi`` and ``Lz``
are positive for prograde (disc-like) orbits. The azimuthal action, angle and
frequency (``Jphi``, ``thetaphi``, ``Omegaphi``) use the same orientation, so
``Jphi = Lz`` in an axisymmetric potential. AGAMA's own ``Jphi`` has the
opposite sign in this frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import polars as pl

from . import coords, galactic
from ._utils import require
from .galactic import SolarParameters

__all__ = [
    "DEFAULT_POTENTIAL",
    "GalaxyModel",
    "load_potential",
    "get_model",
    "set_default_potential",
    "available",
    "calc_exprs",
    "Galactic",
    "ALIASES",
]

#: AGAMA potential used when none is given (Cautun et al. 2020, MNRAS 494, 4291).
DEFAULT_POTENTIAL = "Cautun20"
_default_potential = [DEFAULT_POTENTIAL]

#: Column names of the Galactocentric phase space used by all dynamics quantities.
PHASE_SPACE = ("X", "Y", "Z", "vX", "vY", "vZ")


# =============================================================================
# AGAMA backend
# =============================================================================
def _agama():
    # keep AGAMA quiet unless the user asked for its log
    os.environ.setdefault("LOGLEVEL", "0")
    agama = require("agama")
    return agama


def _potential_search_path() -> list[Path]:
    paths = []
    if os.environ.get("AGAMA_DATA"):
        paths.append(Path(os.environ["AGAMA_DATA"]))
    paths.append(Path.home() / ".config" / "agama")
    try:
        import agama  # noqa: F401

        root = Path(agama.__file__).resolve().parent
        paths += [root / "data", root.parent / "data", root.parent / "agama" / "data"]
    except ImportError:
        pass
    return paths


def load_potential(potential=None):
    """Return an ``agama.Potential``.

    ``potential`` can be an ``agama.Potential`` (returned as is), a path to an
    ``.ini`` file, or the name of a potential shipped with AGAMA (e.g.
    ``"Cautun20"``, ``"McMillan17"``, ``"MWPotential2014"``). Names are
    looked up in ``$AGAMA_DATA``, ``~/.config/agama`` and AGAMA's ``data``
    folder. ``None`` means the current default (`set_default_potential`).

    Before creating a potential from a file, AGAMA's units are set to kpc,
    km/s and solar masses (``agama.setUnits(length=1, velocity=1, mass=1)``),
    the units of the shipped ``.ini`` files.
    """
    if potential is None:
        potential = _default_potential[0]
    if not isinstance(potential, (str, Path)):
        return potential
    agama = _agama()
    name = str(potential)
    candidates = [Path(name)]
    fname = name if name.endswith(".ini") else name + ".ini"
    candidates += [p / fname for p in _potential_search_path()]
    for path in candidates:
        if path.is_file():
            agama.setUnits(length=1, velocity=1, mass=1)
            return agama.Potential(str(path))
    raise FileNotFoundError(
        f"no AGAMA potential {name!r}; looked in: " + ", ".join(str(c) for c in candidates)
    )


@dataclass
class GalaxyModel:
    """A potential plus the cached helpers built from it (action finder, circular orbits).

    Parameters
    ----------
    potential
        Anything `load_potential` accepts, or any object with AGAMA's
        ``Potential`` methods (``potential``, ``force``, ``Rcirc``,
        ``Tcirc``, ``Rperiapo``).
    interp : bool
        Use AGAMA's interpolated action finder (faster, slightly less accurate).
    action_finder
        Optional ready-made action finder (callable like ``agama.ActionFinder``).
    """

    potential: object = None
    interp: bool = False
    action_finder: object = None
    _circular: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.potential = load_potential(self.potential)

    # -- helpers --------------------------------------------------------------
    def finder(self):
        if self.action_finder is None:
            self.action_finder = _agama().ActionFinder(self.potential, interp=self.interp)
        return self.action_finder

    def circular_table(self, rmin: float = 1e-3, rmax: float = 1e3, n: int = 2000):
        """Energy and angular momentum of circular orbits in the plane, on a log grid in R."""
        if self._circular is None:
            R = np.geomspace(rmin, rmax, n)
            xyz = np.column_stack([R, np.zeros_like(R), np.zeros_like(R)])
            vc = np.sqrt(-R * np.asarray(self.potential.force(xyz))[:, 0])
            E = np.asarray(self.potential.potential(xyz)) + 0.5 * vc**2
            self._circular = (E, R * vc)
        return self._circular

    # -- numpy kernels (N, k) -> arrays ------------------------------------------
    def epot(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(self.potential.potential(xyz), dtype=np.float64)

    def actions(self, xv: np.ndarray, angles: bool = False) -> dict[str, np.ndarray]:
        """Actions (and angles, frequencies) in the sign convention of this module."""
        out = {}
        if angles:
            J, theta, omega = self.finder()(xv, actions=True, angles=True, frequencies=True)
        else:
            J = self.finder()(xv)
        J = np.asarray(J, dtype=np.float64).reshape(-1, 3)
        out.update(JR=J[:, 0], Jz=J[:, 1], Jphi=-J[:, 2])
        if angles:
            theta = np.asarray(theta, dtype=np.float64).reshape(-1, 3)
            omega = np.asarray(omega, dtype=np.float64).reshape(-1, 3)
            out.update(
                thetaR=theta[:, 0],
                thetaz=theta[:, 1],
                thetaphi=np.mod(-theta[:, 2], 2 * np.pi),
                OmegaR=omega[:, 0],
                Omegaz=omega[:, 1],
                Omegaphi=-omega[:, 2],
            )
        return out

    def peri_apo(self, EL: np.ndarray) -> dict[str, np.ndarray]:
        pa = np.asarray(self.potential.Rperiapo(EL), dtype=np.float64).reshape(-1, 2)
        return {"peri": pa[:, 0], "apo": pa[:, 1]}

    def rcirc_E(self, E: np.ndarray) -> np.ndarray:
        return np.asarray(self.potential.Rcirc(E=E), dtype=np.float64)

    def rcirc_L(self, L: np.ndarray) -> np.ndarray:
        return np.asarray(self.potential.Rcirc(L=L), dtype=np.float64)

    def tcirc(self, E: np.ndarray) -> np.ndarray:
        return np.asarray(self.potential.Tcirc(E), dtype=np.float64)

    def lcirc(self, E: np.ndarray) -> np.ndarray:
        Ec, Lc = self.circular_table()
        out = np.interp(E, Ec, Lc, left=np.nan, right=np.nan)
        return np.where(np.isfinite(E), out, np.nan)


_models: dict[tuple, GalaxyModel] = {}


def get_model(potential=None, interp: bool = False) -> GalaxyModel:
    """A cached `GalaxyModel` for ``potential`` (a name, path, model or potential object)."""
    if isinstance(potential, GalaxyModel):
        return potential
    key_pot = _default_potential[0] if potential is None else potential
    key = (str(key_pot) if isinstance(key_pot, (str, Path)) else id(key_pot), interp)
    if key not in _models:
        _models[key] = GalaxyModel(key_pot, interp=interp)
    return _models[key]


def set_default_potential(potential) -> None:
    """Set the potential used when ``potential=None`` (a name, path or object)."""
    _default_potential[0] = potential


# =============================================================================
# Quantity registry
# =============================================================================
@dataclass(frozen=True)
class _Quantity:
    name: str
    requires: tuple[str, ...]
    build: Callable[[_Context], pl.Expr]
    doc: str
    potential: bool = False


_REGISTRY: dict[str, _Quantity] = {}

#: Alternative names accepted by `calc` (names used in the original polars_astro).
ALIASES = {
    "U": "Epot",
    "ke": "Ekin",
    "En": "E",
    "vT": "vphi",
    "Lcirc_E": "Lcirc",
    "AR": "thetaR",
    "Az": "thetaz",
    "Aphi": "thetaphi",
    "FreqR": "OmegaR",
    "Freqz": "Omegaz",
    "Freqphi": "Omegaphi",
    "distance_mod_mag": "distance_modulus",
}


class _Context:
    def __init__(self, solar, potential, interp, columns, full_angles):
        self.solar = galactic._solar(solar)
        self._potential = potential
        self._interp = interp
        self._columns = columns
        self.full_angles = full_angles

    def col(self, name: str) -> pl.Expr:
        return pl.col(self._columns.get(name, name))

    @property
    def model(self) -> GalaxyModel:
        return get_model(self._potential, self._interp)


def _quantity(name: str, requires: Sequence[str], potential: bool = False):
    def deco(fn):
        _REGISTRY[name] = _Quantity(
            name, tuple(requires), fn, (fn.__doc__ or "").strip(), potential
        )
        return fn

    return deco


def _numpy_map(
    ctx: _Context, cols: Sequence[str], fn: Callable[[np.ndarray], object], dtype
) -> pl.Expr:
    """Apply ``fn`` to an ``(N, len(cols))`` float array built from columns ``cols``.

    ``fn`` returns a 1-D array, or a dict of 1-D arrays for a struct ``dtype``.
    """

    def run(s: pl.Series) -> pl.Series:
        arr = np.column_stack(
            [s.struct.field(c).cast(pl.Float64).to_numpy() for c in s.struct.fields]
        )
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        if len(arr) == 0:
            return pl.Series(s.name, [], dtype=dtype)
        res = fn(arr)
        if isinstance(res, dict):
            return pl.DataFrame(res).to_struct(s.name)
        return pl.Series(s.name, np.asarray(res, dtype=np.float64))

    inputs = [ctx.col(c).alias(f"c{i}") for i, c in enumerate(cols)]
    return pl.struct(inputs).map_batches(run, return_dtype=dtype, is_elementwise=True)


def _f(name: str) -> pl.Expr:
    return pl.col(name)


# -- observables -> Galactocentric ----------------------------------------------
@_quantity("distance", ["parallax"])
def _distance(ctx):
    """Heliocentric distance (kpc) as 1/parallax (parallax in mas)."""
    return 1.0 / ctx.col("parallax")


@_quantity("_lb", ["ra", "dec", "pmra", "pmdec"])
def _lb(ctx):
    """Galactic l, b, pml, pmb (struct)."""
    return coords.rotate_pm(
        ctx.col("ra"), ctx.col("dec"), ctx.col("pmra"), ctx.col("pmdec"), coords.ICRS_TO_GALACTIC
    )


@_quantity("l", ["_lb"])
def _l(ctx):
    """Galactic longitude (deg)."""
    return _f("_lb").struct.field("lon")


@_quantity("b", ["_lb"])
def _b(ctx):
    """Galactic latitude (deg)."""
    return _f("_lb").struct.field("lat")


@_quantity("pml", ["_lb"])
def _pml(ctx):
    """Proper motion in Galactic longitude, including cos(b) (mas/yr)."""
    return _f("_lb").struct.field("pmlon")


@_quantity("pmb", ["_lb"])
def _pmb(ctx):
    """Proper motion in Galactic latitude (mas/yr)."""
    return _f("_lb").struct.field("pmlat")


@_quantity("_gcpos", ["ra", "dec", "distance"])
def _gcpos(ctx):
    """Galactocentric X, Y, Z (struct)."""
    return galactic.galactocentric(
        ctx.col("ra"), ctx.col("dec"), ctx.col("distance"), None, None, None, ctx.solar
    )


@_quantity("_gcvel", ["ra", "dec", "distance", "pmra", "pmdec", "radial_velocity"])
def _gcvel(ctx):
    """Galactocentric X, Y, Z, vX, vY, vZ (struct)."""
    return galactic.galactocentric(
        ctx.col("ra"),
        ctx.col("dec"),
        ctx.col("distance"),
        ctx.col("pmra"),
        ctx.col("pmdec"),
        ctx.col("radial_velocity"),
        ctx.solar,
    )


def _field_of(struct_name: str, field_name: str) -> Callable[[_Context], pl.Expr]:
    return lambda ctx: _f(struct_name).struct.field(field_name)


for _c in ("X", "Y", "Z"):
    _REGISTRY[_c] = _Quantity(
        _c, ("_gcpos",), _field_of("_gcpos", _c), f"Galactocentric {_c} (kpc)."
    )
for _c in ("vX", "vY", "vZ"):
    _REGISTRY[_c] = _Quantity(
        _c, ("_gcvel",), _field_of("_gcvel", _c), f"Galactocentric {_c} (km/s)."
    )

_X, _Y, _Z, _vX, _vY, _vZ = (pl.col(c) for c in PHASE_SPACE)


# -- spherical and cylindrical -------------------------------------------------------
@_quantity("r", ["X", "Y", "Z"])
def _r(ctx):
    """Galactocentric spherical radius (kpc)."""
    return (_X**2 + _Y**2 + _Z**2).sqrt()


@_quantity("R", ["X", "Y"])
def _R(ctx):
    """Galactocentric cylindrical radius (kpc)."""
    return (_X**2 + _Y**2).sqrt()


@_quantity("phi", ["X", "Y"])
def _phi(ctx):
    """Galactocentric azimuth atan2(Y, X) in degrees (the Sun is at 180)."""
    return pl.arctan2(_Y, _X).degrees()


@_quantity("vr", ["X", "Y", "Z", "vX", "vY", "vZ", "r"])
def _vr(ctx):
    """Galactocentric spherical radial velocity (km/s)."""
    return (_X * _vX + _Y * _vY + _Z * _vZ) / _f("r")


@_quantity("vR", ["X", "Y", "vX", "vY", "R"])
def _vR(ctx):
    """Galactocentric cylindrical radial velocity (km/s)."""
    return (_X * _vX + _Y * _vY) / _f("R")


@_quantity("vphi", ["X", "Y", "vX", "vY", "R"])
def _vphi(ctx):
    """Azimuthal velocity, positive in the direction of Galactic rotation (km/s)."""
    return (_Y * _vX - _X * _vY) / _f("R")


@_quantity("vtheta", ["Z", "vX", "vY", "vZ", "X", "Y", "r", "R"])
def _vtheta(ctx):
    """Polar velocity, positive towards the South Galactic Pole (km/s)."""
    return (_Z * (_X * _vX + _Y * _vY) / _f("R") - _f("R") * _vZ) / _f("r")


@_quantity("vt", ["vtheta", "vphi"])
def _vt(ctx):
    """Galactocentric tangential speed sqrt(vtheta^2 + vphi^2) (km/s)."""
    return (_f("vtheta") ** 2 + _f("vphi") ** 2).sqrt()


@_quantity("v_toomre", ["vphi", "vR", "vZ"])
def _v_toomre(ctx):
    """Speed relative to the local standard of rest, sqrt((vphi - vcirc)^2 + vR^2 + vZ^2) (km/s)."""
    return ((_f("vphi") - ctx.solar.vcirc) ** 2 + _f("vR") ** 2 + _vZ**2).sqrt()


# -- on the sky ----------------------------------------------------------------------
@_quantity("v_l", ["distance", "pml"])
def _v_l(ctx):
    """Heliocentric tangential velocity along l (km/s)."""
    return galactic.K_PM * ctx.col("distance") * _f("pml")


@_quantity("v_b", ["distance", "pmb"])
def _v_b(ctx):
    """Heliocentric tangential velocity along b (km/s)."""
    return galactic.K_PM * ctx.col("distance") * _f("pmb")


@_quantity("vtan", ["v_l", "v_b"])
def _vtan(ctx):
    """Heliocentric tangential speed (km/s)."""
    return (_f("v_l") ** 2 + _f("v_b") ** 2).sqrt()


@_quantity("distance_modulus", ["distance"])
def _dm(ctx):
    """Distance modulus 5 log10(d / 10 pc) (mag)."""
    return 5.0 * ctx.col("distance").log10() + 10.0


# -- angular momentum and energy -------------------------------------------------------
@_quantity("Lx", ["Y", "Z", "vY", "vZ"])
def _Lx(ctx):
    """Angular momentum x component, L = -(r x v) (kpc km/s)."""
    return _Z * _vY - _Y * _vZ


@_quantity("Ly", ["X", "Z", "vX", "vZ"])
def _Ly(ctx):
    """Angular momentum y component (kpc km/s)."""
    return _X * _vZ - _Z * _vX


@_quantity("Lz", ["X", "Y", "vX", "vY"])
def _Lz(ctx):
    """Angular momentum z component, R * vphi, positive for prograde orbits (kpc km/s)."""
    return _Y * _vX - _X * _vY


@_quantity("L", ["Lx", "Ly", "Lz"])
def _L(ctx):
    """Total angular momentum (kpc km/s)."""
    return (_f("Lx") ** 2 + _f("Ly") ** 2 + _f("Lz") ** 2).sqrt()


@_quantity("Lp", ["Lx", "Ly"])
def _Lp(ctx):
    """Angular momentum perpendicular to Z, sqrt(Lx^2 + Ly^2) (kpc km/s)."""
    return (_f("Lx") ** 2 + _f("Ly") ** 2).sqrt()


@_quantity("Ekin", ["vX", "vY", "vZ"])
def _Ekin(ctx):
    """Kinetic energy per unit mass (km/s)^2."""
    return 0.5 * (_vX**2 + _vY**2 + _vZ**2)


@_quantity("Epot", ["X", "Y", "Z"], potential=True)
def _Epot(ctx):
    """Potential energy per unit mass (km/s)^2."""
    m = ctx.model
    return _numpy_map(ctx, ["X", "Y", "Z"], m.epot, pl.Float64)


@_quantity("E", ["Epot", "Ekin"])
def _E(ctx):
    """Total energy per unit mass (km/s)^2."""
    return _f("Epot") + _f("Ekin")


# -- orbit extent and circular orbits ---------------------------------------------------
@_quantity("_periapo", ["E", "L"], potential=True)
def _periapo(ctx):
    """Pericentre and apocentre (struct)."""
    m = ctx.model
    dtype = pl.Struct({"peri": pl.Float64, "apo": pl.Float64})
    return _numpy_map(ctx, ["E", "L"], m.peri_apo, dtype)


@_quantity("peri", ["_periapo"])
def _peri(ctx):
    """Pericentre radius from (E, L) (kpc); exact for spherical potentials."""
    return _f("_periapo").struct.field("peri")


@_quantity("apo", ["_periapo"])
def _apo(ctx):
    """Apocentre radius from (E, L) (kpc); exact for spherical potentials."""
    return _f("_periapo").struct.field("apo")


@_quantity("ecc", ["peri", "apo"])
def _ecc(ctx):
    """Eccentricity (apo - peri) / (apo + peri)."""
    return (_f("apo") - _f("peri")) / (_f("apo") + _f("peri"))


@_quantity("RcircE", ["E"], potential=True)
def _RcircE(ctx):
    """Radius of the circular orbit with energy E (kpc)."""
    m = ctx.model
    return _numpy_map(ctx, ["E"], lambda a: m.rcirc_E(a[:, 0]), pl.Float64)


@_quantity("RcircL", ["L"], potential=True)
def _RcircL(ctx):
    """Radius of the circular orbit with angular momentum L (kpc)."""
    m = ctx.model
    return _numpy_map(ctx, ["L"], lambda a: m.rcirc_L(a[:, 0]), pl.Float64)


@_quantity("Tcirc", ["E"], potential=True)
def _Tcirc(ctx):
    """Period of the circular orbit with energy E (kpc / (km/s), about 0.978 Gyr)."""
    m = ctx.model
    return _numpy_map(ctx, ["E"], lambda a: m.tcirc(a[:, 0]), pl.Float64)


@_quantity("Lcirc", ["E"], potential=True)
def _Lcirc(ctx):
    """Angular momentum of the circular orbit with energy E (kpc km/s)."""
    m = ctx.model
    return _numpy_map(ctx, ["E"], lambda a: m.lcirc(a[:, 0]), pl.Float64)


@_quantity("circ", ["Lz", "Lcirc"])
def _circ(ctx):
    """Circularity Lz / Lcirc(E): +1 prograde circular, -1 retrograde circular."""
    return _f("Lz") / _f("Lcirc")


# -- actions, angles, frequencies -------------------------------------------------------
_ACTIONS = ("JR", "Jz", "Jphi")
_ANGLES = ("thetaR", "thetaz", "thetaphi")
_FREQS = ("OmegaR", "Omegaz", "Omegaphi")


@_quantity("_aa", list(PHASE_SPACE), potential=True)
def _aa(ctx):
    """Actions (and angles and frequencies when needed) from the action finder (struct)."""
    m = ctx.model
    names = _ACTIONS + (_ANGLES + _FREQS if ctx.full_angles else ())
    dtype = pl.Struct({n: pl.Float64 for n in names})
    return _numpy_map(ctx, list(PHASE_SPACE), lambda a: m.actions(a, angles=ctx.full_angles), dtype)


_DOCS = {
    "JR": "Radial action (kpc km/s).",
    "Jz": "Vertical action (kpc km/s).",
    "Jphi": "Azimuthal action, = Lz in axisymmetric potentials (kpc km/s).",
    "thetaR": "Radial angle (rad).",
    "thetaz": "Vertical angle (rad).",
    "thetaphi": "Azimuthal angle, increasing along Galactic rotation (rad).",
    "OmegaR": "Radial frequency (km/s/kpc).",
    "Omegaz": "Vertical frequency (km/s/kpc).",
    "Omegaphi": "Azimuthal frequency, positive for prograde orbits (km/s/kpc).",
}
for _c in _ACTIONS + _ANGLES + _FREQS:
    _REGISTRY[_c] = _Quantity(_c, ("_aa",), _field_of("_aa", _c), _DOCS[_c])


@_quantity("Jtot", ["JR", "Jz", "Jphi"])
def _Jtot(ctx):
    """JR + Jz + |Jphi| (kpc km/s)."""
    return _f("JR") + _f("Jz") + _f("Jphi").abs()


@_quantity("Jdiamond", ["JR", "Jz", "Jtot"])
def _Jdiamond(ctx):
    """(Jz - JR) / Jtot: vertical axis of the action diamond."""
    return (_f("Jz") - _f("JR")) / _f("Jtot")


@_quantity("Jphi_Jtot", ["Jphi", "Jtot"])
def _Jphi_Jtot(ctx):
    """Jphi / Jtot: horizontal axis of the action diamond."""
    return _f("Jphi") / _f("Jtot")


def _signed_sqrt(e: pl.Expr) -> pl.Expr:
    return e.abs().sqrt() * e.sign()


@_quantity("sqrt_JR", ["JR"])
def _sqrt_JR(ctx):
    """sqrt(JR)."""
    return _f("JR").sqrt()


@_quantity("sqrt_Jz", ["Jz"])
def _sqrt_Jz(ctx):
    """sqrt(Jz)."""
    return _f("Jz").sqrt()


@_quantity("sqrt_Jphi", ["Jphi"])
def _sqrt_Jphi(ctx):
    """sign(Jphi) sqrt(|Jphi|)."""
    return _signed_sqrt(_f("Jphi"))


@_quantity("sqrt_Lz", ["Lz"])
def _sqrt_Lz(ctx):
    """sign(Lz) sqrt(|Lz|)."""
    return _signed_sqrt(_f("Lz"))


del _c


# =============================================================================
# Resolution
# =============================================================================
def available() -> pl.DataFrame:
    """Every quantity `calc` can add: name, inputs, whether it needs a potential, description."""
    rows = [q for n, q in _REGISTRY.items() if not n.startswith("_")]
    return pl.DataFrame(
        {
            "name": [q.name for q in rows],
            "requires": [list(q.requires) for q in rows],
            "potential": [q.potential or _needs_potential(q.name) for q in rows],
            "description": [q.doc for q in rows],
        }
    )


def _needs_potential(name: str) -> bool:
    q = _REGISTRY.get(name)
    if q is None:
        return False
    if q.potential:
        return True
    return any(_needs_potential(r) for r in q.requires)


def _plan(targets: Sequence[str], existing: set[str], overwrite: bool) -> list[list[str]]:
    """Quantities to compute, grouped in waves that only depend on earlier waves."""
    needed: dict[str, int] = {}  # name -> wave
    missing: dict[str, str] = {}

    def visit(name: str, stack: tuple, is_target: bool) -> int:
        if name in needed:
            return needed[name]
        if name in existing and not (is_target and overwrite):
            return -1
        q = _REGISTRY.get(name)
        if q is None:
            missing[name] = stack[-1] if stack else name
            return -1
        if name in stack:
            raise ValueError(f"circular dependency: {' -> '.join(stack + (name,))}")
        wave = 1 + max([visit(r, stack + (name,), False) for r in q.requires], default=-1)
        needed[name] = wave
        return wave

    for t in targets:
        visit(t, (), True)
    if missing:
        detail = ", ".join(
            f"{m!r} (needed by {by!r})" if by != m else repr(m) for m, by in missing.items()
        )
        raise ValueError(f"cannot compute: missing column or unknown quantity {detail}")
    waves: list[list[str]] = [[] for _ in range(1 + max(needed.values(), default=-1))]
    for n, w in needed.items():
        waves[w].append(n)
    return waves


def _canonical(names: str | Iterable[str]) -> list[str]:
    names = [names] if isinstance(names, str) else list(names)
    return [ALIASES.get(n, n) for n in names]


def calc_exprs(
    names: str | Iterable[str],
    schema: Iterable[str] = (),
    *,
    solar: SolarParameters | None = None,
    potential=None,
    interp: bool = False,
    columns: dict[str, str] | None = None,
    overwrite: bool = False,
) -> list[list[pl.Expr]]:
    """Expressions for `calc`, in waves: apply each list with one ``with_columns``.

    ``schema`` holds the column names already present (they are reused, not
    recomputed). Intermediate columns start with ``_``.
    """
    targets = _canonical(names)
    columns = dict(columns or {})
    existing = set(schema) | set(columns)
    waves = _plan(targets, existing, overwrite)
    planned = {n for w in waves for n in w}
    full = any(n in planned for n in _ANGLES + _FREQS)
    ctx = _Context(solar, potential, interp, columns, full)
    return [[_REGISTRY[n].build(ctx).alias(n) for n in w] for w in waves]


@pl.api.register_dataframe_namespace("galactic")
@pl.api.register_lazyframe_namespace("galactic")
class Galactic:
    """Galactocentric coordinates and dynamics on a DataFrame or LazyFrame.

    Distances in kpc, proper motions in mas/yr, velocities in km/s.
    """

    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        self._df = df

    def calc(
        self,
        quantities: str | Iterable[str],
        *,
        potential=None,
        solar: SolarParameters | None = None,
        columns: dict[str, str] | None = None,
        interp: bool = False,
        overwrite: bool = False,
        keep_intermediate: bool = False,
    ):
        """Add derived kinematic and dynamical quantities, with their dependencies.

        Existing columns are reused, so ``calc`` works from observables
        (``ra, dec, distance`` or ``parallax, pmra, pmdec, radial_velocity``)
        or from Galactocentric ``X, Y, Z, vX, vY, vZ``.

        Parameters
        ----------
        quantities
            Names from `available` (e.g. ``["R", "vphi", "Lz", "E", "JR"]``);
            the names of the original polars_astro (``U``, ``ke``, ``AR``,
            ``FreqR``, ...) are accepted too, see `ALIASES`.
        potential
            AGAMA potential (name, ``.ini`` path, ``agama.Potential`` or
            `GalaxyModel`); default `DEFAULT_POTENTIAL`. Only loaded if a
            requested quantity needs it.
        solar
            `astro_polars.galactic.SolarParameters`; default the package default.
        columns
            Map from input names (``ra``, ``dec``, ``distance``, ``parallax``,
            ``pmra``, ``pmdec``, ``radial_velocity``) to your column names, e.g.
            ``{"radial_velocity": "rv_corrected"}``.
        interp
            Use AGAMA's interpolated action finder.
        overwrite
            Recompute requested quantities that already exist as columns.
        keep_intermediate
            Keep helper columns (``_aa``, ``_periapo``, ...) instead of dropping them.
        """
        df = self._df
        names = df.collect_schema().names()
        waves = calc_exprs(
            quantities,
            names,
            solar=solar,
            potential=potential,
            interp=interp,
            columns=columns,
            overwrite=overwrite,
        )
        added: list[str] = []
        for exprs in waves:
            df = df.with_columns(exprs)
            added += [e.meta.output_name() for e in exprs]
        aliases = {a: c for a, c in ALIASES.items() if a in _as_list(quantities)}
        if aliases:
            df = df.with_columns(pl.col(c).alias(a) for a, c in aliases.items())
        if not keep_intermediate:
            drop = [n for n in added if n.startswith("_")]
            if drop:
                df = df.drop(drop)
        return df

    def from_observed(
        self,
        ra: str = "ra",
        dec: str = "dec",
        distance: str | pl.Expr | None = "distance",
        pmra: str | None = "pmra",
        pmdec: str | None = "pmdec",
        radial_velocity: str | None = "radial_velocity",
        *,
        parallax: str | None = None,
        solar: SolarParameters | None = None,
        out: Sequence[str] = PHASE_SPACE,
    ):
        """Add Galactocentric ``X, Y, Z`` (kpc) and ``vX, vY, vZ`` (km/s).

        Give ``parallax`` (mas) instead of ``distance`` to use 1/parallax.
        Velocities are skipped when ``pmra``, ``pmdec`` or ``radial_velocity`` is None.
        """
        d = 1.0 / pl.col(parallax) if parallax is not None else distance
        s = galactic.galactocentric(ra, dec, d, pmra, pmdec, radial_velocity, solar)
        nfields = 6 if None not in (pmra, pmdec, radial_velocity) else 3
        return _unpack(self._df, s, PHASE_SPACE[:nfields], out[:nfields])

    def to_observed(
        self,
        X: str = "X",
        Y: str = "Y",
        Z: str = "Z",
        vX: str | None = "vX",
        vY: str | None = "vY",
        vZ: str | None = "vZ",
        *,
        solar: SolarParameters | None = None,
        out: Sequence[str] = ("ra", "dec", "distance", "pmra", "pmdec", "radial_velocity"),
    ):
        """Add ICRS ``ra, dec, distance, pmra, pmdec, radial_velocity`` from Galactocentric columns."""
        s = galactic.from_galactocentric(X, Y, Z, vX, vY, vZ, solar)
        fields = ("ra", "dec", "distance", "pmra", "pmdec", "radial_velocity")
        n = 6 if None not in (vX, vY, vZ) else 3
        return _unpack(self._df, s, fields[:n], out[:n])

    def reflex_correct(
        self,
        lon: str = "ra",
        lat: str = "dec",
        distance: str | float | pl.Expr = "distance",
        pmlon: str = "pmra",
        pmlat: str = "pmdec",
        *,
        solar: SolarParameters | None = None,
        frame: str = "icrs",
        out: Sequence[str] | None = None,
    ):
        """Add proper motions corrected for the solar reflex motion.

        Output names default to ``{pmlon}_gsr, {pmlat}_gsr`` (``pmra_gsr``,
        ``pmdec_gsr``). See `astro_polars.galactic.reflex_correct`.
        """
        out = out or (f"{pmlon}_gsr", f"{pmlat}_gsr")
        s = galactic.reflex_correct(lon, lat, distance, pmlon, pmlat, solar, frame)
        return _unpack(self._df, s, ("pmlon", "pmlat"), out)

    def vgsr(
        self,
        lon: str = "ra",
        lat: str = "dec",
        radial_velocity: str = "radial_velocity",
        *,
        solar: SolarParameters | None = None,
        frame: str = "icrs",
        name: str = "vgsr",
    ):
        """Add the line-of-sight velocity in the Galactic standard of rest."""
        e = galactic.vgsr(lon, lat, radial_velocity, solar, frame)
        return self._df.with_columns(e.alias(name))


def _as_list(x) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def _unpack(df, struct_expr: pl.Expr, fields: Sequence[str], out: Sequence[str]):
    tmp = "__ap_gal"
    return (
        df.with_columns(struct_expr.alias(tmp))
        .with_columns(pl.col(tmp).struct.field(f).alias(o) for f, o in zip(fields, out))
        .drop(tmp)
    )
