import numpy as np
import polars as pl
import pytest

import astro_polars  # noqa: F401
from astro_polars import dynamics
from astro_polars.dynamics import GalaxyModel

G = 4.300917270038e-06  # kpc (km/s)^2 / Msun


class Kepler:
    """Point-mass potential with AGAMA's Potential interface (analytic everything)."""

    def __init__(self, mass=1e12):
        self.gm = G * mass

    def potential(self, xyz):
        return -self.gm / np.linalg.norm(np.atleast_2d(xyz), axis=1)

    def force(self, xyz):
        xyz = np.atleast_2d(xyz)
        r = np.linalg.norm(xyz, axis=1)[:, None]
        return -self.gm * xyz / r**3

    def Rcirc(self, E=None, L=None):
        return -self.gm / (2 * np.asarray(E)) if E is not None else np.asarray(L) ** 2 / self.gm

    def Tcirc(self, E):
        a = -self.gm / (2 * np.asarray(E))
        return 2 * np.pi * np.sqrt(a**3 / self.gm)

    def Rperiapo(self, EL):
        E, L = EL[:, 0], EL[:, 1]
        a = -self.gm / (2 * E)
        e = np.sqrt(1 + 2 * E * L**2 / self.gm**2)
        return np.column_stack([a * (1 - e), a * (1 + e)])


class KeplerActions:
    """Kepler actions in AGAMA's convention (Jphi = (r x v)_z)."""

    def __init__(self, gm):
        self.gm = gm

    def __call__(self, xv, actions=True, angles=False, frequencies=None):
        x, v = xv[:, :3], xv[:, 3:]
        L = np.cross(x, v)
        Ltot = np.linalg.norm(L, axis=1)
        E = 0.5 * (v**2).sum(1) - self.gm / np.linalg.norm(x, axis=1)
        J = np.column_stack([self.gm / np.sqrt(-2 * E) - Ltot, Ltot - np.abs(L[:, 2]), L[:, 2]])
        if not angles:
            return J
        om = (-2 * E) ** 1.5 / self.gm
        omega = np.column_stack([om, om, om * np.sign(L[:, 2])])
        theta = np.column_stack([np.full_like(om, 0.5)] * 3)
        return J, theta, omega


@pytest.fixture
def model():
    pot = Kepler()
    return GalaxyModel(pot, action_finder=KeplerActions(pot.gm))


@pytest.fixture
def halo(rng):
    """Bound orbits around a 1e12 Msun point mass."""
    n = 500
    xyz = rng.normal(0, 10, (n, 3))
    v = rng.normal(0, 120, (n, 3))
    return pl.DataFrame(dict(zip(dynamics.PHASE_SPACE, np.hstack([xyz, v]).T)))


def test_kinematics_conventions():
    # the Sun: X < 0, moving towards +Y -> prograde, vphi and Lz positive
    df = pl.DataFrame({"X": [-8.2], "Y": [0.0], "Z": [0.0], "vX": [11.1], "vY": [245.0]})
    df = df.with_columns(vZ=7.25)
    out = df.galactic.calc(["R", "r", "phi", "vR", "vphi", "vT", "Lz", "L", "v_toomre"])
    row = out.row(0, named=True)
    assert row["vphi"] == pytest.approx(245.0) and row["vT"] == row["vphi"]
    assert row["Lz"] == pytest.approx(8.2 * 245.0)
    assert row["phi"] == pytest.approx(180.0)
    assert row["vR"] == pytest.approx(-11.1)  # towards the centre
    assert row["v_toomre"] == pytest.approx(np.sqrt((245 - 232.8) ** 2 + 11.1**2 + 7.25**2))


def test_angular_momentum_is_minus_r_cross_v(halo):
    out = halo.galactic.calc(["Lx", "Ly", "Lz", "L", "Lp", "vr", "vtheta", "vt", "Ekin"])
    x = halo.select("X", "Y", "Z").to_numpy()
    v = halo.select("vX", "vY", "vZ").to_numpy()
    L = -np.cross(x, v)
    np.testing.assert_allclose(out.select("Lx", "Ly", "Lz").to_numpy(), L, rtol=1e-12)
    np.testing.assert_allclose(out["L"], np.linalg.norm(L, axis=1), rtol=1e-12)
    np.testing.assert_allclose(out["Lp"], np.hypot(L[:, 0], L[:, 1]), rtol=1e-12)
    # speed decomposes into radial + tangential
    np.testing.assert_allclose(out["vr"] ** 2 + out["vt"] ** 2, (v**2).sum(1), rtol=1e-10)
    np.testing.assert_allclose(out["Ekin"], 0.5 * (v**2).sum(1), rtol=1e-12)


def test_energy_and_orbit_extent(halo, model):
    out = halo.galactic.calc(
        ["E", "peri", "apo", "ecc", "RcircE", "RcircL", "Tcirc", "Lcirc", "circ"],
        potential=model,
    )
    r = np.linalg.norm(halo.select("X", "Y", "Z").to_numpy(), axis=1)
    v2 = (halo.select("vX", "vY", "vZ").to_numpy() ** 2).sum(1)
    E = 0.5 * v2 - model.potential.gm / r
    np.testing.assert_allclose(out["E"], E, rtol=1e-12)
    bound = out.filter(pl.col("E") < 0)
    assert len(bound) > 100
    rr = np.linalg.norm(bound.select("X", "Y", "Z").to_numpy(), axis=1)
    assert np.all(bound["peri"].to_numpy() <= rr * (1 + 1e-9))
    assert np.all(bound["apo"].to_numpy() >= rr * (1 - 1e-9))
    assert bound["ecc"].is_between(0, 1).all()
    # Kepler: Lcirc(E) = GM / sqrt(-2E); |circ| <= 1
    np.testing.assert_allclose(
        bound["Lcirc"], model.potential.gm / np.sqrt(-2 * bound["E"]), rtol=1e-4
    )
    assert (bound["circ"].abs() <= 1 + 1e-4).all()
    assert "_periapo" not in out.columns


def test_actions_signs_and_aliases(halo, model):
    out = halo.galactic.calc(
        ["JR", "Jz", "Jphi", "Jtot", "Jdiamond", "Lz", "U", "E"], potential=model
    )
    bound = out.filter(pl.col("E") < 0)
    np.testing.assert_allclose(bound["Jphi"], bound["Lz"], rtol=1e-10)
    assert (bound["JR"] >= -1e-9).all() and (bound["Jz"] >= -1e-9).all()
    assert bound["Jdiamond"].is_between(-1, 1).all()
    np.testing.assert_allclose(out["U"], out["Epot"])
    assert "_aa" not in out.columns


def test_angles_and_frequencies_in_one_call(halo, model):
    calls = []
    finder = model.action_finder

    def counting(*a, **k):
        calls.append(k)
        return finder(*a, **k)

    model.action_finder = counting
    out = halo.galactic.calc(["JR", "thetaphi", "Omegaphi", "OmegaR", "E", "Lz"], potential=model)
    assert len(calls) == 1 and calls[0]["angles"]
    # Omegaphi follows the sign of Jphi (prograde positive)
    bound = out.filter(pl.col("E") < 0)
    assert (np.sign(bound["Omegaphi"]) == np.sign(bound["Lz"])).all()
    np.testing.assert_allclose(bound["thetaphi"], 2 * np.pi - 0.5)


def test_calc_from_observables(rng, model):
    n = 50
    df = pl.DataFrame(
        {
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-60, 60, n),
            "parallax": rng.uniform(0.2, 2, n),
            "pmra": rng.normal(0, 5, n),
            "pmdec": rng.normal(0, 5, n),
            "rv_corr": rng.normal(0, 50, n),
        }
    )
    out = df.lazy().galactic.calc(
        ["Lz", "E", "vtan", "l"], columns={"radial_velocity": "rv_corr"}, potential=model
    )
    assert isinstance(out, pl.LazyFrame)
    out = out.collect()
    ref = df.galactic.from_observed(
        distance=1 / pl.col("parallax"), radial_velocity="rv_corr"
    ).galactic.calc("Lz")
    np.testing.assert_allclose(out["Lz"], ref["Lz"], rtol=1e-12)
    assert {"distance", "X", "vZ", "l", "vtan"} <= set(out.columns)
    assert not any(c.startswith("_") for c in out.columns)


def test_existing_columns_are_reused():
    df = pl.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0], "R": [99.0]})
    assert df.galactic.calc("R")["R"].item() == 99.0
    assert df.galactic.calc("R", overwrite=True)["R"].item() == pytest.approx(np.sqrt(5))


def test_missing_inputs_message():
    with pytest.raises(ValueError, match="radial_velocity"):
        pl.DataFrame({"ra": [1.0], "dec": [1.0], "distance": [1.0]}).galactic.calc("vX")
    with pytest.raises(ValueError, match="nope"):
        pl.DataFrame({"X": [1.0]}).galactic.calc("nope")


def test_available_lists_everything():
    av = dynamics.available()
    names = set(av["name"])
    for n in ("E", "Lz", "JR", "Jz", "Jphi", "peri", "apo", "ecc", "circ", "v_toomre"):
        assert n in names
    assert not any(n.startswith("_") for n in names)
    assert av.filter(pl.col("name") == "JR")["potential"].item()
    assert not av.filter(pl.col("name") == "Lz")["potential"].item()


def test_streaming_engine(halo, model):
    lf = halo.lazy().galactic.calc(["E", "JR"], potential=model)
    a = lf.collect(engine="streaming")
    b = lf.collect()
    np.testing.assert_allclose(a["JR"], b["JR"])


def test_load_potential_errors():
    obj = Kepler()
    assert dynamics.load_potential(obj) is obj
    pytest.importorskip("agama")
    with pytest.raises(FileNotFoundError):
        dynamics.load_potential("no_such_potential_xyz")


# -- with AGAMA ---------------------------------------------------------------------------
def test_agama_milky_way():
    pytest.importorskip("agama")
    m = GalaxyModel("McMillan17")
    sol = astro_polars.galactic.SolarParameters()
    # the Sun and a star on a near-circular disc orbit at 8.2 kpc
    df = pl.DataFrame(
        {"X": [-sol.R0], "Y": [0.0], "Z": [0.0], "vX": [0.0], "vY": [233.0], "vZ": [0.0]}
    )
    out = df.galactic.calc(["E", "JR", "Jz", "Jphi", "Lz", "circ", "peri", "apo"], potential=m)
    row = out.row(0, named=True)
    assert row["E"] < 0
    assert row["Jphi"] == pytest.approx(row["Lz"], rel=1e-6)
    assert row["Jphi"] > 0
    assert row["JR"] < 20 and row["Jz"] < 1
    assert row["circ"] == pytest.approx(1.0, abs=0.02)
    assert row["peri"] <= sol.R0 * 1.001 <= row["apo"] * 1.002
