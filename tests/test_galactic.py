import numpy as np
import polars as pl
import pytest

import astro_polars  # noqa: F401
from astro_polars import galactic
from astro_polars.galactic import SolarParameters, from_galactocentric, galactocentric


@pytest.fixture
def stars(rng):
    n = 2000
    return pl.DataFrame(
        {
            "ra": rng.uniform(0, 360, n),
            "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
            "distance": rng.uniform(0.05, 30, n),
            "pmra": rng.normal(0, 10, n),
            "pmdec": rng.normal(0, 10, n),
            "radial_velocity": rng.normal(0, 100, n),
        }
    )


def test_sun_position_and_velocity():
    sol = SolarParameters()
    # a star at zero distance with zero motion is the Sun
    df = pl.DataFrame(
        {"ra": [0.0], "dec": [0.0], "d": [0.0], "pmra": [0.0], "pmdec": [0.0], "rv": [0.0]}
    )
    out = df.select(galactocentric("ra", "dec", "d", "pmra", "pmdec", "rv").struct.unnest())
    np.testing.assert_allclose(out.row(0), (-sol.R0, 0, 0, *sol.velocity), atol=1e-12)


def test_galactic_centre_direction():
    # the Galactic centre (l = b = 0) at R0 lands on the origin
    sol = SolarParameters(z0=0.0)
    ra, dec = 266.4049882865447, -28.936177761791473  # l = b = 0 (Hipparcos matrix)
    df = pl.DataFrame({"ra": [ra], "dec": [dec], "d": [sol.R0]})
    out = df.select(galactocentric("ra", "dec", "d", None, None, None, sol).struct.unnest())
    np.testing.assert_allclose(out.row(0), (0, 0, 0), atol=1e-6)


@pytest.mark.parametrize("z0", [0.0, 0.0208])
def test_matches_astropy(stars, z0):
    coord = pytest.importorskip("astropy.coordinates")
    u = pytest.importorskip("astropy.units")
    sol = SolarParameters(z0=z0)
    gc = coord.SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic").icrs
    frame = coord.Galactocentric(
        galcen_coord=gc,
        galcen_distance=sol.R0 * u.kpc,
        galcen_v_sun=coord.CartesianDifferential(list(sol.velocity) * u.km / u.s),
        z_sun=z0 * u.kpc,
        roll=0 * u.deg,
    )
    c = coord.SkyCoord(
        ra=stars["ra"].to_numpy() * u.deg,
        dec=stars["dec"].to_numpy() * u.deg,
        distance=stars["distance"].to_numpy() * u.kpc,
        pm_ra_cosdec=stars["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=stars["pmdec"].to_numpy() * u.mas / u.yr,
        radial_velocity=stars["radial_velocity"].to_numpy() * u.km / u.s,
    ).transform_to(frame)
    out = stars.select(galactocentric(solar=sol).struct.unnest())
    # the Hipparcos matrix and astropy's Galactic frame differ by ~6e-7 rad,
    # so compare with a tolerance proportional to the heliocentric vectors
    pos = out.select("X", "Y", "Z").to_numpy()
    vel = out.select("vX", "vY", "vZ").to_numpy()
    kms = u.km / u.s
    pos_ref = np.column_stack([c.x.to_value(u.kpc), c.y.to_value(u.kpc), c.z.to_value(u.kpc)])
    vel_ref = np.column_stack([c.v_x.to_value(kms), c.v_y.to_value(kms), c.v_z.to_value(kms)])
    helio_pos = np.linalg.norm(pos_ref - sol._sun_position(), axis=1)
    helio_vel = np.linalg.norm(vel_ref - np.array(sol.velocity), axis=1)
    assert np.all(np.linalg.norm(pos - pos_ref, axis=1) <= 1e-6 * helio_pos + 1e-12)
    assert np.all(np.linalg.norm(vel - vel_ref, axis=1) <= 1e-6 * helio_vel + 1e-9)


@pytest.mark.parametrize("z0", [0.0, 0.025])
def test_round_trip(stars, z0):
    sol = SolarParameters(z0=z0, R0=8.122)
    gc = stars.select(galactocentric(solar=sol).struct.unnest())
    back = gc.select(from_galactocentric(solar=sol).struct.unnest())
    ra_diff = (back["ra"] - stars["ra"] + 180) % 360 - 180
    np.testing.assert_allclose(ra_diff, 0, atol=1e-8)
    for c in ("dec", "distance", "pmra", "pmdec", "radial_velocity"):
        np.testing.assert_allclose(back[c], stars[c], rtol=1e-8, atol=1e-8)


def test_reflex_correction_star_at_rest(stars):
    # a star at rest in the Galactocentric frame has zero corrected PM and v_gsr
    gc = stars.select(galactocentric().struct.unnest()).with_columns(vX=0.0, vY=0.0, vZ=0.0)
    obs = gc.select(from_galactocentric().struct.unnest())
    out = obs.galactic.reflex_correct().galactic.vgsr()
    np.testing.assert_allclose(out["pmra_gsr"], 0, atol=1e-9)
    np.testing.assert_allclose(out["pmdec_gsr"], 0, atol=1e-9)
    np.testing.assert_allclose(out["vgsr"], 0, atol=1e-9)


def test_reflex_correction_galactic_frame(stars):
    gal = stars.sky.to_galactic()
    icrs = gal.galactic.reflex_correct()
    lb = gal.galactic.reflex_correct("l", "b", pmlon="pml", pmlat="pmb", frame="galactic")
    # rotate the ICRS-corrected PMs to Galactic and compare
    rot = icrs.sky.to_galactic(pmra="pmra_gsr", pmdec="pmdec_gsr", out=("l2", "b2", "a", "b_"))
    np.testing.assert_allclose(rot["a"], lb["pml_gsr"], atol=1e-9)
    np.testing.assert_allclose(rot["b_"], lb["pmb_gsr"], atol=1e-9)


def test_reflex_null_for_bad_distance():
    df = pl.DataFrame(
        {"ra": [1.0, 2.0], "dec": [3.0, 4.0], "distance": [0.0, -1.0], "pmra": [1.0, 1.0]}
    ).with_columns(pmdec=pl.col("pmra"))
    out = df.galactic.reflex_correct()
    assert out["pmra_gsr"].null_count() == 2


def test_default_solar_roundtrip():
    old = galactic.get_default_solar()
    try:
        new = galactic.set_default_solar(R0=8.0)
        assert new.R0 == 8.0 and galactic.get_default_solar().R0 == 8.0
        df = pl.DataFrame({"ra": [0.0], "dec": [0.0], "d": [0.0]})
        out = df.select(galactocentric("ra", "dec", "d", None, None, None).struct.field("X"))
        assert out.item() == pytest.approx(-8.0)
    finally:
        galactic.set_default_solar(old)
    assert galactic.get_default_solar() == old


def test_lazy_and_namespace(stars):
    lf = stars.lazy().galactic.from_observed()
    assert isinstance(lf, pl.LazyFrame)
    out = lf.galactic.to_observed(out=("ra2", "dec2", "d2", "pmra2", "pmdec2", "rv2")).collect()
    np.testing.assert_allclose(out["d2"], out["distance"], rtol=1e-10)
    np.testing.assert_allclose(out["rv2"], out["radial_velocity"], atol=1e-8)
    pos = stars.galactic.from_observed(pmra=None)
    assert "X" in pos.columns and "vX" not in pos.columns


def test_from_parallax(stars):
    df = stars.with_columns(parallax=1 / pl.col("distance"))
    a = df.galactic.from_observed()
    b = df.drop("distance").galactic.from_observed(parallax="parallax")
    np.testing.assert_allclose(a["vY"], b["vY"], rtol=1e-12)


def test_galactic_pm_errors(stars, rng):
    df = stars.with_columns(
        pmra_error=pl.lit(0.1), pmdec_error=pl.lit(0.2), pmra_pmdec_corr=pl.lit(0.3)
    ).sky.galactic_pm_errors()
    # total variance (trace of the covariance) is invariant under rotation
    tr_in = 0.1**2 + 0.2**2
    tr_out = df["pml_error"] ** 2 + df["pmb_error"] ** 2
    np.testing.assert_allclose(tr_out, tr_in, rtol=1e-10)
    no_corr = stars.with_columns(pmra_error=pl.lit(0.1), pmdec_error=pl.lit(0.1))
    out = no_corr.sky.galactic_pm_errors()
    np.testing.assert_allclose(out["pml_pmb_corr"], 0, atol=1e-10)


def test_heliocentric_cartesian_matches_galactocentric(stars):
    sol = SolarParameters()
    gal = stars.sky.to_galactic()
    h = gal.select(
        galactic.heliocentric_cartesian(
            "l", "b", "distance", "pml", "pmb", "radial_velocity"
        ).struct.unnest()
    )
    g = stars.select(galactocentric(solar=sol).struct.unnest())
    np.testing.assert_allclose(h["x"] - sol.R0, g["X"], atol=1e-9)
    np.testing.assert_allclose(h["vy"] + sol.velocity[1], g["vY"], atol=1e-7)
    np.testing.assert_allclose(h["vz"] + sol.W, g["vZ"], atol=1e-7)
