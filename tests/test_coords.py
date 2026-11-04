import numpy as np
import polars as pl
import pytest

from astro_polars import (
    GD1_KOPOSOV10,
    ICRS_TO_GALACTIC,
    frame_from_pole,
    rotate,
    rotate_pm,
    rotate_pm_errors,
)


def _rot(df, *args, **kw):
    return df.select(rotate(*args, **kw).struct.unnest())


def test_matrices_orthonormal():
    for m in (ICRS_TO_GALACTIC, GD1_KOPOSOV10):
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-9)


def test_galactic_landmarks():
    df = pl.DataFrame({"ra": [266.40499, 192.85948], "dec": [-28.93617, 27.12825]})
    out = _rot(df, "ra", "dec", ICRS_TO_GALACTIC, wrap180=True)
    # Galactic centre and north Galactic pole
    assert out["lon"][0] == pytest.approx(0.0, abs=1e-3)
    assert out["lat"][0] == pytest.approx(0.0, abs=1e-3)
    assert out["lat"][1] == pytest.approx(90.0, abs=1e-3)


def test_round_trip(sky):
    gal = sky.sky.to_galactic(pmra=None, pmdec=None)
    back = gal.sky.to_equatorial(pml=None, pmb=None, out=("ra2", "dec2"))
    np.testing.assert_allclose(back["dec2"], sky["dec"], atol=1e-9)
    dra = (back["ra2"] - sky["ra"] + 180) % 360 - 180
    assert np.abs(dra.to_numpy()).max() < 1e-8


def test_pm_preserves_total_and_round_trips(sky, rng):
    df = sky.with_columns(
        pmra=pl.lit(rng.normal(0, 5, len(sky))), pmdec=pl.lit(rng.normal(0, 5, len(sky)))
    )
    g = df.sky.to_galactic()
    tot_in = np.hypot(df["pmra"], df["pmdec"])
    tot_out = np.hypot(g["pml"], g["pmb"])
    np.testing.assert_allclose(tot_out, tot_in, rtol=1e-9, atol=1e-9)
    back = g.sky.to_equatorial(out=("ra2", "dec2", "pmra2", "pmdec2"))
    np.testing.assert_allclose(back["pmra2"], df["pmra"], atol=1e-8)
    np.testing.assert_allclose(back["pmdec2"], df["pmdec"], atol=1e-8)


def test_pm_matches_finite_difference():
    """Proper motion = derivative of the rotated position."""
    ra, dec, pmra, pmdec, dt = 150.0, 30.0, 3.0, -2.0, 1e-6
    df = pl.DataFrame(
        {
            "ra": [ra, ra + pmra * dt / np.cos(np.radians(dec))],
            "dec": [dec, dec + pmdec * dt],
            "pmra": [pmra, pmra],
            "pmdec": [pmdec, pmdec],
        }
    )
    out = df.select(rotate_pm("ra", "dec", "pmra", "pmdec", ICRS_TO_GALACTIC).struct.unnest())
    dl = (out["lon"][1] - out["lon"][0]) * np.cos(np.radians(out["lat"][0])) / dt
    db = (out["lat"][1] - out["lat"][0]) / dt
    assert out["pmlon"][0] == pytest.approx(dl, rel=1e-4)
    assert out["pmlat"][0] == pytest.approx(db, rel=1e-4)


def test_against_astropy(sky, rng):
    try:
        import astropy.coordinates as coord
        import astropy.units as u
    except ImportError:
        pytest.skip("astropy not installed")
    except AttributeError as err:  # old astropy with NumPy 2 (Python 3.9)
        pytest.skip(f"astropy does not work with this NumPy: {err}")
    df = sky.head(500).with_columns(
        pmra=pl.lit(rng.normal(0, 5, 500)), pmdec=pl.lit(rng.normal(0, 5, 500))
    )
    try:
        c = coord.SkyCoord(
            ra=df["ra"].to_numpy() * u.deg,
            dec=df["dec"].to_numpy() * u.deg,
            pm_ra_cosdec=df["pmra"].to_numpy() * u.mas / u.yr,
            pm_dec=df["pmdec"].to_numpy() * u.mas / u.yr,
            frame="icrs",
        ).galactic
    except AttributeError as err:  # old astropy with NumPy 2 (Python 3.9)
        pytest.skip(f"astropy does not work with this NumPy: {err}")
    g = df.sky.to_galactic()
    # astropy's Galactic frame comes from FK4/FK5 and differs from the Hipparcos
    # ICRS->Galactic matrix by a few tens of mas
    np.testing.assert_allclose(g["b"], c.b.deg, atol=2e-5)
    np.testing.assert_allclose(g["pml"], c.pm_l_cosb.value, atol=1e-4)
    np.testing.assert_allclose(g["pmb"], c.pm_b.value, atol=1e-4)


def test_error_propagation_monte_carlo(rng):
    n = 200_000
    ra, dec = 210.0, -40.0
    s1, s2, rho = 0.3, 0.1, 0.5
    cov = [[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]]
    pm = rng.multivariate_normal([0, 0], cov, n)
    df = pl.DataFrame(
        {"ra": np.full(n, ra), "dec": np.full(n, dec), "pmra": pm[:, 0], "pmdec": pm[:, 1]}
    )
    mc = df.select(rotate_pm("ra", "dec", "pmra", "pmdec", ICRS_TO_GALACTIC).struct.unnest())
    pred = pl.DataFrame({"ra": [ra], "dec": [dec]}).select(
        rotate_pm_errors("ra", "dec", s1, s2, rho, ICRS_TO_GALACTIC).struct.unnest()
    )
    assert pred["pmlon_error"][0] == pytest.approx(mc["pmlon"].std(), rel=0.01)
    assert pred["pmlat_error"][0] == pytest.approx(mc["pmlat"].std(), rel=0.01)
    assert pred["pm_corr"][0] == pytest.approx(
        np.corrcoef(mc["pmlon"], mc["pmlat"])[0, 1], abs=0.01
    )


def test_frame_from_pole():
    m = frame_from_pole(30.0, 60.0, 120.0, 10.0)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    df = pl.DataFrame({"ra": [30.0, 120.0], "dec": [60.0, 10.0]})
    out = _rot(df, "ra", "dec", m, wrap180=True)
    assert out["lat"][0] == pytest.approx(90.0)
    assert out["lon"][1] == pytest.approx(0.0, abs=1e-9)


def test_gd1_frame_origin():
    # phi1 = phi2 = 0 of Koposov+10 lies at (RA, Dec) ~ (200.0, 59.4)
    df = pl.DataFrame({"phi1": [0.0], "phi2": [0.0]})
    out = _rot(df, "phi1", "phi2", GD1_KOPOSOV10.T)
    assert out["lon"][0] == pytest.approx(200.0, abs=0.1)
    assert out["lat"][0] == pytest.approx(59.4, abs=0.1)


def test_to_frame_namespace(sky):
    out = sky.lazy().sky.to_frame(GD1_KOPOSOV10, pmlon=None, pmlat=None).collect()
    assert {"phi1", "phi2"} <= set(out.columns)
    assert out["phi1"].min() >= -180 and out["phi1"].max() <= 180
