import numpy as np
import polars as pl
import pytest

from astro_polars.surveys import gaia


def test_radial_velocity_corrections():
    df = pl.DataFrame(
        {
            "radial_velocity": [10.0, 10.0, 10.0, 10.0, 10.0],
            "grvs_mag": [9.0, 11.0, 14.0, 10.0, 13.0],
            "rv_template_teff": [5000.0, 5000.0, 5000.0, 9000.0, 9000.0],
        }
    )
    out = df.select(gaia.correct_radial_velocity()).to_series().to_numpy()
    # bright cool: none; g = 11: continuous (~0); g = 14: ~0.4 km/s;
    # hot g = 10: -7.98 + 11.35; hot g = 13: outside range
    g = 14.0
    expected = [10, 10 - (0.02755 * 121 - 0.55863 * 11 + 2.81129), 10]
    expected[2] = 10 - (0.02755 * g**2 - 0.55863 * g + 2.81129)
    expected += [10 - 7.98 + 11.35, 10]
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert abs(out[1] - 10) < 1e-3
    assert out[2] == pytest.approx(10 - 0.39, abs=0.01)


def test_radial_velocity_error_factor():
    g = np.array([5.0, 8.0, 10.0, 12.0, 14.0])
    df = pl.DataFrame({"radial_velocity_error": np.ones(5), "grvs_mag": g})
    out = df.select(gaia.correct_radial_velocity_error()).to_series().to_numpy()
    f8 = 0.318 + 0.3884 * 8 - 0.02778 * 64
    assert out[0] == pytest.approx(f8) and out[1] == pytest.approx(f8)
    assert out[2] == pytest.approx(0.318 + 0.3884 * 10 - 0.02778 * 100)
    assert out[4] == pytest.approx(16.554 - 2.4899 * 14 + 0.09933 * 196)


def test_estimate_grvs_range():
    df = pl.DataFrame(
        {"phot_g_mean_mag": [12.0, 12.0, 12.0], "phot_rp_mean_mag": [11.3, 10.5, 11.95]}
    )
    out = df.select(gaia.estimate_grvs()).to_series()
    assert np.isfinite(out[0]) and np.isfinite(out[1]) and np.isnan(out[2])
    # Gaia Collaboration 2018 (A&A 616, A1) Eq. 2 gives G_RVS - G_RP, not G_RVS - G
    x = 0.7
    poly = 0.042319 - 0.65124 * x + 1.0215 * x**2 - 1.3947 * x**3 + 0.53768 * x**4
    assert out[0] == pytest.approx(11.3 + poly)
    # a red star (G - G_RP = 1.5) is brighter in RVS than in RP
    assert out[1] < 10.5


def test_deredden_zero_and_consistency():
    df = pl.DataFrame(
        {
            "ebv": [0.0, 0.3],
            "phot_g_mean_mag": [15.0, 15.0],
            "phot_bp_mean_mag": [15.5, 15.8],
            "phot_rp_mean_mag": [14.4, 14.3],
        }
    )
    out = gaia.deredden(df.lazy()).collect()
    assert out["A_G"][0] == 0 and out["phot_g_mean_mag_0"][0] == 15.0
    # converged: intrinsic colour = observed - (A_BP - A_RP)
    row = out.row(1, named=True)
    assert row["bp_rp_0"] == pytest.approx(1.5 - (row["A_BP"] - row["A_RP"]), abs=1e-8)
    assert row["phot_bp_mean_mag_0"] - row["phot_rp_mean_mag_0"] == pytest.approx(row["bp_rp_0"])
    assert 0.5 < row["A_G"] / (3.1 * 0.3) < 1.0


def test_dodd23_quality():
    df = pl.DataFrame(
        {
            "ruwe": [1.0, 1.5, 1.0, 1.0],
            "radial_velocity_error": [1.0, 1.0, 1.0, 1.0],
            "grvs_mag": [12.0, 12.0, 12.0, 12.0],
            "phot_g_mean_mag": [12.5, 12.5, 12.5, 12.5],
            "rv_expected_sig_to_noise": [10.0, 10.0, 3.0, 10.0],
            "b": [30.0, 30.0, 5.0, 30.0],
            "parallax_over_error": [10.0, 10.0, 10.0, 3.0],
        }
    )
    assert df.select(gaia.dodd23_quality()).to_series().to_list() == [True, False, False, False]
    m = df.select(gaia.dodd23_quality(parallax_over_error=None)).to_series().to_list()
    assert m == [True, False, False, True]


def test_distance_from_parallax():
    df = pl.DataFrame({"parallax": [2.0, -1.0, 0.0]})
    out = df.select(gaia.distance_from_parallax(zero_point=-0.017)).to_series()
    assert (
        out[0] == pytest.approx(1 / 2.017) and out[1] is None and out[2] == pytest.approx(1 / 0.017)
    )


def test_lindegren21_zero_point():
    pytest.importorskip("zero_point")
    df = pl.DataFrame(
        {
            "phot_g_mean_mag": [15.0, 17.0],
            "nu_eff_used_in_astrometry": [1.5, 1.55],
            "pseudocolour": [np.nan, np.nan],
            "ecl_lat": [10.0, -40.0],
            "astrometric_params_solved": [31, 31],
        }
    )
    z = df.select(gaia.lindegren21_zero_point()).to_series()
    assert z.is_finite().all() and (z.abs() < 0.1).all()


def test_process_pipeline(rng):
    n = 100
    df = pl.DataFrame(
        {
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-80, 80, n),
            "parallax": rng.uniform(0.3, 3, n),
            "pmra": rng.normal(0, 5, n),
            "pmdec": rng.normal(0, 5, n),
            "pmra_error": np.full(n, 0.02),
            "pmdec_error": np.full(n, 0.03),
            "pmra_pmdec_corr": np.full(n, 0.1),
            "radial_velocity": rng.normal(0, 40, n),
            "radial_velocity_error": np.full(n, 1.0),
            "grvs_mag": rng.uniform(8, 14, n),
            "rv_template_teff": np.full(n, 5000.0),
        }
    )
    out = gaia.process(
        df.lazy(), correct_rv=True, correct_rv_error=True, zero_point=-0.017, pm_errors=True
    ).collect()
    for c in gaia.DEFAULT_QUANTITIES + ("distance", "radial_velocity_corrected", "pml_error"):
        assert c in out.columns, c
    ref = df.with_columns(
        distance=1 / (pl.col("parallax") + 0.017),
        radial_velocity=gaia.correct_radial_velocity(),
    ).galactic.from_observed()
    np.testing.assert_allclose(out["vZ"], ref["vZ"], rtol=1e-12)
    # galactic PMs agree with df.sky.to_galactic
    np.testing.assert_allclose(out["pmb"], df.sky.to_galactic()["pmb"], rtol=1e-12)
