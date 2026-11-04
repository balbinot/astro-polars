import math

import numpy as np
import polars as pl
import pytest

import astro_polars as ap
from astro_polars.photometry import AB_ZEROPOINTS


def test_zeropoints():
    assert ap.ab_zeropoint("Jy") == pytest.approx(2.5 * math.log10(3631))
    assert ap.ab_zeropoint("uJy") == pytest.approx(23.9, abs=1e-3)
    assert ap.ab_zeropoint("nJy") == pytest.approx(31.4, abs=1e-3)
    assert ap.ab_zeropoint("nanomaggy") == pytest.approx(22.5, abs=1e-12)
    assert ap.ab_zeropoint(1e-6) == AB_ZEROPOINTS["uJy"]
    assert ap.MAGERR_FACTOR == pytest.approx(1.0857362, rel=1e-7)


def test_flux_mag_round_trip():
    df = pl.DataFrame({"f": [1.0, 10.0, 3631e6], "zp": [20.0, 21.0, 22.0]})
    out = df.select(
        m=ap.flux_to_mag("f"),
        back=ap.mag_to_flux(ap.flux_to_mag("f")),
        m_nmgy=ap.flux_to_mag("f", "nanomaggy"),
        m_col=ap.flux_to_mag("f", pl.col("zp")),
    )
    np.testing.assert_allclose(out["m"], [23.9, 21.4, 0.0], atol=1e-3)
    np.testing.assert_allclose(out["back"], df["f"], rtol=1e-12)
    np.testing.assert_allclose(out["m_nmgy"], [22.5, 20.0, 22.5 - 2.5 * math.log10(3631e6)])
    np.testing.assert_allclose(out["m_col"], [20.0, 18.5, 22.0 - 2.5 * math.log10(3631e6)])


def test_nonpositive_flux_is_not_finite():
    out = pl.DataFrame({"f": [-1.0, 0.0]}).select(ap.flux_to_mag("f")).to_series()
    assert not out.is_finite().any()


def test_float32_kept():
    df = pl.DataFrame({"f": pl.Series([1.0, 2.0], dtype=pl.Float32)})
    assert df.select(ap.flux_to_mag("f")).dtypes == [pl.Float32]
    wide = ap.flux_to_mag("f", pl.lit(23.9, dtype=pl.Float64))
    assert df.select(wide).dtypes == [pl.Float64]


def test_errors_against_monte_carlo(rng):
    flux, err = 100.0, 2.0
    mags = -2.5 * np.log10(rng.normal(flux, err, 200_000))
    df = pl.DataFrame({"f": [flux], "e": [err]})
    magerr = df.select(ap.fluxerr_to_magerr("e", "f")).item()
    assert magerr == pytest.approx(mags.std(), rel=0.01)
    back = df.with_columns(me=ap.fluxerr_to_magerr("e", "f")).select(
        ap.magerr_to_fluxerr("me", "f")
    )
    assert back.item() == pytest.approx(err)


def test_color_aperture_dereddening():
    df = pl.DataFrame(
        {"a": [10.0], "b": [1.0], "tot": [3.0], "ap": [2.0], "g": [20.0], "ebv": [0.1]}
    )
    out = df.select(
        c=ap.flux_color("a", "b"),
        c2=ap.flux_to_mag("a") - ap.flux_to_mag("b"),
        corr=ap.aperture_correction("tot", "ap"),
        g0=ap.deredden("g", "ebv", 3.2),
    ).row(0, named=True)
    assert out["c"] == pytest.approx(-2.5) == pytest.approx(out["c2"])
    assert out["corr"] == pytest.approx(1.5)
    assert out["g0"] == pytest.approx(19.68)


def test_phot_namespace_skips_missing_columns():
    df = pl.DataFrame({"fg": [10.0], "fr": [1.0], "eg": [0.1], "ebv": [0.1]})
    out = (
        df.lazy()
        .phot.add_mags({"g": "fg", "r": "fr", "i": "fi"})
        .phot.add_magerrs({"g_err": ("eg", "fg"), "r_err": ("er", "fr")})
        .phot.add_colors({"g_r": ("fg", "fr")})
        .phot.deredden({"g": 3.2, "r": 2.2, "i": 1.6})
        .collect()
    )
    assert {"g", "r", "g_err", "g_r", "g_0", "r_0"} <= set(out.columns)
    assert not {"i", "r_err", "i_0"} & set(out.columns)
    assert out["g_r"].item() == pytest.approx(out["g"].item() - out["r"].item())
    assert out["g_0"].item() == pytest.approx(out["g"].item() - 0.32)


def test_phot_namespace_strict():
    df = pl.DataFrame({"fg": [10.0]})
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="fi"):
        df.phot.add_mags({"i": "fi"}, strict=True)
