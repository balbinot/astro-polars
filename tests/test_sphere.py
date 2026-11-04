import numpy as np
import polars as pl
import pytest

from astro_polars import angsep, angular_separation, in_cone


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.degrees(2 * np.arcsin(np.sqrt(a)))


def test_known_values():
    assert angular_separation(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0)
    assert angular_separation(0.0, 90.0, 123.0, -90.0) == pytest.approx(180.0)
    assert angular_separation(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)
    # antipodes, where haversine/cosine formulas lose precision
    assert angular_separation(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0)


def test_arrays_match_haversine(sky):
    ra, dec = sky["ra"].to_numpy(), sky["dec"].to_numpy()
    sep = angular_separation(ra, dec, 150.0, 2.0)
    np.testing.assert_allclose(sep, haversine(ra, dec, 150.0, 2.0), atol=1e-9)


def test_expression_matches_numba(sky):
    out = sky.with_columns(sep=angsep("ra", "dec", 150.0, 2.0))
    ref = angular_separation(sky["ra"].to_numpy(), sky["dec"].to_numpy(), 150.0, 2.0)
    np.testing.assert_allclose(out["sep"].to_numpy(), ref, atol=1e-10)


def test_expression_between_columns(sky):
    df = sky.with_columns(ra2=pl.col("ra") + 0.001, dec2=pl.col("dec"))
    out = df.select(angsep("ra", "dec", "ra2", "dec2")).to_series().to_numpy()
    ref = angular_separation(
        df["ra"].to_numpy(), df["dec"].to_numpy(), df["ra2"].to_numpy(), df["dec2"].to_numpy()
    )
    np.testing.assert_allclose(out, ref, atol=1e-12)


def test_in_cone(sky):
    sel = sky.lazy().filter(in_cone("ra", "dec", 30.0, -10.0, 15.0)).collect()
    ref = angular_separation(sky["ra"].to_numpy(), sky["dec"].to_numpy(), 30.0, -10.0) <= 15.0
    assert len(sel) == ref.sum() > 0
