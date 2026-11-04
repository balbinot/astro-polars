import numpy as np
import polars as pl
import pytest

from astro_polars import K_NMAD, nmad


def test_nmad_gaussian(rng):
    df = pl.DataFrame({"x": rng.normal(5, 2, 200_000)})
    assert df.select(pl.col("x").robust.nmad()).item() == pytest.approx(2.0, rel=0.02)
    assert df.select(nmad("x")).columns == ["x_nmad"]


def test_nmad_ignores_outliers():
    x = np.r_[np.linspace(-1, 1, 101), 1e6]
    s = pl.DataFrame({"x": x}).select(pl.col("x").robust.nmad()).item()
    med = np.median(x)
    assert s == pytest.approx(K_NMAD * np.median(np.abs(x - med)))


def test_zscore_grouped(rng):
    df = pl.DataFrame(
        {"g": [0] * 1000 + [1] * 1000, "x": np.r_[rng.normal(0, 1, 1000), rng.normal(10, 5, 1000)]}
    )
    z = df.with_columns(z=pl.col("x").robust.zscore().over("g"))
    assert z.group_by("g").agg(pl.col("z").median().abs() < 0.1)["z"].all()
