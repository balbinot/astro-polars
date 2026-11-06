import numpy as np
import polars as pl
import pytest

from astro_polars.healpix import nside_to_order
from astro_polars.viz import healpix_map, heatmap


def test_counts_match_histogram2d(rng):
    df = pl.DataFrame({"x": rng.normal(0, 1, 50_000), "y": rng.normal(0, 2, 50_000)})
    lims = ((-3, 3), (-6, 6))
    H, extent = heatmap(df.lazy(), "x", "y", lims=lims, shape=(30, 40))
    sel = df.filter(pl.col("x").is_between(-3, 3) & pl.col("y").is_between(-6, 6))
    ref, _, _ = np.histogram2d(sel["x"], sel["y"], bins=(30, 40), range=lims)
    assert H.shape == (30, 40)
    np.testing.assert_array_equal(H, ref)
    np.testing.assert_array_equal(extent, [-3, 3, -6, 6])


def test_mean_and_fill(rng):
    df = pl.DataFrame({"x": [0.1, 0.2, 0.9], "y": [0.1, 0.1, 0.9], "z": [1.0, 3.0, 5.0]})
    H, _ = heatmap(df, "x", "y", what="mean", z="z", lims=((0, 1), (0, 1)), shape=2, fill=np.nan)
    assert H[0, 0] == 2.0 and H[1, 1] == 5.0
    assert np.isnan(H[0, 1]) and np.isnan(H[1, 0])


def test_infers_limits_and_drops_nonfinite():
    df = pl.DataFrame({"x": [0.0, 1.0, np.nan, np.inf], "y": [0.0, 1.0, 0.5, 0.5]})
    H, extent = heatmap(df, "x", "y", shape=2)
    assert H.sum() == 2
    np.testing.assert_array_equal(extent, [0, 1, 0, 1])


def test_requires_z():
    with pytest.raises(ValueError):
        heatmap(pl.DataFrame({"x": [1.0], "y": [1.0]}), "x", "y", what="mean")


def test_nside_to_order():
    assert nside_to_order(1) == 0 and nside_to_order(4096) == 12
    with pytest.raises(ValueError):
        nside_to_order(100)


def test_healpix_counts(sky):
    m = healpix_map(sky, "ra", "dec", nside=8)
    assert m.shape == (12 * 64,)
    assert m.sum() == len(sky)
    # same via heatmap(nside=...)
    np.testing.assert_array_equal(heatmap(sky.lazy(), "ra", "dec", nside=8), m)


def test_healpix_degrade_matches_direct(sky):
    fine = sky.sky.add_healpix(4096)
    direct = healpix_map(fine, "ra", "dec", nside=32, what="mean", z="mag", fill=np.nan)
    degraded = healpix_map(
        fine, nside=32, what="mean", z="mag", fill=np.nan, ipix="hpx4096", ipix_nside=4096
    )
    np.testing.assert_allclose(direct, degraded, equal_nan=True)


def test_healpix_sparse(sky):
    pix, val = healpix_map(sky, nside=64, sparse=True)
    assert (np.diff(pix) > 0).all() and val.sum() == len(sky)


def test_healpix_matches_bincount(sky):
    hpgeom = pytest.importorskip("hpgeom")
    for nest in (True, False):
        m = healpix_map(sky, nside=16, nest=nest)
        pix = hpgeom.angle_to_pixel(16, sky["ra"].to_numpy(), sky["dec"].to_numpy(), nest=nest)
        np.testing.assert_array_equal(m, np.bincount(pix, minlength=12 * 256))
