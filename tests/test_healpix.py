import numpy as np
import polars as pl
import pytest

import astro_polars as ap
from astro_polars.healpix import nside_to_order, order_to_nside, pixel_area


def test_orders():
    assert nside_to_order(4096) == 12 and order_to_nside(12) == 4096
    with pytest.raises(ValueError):
        nside_to_order(3)
    assert pixel_area(1) * 12 == pytest.approx(41252.96, rel=1e-5)


def test_index_matches_hpgeom(sky):
    hpgeom = pytest.importorskip("hpgeom")
    for nest in (True, False):
        out = sky.select(ap.healpix_index("ra", "dec", 64, nest=nest)).to_series().to_numpy()
        ref = hpgeom.angle_to_pixel(64, sky["ra"].to_numpy(), sky["dec"].to_numpy(), nest=nest)
        np.testing.assert_array_equal(out, ref)


def test_nulls_propagate():
    df = pl.DataFrame({"ra": [10.0, None, 20.0], "dec": [0.0, 5.0, None]})
    out = df.select(ap.healpix_index("ra", "dec", 32)).to_series()
    assert out.null_count() == 2 and out[0] is not None


def test_degrade_is_parent(sky):
    df = sky.sky.add_healpix(4096).sky.add_healpix(32)
    out = df.select(ap.degrade("hpx4096", 4096, 32)).to_series()
    assert out.equals(df["hpx32"], check_names=False)


def test_gaia_source_id():
    # a source in nested pixel p at order 12 has source_id in [p*2**35, (p+1)*2**35)
    p = 1234567
    sid = p * 2**35 + 98765
    df = pl.DataFrame({"source_id": [sid]})
    assert df.select(ap.gaia_source_id_to_healpix("source_id", 4096)).item() == p
    assert df.select(ap.gaia_source_id_to_healpix("source_id", 32)).item() == p // 4**7


def test_cone_pixels_cover_cone(sky):
    pix = ap.cone_pixels(40.0, -20.0, 5.0, nside=64)
    inside = sky.sky.cone(40.0, -20.0, 5.0).sky.add_healpix(64)
    assert set(inside["hpx64"].to_list()) <= set(pix.tolist())


def test_nan_and_inf_give_null():
    df = pl.DataFrame({"ra": [10.0, float("nan"), float("inf")], "dec": [0.0, 5.0, 5.0]})
    out = df.select(ap.healpix_index("ra", "dec", 32)).to_series()
    assert out.null_count() == 2 and out[0] is not None
