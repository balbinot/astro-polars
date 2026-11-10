import numpy as np
import polars as pl
import pytest
from mer_fake import astropy_fits

from astro_polars.io.fits import read_fits_table


@pytest.fixture
def table(tmp_path):
    fits = astropy_fits()
    path = tmp_path / "t.fits"
    cols = [
        fits.Column(name="ID", format="K", array=np.arange(4, dtype=">i8")),
        fits.Column(name="RA", format="D", array=np.array([1.0, 2.0, 3.0, 4.0], dtype=">f8")),
        fits.Column(name="FLUX", format="E", array=np.array([1, 2, 3, np.nan], dtype=">f4")),
        fits.Column(name="VEC", format="3E", array=np.arange(12, dtype=">f4").reshape(4, 3)),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path)
    return path


def test_read_all(table):
    df = read_fits_table(table)
    assert df.columns == ["ID", "RA", "FLUX", "VEC"]
    assert df.schema["ID"] == pl.Int64 and df.schema["FLUX"] == pl.Float32
    assert df["RA"].to_list() == [1.0, 2.0, 3.0, 4.0]
    assert df["FLUX"].is_nan().to_list() == [False, False, False, True]
    assert df["VEC"][1].to_list() == [3.0, 4.0, 5.0]
    # data are in native byte order and can be computed on
    assert df.select(pl.col("ID").sum()).item() == 6


def test_columns_and_missing(table):
    with pytest.warns(UserWarning, match="1 requested columns missing"):
        df = read_fits_table(table, columns=["RA", "NOPE", "ID"])
    assert df.columns == ["RA", "ID"]
    df = read_fits_table(table, columns=["NOPE"], warn_missing=False)
    assert df.shape == (0, 0)


def test_integer_tnull_becomes_null(tmp_path):
    fits = astropy_fits()
    path = tmp_path / "n.fits"
    col = fits.Column(name="N", format="J", null=-99, array=np.array([1, -99, 3], dtype=">i4"))
    fits.BinTableHDU.from_columns([col]).writeto(path)
    assert read_fits_table(path)["N"].to_list() == [1, None, 3]
