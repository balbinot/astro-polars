import json
import os

import numpy as np
import polars as pl
import pytest
from mer_fake import DROPPED, astropy_fits, mer_columns, write_mer_tile

import astro_polars as ap
from astro_polars import cli
from astro_polars.io import scan_hive
from astro_polars.surveys import euclid


@pytest.fixture
def mer(rng):
    return pl.DataFrame(mer_columns(3000, rng, 150.0, 2.0, tile=102018))


def test_constants():
    assert euclid.ZPT == pytest.approx(23.9, abs=1e-3)
    assert len(euclid.KEEPCOLS) == len(set(euclid.KEEPCOLS))
    for name, flux in euclid.MAG_MAP.items():
        assert name.startswith("mag_") and flux in euclid.KEEPCOLS
    for err, flux in euclid.MAGERR_MAP.values():
        assert err in euclid.KEEPCOLS and flux in euclid.KEEPCOLS
    for sign, flux, visto in euclid.VISTO_MAP.values():
        assert sign in (1, -1) and {flux, visto} <= set(euclid.KEEPCOLS)


def test_magnitudes(mer):
    out = euclid.add_magerrs(
        euclid.add_aper_mags(euclid.add_mags(euclid.add_aperture_corrections(mer)))
    )
    f = mer["FLUX_DETECTION_TOTAL"].to_numpy().astype(np.float64)
    np.testing.assert_allclose(out["mag_VIS_auto"], -2.5 * np.log10(f) + 23.9, atol=2e-4)
    assert out.schema["mag_VIS_auto"] == pl.Float64

    corr1 = f / mer["FLUX_VIS_1FWHM_APER"].to_numpy()
    np.testing.assert_allclose(out["corr1"], corr1, rtol=1e-6)
    # the 1-FWHM VIS aperture magnitude corrected to total is the total magnitude
    np.testing.assert_allclose(out["mag_VIS_1fwhm"], out["mag_VIS_auto"], atol=1e-5)

    e = mer["FLUXERR_Y_TEMPLFIT"].to_numpy() / mer["FLUX_Y_TEMPLFIT"].to_numpy()
    np.testing.assert_allclose(out["magerr_Y"], e * 2.5 / np.log(10), rtol=1e-6)


def test_visto_colours(mer):
    out = euclid.add_visto_colors(mer)
    h, vh = mer["FLUX_H_TEMPLFIT"].to_numpy(), mer["FLUX_VIS_TO_H_TEMPLFIT"].to_numpy()
    g, vg = mer["FLUX_G_EXT_DECAM_TEMPLFIT"], mer["FLUX_VIS_TO_G_EXT_DECAM_TEMPLFIT"]
    with np.errstate(invalid="ignore"):  # negative fluxes
        ref_h, ref_g = 2.5 * np.log10(h / vh), -2.5 * np.log10((g / vg).to_numpy())
    np.testing.assert_allclose(out["VIStoH"], ref_h, rtol=1e-5, equal_nan=True)
    np.testing.assert_allclose(out["GtoVIS"], ref_g, rtol=1e-5, equal_nan=True)


def test_missing_columns_are_skipped(mer):
    out = euclid.transform(mer)
    assert DROPPED and not any("JPCAM" in c for c in out.columns if c not in DROPPED)
    smaller = euclid.transform(mer.drop("FLUX_H_TEMPLFIT"))
    assert "mag_H" not in smaller.columns and "VIStoH" not in smaller.columns
    assert "mag_J" in smaller.columns


def test_transform_cuts(mer):
    out = euclid.transform(mer)
    assert 0 < len(out) < len(mer)
    assert out["MUMAX_MINUS_MAG"].is_finite().all() and out["POINT_LIKE_PROB"].is_finite().all()
    assert (out["mag_VIS_auto"] < euclid.MAG_VIS_AUTO_LIM).all()
    assert out["OBJECT_ID"].is_unique().all()
    w = 0.6 + np.exp((out["mag_VIS_auto"].to_numpy() - 27.0) / 1.5)
    assert (np.abs(out["MUMAX_MINUS_MAG"].to_numpy() + 3) <= w + 1e-6).all()

    # lazy and eager agree, and order is kept
    lazy = euclid.transform(mer.lazy()).collect()
    assert lazy.equals(out)
    assert out["OBJECT_ID"].is_sorted()

    loose = euclid.transform(mer, horn=None, mag_vis_auto_max=None)
    assert len(loose) > len(out)


def test_add_healpix(mer):
    out = euclid.add_healpix(mer, nside=32)
    fine = mer.select(ap.healpix_index(euclid.RA, euclid.DEC, 4096)).to_series()
    np.testing.assert_array_equal(out["hpx4096"], fine)
    np.testing.assert_array_equal(out["hpx32"], fine // 4 ** (12 - 5))
    with pytest.raises(ValueError):
        euclid.add_healpix(mer, nside=8192)


# --------------------------------------------------------------------------
# FITS tiles -> hive dataset
# --------------------------------------------------------------------------


@pytest.fixture
def tiles(tmp_path):
    astropy_fits()
    root = tmp_path / "tiles"
    paths = [
        write_mer_tile(root, 102018, n=1500, ra0=150.0, dec0=2.0, seed=1),
        write_mer_tile(root, 102019, n=1500, ra0=151.5, dec0=2.5, seed=2),
    ]
    return root, paths


def _quiet(*args, **kw):
    return euclid.convert_tiles(*args, progress=None, **kw)


@pytest.mark.filterwarnings("ignore:.*requested columns missing")
def test_read_mer(tiles):
    _, paths = tiles
    df = euclid.read_mer(paths[0])
    assert df.columns == [c for c in euclid.KEEPCOLS if c not in DROPPED]
    assert len(df) == 1500 + 30


@pytest.mark.filterwarnings("ignore:.*requested columns missing")
def test_convert_tiles(tiles, tmp_path):
    root, paths = tiles
    out = tmp_path / "hive"
    summary = _quiet(root, out, nside=32)
    assert summary == {"found": 2, "processed": 2, "failed": 0, "skipped": 0, "errors": {}}

    manifest = json.loads((out / "_manifest.json").read_text())
    assert set(manifest) == {os.path.basename(p) for p in paths}
    rows = sum(m["rows"] for m in manifest.values())

    lf = scan_hive(out)
    df = lf.collect()
    assert len(df) == rows
    assert {"hpx32", "hpx4096", "mag_VIS_auto", "VIStoH"} <= set(df.columns)
    np.testing.assert_array_equal(df["hpx32"], df["hpx4096"] // 4**7)

    # same rows as running the steps by hand
    ref = euclid.transform(euclid.read_mer(paths[0]))
    got = df.filter(pl.col("TILE_INDEX") == 102018).sort("OBJECT_ID")
    assert got.select(ref.columns).equals(ref.sort("OBJECT_ID"))

    # a second run skips everything
    assert _quiet(root, out)["skipped"] == 2

    # a cone read touches fewer partitions and loses no source in the cone
    cone = (150.0, 2.0, 0.5)
    exact = scan_hive(out, cone=cone).sky.cone(*cone, lon=euclid.RA, lat=euclid.DEC).collect()
    assert len(exact) == len(df.sky.cone(*cone, lon=euclid.RA, lat=euclid.DEC)) > 0


@pytest.mark.filterwarnings("ignore:.*requested columns missing")
def test_convert_tiles_redo_and_force(tiles, tmp_path):
    root, paths = tiles
    out = tmp_path / "hive"
    _quiet(root, out, nside=16)
    before = scan_hive(out).select(pl.len()).collect().item()

    # a stale file from an earlier attempt at a tile is removed when it is redone
    stale = out / "hpx16=0" / "part-102018-0.parquet"
    stale.parent.mkdir()
    pl.DataFrame({"OBJECT_ID": [1]}).write_parquet(stale)
    write_mer_tile(root, 102018, n=1000, ra0=150.0, dec0=2.0, seed=3)  # new size -> redone
    s = _quiet(root, out, nside=16)
    assert (s["processed"], s["skipped"]) == (1, 1)
    assert not stale.exists()
    assert scan_hive(out).select(pl.len()).collect().item() < before

    assert _quiet(root, out, nside=16, force=True)["processed"] == 2


@pytest.mark.filterwarnings("ignore:.*requested columns missing")
def test_convert_tiles_reports_failures(tiles, tmp_path):
    root, _ = tiles
    bad = root / "999" / "EUC_MER_FINAL-CAT_TILE999-X.fits"
    bad.parent.mkdir()
    bad.write_bytes(b"not a fits file")
    s = _quiet(root, tmp_path / "hive")
    assert s["processed"] == 2 and s["failed"] == 1 and bad.name in s["errors"]
    manifest = json.loads((tmp_path / "hive" / "_manifest.json").read_text())
    assert bad.name not in manifest


def test_convert_tiles_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        _quiet(tmp_path, tmp_path / "out")
    with pytest.raises(ValueError):
        _quiet(tmp_path, tmp_path / "out", nside=3)


@pytest.mark.filterwarnings("ignore:.*requested columns missing")
def test_cli(tiles, tmp_path, capsys):
    root, _ = tiles
    out = tmp_path / "hive"
    assert (
        cli.main(["euclid-hive", "--indir", str(root), "--outdir", str(out), "--nside", "32"]) == 0
    )
    text = capsys.readouterr().out
    assert "2 processed" in text and "scan_hive" in text
    assert cli.main(["euclid-hive", "--indir", str(tmp_path / "none"), "--outdir", str(out)]) == 2
