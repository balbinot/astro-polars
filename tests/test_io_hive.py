import json
import os

import numpy as np
import polars as pl
import pytest

import astro_polars as ap
from astro_polars.io import Manifest, remove_partitions, scan_hive, write_partitions
from astro_polars.io.hive import partition_nside, partition_pixels


@pytest.fixture
def cat(rng):
    n = 5000
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return pl.DataFrame({"id": np.arange(n), "ra": ra, "dec": dec})


def _files(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(root)
        for f in fs
        if not f.startswith("_")
    )


def test_write_and_scan(tmp_path, cat):
    n = write_partitions(cat, tmp_path, nside=4, tag="a")
    files = _files(tmp_path)
    assert n == len(files) == cat.select(ap.healpix_index("ra", "dec", 4).n_unique()).item()
    assert all(f.endswith(".parquet") and "part-a-" in f for f in files)
    assert not any(f.endswith(".tmp") for f in files)
    assert partition_nside(tmp_path) == 4

    lf = scan_hive(tmp_path)
    assert lf.collect_schema()["hpx4"] == pl.Int64
    out = lf.collect().sort("id")
    assert out.drop("hpx4").equals(cat)
    expected = cat.select(ap.healpix_index("ra", "dec", 4)).to_series()
    np.testing.assert_array_equal(out["hpx4"], expected)


def test_partition_from_existing_index(tmp_path, cat):
    cat = cat.with_columns(ap.healpix_index("ra", "dec", 64).alias("hpx64"))
    write_partitions(cat, tmp_path, nside=8, ipix="hpx64", ipix_nside=64, lon=None, lat=None)
    out = scan_hive(tmp_path).collect()
    np.testing.assert_array_equal(out["hpx8"], out["hpx64"] // 64)
    with pytest.raises(ValueError):
        write_partitions(cat, tmp_path / "x", nside=8, ipix="hpx64")


def test_inputs_do_not_collide_and_can_be_redone(tmp_path, cat):
    a, b = cat[:2500], cat[2500:]
    write_partitions(a, tmp_path, nside=2, tag="a")
    write_partitions(b, tmp_path, nside=2, tag="b")
    assert scan_hive(tmp_path).select(pl.len()).collect().item() == len(cat)

    removed = remove_partitions(tmp_path, nside=2, tag="a")
    assert removed > 0 and all("part-b-" in f for f in _files(tmp_path))
    assert scan_hive(tmp_path).select(pl.len()).collect().item() == len(b)


def test_cone_pruning(tmp_path, cat):
    write_partitions(cat, tmp_path, nside=8)
    cone = (120.0, 30.0, 5.0)
    pruned = scan_hive(tmp_path, cone=cone).collect()
    assert 0 < len(pruned) < len(cat)
    assert set(pruned["hpx8"].unique()) <= set(ap.cone_pixels(*cone, nside=8).tolist())
    # every row inside the cone survives the pruning
    inside = cat.sky.cone(*cone)
    assert set(inside["id"]) <= set(pruned["id"])
    exact = scan_hive(tmp_path, cone=cone).sky.cone(*cone).collect()
    assert sorted(exact["id"]) == sorted(inside["id"])


def test_pixels_and_empty_selection(tmp_path, cat):
    write_partitions(cat, tmp_path, nside=2)
    pix = partition_pixels(tmp_path)
    assert pix.tolist() == sorted(pix.tolist())
    one = scan_hive(tmp_path, pixels=[int(pix[0])]).collect()
    assert one["hpx2"].unique().to_list() == [pix[0]]

    empty = scan_hive(tmp_path, pixels=[-1]).collect()
    assert len(empty) == 0 and "ra" in empty.columns

    with pytest.raises(FileNotFoundError):
        scan_hive(tmp_path / "nothing")


def test_manifest(tmp_path):
    src = tmp_path / "input.fits"
    src.write_bytes(b"x" * 10)
    m = Manifest.load(tmp_path / "out" / "_manifest.json")
    assert len(m) == 0 and not m.is_done(src)

    m.record(src, rows=3)
    m.save()
    again = Manifest.load(tmp_path / "out" / "_manifest.json")
    assert again.is_done(src) and src in again
    assert json.loads((tmp_path / "out" / "_manifest.json").read_text())["input.fits"]["rows"] == 3
    assert not list((tmp_path / "out").glob("*.tmp"))

    src.write_bytes(b"x" * 11)  # a re-download with a different size is redone
    assert not again.is_done(src)


def test_rows_without_partition_raise(tmp_path, cat):
    bad = cat.with_columns(pl.when(pl.col("id") == 3).then(None).otherwise("ra").alias("ra"))
    with pytest.raises(ValueError, match="no hpx32 partition"):
        write_partitions(bad, tmp_path, nside=32, tag="t")


def test_remove_partitions_matches_tag_exactly(tmp_path, cat):
    write_partitions(cat, tmp_path, nside=8, tag="a")
    write_partitions(cat, tmp_path, nside=8, tag="a-b")
    n_ab = len([f for f in _files(tmp_path) if "part-a-b-" in f])
    assert remove_partitions(tmp_path, 8, "a") == n_ab
    left = _files(tmp_path)
    assert len(left) == n_ab and all("part-a-b-" in f for f in left)
