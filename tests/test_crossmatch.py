import numpy as np
import polars as pl
import pytest

from astro_polars import (
    angular_separation,
    crossmatch,
    crossmatch_polars,
    find_duplicates,
    remove_duplicates,
)


@pytest.fixture
def cats(rng):
    n = 5000
    ra = rng.uniform(10, 20, n)
    dec = rng.uniform(-5, 5, n)
    cat1 = pl.DataFrame({"id": np.arange(n), "ra": ra, "dec": dec})
    # cat2: first 3000 sources shifted by < 0.2", plus 1000 unrelated sources
    m = 3000
    off = rng.uniform(-0.1, 0.1, (m, 2)) / 3600
    ra2 = np.r_[ra[:m] + off[:, 0] / np.cos(np.radians(dec[:m])), rng.uniform(30, 40, 1000)]
    dec2 = np.r_[dec[:m] + off[:, 1], rng.uniform(-5, 5, 1000)]
    cat2 = pl.DataFrame({"oid": np.arange(len(ra2)) + 100_000, "RA": ra2, "DEC": dec2})
    return cat1, cat2


def test_inner(cats):
    cat1, cat2 = cats
    out = crossmatch(
        cat1.lazy(), cat2.lazy(), x2="RA", y2="DEC", radius_arcsec=0.5, how="inner"
    ).collect()
    assert len(out) == 3000
    assert (out["oid"] - 100_000 == out["id"]).all()
    sep = (
        angular_separation(
            out["ra"].to_numpy(), out["dec"].to_numpy(), out["RA"].to_numpy(), out["DEC"].to_numpy()
        )
        * 3600
    )
    np.testing.assert_allclose(out["separation"].to_numpy(), sep, atol=1e-6)
    assert (out["multiplicity"] == 1).all()


def test_left_keeps_all_of_df1(cats):
    cat1, cat2 = cats
    out = crossmatch_polars(cat1, cat2, x2="RA", y2="DEC", radius_arcsec=0.5).collect()
    assert len(out) == len(cat1)
    assert out["oid"].null_count() == len(cat1) - 3000


def test_keep_and_suffix(cats):
    cat1, cat2 = cats
    out = crossmatch(
        cat1,
        cat2,
        x2="RA",
        y2="DEC",
        keep1=["id"],
        keep2=["oid", "nope"],
        suffix1="_a",
        suffix2="_b",
        how="inner",
    ).collect()
    assert out.columns == ["id_a", "oid_b", "separation", "multiplicity"]


@pytest.mark.parametrize("method", ["kdtree", "balltree"])
def test_threads_match_serial(cats, method):
    cat1, cat2 = cats
    kw = dict(x2="RA", y2="DEC", k=3, radius_arcsec=3600, how="inner", method=method)
    a = crossmatch(cat1, cat2, n_jobs=1, **kw).collect()
    b = crossmatch(cat1, cat2, n_jobs=4, **kw).collect()
    assert a.sort("id", "oid").equals(b.sort("id", "oid"))


def test_kdtree_matches_balltree(cats):
    cat1, cat2 = cats
    kw = dict(x2="RA", y2="DEC", k=2, radius_arcsec=60, how="inner")
    a = crossmatch(cat1, cat2, method="kdtree", **kw).collect().sort("oid", "id")
    b = crossmatch(cat1, cat2, method="balltree", **kw).collect().sort("oid", "id")
    assert a.select("id", "oid", "multiplicity").equals(b.select("id", "oid", "multiplicity"))
    np.testing.assert_allclose(a["separation"], b["separation"], atol=1e-6)


def test_remove_duplicates(rng):
    n = 2000
    df = pl.DataFrame(
        {"source_id": np.arange(n), "ra": rng.uniform(0, 1, n), "dec": rng.uniform(0, 1, n)}
    )
    dup = df.head(50).with_columns(pl.col("ra") + 0.3 / 3600, pl.col("source_id") + 10_000)
    both = pl.concat([df, dup])
    out = remove_duplicates(both, radius_arcsec=1.0)
    assert out.columns == both.columns
    # every injected duplicate (later rows) is removed, originals are kept
    assert out["source_id"].max() < 10_000
    assert len(out) <= n and len(out) >= n - 60


def test_same_column_names_get_suffix(cats):
    cat1, _ = cats
    cat2 = cat1.with_columns(pl.col("ra") + 0.1 / 3600)
    out = crossmatch(cat1, cat2, radius_arcsec=1, how="inner").collect()
    assert out.columns == ["id", "ra", "dec", "id_2", "ra_2", "dec_2", "separation", "multiplicity"]
    assert (out["id"] == out["id_2"]).all()


ARCSEC = 1 / 3600


def _line(xs, **cols):
    """Sources on the equator at ``xs`` arcsec east of RA = 10 deg."""
    return pl.DataFrame(
        {
            "id": list("ABCDEFGH"[: len(xs)]),
            "ra": [10 + v * ARCSEC for v in xs],
            "dec": [0.0] * len(xs),
            **cols,
        }
    )


@pytest.mark.parametrize("method", ["kdtree", "balltree"])
@pytest.mark.parametrize(
    "xs, kept",
    [
        ([0, 0, 0], ["A"]),  # exact triple
        ([0, 0, 0, 0], ["A"]),  # exact quadruple
        ([0, 0.9, 0.45], ["A"]),  # triple where nearest neighbours skip A-B
        ([0, 0.9, 0.45, 0.2], ["A"]),  # quadruple, same
        ([0, 0.8, 1.6], ["A"]),  # chain A~B~C with A, C 1.6" apart: one group
        ([0, 0.8, 1.6, 2.4], ["A"]),
        ([0, 0.5, 5, 5.5, 10], ["A", "C", "E"]),  # two pairs and a single
    ],
)
def test_remove_duplicates_multiples(xs, kept, method):
    out = remove_duplicates(_line(xs), radius_arcsec=1.0, method=method)
    assert out["id"].to_list() == kept


def test_find_duplicates_groups():
    out = find_duplicates(_line([0, 0.8, 1.6, 5, 10, 10.5]), radius_arcsec=1.0)
    assert out["dup_group"].to_list() == [0, 0, 0, 1, 2, 2]
    assert out["dup_n"].to_list() == [3, 3, 3, 1, 2, 2]


def test_remove_duplicates_priority():
    df = _line([0, 0.3, 0.6, 5], err=[0.3, 0.1, 0.2, 1.0])
    assert remove_duplicates(df, priority="err")["id"].to_list() == ["B", "D"]
    assert remove_duplicates(df, priority="err", descending=True)["id"].to_list() == ["A", "D"]


def test_remove_duplicates_agg():
    df = _line([0, 0.3, 0.6, 5], g=[20.0, 21.0, 25.0, 18.0])
    med = remove_duplicates(df, agg="median", size="n")
    assert med.columns == ["id", "ra", "dec", "g", "n"]
    assert med["g"].to_list() == [21.0, 18.0]
    assert med["n"].to_list() == [3, 1]
    assert med["id"].to_list() == ["A", "D"]  # non-numeric columns from the representative
    np.testing.assert_allclose(med["ra"][0], 10 + 0.3 * ARCSEC, atol=1e-9)

    custom = remove_duplicates(df, agg=[pl.col("g").mean(), pl.col("g").std().alias("g_std")])
    assert custom.columns == ["id", "ra", "dec", "g", "g_std"]
    np.testing.assert_allclose(custom["g"].to_list(), [22.0, 18.0])

    by_col = remove_duplicates(df, agg={"g": "max"})
    assert by_col["g"].to_list() == [25.0, 18.0]


def test_remove_duplicates_centroid_wraps_ra():
    df = pl.DataFrame({"ra": [359.9999, 0.0001], "dec": [0.0, 0.0]})
    out = remove_duplicates(df, agg="mean")
    assert len(out) == 1
    assert min(out["ra"][0], 360 - out["ra"][0]) < 1e-6


def test_remove_duplicates_leaves_no_close_pairs(rng):
    n = 20_000
    ra, dec = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    base = rng.choice(n, 1000, replace=False)
    mult = rng.integers(1, 5, 1000)  # 1-4 extra copies
    ang = rng.uniform(0, 2 * np.pi, mult.sum())
    r = rng.uniform(0, 0.45, mult.sum()) * ARCSEC
    df = pl.DataFrame(
        {
            "ra": np.r_[ra, np.repeat(ra[base], mult) + r * np.cos(ang)],
            "dec": np.r_[dec, np.repeat(dec[base], mult) + r * np.sin(ang)],
        }
    ).sample(fraction=1, shuffle=True, seed=0)
    out = remove_duplicates(df, radius_arcsec=1.0)
    assert len(find_duplicates(out, radius_arcsec=1.0).filter(pl.col("dup_n") > 1)) == 0


def test_namespace_remove_duplicates():
    df = _line([0, 0.3, 5], g=[1.0, 3.0, 5.0])
    assert df.sky.remove_duplicates(agg="mean")["g"].to_list() == [2.0, 5.0]
    assert df.sky.find_duplicates()["dup_n"].to_list() == [2, 2, 1]
