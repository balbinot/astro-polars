import numpy as np
import polars as pl

from astro_polars import close_polygon, in_polygon, inpoly, inpoly_parallel, pinside

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_inpoly_scalar():
    poly = close_polygon(SQUARE)
    assert inpoly(poly, 0.5, 0.5) == 1
    assert inpoly(poly, 1.5, 0.5) == 0
    assert inpoly(poly, 1.0, 0.5) == 2  # on an edge


def test_edges_count_as_inside():
    poly = close_polygon(SQUARE)
    res = inpoly_parallel(np.array([1.0, 0.5, 2.0]), np.array([0.5, 0.0, 0.5]), poly)
    assert res.tolist() == [True, True, False]


def test_open_and_closed_agree(rng):
    df = pl.DataFrame({"x": rng.uniform(-1, 2, 5000), "y": rng.uniform(-1, 2, 5000)})
    a = df.select(in_polygon(SQUARE, "x", "y")).to_series()
    b = df.select(in_polygon(close_polygon(SQUARE), "x", "y")).to_series()
    assert a.equals(b)
    inside = df["x"].is_between(0, 1) & df["y"].is_between(0, 1)
    assert (a == inside).all()


def test_concave(rng):
    # a "U" shape: the notch (1 < x < 2, y > 1) is outside
    poly = np.array([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]], float)
    df = pl.DataFrame({"x": [0.5, 1.5, 1.5, 2.5], "y": [2.0, 2.0, 0.5, 2.0]})
    assert df.select(in_polygon(poly, "x", "y")).to_series().to_list() == [True, False, True, True]


def test_pinside_and_namespace(rng):
    lf = pl.LazyFrame({"x": rng.uniform(-1, 2, 1000), "y": rng.uniform(-1, 2, 1000)})
    out = lf.with_columns(pinside(None, SQUARE, "x", "y")).collect()
    assert out["inside"].dtype == pl.Boolean
    sel = lf.sky.polygon(SQUARE).collect()
    assert len(sel) == out["inside"].sum()


def test_streaming(rng):
    lf = pl.LazyFrame({"x": rng.uniform(-1, 2, 100_000), "y": rng.uniform(-1, 2, 100_000)})
    a = lf.filter(in_polygon(SQUARE, "x", "y")).collect(engine="streaming")
    b = lf.filter(in_polygon(SQUARE, "x", "y")).collect()
    assert len(a) == len(b)
