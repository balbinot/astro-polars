import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from astro_polars import viz  # noqa: E402


def test_use_style():
    viz.use_style()
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    assert colors == viz.TAB10


def test_df_viz_heatmap(sky):
    im = sky.lazy().viz.heatmap("ra", "dec", shape=64, log=True)
    assert im.get_array().shape == (64, 64)
    plt.close("all")


def test_df_viz_healpix(sky):
    pytest.importorskip("skyproj")
    sp = sky.viz.healpix("ra", "dec", nside=16, what="mean", z="mag", projection="mcbryde")
    assert sp is not None
    sp2 = viz.skymap(viz.healpix_map(sky, nside=8, fill=np.nan), projection="mollweide")
    assert sp2 is not None
    plt.close("all")
