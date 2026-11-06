"""Visualisation, in the spirit of ``vaex.viz``.

Binning (``heatmap``, ``healpix_map``) needs only Polars and NumPy and
streams over LazyFrames. Plotting (``plot_heatmap``, ``skymap``, the
``df.viz`` namespace) needs the ``viz`` extra (matplotlib and skyproj):

    pip install "astro-polars[viz]"

Example:

    import astro_polars as ap
    lf.viz.heatmap("bp_rp", "phot_g_mean_mag", log=True)
    lf.viz.healpix("ra", "dec", nside=128, what="mean", z="parallax")
    H, extent = ap.viz.heatmap(lf, "ra", "dec", shape=512)
    m = ap.viz.heatmap(lf, "ra", "dec", nside=256)   # HEALPix map
"""

from .binning import AGGREGATIONS, healpix_map, heatmap
from .namespace import Viz
from .plot import PROJECTIONS, TAB10, plot_heatmap, skymap, use_style

__all__ = [
    "AGGREGATIONS",
    "PROJECTIONS",
    "TAB10",
    "Viz",
    "healpix_map",
    "heatmap",
    "plot_heatmap",
    "skymap",
    "use_style",
]
