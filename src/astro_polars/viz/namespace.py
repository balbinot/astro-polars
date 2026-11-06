"""``df.viz`` / ``lf.viz``: plot straight from a DataFrame or LazyFrame, as in ``vaex.viz``."""

from __future__ import annotations

from typing import Any

import polars as pl

from .binning import healpix_map, heatmap


@pl.api.register_dataframe_namespace("viz")
@pl.api.register_lazyframe_namespace("viz")
class Viz:
    """Plotting namespace. Binning is streamed through Polars; only the binned map
    reaches matplotlib."""

    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        self._df = df

    def heatmap(
        self,
        x: str,
        y: str,
        what: str = "len",
        z: str | None = None,
        lims=None,
        shape=256,
        fill: float | None = None,
        log: bool = False,
        ax=None,
        colorbar: bool = True,
        label: str | None = None,
        *,
        nside: int | None = None,
        **kwargs: Any,
    ):
        """Bin and plot ``what(z)`` on an ``x``-``y`` grid with ``imshow``.

        Give ``nside`` to bin ``x``/``y`` as lon/lat on HEALPix and draw a sky map
        instead (same as ``healpix``). Empty bins are NaN (blank) unless
        ``fill`` is given. Other keywords go to ``imshow`` or to
        ``astro_polars.viz.skymap``.
        """
        from .plot import plot_heatmap

        if nside is not None:
            return self.healpix(
                x,
                y,
                nside=nside,
                what=what,
                z=z,
                log=log,
                ax=ax,
                colorbar=colorbar,
                label=label,
                **kwargs,
            )
        fill_value = float("nan") if fill is None else fill
        H, extent = heatmap(self._df, x, y, what=what, z=z, lims=lims, shape=shape, fill=fill_value)
        im = plot_heatmap(
            H,
            extent,
            ax=ax,
            log=log,
            colorbar=colorbar,
            label=label if label is not None else _label(what, z),
            **kwargs,
        )
        im.axes.set_xlabel(x)
        im.axes.set_ylabel(y)
        return im

    def healpix(
        self,
        lon: str = "ra",
        lat: str = "dec",
        nside: int = 64,
        what: str = "len",
        z: str | None = None,
        *,
        ipix: str | None = None,
        ipix_nside: int | None = None,
        projection="mollweide",
        log: bool = False,
        ax=None,
        colorbar: bool = True,
        label: str | None = None,
        **kwargs: Any,
    ):
        """Bin on HEALPix pixels and draw a sky map with skyproj.

        Only non-empty pixels are drawn. With ``ipix``/``ipix_nside`` an existing
        nested pixel column is degraded instead of recomputing from lon/lat
        (fastest). Returns the skyproj object.
        """
        from .plot import skymap

        pixels, values = healpix_map(
            self._df,
            lon,
            lat,
            nside=nside,
            what=what,
            z=z,
            ipix=ipix,
            ipix_nside=ipix_nside,
            sparse=True,
        )
        return skymap(
            (pixels, values),
            nside=nside,
            nest=True,
            projection=projection,
            log=log,
            ax=ax,
            colorbar=colorbar,
            label=label if label is not None else _label(what, z),
            **kwargs,
        )

    #: Alias of ``healpix``.
    sky = healpix


def _label(what: str, z: str | None) -> str:
    return "counts" if what in ("len", "count") else f"{what}({z})"
