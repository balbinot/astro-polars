"""Plotting helpers. Needs the ``viz`` extra: ``pip install astro-polars[viz]``."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import require

__all__ = ["TAB10", "use_style", "plot_heatmap", "skymap", "PROJECTIONS"]

#: Matplotlib's tab10 colours, the cycle set by ``use_style``.
TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

#: Short names for skyproj projections.
PROJECTIONS = {
    "mollweide": "MollweideSkyproj",
    "mcbryde": "McBrydeSkyproj",
    "hammer": "HammerSkyproj",
    "equalearth": "EqualEarthSkyproj",
    "laea": "LaeaSkyproj",
    "gnomonic": "GnomonicSkyproj",
    "albers": "AlbersSkyproj",
    "cyl": "Skyproj",
    "car": "Skyproj",
    "des": "DESSkyproj",
    "des-mcbryde": "DESMcBrydeSkyproj",
    "decals": "DecalsSkyproj",
}


def use_style(colors: list[str] | None = None) -> None:
    """Set the matplotlib colour cycle (tab10 by default) for all later plots."""
    mpl = require("matplotlib", "viz")
    from cycler import cycler

    mpl.rcParams["axes.prop_cycle"] = cycler("color", list(colors or TAB10))


def plot_heatmap(
    H: np.ndarray,
    extent,
    ax=None,
    log: bool = False,
    colorbar: bool = True,
    label: str | None = None,
    **imshow_kwargs: Any,
):
    """Show the output of ``astro_polars.viz.heatmap`` with ``imshow``.

    ``H`` is x-first (as returned by ``heatmap``); it is transposed here.
    Returns the ``AxesImage``.
    """
    require("matplotlib", "viz")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ax = ax if ax is not None else plt.gca()
    kw = dict(origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    if log:
        # matplotlib refuses vmin/vmax next to a norm, so they go into the norm
        vmin = imshow_kwargs.pop("vmin", None)
        vmax = imshow_kwargs.pop("vmax", None)
        kw["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    kw.update(imshow_kwargs)
    im = ax.imshow(np.asarray(H).T, **kw)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=label)
    return im


def _projection_class(projection):
    skyproj = require("skyproj", "viz")
    if isinstance(projection, str):
        key = projection.lower()
        if key not in PROJECTIONS:
            raise ValueError(f"Unknown projection {projection!r}; use one of {sorted(PROJECTIONS)}")
        return getattr(skyproj, PROJECTIONS[key])
    return projection


def _need_positive(good: np.ndarray) -> None:
    if not good.any():
        raise ValueError("log=True needs at least one pixel with a positive, finite value")


def skymap(
    data,
    nside: int | None = None,
    nest: bool = True,
    projection="mollweide",
    ax=None,
    lon_0: float = 0.0,
    galactic: bool = False,
    colorbar: bool = True,
    label: str | None = None,
    zoom: bool = False,
    log: bool = False,
    figsize=(10, 6),
    **kwargs: Any,
):
    """Draw a HEALPix map with [skyproj](https://skyproj.readthedocs.io).

    Parameters
    ----------
    data
        A full-sky map (1D array) or a ``(pixels, values)`` tuple as returned
        by ``healpix_map(..., sparse=True)``.
    nside
        Required for ``(pixels, values)`` input.
    nest
        Pixel ordering of ``data``.
    projection
        A name from ``PROJECTIONS`` (``mollweide``, ``mcbryde``, ``hammer``,
        ``des``...) or a skyproj class.
    ax
        Matplotlib axes; a new figure is made if ``None``.
    lon_0, galactic
        Central longitude and whether to label axes as Galactic.
    zoom
        Zoom on the region with data (skyproj's ``zoom``).
    log
        Logarithmic colour scale.
    **kwargs
        Passed to ``draw_hpxmap``/``draw_hpxpix`` (``cmap``, ``vmin``, ``vmax``...).

    Returns
    -------
    skyproj.Skyproj
        The skyproj object; use it to overlay more layers.
    """
    require("matplotlib", "viz")
    import matplotlib.pyplot as plt

    cls = _projection_class(projection)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sp = cls(ax=ax, lon_0=lon_0, galactic=galactic)
    norm = kwargs.pop("norm", "linear")
    if log:
        norm = "log"

    if isinstance(data, tuple):
        if nside is None:
            raise ValueError("nside is required for (pixels, values) input")
        pixels, values = data
        values = np.asarray(values, dtype=np.float64)
        good = np.isfinite(values)
        if log:
            good &= values > 0
            _need_positive(good)
        sp.draw_hpxpix(
            nside,
            np.asarray(pixels)[good],
            np.asarray(values)[good],
            nest=nest,
            zoom=zoom,
            norm=norm,
            **kwargs,
        )
    else:
        m = np.asarray(data, dtype=np.float64)
        if log:
            m = np.where(m > 0, m, np.nan)
            _need_positive(np.isfinite(m))
        sp.draw_hpxmap(m, nest=nest, zoom=zoom, norm=norm, **kwargs)

    if colorbar:
        sp.draw_colorbar(label=label)
    return sp
