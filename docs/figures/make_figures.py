"""Make the figures in the documentation, from synthetic catalogues.

    python docs/figures/make_figures.py

Needs the ``viz`` extra. Writes PNG files next to this script. The data are
random draws meant to look like a survey, not real measurements.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import astro_polars as ap  # noqa: E402
from astro_polars.surveys import euclid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 110
rng = np.random.default_rng(2026)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------
# A mock all-sky catalogue: disc, bulge, halo, two satellites and a stream
# --------------------------------------------------------------------------


def _uniform_sphere(n):
    return rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))


def _galactic_components():
    parts = []
    # thin disc: concentrated towards the centre and the plane, reddened near b = 0
    n = 1_500_000
    glon = (rng.vonmises(0.0, 0.6, n) * 180 / np.pi) % 360
    glat = np.clip(rng.laplace(0, 2.5, n), -89.9, 89.9)
    parts.append(("disc", glon, glat, 0.8 + 0.6 * np.exp(-np.abs(glat) / 4), 0.25))
    # bulge
    n = 250_000
    glon, glat = rng.normal(0, 6, n) % 360, np.clip(rng.normal(0, 4, n), -89.9, 89.9)
    parts.append(("bulge", glon, glat, 1.1 + 0.5 * np.exp(-np.abs(glat) / 4), 0.2))
    # halo
    glon, glat = _uniform_sphere(1_500_000)
    parts.append(("halo", glon, glat, np.full(len(glon), 0.55), 0.15))
    # two satellites
    for lon0, lat0, size, n in [(280.5, -32.9, 2.5, 80_000), (302.8, -44.3, 1.2, 25_000)]:
        glon = (lon0 + rng.normal(0, size, n) / np.cos(np.radians(lat0))) % 360
        glat = lat0 + rng.normal(0, size, n)
        parts.append(("satellite", glon, glat, np.full(n, 0.6), 0.1))

    frames = [
        pl.DataFrame(
            {"l": glon, "b": glat, "g_r": col + rng.normal(0, sig, len(glon)), "component": name}
        )
        for name, glon, glat, col, sig in parts
    ]
    return (
        pl.concat(frames)
        .sky.to_equatorial(pml=None, pmb=None)
        .select("ra", "dec", "g_r", "component")
    )


def _stream(n=40_000):
    """A thin stream along phi2 = 0 in the GD-1 frame, with a gentle track and two gaps."""
    phi1 = rng.uniform(-90, 10, 3 * n)
    density = (
        (1 - 0.85 * np.exp(-(((phi1 + 40) / 2.0) ** 2)) - 0.7 * np.exp(-(((phi1 + 20) / 1.5) ** 2)))
        * np.clip((phi1 + 90) / 15, 0, 1)
        * np.clip((10 - phi1) / 15, 0, 1)
    )
    phi1 = phi1[rng.random(len(phi1)) < density][:n]
    n = len(phi1)
    phi2 = 0.4 * np.sin(np.radians(phi1 * 3)) + rng.normal(0, 0.25, n)
    df = pl.DataFrame({"phi1": phi1, "phi2": phi2})
    radec = ap.rotate("phi1", "phi2", ap.GD1_KOPOSOV10.T).struct.unnest()
    return (
        df.select(radec)
        .rename({"lon": "ra", "lat": "dec"})
        .with_columns(g_r=pl.lit(0.3) + rng.normal(0, 0.04, n), component=pl.lit("stream"))
    )


def mock_sky() -> pl.DataFrame:
    return pl.concat([_galactic_components(), _stream()])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_sky_galactic(sky):
    lf = sky.lazy().sky.to_galactic(pmra=None, pmdec=None)
    sp = lf.viz.healpix("l", "b", nside=64, galactic=True, log=True, cmap="magma")
    sp.ax.set_title("Source counts, Galactic coordinates (Mollweide, nside 64)")
    save(sp.ax.figure, "sky_galactic.png")


def fig_sky_mean(sky):
    sp = sky.lazy().viz.healpix(
        "ra",
        "dec",
        nside=64,
        what="median",
        z="g_r",
        projection="mcbryde",
        lon_0=180,
        cmap="RdYlBu_r",
        vmin=0.3,
        vmax=1.3,
    )
    sp.ax.set_title("Median g - r per pixel, equatorial (McBryde, nside 64)", pad=24)
    save(sp.ax.figure, "sky_mean.png")


def fig_stream(sky):
    lf = (
        sky.lazy()
        .sky.to_frame(ap.GD1_KOPOSOV10, pmlon=None, pmlat=None)
        .filter(pl.col("phi2").is_between(-8, 4), pl.col("phi1").is_between(-95, 15))
    )
    fig, ax = plt.subplots(figsize=(10, 3.2))
    lf.viz.heatmap(
        "phi1", "phi2", lims=((-95, 15), (-8, 4)), shape=(220, 36), log=True, ax=ax, cmap="magma"
    )
    ax.set_title("A stream in its own frame: lf.sky.to_frame(ap.GD1_KOPOSOV10)")
    save(fig, "stream_heatmap.png")


def mock_cluster(n=120_000, n_field=60_000) -> pl.DataFrame:
    """Fluxes (nanomaggies) of a mock old cluster plus field stars."""
    # main sequence below the turn-off, more stars at faint magnitudes
    g_ms = 18.6 + 5.9 * rng.power(2.2, n)
    gr_ms = 0.32 + 0.95 * ((g_ms - 18.6) / 5.6) ** 1.6
    # subgiant branch, red giant branch and blue horizontal branch
    t = rng.random(n // 40)
    g_sgb, gr_sgb = 18.55 - 0.35 * t, 0.33 + 0.25 * t
    t = rng.random(n // 15) ** 1.8
    g_rgb, gr_rgb = 18.2 - 3.8 * t, 0.58 + 0.15 * t + 0.4 * t**1.8
    gr_hb = rng.uniform(-0.25, 0.2, n // 60)
    g_hb = 15.1 + 2.5 * np.clip(-gr_hb, 0, None) ** 1.5
    # field: blue halo turn-off stars and red disc dwarfs, more at faint magnitudes
    g_f = 14.0 + 10.5 * rng.power(2.5, n_field)
    gr_f = np.where(
        rng.random(n_field) < 0.45, rng.normal(0.4, 0.08, n_field), rng.normal(1.2, 0.2, n_field)
    )
    n_cl = len(g_ms) + len(g_sgb) + len(g_rgb) + len(g_hb)
    g = np.concatenate([g_ms, g_sgb, g_rgb, g_hb, g_f])
    gr = np.concatenate([gr_ms, gr_sgb, gr_rgb, gr_hb, gr_f])
    gr[:n_cl] += rng.normal(0, 0.015, n_cl)  # intrinsic width of the cluster sequences
    r = g - gr

    def to_flux(m):
        return 10 ** (-0.4 * (m - 22.5))

    err_g = to_flux(g) * (0.005 + 0.08 * 10 ** (0.4 * (g - 24)))
    err_r = to_flux(r) * (0.005 + 0.08 * 10 ** (0.4 * (r - 24)))
    return pl.DataFrame(
        {
            "flux_g": to_flux(g) + rng.normal(0, 1, len(g)) * err_g,
            "flux_r": to_flux(r) + rng.normal(0, 1, len(r)) * err_r,
            "flux_g_err": err_g,
            "flux_r_err": err_r,
        }
    )


def fig_cmd():
    lf = (
        mock_cluster()
        .lazy()
        .phot.add_mags({"g": "flux_g", "r": "flux_r"}, zeropoint="nanomaggy")
        .phot.add_colors({"g_r": ("flux_g", "flux_r")})
        .phot.add_magerrs({"g_err": ("flux_g_err", "flux_g")})
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    lf.viz.heatmap("g_r", "g", lims=((-0.4, 1.8), (14.2, 24.6)), shape=220, log=True, ax=ax1)
    lf.viz.heatmap(
        "g_r",
        "g",
        what="median",
        z="g_err",
        lims=((-0.4, 1.8), (14.2, 24.6)),
        shape=110,
        ax=ax2,
        cmap="viridis",
        vmax=0.1,
    )
    ax1.invert_yaxis()
    ax1.set_title("Counts (log)")
    ax2.set_title("Median g error")
    for ax in (ax1, ax2):
        ax.set_xlabel("g - r")
    fig.suptitle("Colour-magnitude diagram from fluxes with df.phot and df.viz.heatmap")
    save(fig, "cmd.png")


def fig_polygon():
    n = 30_000
    df = pl.DataFrame({"ra": rng.uniform(148, 152, n), "dec": rng.uniform(0, 4, n)})
    poly = np.array([[148.6, 0.5], [151.4, 0.9], [151.0, 3.4], [149.8, 2.2], [148.9, 3.5]])
    inside = df.sky.polygon(poly, x="ra", y="dec")

    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.scatter(df["ra"], df["dec"], s=1, c="0.8", rasterized=True)
    ax.scatter(inside["ra"], inside["dec"], s=1, c="C0", rasterized=True)
    ax.plot(*ap.close_polygon(poly).T, c="C3", lw=2)
    ax.set(xlabel="ra", ylabel="dec", title=f"df.sky.polygon: {len(inside)} of {n} inside")
    ax.invert_xaxis()
    save(fig, "polygon.png")


def mock_mer(n=300_000) -> pl.DataFrame:
    """Two loci in MUMAX_MINUS_MAG: point-like sources near -3, extended above."""
    star = rng.random(n) < 0.3
    mag = np.where(star, 17 + 11 * rng.power(2.5, n), 19 + 9.5 * rng.power(3.0, n))
    scatter = 0.04 + 0.5 * np.exp((mag - 27.0) / 1.5)
    mumax = np.where(
        star,
        -3 + rng.normal(0, 1, n) * scatter,
        -3 + 0.6 + rng.gamma(2.0, 0.45, n) + rng.normal(0, 1, n) * scatter,
    )
    return pl.DataFrame({"mag_VIS_auto": mag, "MUMAX_MINUS_MAG": mumax})


def fig_horn():
    df = mock_mer()
    kept = df.filter(euclid.horn_mask())
    lims = ((17, 28), (-5.5, 3.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    df.viz.heatmap(
        "mag_VIS_auto",
        "MUMAX_MINUS_MAG",
        lims=lims,
        shape=(220, 170),
        log=True,
        ax=ax,
        cmap="Greys",
        colorbar=False,
    )
    m = np.linspace(17, 28, 300)
    w = euclid.HORN_A + np.exp((m - euclid.HORN_B) / euclid.HORN_C)
    ax.plot(m, -3 + w, c="C3", lw=1.5, label="horn_cut limits (a=0.6, b=27, c=1.5)")
    ax.plot(m, -3 - w, c="C3", lw=1.5)
    ax.axvline(euclid.MAG_VIS_AUTO_LIM, c="C0", ls="--", label="magnitude_cut (27.5)")
    ax.set_ylim(*lims[1])
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(
        f"Horn and magnitude cuts on synthetic data: {len(kept) / len(df):.0%} inside the horn"
    )
    save(fig, "euclid_horn.png")


if __name__ == "__main__":
    ap.viz.use_style()
    sky = mock_sky()
    fig_sky_galactic(sky)
    fig_sky_mean(sky)
    fig_stream(sky)
    fig_cmd()
    fig_polygon()
    fig_horn()
