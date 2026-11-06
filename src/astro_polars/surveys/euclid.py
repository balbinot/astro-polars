"""Euclid MER final catalogues: magnitudes, colours, quality cuts and HEALPix datasets.

The column lists, maps and cut parameters are module constants. Every step
is a function on a DataFrame or LazyFrame, and ``transform`` runs them
in order.

Typical use:

    from astro_polars.surveys import euclid

    lf = euclid.read_mer("EUC_MER_FINAL-CAT_TILE....fits").lazy()
    lf = euclid.transform(lf)

    # whole download -> HEALPix hive dataset, restartable
    euclid.convert_tiles("tiles/", "mer_hive/", nside=32)

or from a shell: ``astro-polars euclid-hive --indir tiles --outdir mer_hive``.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Callable

import polars as pl

from ..healpix import degrade, healpix_index, nside_to_order
from ..io.hive import Manifest, remove_partitions, write_partitions
from ..photometry import MAGERR_FACTOR, ab_zeropoint, flux_to_mag

__all__ = [
    "ZPT",
    "CCC",
    "FITS_PATTERN",
    "KEEPCOLS",
    "MAG_MAP",
    "APER_MAG_MAP",
    "MAGERR_MAP",
    "VISTO_MAP",
    "HORN_A",
    "HORN_B",
    "HORN_C",
    "MAG_VIS_AUTO_LIM",
    "MAP_NSIDE",
    "RA",
    "DEC",
    "read_mer",
    "add_aperture_corrections",
    "add_mags",
    "add_aper_mags",
    "add_magerrs",
    "add_visto_colors",
    "clean",
    "horn_mask",
    "horn_cut",
    "magnitude_cut",
    "transform",
    "add_healpix",
    "find_tiles",
    "convert_tiles",
]

log = logging.getLogger(__name__)

#: MER fluxes are in microjansky: AB zero point ``-2.5 log10(1e-6 / 3631)`` = 23.9.
ZPT = ab_zeropoint("uJy")
#: ``2.5 / ln(10)``, flux error to magnitude error.
CCC = MAGERR_FACTOR

FITS_PATTERN = "EUC_MER_FINAL-CAT_TILE*.fits*"

RA = "RIGHT_ASCENSION"
DEC = "DECLINATION"

# fmt: off
#: Columns kept from the MER final catalogue.
KEEPCOLS = [
    'OBJECT_ID', 'RIGHT_ASCENSION', 'DECLINATION', 'TILE_INDEX',
    'FLUX_VIS_1FWHM_APER', 'FLUX_VIS_2FWHM_APER',
    'FLUX_VIS_3FWHM_APER', 'FLUX_VIS_4FWHM_APER',
    'FLUXERR_VIS_1FWHM_APER', 'FLUXERR_VIS_2FWHM_APER',
    'FLUXERR_VIS_3FWHM_APER', 'FLUXERR_VIS_4FWHM_APER',
    'FLUX_Y_TEMPLFIT', 'FLUX_J_TEMPLFIT', 'FLUX_H_TEMPLFIT',
    'FLUX_G_EXT_DECAM_TEMPLFIT', 'FLUX_R_EXT_DECAM_TEMPLFIT',
    'FLUX_I_EXT_DECAM_TEMPLFIT', 'FLUX_Z_EXT_DECAM_TEMPLFIT',
    'FLUX_U_EXT_MEGACAM_TEMPLFIT', 'FLUX_R_EXT_MEGACAM_TEMPLFIT',
    'FLUX_G_EXT_JPCAM_TEMPLFIT',
    'FLUX_I_EXT_PANSTARRS_TEMPLFIT', 'FLUX_Z_EXT_PANSTARRS_TEMPLFIT',
    'FLUX_G_EXT_HSC_TEMPLFIT', 'FLUX_Z_EXT_HSC_TEMPLFIT',
    'FLUXERR_Y_TEMPLFIT', 'FLUXERR_J_TEMPLFIT', 'FLUXERR_H_TEMPLFIT',
    'FLUXERR_G_EXT_DECAM_TEMPLFIT', 'FLUXERR_R_EXT_DECAM_TEMPLFIT',
    'FLUXERR_I_EXT_DECAM_TEMPLFIT', 'FLUXERR_Z_EXT_DECAM_TEMPLFIT',
    'FLUXERR_U_EXT_MEGACAM_TEMPLFIT', 'FLUXERR_R_EXT_MEGACAM_TEMPLFIT',
    'FLUXERR_G_EXT_JPCAM_TEMPLFIT',
    'FLUXERR_I_EXT_PANSTARRS_TEMPLFIT', 'FLUXERR_Z_EXT_PANSTARRS_TEMPLFIT',
    'FLUXERR_G_EXT_HSC_TEMPLFIT', 'FLUXERR_Z_EXT_HSC_TEMPLFIT',
    'FLUX_VIS_PSF', 'FLUXERR_VIS_PSF',
    'FLAG_VIS', 'FLAG_Y', 'FLAG_J', 'FLAG_H', 'FLAG_NIR_STACK',
    'FLAG_U_EXT_DECAM', 'FLAG_G_EXT_DECAM', 'FLAG_R_EXT_DECAM',
    'FLAG_I_EXT_DECAM', 'FLAG_Z_EXT_DECAM', 'FLAG_U_EXT_MEGACAM',
    'FLAG_R_EXT_MEGACAM', 'FLAG_G_EXT_JPCAM',
    'FLAG_I_EXT_PANSTARRS', 'FLAG_Z_EXT_PANSTARRS',
    'FLAG_G_EXT_HSC', 'FLAG_Z_EXT_HSC',
    'POINT_LIKE_FLAG', 'POINT_LIKE_PROB',
    'EXTENDED_FLAG', 'EXTENDED_PROB', 'SPURIOUS_FLAG', 'SPURIOUS_PROB',
    'MAG_STARGAL_SEP', 'DET_QUALITY_FLAG', 'MU_MAX', 'MUMAX_MINUS_MAG',
    'SEMIMAJOR_AXIS', 'SEMIMAJOR_AXIS_ERR',
    'POSITION_ANGLE', 'POSITION_ANGLE_ERR',
    'ELLIPTICITY', 'ELLIPTICITY_ERR',
    'KRON_RADIUS', 'KRON_RADIUS_ERR', 'FWHM',
    'GAL_EBV', 'GAL_EBV_ERR', 'GAIA_ID',
    'FLUX_Y_1FWHM_APER', 'FLUX_Y_2FWHM_APER',
    'FLUX_J_1FWHM_APER', 'FLUX_J_2FWHM_APER',
    'FLUX_H_1FWHM_APER', 'FLUX_H_2FWHM_APER',
    'FLUX_NIR_STACK_1FWHM_APER', 'FLUX_NIR_STACK_2FWHM_APER',
    'FLUX_NIR_STACK_3FWHM_APER', 'FLUX_NIR_STACK_4FWHM_APER',
    'FLUXERR_Y_1FWHM_APER', 'FLUXERR_Y_2FWHM_APER',
    'FLUXERR_J_1FWHM_APER', 'FLUXERR_J_2FWHM_APER',
    'FLUXERR_H_1FWHM_APER', 'FLUXERR_H_2FWHM_APER',
    'FLUXERR_NIR_STACK_1FWHM_APER', 'FLUXERR_NIR_STACK_2FWHM_APER',
    'FLUX_G_EXT_DECAM_1FWHM_APER', 'FLUX_G_EXT_DECAM_2FWHM_APER',
    'FLUX_R_EXT_DECAM_1FWHM_APER', 'FLUX_R_EXT_DECAM_2FWHM_APER',
    'FLUX_I_EXT_DECAM_1FWHM_APER', 'FLUX_I_EXT_DECAM_2FWHM_APER',
    'FLUX_Z_EXT_DECAM_1FWHM_APER', 'FLUX_Z_EXT_DECAM_2FWHM_APER',
    'FLUXERR_G_EXT_DECAM_1FWHM_APER', 'FLUXERR_G_EXT_DECAM_2FWHM_APER',
    'FLUXERR_R_EXT_DECAM_1FWHM_APER', 'FLUXERR_R_EXT_DECAM_2FWHM_APER',
    'FLUXERR_I_EXT_DECAM_1FWHM_APER', 'FLUXERR_I_EXT_DECAM_2FWHM_APER',
    'FLUXERR_Z_EXT_DECAM_1FWHM_APER', 'FLUXERR_Z_EXT_DECAM_2FWHM_APER',
    'FLUX_U_EXT_MEGACAM_1FWHM_APER', 'FLUX_U_EXT_MEGACAM_2FWHM_APER',
    'FLUX_R_EXT_MEGACAM_1FWHM_APER', 'FLUX_R_EXT_MEGACAM_2FWHM_APER',
    'FLUX_G_EXT_JPCAM_1FWHM_APER', 'FLUX_G_EXT_JPCAM_2FWHM_APER',
    'FLUX_I_EXT_PANSTARRS_1FWHM_APER', 'FLUX_I_EXT_PANSTARRS_2FWHM_APER',
    'FLUX_G_EXT_HSC_1FWHM_APER', 'FLUX_G_EXT_HSC_2FWHM_APER',
    'FLUX_Z_EXT_HSC_1FWHM_APER', 'FLUX_Z_EXT_HSC_2FWHM_APER',
    'FLUXERR_U_EXT_MEGACAM_1FWHM_APER', 'FLUXERR_U_EXT_MEGACAM_2FWHM_APER',
    'FLUXERR_R_EXT_MEGACAM_1FWHM_APER', 'FLUXERR_R_EXT_MEGACAM_2FWHM_APER',
    'FLUXERR_G_EXT_JPCAM_1FWHM_APER', 'FLUXERR_G_EXT_JPCAM_2FWHM_APER',
    'FLUXERR_I_EXT_PANSTARRS_1FWHM_APER', 'FLUXERR_I_EXT_PANSTARRS_2FWHM_APER',
    'FLUXERR_G_EXT_HSC_1FWHM_APER', 'FLUXERR_G_EXT_HSC_2FWHM_APER',
    'FLUXERR_Z_EXT_HSC_1FWHM_APER', 'FLUXERR_Z_EXT_HSC_2FWHM_APER',
    'FLUX_DETECTION_TOTAL', 'FLUXERR_DETECTION_TOTAL', 'VIS_DET',
    'FLUX_VIS_TO_Y_TEMPLFIT', 'FLUX_VIS_TO_J_TEMPLFIT',
    'FLUX_VIS_TO_H_TEMPLFIT',
    'FLUX_VIS_TO_G_EXT_DECAM_TEMPLFIT', 'FLUX_VIS_TO_R_EXT_DECAM_TEMPLFIT',
    'FLUX_VIS_TO_I_EXT_DECAM_TEMPLFIT', 'FLUX_VIS_TO_Z_EXT_DECAM_TEMPLFIT',
    'FLUX_VIS_TO_U_EXT_MEGACAM_TEMPLFIT', 'FLUX_VIS_TO_R_EXT_MEGACAM_TEMPLFIT',
    'FLUX_VIS_TO_I_EXT_PANSTARRS_TEMPLFIT',
    'FLUX_VIS_TO_G_EXT_HSC_TEMPLFIT', 'FLUX_VIS_TO_Z_EXT_HSC_TEMPLFIT',
    'FLUXERR_VIS_TO_Y_TEMPLFIT', 'FLUXERR_VIS_TO_J_TEMPLFIT',
    'FLUXERR_VIS_TO_H_TEMPLFIT',
    'FLUXERR_VIS_TO_G_EXT_DECAM_TEMPLFIT', 'FLUXERR_VIS_TO_R_EXT_DECAM_TEMPLFIT',
    'FLUXERR_VIS_TO_I_EXT_DECAM_TEMPLFIT', 'FLUXERR_VIS_TO_Z_EXT_DECAM_TEMPLFIT',
    'FLUXERR_VIS_TO_U_EXT_MEGACAM_TEMPLFIT', 'FLUXERR_VIS_TO_R_EXT_MEGACAM_TEMPLFIT',
    'FLUXERR_VIS_TO_I_EXT_PANSTARRS_TEMPLFIT',
    'FLUXERR_VIS_TO_G_EXT_HSC_TEMPLFIT', 'FLUXERR_VIS_TO_Z_EXT_HSC_TEMPLFIT',
]

# name -> flux column, for TEMPLFIT-style total magnitudes
MAG_MAP = {
    'mag_VIS':       'FLUX_VIS_PSF',
    'mag_VIS_auto':  'FLUX_DETECTION_TOTAL',
    'mag_Y':         'FLUX_Y_TEMPLFIT',
    'mag_J':         'FLUX_J_TEMPLFIT',
    'mag_H':         'FLUX_H_TEMPLFIT',
    'mag_G':         'FLUX_G_EXT_DECAM_TEMPLFIT',
    'mag_R':         'FLUX_R_EXT_DECAM_TEMPLFIT',
    'mag_I':         'FLUX_I_EXT_DECAM_TEMPLFIT',
    'mag_Z':         'FLUX_Z_EXT_DECAM_TEMPLFIT',
    'mag_MEGACAM-R': 'FLUX_R_EXT_MEGACAM_TEMPLFIT',
    'mag_PS-I':      'FLUX_I_EXT_PANSTARRS_TEMPLFIT',
    'mag_HSC-G':     'FLUX_G_EXT_HSC_TEMPLFIT',
    'mag_HSC-Z':     'FLUX_Z_EXT_HSC_TEMPLFIT',
}

# name stem -> aperture flux column stem, for corr1/corr2 aperture mags
APER_MAG_MAP = {
    'mag_VIS':       'FLUX_VIS',
    'mag_NIR':       'FLUX_NIR_STACK',
    'mag_Y':         'FLUX_Y',
    'mag_J':         'FLUX_J',
    'mag_H':         'FLUX_H',
    'mag_G':         'FLUX_G_EXT_DECAM',
    'mag_R':         'FLUX_R_EXT_DECAM',
    'mag_I':         'FLUX_I_EXT_DECAM',
    'mag_Z':         'FLUX_Z_EXT_DECAM',
    'mag_MEGACAM-U': 'FLUX_U_EXT_MEGACAM',
    'mag_MEGACAM-R': 'FLUX_R_EXT_MEGACAM',
    'mag_PS-I':      'FLUX_I_EXT_PANSTARRS',
    'mag_HSC-G':     'FLUX_G_EXT_HSC',
    'mag_HSC-Z':     'FLUX_Z_EXT_HSC',
}

# magerr name -> (fluxerr, flux)
MAGERR_MAP = {
    'magerr_VIS':       ('FLUXERR_VIS_PSF', 'FLUX_VIS_PSF'),
    'magerr_Y':         ('FLUXERR_Y_TEMPLFIT', 'FLUX_Y_TEMPLFIT'),
    'magerr_J':         ('FLUXERR_J_TEMPLFIT', 'FLUX_J_TEMPLFIT'),
    'magerr_H':         ('FLUXERR_H_TEMPLFIT', 'FLUX_H_TEMPLFIT'),
    'magerr_G':         ('FLUXERR_G_EXT_DECAM_TEMPLFIT', 'FLUX_G_EXT_DECAM_TEMPLFIT'),
    'magerr_R':         ('FLUXERR_R_EXT_DECAM_TEMPLFIT', 'FLUX_R_EXT_DECAM_TEMPLFIT'),
    'magerr_I':         ('FLUXERR_I_EXT_DECAM_TEMPLFIT', 'FLUX_I_EXT_DECAM_TEMPLFIT'),
    'magerr_Z':         ('FLUXERR_Z_EXT_DECAM_TEMPLFIT', 'FLUX_Z_EXT_DECAM_TEMPLFIT'),
    'magerr_MEGACAM-R': ('FLUXERR_R_EXT_MEGACAM_TEMPLFIT', 'FLUX_R_EXT_MEGACAM_TEMPLFIT'),
    'magerr_PS-I':      ('FLUXERR_I_EXT_PANSTARRS_TEMPLFIT', 'FLUX_I_EXT_PANSTARRS_TEMPLFIT'),
    'magerr_HSC-G':     ('FLUXERR_G_EXT_HSC_TEMPLFIT', 'FLUX_G_EXT_HSC_TEMPLFIT'),
    'magerr_HSC-Z':     ('FLUXERR_Z_EXT_HSC_TEMPLFIT', 'FLUX_Z_EXT_HSC_TEMPLFIT'),
}

# VIS-to-X template colors: name -> (sign, flux, vis_to_flux)
VISTO_MAP = {
    'VIStoH':          (+1, 'FLUX_H_TEMPLFIT', 'FLUX_VIS_TO_H_TEMPLFIT'),
    'VIStoY':          (+1, 'FLUX_Y_TEMPLFIT', 'FLUX_VIS_TO_Y_TEMPLFIT'),
    'VIStoJ':          (+1, 'FLUX_J_TEMPLFIT', 'FLUX_VIS_TO_J_TEMPLFIT'),
    'GtoVIS':          (-1, 'FLUX_G_EXT_DECAM_TEMPLFIT', 'FLUX_VIS_TO_G_EXT_DECAM_TEMPLFIT'),
    'RtoVIS':          (-1, 'FLUX_R_EXT_DECAM_TEMPLFIT', 'FLUX_VIS_TO_R_EXT_DECAM_TEMPLFIT'),
    'ItoVIS':          (-1, 'FLUX_I_EXT_DECAM_TEMPLFIT', 'FLUX_VIS_TO_I_EXT_DECAM_TEMPLFIT'),
    'VIStoZ':          (+1, 'FLUX_Z_EXT_DECAM_TEMPLFIT', 'FLUX_VIS_TO_Z_EXT_DECAM_TEMPLFIT'),
    'MEGACAM-UtoVIS':  (-1, 'FLUX_U_EXT_MEGACAM_TEMPLFIT', 'FLUX_VIS_TO_U_EXT_MEGACAM_TEMPLFIT'),
    'MEGACAM-RtoVIS':  (-1, 'FLUX_R_EXT_MEGACAM_TEMPLFIT', 'FLUX_VIS_TO_R_EXT_MEGACAM_TEMPLFIT'),
    'PS-ItoVIS':       (-1, 'FLUX_I_EXT_PANSTARRS_TEMPLFIT', 'FLUX_VIS_TO_I_EXT_PANSTARRS_TEMPLFIT'),
    'HSC_GtoVIS':      (-1, 'FLUX_G_EXT_HSC_TEMPLFIT', 'FLUX_VIS_TO_G_EXT_HSC_TEMPLFIT'),
    'VIStoHSC-Z':      (+1, 'FLUX_Z_EXT_HSC_TEMPLFIT', 'FLUX_VIS_TO_Z_EXT_HSC_TEMPLFIT'),
}
# fmt: on

#: Horn filter parameters (``MUMAX_MINUS_MAG`` vs ``mag_VIS_auto``).
HORN_A, HORN_B, HORN_C = 0.6, 27.0, 1.5
#: Faint limit on ``mag_VIS_auto``.
MAG_VIS_AUTO_LIM = 27.5

#: High-resolution nested HEALPix index stored in every output file, for maps.
MAP_NSIDE = 4096


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_mer(path: str | os.PathLike, columns=KEEPCOLS, warn_missing: bool = True) -> pl.DataFrame:
    """Read a MER final catalogue FITS file, keeping ``columns`` (default ``KEEPCOLS``).

    Columns missing from the file are skipped with a warning. Needs the ``fits`` extra.
    """
    from ..io.fits import read_fits_table

    return read_fits_table(path, columns=columns, warn_missing=warn_missing)


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------


def _names(df) -> set[str]:
    return set(df.collect_schema().names())


def _f64(x: float) -> pl.Expr:
    # Magnitudes and errors are Float64 whatever the flux dtype, so that
    # datasets written at different times share one schema.
    return pl.lit(x, dtype=pl.Float64)


def add_aperture_corrections(df):
    """Add ``corr1``/``corr2``: ``FLUX_DETECTION_TOTAL`` over the 1/2 FWHM VIS aperture flux."""
    have = _names(df)
    exprs = []
    for n in (1, 2):
        aper = f"FLUX_VIS_{n}FWHM_APER"
        if {"FLUX_DETECTION_TOTAL", aper} <= have:
            exprs.append((pl.col("FLUX_DETECTION_TOTAL") / pl.col(aper)).alias(f"corr{n}"))
    return df.with_columns(exprs) if exprs else df


def add_mags(df, mapping=MAG_MAP, zeropoint: float = ZPT):
    """Total magnitudes ``name = -2.5 log10(flux) + ZPT`` for each ``name: flux`` present."""
    have = _names(df)
    exprs = [
        flux_to_mag(flux, _f64(zeropoint)).alias(name)
        for name, flux in mapping.items()
        if flux in have
    ]
    return df.with_columns(exprs) if exprs else df


def add_aper_mags(df, mapping=APER_MAG_MAP, zeropoint: float = ZPT):
    """Aperture magnitudes corrected to total with ``corr1``/``corr2``.

    For each ``stem: fluxstem`` adds ``{stem}_{n}fwhm`` from
    ``{fluxstem}_{n}FWHM_APER * corr{n}``, n = 1, 2. Run
    ``add_aperture_corrections`` first.
    """
    have = _names(df)
    exprs = []
    for stem, fluxstem in mapping.items():
        for n, corr in ((1, "corr1"), (2, "corr2")):
            flux = f"{fluxstem}_{n}FWHM_APER"
            if flux in have and corr in have:
                mag = flux_to_mag(pl.col(corr) * pl.col(flux), _f64(zeropoint))
                exprs.append(mag.alias(f"{stem}_{n}fwhm"))
    return df.with_columns(exprs) if exprs else df


def add_magerrs(df, mapping=MAGERR_MAP):
    """Magnitude errors ``name = fluxerr / flux * 2.5 / ln 10`` for each ``name: (fluxerr, flux)``."""
    have = _names(df)
    exprs = [
        (pl.col(err) / pl.col(flux) * _f64(CCC)).alias(name)
        for name, (err, flux) in mapping.items()
        if {err, flux} <= have
    ]
    return df.with_columns(exprs) if exprs else df


def add_visto_colors(df, mapping=VISTO_MAP):
    """VIS-to-X template colours ``name = sign * 2.5 log10(flux / vis_to_flux)``."""
    have = _names(df)
    exprs = [
        (sign * 2.5 * (pl.col(flux) / pl.col(visto)).log10()).alias(name)
        for name, (sign, flux, visto) in mapping.items()
        if {flux, visto} <= have
    ]
    return df.with_columns(exprs) if exprs else df


def clean(df):
    """Drop rows with NaN or infinite ``MUMAX_MINUS_MAG`` or ``POINT_LIKE_PROB``."""
    return df.filter(pl.col("MUMAX_MINUS_MAG").is_finite() & pl.col("POINT_LIKE_PROB").is_finite())


def horn_mask(
    a: float = HORN_A, b: float = HORN_B, c: float = HORN_C, mag: str = "mag_VIS_auto"
) -> pl.Expr:
    """True inside the "horn" in ``MUMAX_MINUS_MAG`` vs ``mag``.

    ``|MUMAX_MINUS_MAG + 3| <= a + exp((mag - b) / c)``: a band around
    ``MUMAX_MINUS_MAG = -3`` that widens at faint magnitudes.
    """
    w = a + ((pl.col(mag) - b) / c).exp()
    return pl.col("MUMAX_MINUS_MAG").is_between(-3 - w, -3 + w)


def horn_cut(df, a: float = HORN_A, b: float = HORN_B, c: float = HORN_C):
    """Keep rows inside the horn (see ``horn_mask``)."""
    return df.filter(horn_mask(a, b, c))


def magnitude_cut(df, mag_max: float = MAG_VIS_AUTO_LIM, unique: bool = True):
    """Keep ``mag_VIS_auto < mag_max`` and, if ``unique``, one row per ``OBJECT_ID``."""
    df = df.filter(pl.col("mag_VIS_auto") < mag_max)
    return df.unique("OBJECT_ID", keep="first", maintain_order=True) if unique else df


def transform(
    df,
    horn: tuple[float, float, float] | None = (HORN_A, HORN_B, HORN_C),
    mag_vis_auto_max: float | None = MAG_VIS_AUTO_LIM,
):
    """The full MER pipeline: magnitudes, errors, colours, then the quality cuts.

    Steps, in order: ``add_aperture_corrections``, ``add_mags``,
    ``add_aper_mags``, ``add_magerrs``, ``add_visto_colors``,
    ``clean``, ``horn_cut`` and ``magnitude_cut``. Pass
    ``horn=None`` or ``mag_vis_auto_max=None`` to skip those cuts. Works on
    DataFrames and LazyFrames.
    """
    df = add_aperture_corrections(df)
    df = add_mags(df)
    df = add_aper_mags(df)
    df = add_magerrs(df)
    df = add_visto_colors(df)
    df = clean(df)
    if horn is not None:
        df = horn_cut(df, *horn)
    if mag_vis_auto_max is not None:
        df = magnitude_cut(df, mag_vis_auto_max)
    return df


def add_healpix(df, nside: int, map_nside: int = MAP_NSIDE, lon: str = RA, lat: str = DEC):
    """Add nested ``hpx{map_nside}`` (fine, for maps) and ``hpx{nside}`` (partition).

    The coarse index is derived from the fine one by integer division, which
    is exact in nested ordering.
    """
    if nside > map_nside:
        raise ValueError(f"nside must be <= {map_nside}")
    nside_to_order(nside)
    fine = f"hpx{map_nside}"
    df = df.with_columns(healpix_index(lon, lat, map_nside).alias(fine))
    if nside != map_nside:
        df = df.with_columns(degrade(fine, map_nside, nside).alias(f"hpx{nside}"))
    return df


# --------------------------------------------------------------------------
# Tiles -> hive dataset
# --------------------------------------------------------------------------


def find_tiles(indir: str | os.PathLike, pattern: str = FITS_PATTERN) -> list[str]:
    """MER files under ``indir/<tile>/``, sorted."""
    return sorted(glob.glob(os.path.join(os.fspath(indir), "*", pattern)))


def convert_tiles(
    indir: str | os.PathLike,
    outdir: str | os.PathLike,
    nside: int = 32,
    force: bool = False,
    pattern: str = FITS_PATTERN,
    map_nside: int = MAP_NSIDE,
    transform_fn: Callable = transform,
    progress: Callable[[str], None] | None = print,
) -> dict:
    """Convert downloaded MER tiles into a HEALPix hive-partitioned Parquet dataset.

    Expects ``indir/<tile>/EUC_MER_FINAL-CAT_TILE*.fits*``. For each file not
    already in ``outdir/_manifest.json`` (or every file with ``force``): read
    ``KEEPCOLS``, run ``transform_fn`` (default ``transform``), add
    ``hpx{map_nside}`` and write one Parquet file per ``hpx{nside}`` partition,
    tagged with the tile directory name. Files from an earlier attempt at the
    same tile are removed first, and the manifest is saved after each tile,
    so the run can be interrupted and resumed.

    Read the result with ``astro_polars.io.scan_hive`` or
    ``pl.scan_parquet(f"{outdir}/**/*.parquet", hive_partitioning=True)``.

    Returns
    -------
    dict
        Counts: ``found``, ``processed``, ``failed``, ``skipped``, and
        ``errors`` (file name -> message).
    """
    say = progress or (lambda msg: None)
    if nside > map_nside or map_nside % nside != 0:
        raise ValueError(f"nside must be a power of 2 <= {map_nside}")
    nside_to_order(nside)

    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(os.fspath(outdir), "_manifest.json")
    manifest = Manifest(manifest_path) if force else Manifest.load(manifest_path)

    files = find_tiles(indir, pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {indir}")
    todo = [f for f in files if force or not manifest.is_done(f)]
    say(f"{len(files)} FITS files found, {len(todo)} to process (nside={nside}, nested)")

    n_ok, errors = 0, {}
    for f in todo:
        key = Manifest.key(f)
        tile = os.path.basename(os.path.dirname(f))
        say(f"[proc] {tile}/{key}")
        try:
            df = read_mer(f)
            n_in = len(df)
            df = transform_fn(df.lazy()).collect()
            df = add_healpix(df, nside, map_nside)
            n_old = remove_partitions(outdir, nside, tile)
            if n_old:
                say(f"    removed {n_old} stale partition files for {tile}")
            nparts = write_partitions(df, outdir, nside, tag=tile)
            say(f"    {n_in} -> {len(df)} rows after cuts, {nparts} partition file(s)")
            manifest.record(f, tile=tile, rows=len(df), partitions=nparts)
            manifest.save()
            n_ok += 1
        except Exception as e:  # keep going; report at the end
            log.exception("failed on %s", f)
            say(f"[fail] {key}: {e}")
            errors[key] = str(e)

    summary = {
        "found": len(files),
        "processed": n_ok,
        "failed": len(errors),
        "skipped": len(files) - len(todo),
        "errors": errors,
    }
    say(
        f"Done. {n_ok} processed, {len(errors)} failed, "
        f"{summary['skipped']} skipped (already in manifest)."
    )
    return summary
