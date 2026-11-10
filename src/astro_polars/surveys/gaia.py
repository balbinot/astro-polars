"""Gaia DR3: radial-velocity and error corrections, extinction, quality cuts and kinematics.

All functions take and return Polars expressions or frames (DataFrame or
LazyFrame) and use Gaia archive column names. `process` chains the usual
steps for a radial-velocity sample:

    from astro_polars.surveys import gaia

    lf = pl.scan_parquet("gaia_dr3_rvs.parquet")
    lf = gaia.process(lf, correct_rv=True)                  # l, b, pml, pmb, X..vZ, R, vphi, ...
    lf = lf.filter(gaia.dodd23_quality())
    lf = lf.galactic.calc(["E", "Lz", "JR", "Jz", "circ"])  # needs AGAMA

The HEALPix index encoded in ``source_id`` is in
`astro_polars.healpix.gaia_source_id_to_healpix`. DR4 support (schema,
epoch astrometry) will be added once its data model is public.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import numpy as np
import polars as pl

from .._utils import require, to_expr
from ..galactic import SolarParameters

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "EXTINCTION_COEFFS",
    "correct_radial_velocity",
    "correct_radial_velocity_error",
    "estimate_grvs",
    "extinction",
    "deredden",
    "dodd23_quality",
    "lindegren21_zero_point",
    "distance_from_parallax",
    "process",
    "DEFAULT_QUANTITIES",
]


# -- radial velocities ---------------------------------------------------------
def correct_radial_velocity(
    radial_velocity: IntoExpr = "radial_velocity",
    grvs_mag: IntoExpr = "grvs_mag",
    rv_template_teff: IntoExpr = "rv_template_teff",
) -> pl.Expr:
    """Gaia DR3 radial velocities with the recommended magnitude corrections (km/s).

    - Cool stars (``rv_template_teff < 8500 K``) with ``grvs_mag >= 11``:
      ``rv - 0.02755 g^2 + 0.55863 g - 2.81129`` (Katz et al. 2023, A&A 674, A5, Eq. 5).
    - Hot stars (``8500 <= rv_template_teff <= 14500 K``) with
      ``6 <= grvs_mag <= 12``: ``rv - 7.98 + 1.135 g`` (Blomme et al. 2023,
      A&A 674, A7).
    - Everything else is left as is.

    These are the ranges given in the papers; the original polars_astro
    also applied the cool-star polynomial below ``grvs_mag = 11`` and a
    constant hot-star offset beyond 12 mag.
    """
    rv, g, teff = to_expr(radial_velocity), to_expr(grvs_mag), to_expr(rv_template_teff)
    hot = ((teff >= 8500) & (teff <= 14500)).fill_null(False)
    cool = (teff < 8500).fill_null(False)
    return (
        pl.when(hot & (g >= 6) & (g <= 12))
        .then(rv - 7.98 + 1.135 * g)
        .when(cool & (g >= 11))
        .then(rv - 0.02755 * g**2 + 0.55863 * g - 2.81129)
        .otherwise(rv)
    )


def correct_radial_velocity_error(
    radial_velocity_error: IntoExpr = "radial_velocity_error",
    grvs_mag: IntoExpr = "grvs_mag",
) -> pl.Expr:
    """Gaia DR3 radial-velocity uncertainties inflated as in Babusiaux et al. (2023, A&A 674, A32).

    The factor is ``0.318 + 0.3884 g - 0.02778 g^2`` for ``8 <= grvs_mag < 12``
    and ``16.554 - 2.4899 g + 0.09933 g^2`` for ``grvs_mag >= 12``; brighter
    than 8 mag the value at 8 mag is used, as the paper recommends. Null
    where ``grvs_mag`` is null.
    """
    err, g = to_expr(radial_velocity_error), to_expr(grvs_mag)
    # not max_horizontal: it ignores nulls, which would treat a missing magnitude as 8 mag
    gb = pl.when(g < 8).then(8.0).otherwise(g)
    factor = (
        pl.when(gb < 12)
        .then(0.318 + 0.3884 * gb - 0.02778 * gb**2)
        .otherwise(16.554 - 2.4899 * gb + 0.09933 * gb**2)
    )
    return err * factor


def estimate_grvs(
    phot_g_mean_mag: IntoExpr = "phot_g_mean_mag",
    phot_rp_mean_mag: IntoExpr = "phot_rp_mean_mag",
) -> pl.Expr:
    """G_RVS estimated from G and G_RP, for stars without ``grvs_mag``.

    ``G_RVS - G_RP`` is a quartic polynomial in ``G - G_RP`` for
    ``0.1 < G - G_RP < 1.4`` and ``1.4 <= G - G_RP < 1.7`` (Gaia Collaboration
    2018, A&A 616, A1, Eqs. 2 and 3); NaN outside that colour range.
    """
    rp = to_expr(phot_rp_mean_mag)
    x = to_expr(phot_g_mean_mag) - rp

    def quartic(c):
        return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))

    low = (0.042319, -0.65124, 1.0215, -1.3947, 0.53768)
    high = (132.32, -377.28, 402.32, -190.97, 34.026)
    return (
        pl.when((x > 0.1) & (x < 1.4))
        .then(rp + quartic(low))
        .when((x >= 1.4) & (x < 1.7))
        .then(rp + quartic(high))
        .otherwise(float("nan"))
    )


# -- extinction ------------------------------------------------------------------
#: Coefficients ``c1..c7`` of ``k_X = A_X / A0`` for G, BP and RP
#: (Gaia Collaboration, Babusiaux et al. 2018, A&A 616, A10, Eq. 1 and Table 1):
#: ``k = c1 + c2 x + c3 x^2 + c4 x^3 + c5 A0 + c6 A0^2 + c7 x A0``, ``x = (BP - RP)_0``.
EXTINCTION_COEFFS = {
    "G": (0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099),
    "BP": (1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043),
    "RP": (0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006),
}


def _k(band: str, x: pl.Expr, a0: pl.Expr) -> pl.Expr:
    c = EXTINCTION_COEFFS[band]
    return c[0] + x * (c[1] + x * (c[2] + x * c[3])) + c[4] * a0 + c[5] * a0**2 + c[6] * x * a0


def extinction(band: str, bp_rp0: IntoExpr, ebv: IntoExpr, r_v: float = 3.1) -> pl.Expr:
    """Extinction ``A_X`` in Gaia ``band`` ("G", "BP", "RP") for intrinsic colour ``bp_rp0``.

    ``A0 = r_v * E(B-V)``. Use `deredden` when only the observed colour is known.
    """
    a0 = r_v * to_expr(ebv)
    return _k(band, to_expr(bp_rp0), a0) * a0


def deredden(
    df: pl.DataFrame | pl.LazyFrame,
    ebv: str = "ebv",
    g: str = "phot_g_mean_mag",
    bp: str = "phot_bp_mean_mag",
    rp: str = "phot_rp_mean_mag",
    *,
    r_v: float = 3.1,
    iterations: int = 10,
    suffix: str = "_0",
):
    """Add extinctions ``A_G, A_BP, A_RP`` and dereddened magnitudes and colour.

    The coefficients depend on the intrinsic colour, so ``(BP - RP)_0`` is
    found by fixed-point iteration ``x <- (BP - RP) - E(BP - RP)(x)`` starting
    from the observed colour (10 iterations converge to well below 1 mmag).
    New columns: ``bp_rp{suffix}``, ``A_G``, ``A_BP``, ``A_RP`` and
    ``{g}{suffix}``, ``{bp}{suffix}``, ``{rp}{suffix}``.
    """
    a0 = r_v * pl.col(ebv)
    obs = pl.col(bp) - pl.col(rp)
    x_name = f"bp_rp{suffix}"
    df = df.with_columns(obs.alias(x_name))
    x = pl.col(x_name)
    for _ in range(iterations):
        df = df.with_columns((obs - (_k("BP", x, a0) - _k("RP", x, a0)) * a0).alias(x_name))
    df = df.with_columns(
        (_k("G", x, a0) * a0).alias("A_G"),
        (_k("BP", x, a0) * a0).alias("A_BP"),
        (_k("RP", x, a0) * a0).alias("A_RP"),
    )
    return df.with_columns(
        (pl.col(g) - pl.col("A_G")).alias(f"{g}{suffix}"),
        (pl.col(bp) - pl.col("A_BP")).alias(f"{bp}{suffix}"),
        (pl.col(rp) - pl.col("A_RP")).alias(f"{rp}{suffix}"),
    )


# -- selections and distances ------------------------------------------------------
def dodd23_quality(
    radial_velocity_error: IntoExpr | None = None,
    *,
    parallax_over_error: float | None = 5.0,
    b: IntoExpr = "b",
) -> pl.Expr:
    """Boolean mask of the Dodd et al. (2023, A&A 670, L2) quality cuts for halo kinematics.

    ``ruwe < 1.4``, radial-velocity error ``< 20 km/s``, ``grvs_mag -
    phot_g_mean_mag >= -3``, not (``rv_expected_sig_to_noise < 5`` and
    ``|b| < 7.5``), and ``parallax_over_error > 5`` (pass None to skip the
    last one). By default the radial-velocity error is the corrected one
    (`correct_radial_velocity_error`). Needs Galactic ``b``.
    """
    err = (
        correct_radial_velocity_error()
        if radial_velocity_error is None
        else to_expr(radial_velocity_error)
    )
    mask = (
        (pl.col("ruwe") < 1.4)
        & (err < 20)
        & ((pl.col("grvs_mag") - pl.col("phot_g_mean_mag")) >= -3)
        & ~((pl.col("rv_expected_sig_to_noise") < 5) & (to_expr(b).abs() < 7.5))
    )
    if parallax_over_error is not None:
        mask = mask & (pl.col("parallax_over_error") > parallax_over_error)
    return mask


def lindegren21_zero_point(
    phot_g_mean_mag: IntoExpr = "phot_g_mean_mag",
    nu_eff_used_in_astrometry: IntoExpr = "nu_eff_used_in_astrometry",
    pseudocolour: IntoExpr = "pseudocolour",
    ecl_lat: IntoExpr = "ecl_lat",
    astrometric_params_solved: IntoExpr = "astrometric_params_solved",
    *,
    extrapolate: bool = True,
) -> pl.Expr:
    """Parallax zero point (mas) of Lindegren et al. (2021, A&A 649, A4).

    Uses the official ``gaiadr3-zeropoint`` package (``pip install
    gaiadr3-zeropoint``). Subtract it: ``parallax - zero_point``. Outside the
    calibrated range (6 < G < 21, ...) the value is extrapolated, or NaN with
    ``extrapolate=False``.
    """
    zpt = require("zero_point.zpt")
    if not getattr(zpt, "_astro_polars_loaded", False):
        zpt.load_tables()
        zpt._astro_polars_loaded = True

    def run(s: pl.Series) -> pl.Series:
        cols = [s.struct.field(f).to_numpy() for f in s.struct.fields]
        if len(cols[0]) == 0:
            return pl.Series(s.name, [], dtype=pl.Float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = zpt.get_zpt(
                np.asarray(cols[0], dtype=float),
                np.asarray(cols[1], dtype=float),
                np.asarray(cols[2], dtype=float),
                np.asarray(cols[3], dtype=float),
                np.asarray(cols[4]).astype(int),
                _warnings=extrapolate,
            )
        return pl.Series(s.name, np.asarray(z, dtype=np.float64))

    args = [
        phot_g_mean_mag,
        nu_eff_used_in_astrometry,
        pseudocolour,
        ecl_lat,
        astrometric_params_solved,
    ]
    struct = pl.struct([to_expr(a).alias(f"a{i}") for i, a in enumerate(args)])
    return struct.map_batches(run, return_dtype=pl.Float64, is_elementwise=True)


def distance_from_parallax(
    parallax: IntoExpr = "parallax", zero_point: IntoExpr | float = 0.0
) -> pl.Expr:
    """Distance in kpc as ``1 / (parallax - zero_point)``, parallax in mas.

    Null for non-positive corrected parallaxes. A plain inverse is only
    sensible for precise parallaxes (e.g. ``parallax_over_error > 5``).
    """
    p = to_expr(parallax) - (zero_point if isinstance(zero_point, pl.Expr) else to_expr(zero_point))
    return pl.when(p > 0).then(1.0 / p)


# -- pipeline ----------------------------------------------------------------------
#: Columns added by `process` by default.
DEFAULT_QUANTITIES = (
    "l",
    "b",
    "pml",
    "pmb",
    "X",
    "Y",
    "Z",
    "vX",
    "vY",
    "vZ",
    "R",
    "phi",
    "vR",
    "vphi",
    "v_toomre",
)


def process(
    df: pl.DataFrame | pl.LazyFrame,
    quantities: Sequence[str] = DEFAULT_QUANTITIES,
    *,
    correct_rv: bool = False,
    correct_rv_error: bool = False,
    zero_point: str | float | None = None,
    pm_errors: bool = False,
    solar: SolarParameters | None = None,
    potential=None,
):
    """Galactic coordinates and Galactocentric kinematics for a Gaia DR3 table.

    Steps, each skipped if its output column already exists:

    1. ``correct_rv``: add ``radial_velocity_corrected``
       (`correct_radial_velocity`) and use it for the velocities.
    2. ``correct_rv_error``: add ``radial_velocity_error_corrected``.
    3. ``distance`` (kpc) from ``parallax`` minus ``zero_point``: None (no
       correction), a number in mas, or ``"L21"`` for
       `lindegren21_zero_point`.
    4. ``quantities`` through ``df.galactic.calc`` (default
       `DEFAULT_QUANTITIES`: Galactic ``l, b, pml, pmb``, Galactocentric
       ``X..vZ``, ``R, phi, vR, vphi, v_toomre``). Add ``"E"``, ``"JR"`` etc.
       to get energies and actions (AGAMA, see `astro_polars.dynamics`).
    5. ``pm_errors``: propagate ``pmra_error, pmdec_error, pmra_pmdec_corr``
       to ``pml_error, pmb_error, pml_pmb_corr``.
    """
    from .. import dynamics  # noqa: F401  (registers df.galactic)

    names = set(df.collect_schema().names())
    rv = "radial_velocity"
    if correct_rv:
        if "radial_velocity_corrected" not in names:
            df = df.with_columns(correct_radial_velocity().alias("radial_velocity_corrected"))
        rv = "radial_velocity_corrected"
    if correct_rv_error and "radial_velocity_error_corrected" not in names:
        df = df.with_columns(
            correct_radial_velocity_error().alias("radial_velocity_error_corrected")
        )
    if "distance" not in names:
        if zero_point is None:
            zp = 0.0
        elif isinstance(zero_point, str):
            if zero_point.upper() != "L21":
                raise ValueError("zero_point must be None, a number or 'L21'")
            zp = lindegren21_zero_point()
        else:
            zp = float(zero_point)
        df = df.with_columns(distance_from_parallax("parallax", zp).alias("distance"))
    df = df.galactic.calc(
        list(quantities),
        columns={"radial_velocity": rv} if rv != "radial_velocity" else None,
        solar=solar,
        potential=potential,
    )
    if pm_errors:
        df = df.sky.galactic_pm_errors()
    return df
