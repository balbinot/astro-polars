"""astro-polars: astronomy on Polars.

Importing the package registers these Polars namespaces:

- ``df.sky`` / ``lf.sky``: cone and polygon selections, separations, HEALPix
  indices, frame rotations, cross-matching, duplicate removal.
- ``df.viz`` / ``lf.viz``: heatmaps and HEALPix sky maps (plotting needs the
  ``viz`` extra).
- ``df.phot`` / ``lf.phot``: magnitudes, magnitude errors, colours and
  extinction corrections from many columns at once.
- ``df.galactic`` / ``lf.galactic``: Galactocentric positions and
  velocities, solar reflex correction, and ``calc`` for kinematic and
  dynamical quantities (angular momenta, energies, actions with AGAMA).
- ``pl.col(...).robust``: NMAD and robust z-scores.

Subpackages: ``astro_polars.io`` (HEALPix hive Parquet datasets, FITS
tables) and ``astro_polars.surveys`` (survey pipelines, e.g.
``from astro_polars.surveys import euclid, gaia``).
"""

from . import (
    coords,
    dynamics,
    galactic,
    healpix,
    io,
    photometry,
    polygon,
    robust,
    sphere,
    surveys,
    viz,
)
from . import crossmatch as _crossmatch_module
from ._version import __version__
from .coords import (
    GALACTIC_TO_ICRS,
    GD1_KOPOSOV10,
    ICRS_TO_GALACTIC,
    frame_from_pole,
    rotate,
    rotate_pm,
    rotate_pm_errors,
)
from .crossmatch import crossmatch, crossmatch_polars, remove_duplicates
from .dynamics import Galactic
from .galactic import SolarParameters, galactocentric, reflex_correct
from .healpix import cone_pixels, degrade, gaia_source_id_to_healpix, healpix_index
from .namespace import Sky
from .photometry import (
    AB_ZEROPOINTS,
    MAGERR_FACTOR,
    Photometry,
    ab_zeropoint,
    aperture_correction,
    deredden,
    flux_color,
    flux_to_mag,
    fluxerr_to_magerr,
    mag_to_flux,
    magerr_to_fluxerr,
)
from .polygon import close_polygon, in_polygon, inpoly, inpoly_parallel, pinside
from .robust import K_NMAD, nmad, zscore
from .sphere import angsep, angular_separation, in_cone

del _crossmatch_module

__all__ = [
    "__version__",
    # modules
    "coords",
    "dynamics",
    "galactic",
    "healpix",
    "io",
    "photometry",
    "polygon",
    "robust",
    "sphere",
    "surveys",
    "viz",
    # sphere
    "angular_separation",
    "angsep",
    "in_cone",
    # polygon
    "inpoly",
    "inpoly_parallel",
    "in_polygon",
    "pinside",
    "close_polygon",
    # crossmatch
    "crossmatch",
    "crossmatch_polars",
    "remove_duplicates",
    # healpix
    "healpix_index",
    "degrade",
    "gaia_source_id_to_healpix",
    "cone_pixels",
    # coords
    "ICRS_TO_GALACTIC",
    "GALACTIC_TO_ICRS",
    "GD1_KOPOSOV10",
    "frame_from_pole",
    "rotate",
    "rotate_pm",
    "rotate_pm_errors",
    # galactic
    "SolarParameters",
    "galactocentric",
    "reflex_correct",
    # photometry
    "AB_ZEROPOINTS",
    "MAGERR_FACTOR",
    "ab_zeropoint",
    "flux_to_mag",
    "mag_to_flux",
    "fluxerr_to_magerr",
    "magerr_to_fluxerr",
    "flux_color",
    "aperture_correction",
    "deredden",
    # robust
    "K_NMAD",
    "nmad",
    "zscore",
    # namespaces
    "Sky",
    "Galactic",
    "Photometry",
]
