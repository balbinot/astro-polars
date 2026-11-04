"""astro-polars: astronomy on Polars.

Importing the package registers these Polars namespaces:

- ``df.sky`` / ``lf.sky``: cone and polygon selections, separations, HEALPix
  indices, frame rotations, cross-matching, duplicate removal.
- ``df.viz`` / ``lf.viz``: heatmaps and HEALPix sky maps (plotting needs the
  ``viz`` extra).
- ``pl.col(...).robust``: NMAD and robust z-scores.
"""

from . import coords, healpix, polygon, robust, sphere, viz
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
from .healpix import cone_pixels, degrade, gaia_source_id_to_healpix, healpix_index
from .namespace import Sky
from .polygon import close_polygon, in_polygon, inpoly, inpoly_parallel, pinside
from .robust import K_NMAD, nmad, zscore
from .sphere import angsep, angular_separation, in_cone

del _crossmatch_module

__all__ = [
    "__version__",
    # modules
    "coords",
    "healpix",
    "polygon",
    "robust",
    "sphere",
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
    # robust
    "K_NMAD",
    "nmad",
    "zscore",
    # namespaces
    "Sky",
]
