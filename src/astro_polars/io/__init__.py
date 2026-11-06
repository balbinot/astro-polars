"""Reading and writing catalogues.

- ``astro_polars.io.hive``: HEALPix hive-partitioned Parquet datasets.
- ``astro_polars.io.fits``: FITS binary tables to Polars (``fits`` extra).
"""

from .hive import Manifest, remove_partitions, scan_hive, write_partitions

__all__ = ["Manifest", "remove_partitions", "scan_hive", "write_partitions"]
