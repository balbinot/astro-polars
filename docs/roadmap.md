# Roadmap

What exists today is listed in the [API](api/namespace.md). This page records
what comes next, where it will live in the package, and the design choices
already made, so new code lands in a predictable place.

## Release plan

| Version | Content |
|---|---|
| 0.1 | Core: separations, cones, polygons, cross-match, duplicates, HEALPix, frame rotations with proper motions and errors, robust statistics, `viz` (heatmaps, HEALPix maps, skyproj). |
| 0.2 (done) | Photometry, Euclid MER pipeline, HEALPix hive-partitioned Parquet with manifest. |
| 0.3 | HATS read and write (basic). |
| 0.4 | Survey modules: Gaia (DR3 now, DR4 when its data model is public), 4MOST, DESI, DES, LSST. |
| later | HATS margins, remote reads and LSDB interop; Rust kernels where profiling shows Python is the bottleneck. |

## Done in 0.2

- **Photometry** (`astro_polars.photometry`): AB zero points by unit, flux to
  magnitude and back with numeric, unit-name or per-row zero points,
  first-order errors, colours, aperture corrections, extinction, and the
  `df.phot` namespace.
- **HEALPix hive datasets** (`astro_polars.io.hive`): the layout below,
  atomic writes, `Manifest` for restartable conversions, and `scan_hive`
  with cone or pixel pruning.
- **Euclid MER** (`astro_polars.surveys.euclid`): the pipeline as composable
  steps, `convert_tiles` and the `astro-polars euclid-hive` command; FITS is
  read with astropy (`fits` extra) without the pandas round trip. See
  [Euclid MER](euclid.md).

Hive layout (fastest to write and to prune when reading):

```
outdir/
  _manifest.json                # input file -> size, mtime, rows, partitions
  hpx32=1234/part-<tag>-1234.parquet
  hpx32=1235/...
```

Possible additions: per-band Euclid extinction coefficients for `deredden`,
a parallel (multi-process) `convert_tiles`, and Euclid SPE/PHZ catalogues.

## HATS: `astro_polars.io.hats`

[HATS](https://hats.readthedocs.io) is the HEALPix-partitioned Parquet format
used by LSDB and by several surveys. The first version covers plain catalogues.

**Read** `scan_hats(path, columns=None, cone=None, pixels=None) -> pl.LazyFrame`

1. Read `properties` (catalogue metadata) and `partition_info.csv`
   (one row per `Norder, Npix`).
2. Select partitions overlapping the region: for each `(Norder, Npix)`, test
   overlap with `cone_pixels` at that order.
3. `pl.scan_parquet` over `dataset/Norder=k/Dir=d/Npix=p.parquet` for the
   selected partitions only, then filter rows on `_healpix_29` (nested,
   order 29) or on coordinates.

**Write** `write_hats(lf, path, catalog_name, ra="ra", dec="dec", threshold=1_000_000, highest_order=10)`

1. Add `_healpix_29` with `healpix_index(ra, dec, 2**29)`.
2. Count rows per pixel at `highest_order` (one streaming `group_by`).
3. Choose partitions bottom-up: merge four children into their parent while
   the total stays under `threshold` (the hats-import rule).
4. Write each partition sorted by `_healpix_29`, then `properties`,
   `partition_info.csv`, `dataset/_common_metadata` and `dataset/_metadata`.

**Extension points** (designed in now, built later):

| Feature | Where it plugs in |
|---|---|
| Margin caches | `write_hats(..., margin_arcsec=...)` writes `<name>_margin/` with the same layout; `scan_hats(..., margin=True)` unions it. Needed for complete cross-matches at partition edges. |
| Remote catalogues | Paths are passed to `pl.scan_parquet`, which already reads `s3://`, `gs://` and `https://`; only `properties` and `partition_info.csv` need an fsspec-style open. |
| `point_map.fits` | Optional output of `write_hats` via the `fits` extra; not needed to read. |
| LSDB interop | Tests open our output with `hats.read_hats` (the `hats` package needs Python 3.11+, so it is a test-only dependency). |
| Partition-wise cross-match | `crossmatch` per HATS partition plus margin, parallel over partitions. |

## Survey modules: `astro_polars.surveys`

One module per survey, each with: column-name maps to the package
conventions, flux/magnitude helpers with the survey's zero points, standard
quality cuts, and readers for the survey's distribution format.

| Module | Notes |
|---|---|
| `gaia` | DR3 done: radial-velocity and error corrections, extinction, Dodd+23 cuts, Lindegren+21 zero point, `process`. DR4: schema, epoch astrometry and photometry, covariance-based errors, once the data model is published. |
| `fourmost` | 4MOST target catalogues and spectroscopic products. |
| `desi` | Legacy Surveys nanomaggie fluxes (zero point 22.5), DESI redshift catalogues. |
| `des` | DES Y6 Gold columns, flags and extinction. |
| `lsst` | Rubin Data Preview / DR object tables (nJy fluxes), HATS-distributed catalogues. |
| `euclid` | see above |

## Streams and dynamics

- Done: `frame_from_pole`, `GD1_KOPOSOV10`, `df.sky.to_frame`; Galactocentric
  positions and velocities with configurable solar parameters, reflex-motion
  correction, `vgsr`, and `df.galactic.calc` with AGAMA energies, actions,
  angles, frequencies and orbit extent ([Galactic dynamics](dynamics.md)).
- Next: orbit integration (`zmax`, orbit-averaged quantities) through
  `agama.orbit`, Monte Carlo error propagation from the Gaia covariances, and a
  stream-track lookup from `galstreams` as an optional extra.

## Rust

The package is pure Python today: Polars does the heavy lifting, numba covers
point-in-polygon and separations, hpgeom covers HEALPix. When a kernel
needs Rust (a candidate is a HEALPix-bucketed cross-match for catalogues that
do not fit in memory), it will go into a separate compiled extension built with
maturin and pyo3-polars as Polars expression plugins, with the pure-Python
version kept as the reference in tests.
