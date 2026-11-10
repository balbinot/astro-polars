# Changelog

## Unreleased

- `remove_duplicates` now groups close sources friends-of-friends. The previous nearest-neighbour version could keep close pairs inside triples and quadruples (A=0", C=0.45", B=0.9" kept A and B at 1"). New `priority`/`descending` pick the surviving row, `agg` returns group statistics (`"mean"`, `"median"`, ..., a `{column: statistic}` dict or Polars expressions) with the spherical centroid as position, and `size` adds the group size. New `find_duplicates` labels the groups. Both are on `df.sky`.
- `galactic`: `SolarParameters` (R0, z0, U, V, W, vcirc; settable default), `galactocentric` and `from_galactocentric` (astropy-compatible frame, including the z0 tilt), `heliocentric_cartesian`, `reflex_correct` (proper motions in the Galactic standard of rest, ICRS or Galactic input), `vgsr`.
- `dynamics`: `df.galactic` namespace with `from_observed`, `to_observed`, `reflex_correct`, `vgsr` and `calc`, which resolves dependencies between ~60 quantities (cylindrical and spherical kinematics, angular momenta, energies, actions, angles, frequencies, pericentre, apocentre, eccentricity, circular-orbit radius, period and angular momentum, circularity, action-diamond coordinates) starting from observables or Galactocentric phase space. Potential-based quantities use AGAMA (`dynamics` extra) through `GalaxyModel`, with cached action finders; `available()` lists every quantity, and the names of polars_astro are accepted as aliases.
- `surveys.gaia`: `correct_radial_velocity` (Katz+23, Blomme+23), `correct_radial_velocity_error` (Babusiaux+23), `estimate_grvs`, `extinction` and `deredden` (Gaia Collaboration 2018 coefficients), `dodd23_quality`, `lindegren21_zero_point` (`gaia` extra), `distance_from_parallax` and the `process` pipeline.
- `df.sky.galactic_pm_errors`: Galactic proper-motion errors and correlation from Gaia's covariance.

## 0.2.0 (unreleased)

- `photometry`: `ab_zeropoint` and `AB_ZEROPOINTS` (Jy, mJy, uJy, nJy, nanomaggy), `flux_to_mag` / `mag_to_flux` with numeric, unit-name or per-row zero points, `fluxerr_to_magerr` / `magerr_to_fluxerr`, `flux_color`, `aperture_correction`, `deredden`; `df.phot` namespace (`add_mags`, `add_magerrs`, `add_colors`, `deredden`) that skips entries whose columns are missing.
- `io.hive`: HEALPix hive-partitioned Parquet. `write_partitions` (atomic writes, one file per input and partition), `remove_partitions`, `scan_hive` with cone or pixel pruning, `Manifest` for restartable conversions.
- `io.fits`: `read_fits_table`, FITS binary tables straight to Polars in native byte order (`fits` extra).
- `surveys.euclid`: Euclid MER final-catalogue pipeline as composable steps (`add_mags`, `add_aper_mags`, `add_magerrs`, `add_visto_colors`, `clean`, `horn_cut`, `magnitude_cut`, `transform`), `add_healpix`, and `convert_tiles` from downloaded tiles to a hive dataset.
- Command line: `astro-polars euclid-hive --indir ... --outdir ... [--nside 32] [--force]`.
- `crossmatch` uses a SciPy KD-tree by default (`method="balltree"` keeps the scikit-learn haversine BallTree); scipy is now a declared dependency.
- The `fits` extra installs `numpy<2` on Python 3.9.

## 0.1.0 (not released separately)

First version.

- `sphere`: `angular_separation` (Vincenty, numba), `angsep` and `in_cone` expressions.
- `polygon`: parallel point-in-polygon (`in_polygon`, `pinside`); edges count as inside; open polygons are closed automatically.
- `crossmatch`: sky cross-match with `how="left"|"inner"` and threaded queries; `remove_duplicates` returns the cleaned catalogue.
- `healpix`: `healpix_index` (hpgeom), `degrade`, `gaia_source_id_to_healpix`, `cone_pixels`.
- `coords`: rotations with proper motions and exact error propagation; ICRS/Galactic matrices, `frame_from_pole`, GD-1 frame.
- `robust`: `.robust.nmad()`, `.robust.zscore()`.
- `viz`: streaming `heatmap` on a grid or HEALPix, `healpix_map`, `skymap` (skyproj), `plot_heatmap`, `use_style`, `df.viz` namespace.
- `df.sky` namespace.
