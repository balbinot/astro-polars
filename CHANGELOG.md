# Changelog

## 0.1.0 (unreleased)

First version.

- `sphere`: `angular_separation` (Vincenty, numba), `angsep` and `in_cone` expressions.
- `polygon`: parallel point-in-polygon (`in_polygon`, `pinside`); edges count as inside; open polygons are closed automatically.
- `crossmatch`: BallTree haversine cross-match with `how="left"|"inner"` and threaded queries; `remove_duplicates` returns the cleaned catalogue.
- `healpix`: `healpix_index` (hpgeom), `degrade`, `gaia_source_id_to_healpix`, `cone_pixels`.
- `coords`: rotations with proper motions and exact error propagation; ICRS/Galactic matrices, `frame_from_pole`, GD-1 frame.
- `robust`: `.robust.nmad()`, `.robust.zscore()`.
- `viz`: streaming `heatmap` on a grid or HEALPix, `healpix_map`, `skymap` (skyproj), `plot_heatmap`, `use_style`, `df.viz` namespace.
- `df.sky` namespace.
