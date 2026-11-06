# Conventions

- **Angles** are in degrees, in and out. Separations from `crossmatch` are in
  arcseconds (column `separation`).
- **Proper motions**: the longitude component includes `cos(lat)`, as Gaia's
  `pmra`. Outputs follow the same rule (`pml` is mu_l*).
- **HEALPix**: nested ordering by default. Resolution is given as `nside`
  (a power of two); columns are named `hpx{nside}`, e.g. `hpx4096`. Changing
  resolution of a nested index is integer division, see
  `astro_polars.healpix.degrade`.
- **Gaia `source_id`** encodes the nested order-12 (nside 4096) pixel:
  `astro_polars.healpix.gaia_source_id_to_healpix`.
- **Frames** are 3x3 rotation matrices `M` with `r_out = M @ r_in`.
  ICRS to Galactic uses the Hipparcos matrix (ESA 1997). It differs from
  astropy's FK5-based Galactic frame by a few tens of mas.
- **Heatmaps** return `H` with x as the first axis (like
  `numpy.histogram2d`); show them with `imshow(H.T, origin="lower", extent=extent)`
  or `astro_polars.viz.plot_heatmap`.
- **Cross-match backends**: the default `method="kdtree"` searches 3D unit
  vectors by chord length (SciPy cKDTree, multithreaded; about 2 s for
  1M x 1M sources on 4 cores). `method="balltree"` uses
  scikit-learn's BallTree with the haversine metric, which expects (lat, lon),
  so the coordinates are flipped before building it. Both return the same
  pairs.
