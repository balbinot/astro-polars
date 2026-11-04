# astro-polars

Astronomy on [Polars](https://pola.rs): cross-matching, sky selections, HEALPix,
frame rotations and sky maps for catalogues with millions to billions of rows.
Everything works on lazy and streaming Polars frames, so the same code runs on a
notebook-sized table or on a partitioned survey on disk.

Documentation: https://balbinot.github.io/astro-polars/

## Install

```bash
pip install astro-polars            # core
pip install "astro-polars[viz]"     # + matplotlib and skyproj for plots
pip install "astro-polars[all]"     # every optional feature
```

Development version:

```bash
pip install "git+https://github.com/balbinot/astro-polars"
```

Python 3.9 or newer. The core depends on polars, numpy, numba, scikit-learn and
hpgeom.

## Quick look

```python
import polars as pl
import astro_polars as ap

lf = pl.scan_parquet("gaia_dr3/*.parquet")

# selections and new columns, all lazy
sel = (
    lf.sky.cone(266.4, -28.9, radius=2.0)           # degrees
      .sky.to_galactic()                             # l, b, pml, pmb
      .sky.add_healpix(4096)                         # hpx4096, nested
      .filter(pl.col("parallax").robust.zscore().abs() < 5)
)

# cross-match two catalogues (BallTree, haversine)
matches = ap.crossmatch(lf, other, radius_arcsec=1.0, how="inner")

# heatmaps and sky maps, binned in streaming Polars
lf.viz.heatmap("bp_rp", "phot_g_mean_mag", log=True)
lf.viz.healpix("ra", "dec", nside=128, what="mean", z="parallax")
```

## What is in it

| Module | Contents |
|---|---|
| `astro_polars.sphere` | `angular_separation` (Vincenty, numba), `angsep` and `in_cone` expressions |
| `astro_polars.polygon` | `in_polygon` / `pinside`, parallel ray casting with numba |
| `astro_polars.crossmatch` | `crossmatch` (alias `crossmatch_polars`), `remove_duplicates` |
| `astro_polars.healpix` | `healpix_index`, `degrade`, `gaia_source_id_to_healpix`, `cone_pixels` |
| `astro_polars.coords` | rotations with proper motions and their errors: ICRS, Galactic, stream frames |
| `astro_polars.robust` | `pl.col(x).robust.nmad()`, `.robust.zscore()` |
| `astro_polars.viz` | `heatmap` on grids or HEALPix, `skymap` with skyproj, `df.viz` namespace |

Planned next: Euclid MER catalogue pipeline, HEALPix hive-partitioned Parquet,
HATS read and write, survey modules (Gaia DR4, 4MOST, DESI, DES, LSST). See the
[roadmap](https://balbinot.github.io/astro-polars/roadmap/).

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).

## Acknowledgements

The idea of a Polars extension for astronomy follows Tom Callingham's
[polars_astro](https://github.com/TomCallingham/polars_astro). This package is
a separate, fresh implementation.
