# Quick start

```python
import numpy as np
import polars as pl
import astro_polars as ap

rng = np.random.default_rng(1)
n = 1_000_000
lf = pl.LazyFrame({
    "ra": rng.uniform(0, 360, n),
    "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
    "pmra": rng.normal(0, 5, n),
    "pmdec": rng.normal(0, 5, n),
    "g": rng.normal(19, 1.5, n),
})
```

## Selections

```python
lf.sky.cone(266.4, -28.9, radius=5)                 # rows within 5 deg
lf.sky.polygon([[10, 0], [20, 0], [20, 10]], x="ra", y="dec")
lf.filter(ap.in_cone("ra", "dec", 266.4, -28.9, 5))   # same, as an expression
```

## New columns

```python
lf = (
    lf.sky.separation(266.4, -28.9)                  # "sep" in degrees
      .sky.to_galactic()                              # l, b, pml, pmb
      .sky.add_healpix(4096)                          # hpx4096 (nested)
      .with_columns(ap.degrade("hpx4096", 4096, 32).alias("hpx32"))
)
```

Stream frames:

```python
M = ap.frame_from_pole(lon_pole=34.5, lat_pole=29.7, lon_origin=200.0, lat_origin=59.4)
lf.sky.to_frame(M)                     # phi1, phi2, pm_phi1, pm_phi2
lf.sky.to_frame(ap.GD1_KOPOSOV10)      # GD-1 frame of Koposov+10
```

Proper-motion errors through any rotation:

```python
lf.with_columns(
    ap.rotate_pm_errors("ra", "dec", "pmra_error", "pmdec_error", "pmra_pmdec_corr",
                        ap.ICRS_TO_GALACTIC).struct.unnest()
)
```

## Robust statistics

```python
lf.group_by("hpx32").agg(
    pl.col("g").median(),
    pl.col("g").robust.nmad().alias("g_nmad"),
)
```

## Cross-match and duplicates

```python
pairs = ap.crossmatch(cat1, cat2, radius_arcsec=1.0, how="inner")   # LazyFrame
clean = ap.remove_duplicates(cat, radius_arcsec=1.0)                # DataFrame
```

## Photometry

```python
lf.with_columns(
    g=ap.flux_to_mag("flux_g", zeropoint="nanomaggy"),         # 22.5
    g_err=ap.fluxerr_to_magerr("flux_g_err", "flux_g"),
    g_r=ap.flux_color("flux_g", "flux_r"),
)

# many bands at once; entries whose columns are missing are skipped
lf.phot.add_mags({"g": "flux_g", "r": "flux_r", "z": "flux_z"}, zeropoint="nanomaggy")
lf.phot.deredden({"g": 3.214, "r": 2.165}, ebv="ebv")          # adds g_0, r_0
```

`zeropoint` is a number, a unit name (`"Jy"`, `"uJy"`, `"nJy"`,
`"nanomaggy"`...) or an expression such as `pl.col("zp")`.

## HEALPix-partitioned Parquet

```python
from astro_polars.io import write_partitions, scan_hive

write_partitions(df, "cat_hive/", nside=32, tag="chunk01")   # hpx32=<pix>/part-chunk01-<pix>.parquet
lf = scan_hive("cat_hive/", cone=(150.1, 2.2, 1.0))          # reads only partitions touching the cone
lf.sky.cone(150.1, 2.2, 1.0).collect()                       # exact cut
```

## Plots

```python
lf.viz.heatmap("ra", "dec", shape=512, log=True)
lf.viz.healpix("ra", "dec", nside=64, what="mean", z="g", projection="mcbryde")

H, extent = ap.viz.heatmap(lf, "ra", "dec", shape=256)        # arrays only
m = ap.viz.heatmap(lf, "ra", "dec", nside=128)                  # HEALPix map
pix, val = ap.viz.healpix_map(lf, nside=1024, sparse=True)      # non-empty pixels
ap.viz.skymap((pix, val), nside=1024, projection="mollweide")
ap.viz.use_style()                                              # tab10 colour cycle
```

If a column already holds nested pixels at high resolution (for example
`hpx4096`), pass it as `ipix` to skip the coordinate maths:

```python
lf.viz.healpix(nside=256, ipix="hpx4096", ipix_nside=4096)
```
