# astro-polars

Astronomy on [Polars](https://pola.rs). Cross-match catalogues, select by cone or
polygon, rotate between sky frames with proper motions and their errors,
attach HEALPix indices, compute magnitudes and colours, store catalogues as
HEALPix-partitioned Parquet and draw heatmaps and sky maps, all on lazy and
streaming Polars frames.

```python
import polars as pl
import astro_polars as ap

lf = pl.scan_parquet("survey/**/*.parquet")
lf.sky.cone(150.1, 2.2, radius=1.0).sky.to_galactic().collect()
```

Importing `astro_polars` adds four namespaces to Polars:

| Namespace | On | What |
|---|---|---|
| `.sky` | DataFrame, LazyFrame | cones, polygons, separations, HEALPix, frame rotations, cross-match, duplicates |
| `.phot` | DataFrame, LazyFrame | magnitudes, magnitude errors, colours and extinction for many bands at once |
| `.viz` | DataFrame, LazyFrame | heatmaps on a grid or on HEALPix, sky maps with skyproj |
| `.robust` | expressions | NMAD and robust z-scores |

Everything is also available as plain functions returning Polars
expressions, for use inside `with_columns`, `filter` or `group_by`.

Survey pipelines live in `astro_polars.surveys`; the first is
[Euclid MER](euclid.md).

Start with [Install](install.md) and the [Quick start](quickstart.md). The
[Roadmap](roadmap.md) lists what comes next (HATS, more survey modules).
