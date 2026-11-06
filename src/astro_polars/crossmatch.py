"""Sky cross-matching and duplicate removal.

Two tree backends give the same pairs:

- ``method="kdtree"`` (default): SciPy's cKDTree on 3D unit vectors, searched
  by chord length with a distance bound and multithreaded queries. Roughly an
  order of magnitude faster than the BallTree.
- ``method="balltree"``: scikit-learn's BallTree with the haversine metric,
  queried in parallel threads.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import polars as pl

__all__ = ["crossmatch", "crossmatch_polars", "remove_duplicates"]

log = logging.getLogger(__name__)


def _say(verbose: bool, msg: str) -> None:
    log.info(msg)
    if verbose:
        print(msg)


def _n_jobs(n_jobs: int) -> int:
    return (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)


def _unit_vectors(lonlat_deg: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lonlat_deg[:, 0]), np.radians(lonlat_deg[:, 1])
    clat = np.cos(lat)
    return np.column_stack([clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)])


def _match_kdtree(c1: np.ndarray, c2: np.ndarray, k: int, radius_arcsec: float, n_jobs: int):
    """Nearest neighbours by chord length; returns (separation_arcsec, idx), shape (n2, k)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(_unit_vectors(c1))
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    dist, idx = tree.query(
        _unit_vectors(c2), k=k, distance_upper_bound=chord, workers=_n_jobs(n_jobs)
    )
    dist, idx = dist.reshape(len(c2), k), idx.reshape(len(c2), k)
    with np.errstate(invalid="ignore"):
        sep = np.degrees(2.0 * np.arcsin(np.minimum(dist, 2.0) / 2.0)) * 3600.0
    return sep, idx


def _match_balltree(c1: np.ndarray, c2: np.ndarray, k: int, n_jobs: int):
    """BallTree (haversine) query; returns (separation_arcsec, idx)."""
    from sklearn.neighbors import BallTree

    # haversine in scikit-learn expects (lat, lon) in radians, hence the flip.
    tree = BallTree(np.deg2rad(c1[:, ::-1]), metric="haversine")
    dist_rad, idx = _query_tree(tree, np.deg2rad(c2[:, ::-1]), k, n_jobs)
    return np.rad2deg(dist_rad) * 3600.0, idx


def _query_tree(tree, coords_rad: np.ndarray, k: int, n_jobs: int):
    """Query ``tree`` in parallel chunks. BallTree releases the GIL, so threads scale."""
    jobs = _n_jobs(n_jobs)
    if jobs <= 1 or len(coords_rad) < 1000:
        return tree.query(coords_rad, k=k)

    from concurrent.futures import ThreadPoolExecutor

    chunks = np.array_split(coords_rad, jobs)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(lambda c: tree.query(c, k=k), chunks))
    dist, idx = zip(*results)
    return np.vstack(dist), np.vstack(idx)


def crossmatch(
    df1: pl.LazyFrame | pl.DataFrame,
    df2: pl.LazyFrame | pl.DataFrame,
    x1: str = "ra",
    y1: str = "dec",
    x2: str = "ra",
    y2: str = "dec",
    k: int = 1,
    radius_arcsec: float = 0.5,
    keep1: Sequence[str] | None = None,
    keep2: Sequence[str] | None = None,
    suffix1: str | None = None,
    suffix2: str | None = None,
    n_jobs: int = -1,
    how: str = "left",
    method: str = "kdtree",
    verbose: bool = False,
) -> pl.LazyFrame:
    """Cross-match two catalogues on the sky.

    A tree is built on ``df1``; for every source in ``df2`` the ``k``
    nearest ``df1`` sources are found and pairs closer than
    ``radius_arcsec`` are kept.

    Parameters
    ----------
    df1
        Catalogue the tree is built on.
    df2
        Catalogue whose sources are matched against ``df1``.
    x1, y1, x2, y2
        Longitude and latitude columns (degrees) in ``df1`` and ``df2``.
    k
        Number of nearest ``df1`` neighbours to consider per ``df2`` source.
    radius_arcsec
        Maximum separation of a match, in arcseconds.
    keep1, keep2
        Columns to keep from each catalogue (default: all). Names that do
        not exist are ignored.
    suffix1, suffix2
        Suffixes appended to the kept columns of each catalogue. A ``df2``
        column whose name would still clash with a ``df1`` column gets an
        extra ``_2``.
    n_jobs
        Threads used for the tree query; ``-1`` uses all CPUs.
    how
        ``"left"`` (default): every ``df1`` row is returned, with ``df2``
        columns null where it has no match. ``"inner"``: matched pairs only.
    method
        ``"kdtree"`` (default, fastest) or ``"balltree"`` (scikit-learn,
        haversine). Both return the same pairs.
    verbose
        Print progress messages.

    Returns
    -------
    polars.LazyFrame
        One row per pair (or unmatched ``df1`` row with ``how="left"``),
        with the kept columns plus ``separation`` (arcsec) and
        ``multiplicity`` (number of matches of that ``df2`` source).

    Notes
    -----
    Coordinates are collected into memory to build the tree; the other
    columns stay lazy and are joined back at the end.
    """
    if how not in ("left", "inner"):
        raise ValueError("how must be 'left' or 'inner'")
    if method not in ("kdtree", "balltree"):
        raise ValueError("method must be 'kdtree' or 'balltree'")

    df1, df2 = df1.lazy(), df2.lazy()
    df1_cols = df1.collect_schema().names()
    df2_cols = df2.collect_schema().names()

    keep1 = [c for c in (keep1 if keep1 is not None else df1_cols) if c in df1_cols]
    keep2 = [c for c in (keep2 if keep2 is not None else df2_cols) if c in df2_cols]
    sfx1 = suffix1 or ""
    sfx2 = suffix2 or ""

    _say(verbose, "Collecting coordinates...")
    coords1 = df1.select(x1, y1).collect().to_numpy().astype(np.float64)
    coords2 = df2.select(x2, y2).collect().to_numpy().astype(np.float64)

    _say(verbose, f"Matching {len(coords2)} sources against {len(coords1)} ({method}, k={k})...")
    if method == "kdtree":
        sep_arcsec, idx = _match_kdtree(coords1, coords2, k, radius_arcsec, n_jobs)
    else:
        sep_arcsec, idx = _match_balltree(coords1, coords2, k, n_jobs)
    _say(verbose, "Query done.")

    n2, k_neighbors = sep_arcsec.shape
    matches = (
        pl.DataFrame(
            {
                "_idx1": idx.ravel().astype(np.int64),
                "_idx2": np.arange(n2, dtype=np.int64).repeat(k_neighbors),
                "separation": sep_arcsec.ravel(),
            }
        )
        .lazy()
        .filter(pl.col("separation") < radius_arcsec)
        .with_columns(pl.len().over("_idx2").cast(pl.UInt32).alias("multiplicity"))
    )

    names1 = [f"{c}{sfx1}" for c in keep1]
    taken = set(names1) | {"separation", "multiplicity"}
    names2 = []
    for c in keep2:
        name = f"{c}{sfx2}"
        if name in taken:  # same column in both catalogues and no suffix to tell them apart
            name = f"{name}_2"
        names2.append(name)
        taken.add(name)

    lf1 = df1.with_row_index("_idx1").select(
        pl.col("_idx1").cast(pl.Int64), *[pl.col(c).alias(n) for c, n in zip(keep1, names1)]
    )
    lf2 = df2.with_row_index("_idx2").select(
        pl.col("_idx2").cast(pl.Int64), *[pl.col(c).alias(n) for c, n in zip(keep2, names2)]
    )

    pairs = matches.join(lf2, on="_idx2")
    out = lf1.join(pairs, on="_idx1", how=how)
    return out.select([*names1, *names2, "separation", "multiplicity"])


#: Alias of ``crossmatch``.
crossmatch_polars = crossmatch


def remove_duplicates(
    df: pl.DataFrame | pl.LazyFrame,
    x: str = "ra",
    y: str = "dec",
    radius_arcsec: float = 1.0,
    n_jobs: int = -1,
    method: str = "kdtree",
) -> pl.DataFrame:
    """Remove sources that have another source within ``radius_arcsec``.

    Each source is matched to its nearest neighbour other than itself. For
    every close pair, the row that comes later in ``df`` is dropped, so
    exactly one of each pair survives.

    Returns the de-duplicated catalogue as a DataFrame with the original
    columns and row order.
    """
    idx = "__dedup_idx"
    lf = df.lazy().with_row_index(idx)
    pairs = crossmatch(
        lf,
        lf,
        x1=x,
        y1=y,
        x2=x,
        y2=y,
        k=2,  # the source itself and its nearest neighbour
        radius_arcsec=radius_arcsec,
        keep1=[idx],
        keep2=[idx],
        suffix1="_1",
        suffix2="_2",
        n_jobs=n_jobs,
        how="inner",
        method=method,
    )
    to_remove = (
        pairs.filter(pl.col(f"{idx}_1") != pl.col(f"{idx}_2"))
        .select(pl.max_horizontal(f"{idx}_1", f"{idx}_2").alias(idx))
        .unique()
    )
    return lf.join(to_remove, on=idx, how="anti").sort(idx).drop(idx).collect()
