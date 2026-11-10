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

from ._utils import wrap360

__all__ = ["crossmatch", "crossmatch_polars", "find_duplicates", "remove_duplicates"]

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


def _finite_rows(lonlat_deg: np.ndarray) -> np.ndarray:
    """Indices of rows whose two coordinates are finite (nulls arrive as NaN)."""
    return np.flatnonzero(np.isfinite(lonlat_deg).all(axis=1)).astype(np.int64)


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
        extra ``_2``. A ``df1`` column called ``separation`` or
        ``multiplicity`` becomes ``separation_1`` / ``multiplicity_1``.
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
    columns stay lazy and are joined back at the end. Rows with a null, NaN
    or infinite coordinate are never matched (with ``how="left"`` they are
    returned with null ``df2`` columns).
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

    # rows with a null, NaN or infinite coordinate never match
    rows1, rows2 = _finite_rows(coords1), _finite_rows(coords2)
    _say(verbose, f"Matching {len(rows2)} sources against {len(rows1)} ({method}, k={k})...")
    if len(rows1) and len(rows2):
        if method == "kdtree":
            sep_arcsec, idx = _match_kdtree(
                coords1[rows1], coords2[rows2], k, radius_arcsec, n_jobs
            )
        else:
            k_eff = min(k, len(rows1))  # BallTree refuses k > number of points
            sep_arcsec, idx = _match_balltree(coords1[rows1], coords2[rows2], k_eff, n_jobs)
        found = idx < len(rows1)  # cKDTree marks missing neighbours with idx = n
        idx1 = np.where(found, rows1[np.minimum(idx, len(rows1) - 1)], -1)
        idx2 = np.broadcast_to(rows2[:, None], idx.shape)
        sep_arcsec = np.where(found, sep_arcsec, np.inf)
    else:
        idx1 = idx2 = np.empty(0, np.int64)
        sep_arcsec = np.empty(0, np.float64)
    _say(verbose, "Query done.")

    matches = (
        pl.DataFrame(
            {
                "_idx1": idx1.ravel().astype(np.int64),
                "_idx2": idx2.ravel().astype(np.int64),
                "separation": sep_arcsec.ravel().astype(np.float64),
            }
        )
        .lazy()
        .filter(pl.col("separation") < radius_arcsec)
        .with_columns(pl.len().over("_idx2").cast(pl.UInt32).alias("multiplicity"))
    )

    # a kept column that would clash with the two added columns gets "_1"
    added = {"separation", "multiplicity"}
    names1 = [f"{c}{sfx1}" for c in keep1]
    names1 = [f"{n}_1" if n in added else n for n in names1]
    taken = set(names1) | added
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


def _close_pairs(lonlat_deg: np.ndarray, radius_arcsec: float, n_jobs: int, method: str):
    """Every pair ``(i, j)``, ``i < j``, closer than ``radius_arcsec``."""
    if method == "kdtree":
        from scipy.spatial import cKDTree

        chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
        pairs = cKDTree(_unit_vectors(lonlat_deg)).query_pairs(chord, output_type="ndarray")
        return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)

    from sklearn.neighbors import BallTree

    rad = np.deg2rad(lonlat_deg[:, ::-1])
    tree = BallTree(rad, metric="haversine")
    nbrs = tree.query_radius(rad, r=np.radians(radius_arcsec / 3600.0))
    counts = np.fromiter((len(a) for a in nbrs), np.int64, len(nbrs))
    i = np.repeat(np.arange(len(nbrs), dtype=np.int64), counts)
    j = np.concatenate(nbrs).astype(np.int64) if len(nbrs) else np.empty(0, np.int64)
    keep = i < j
    return i[keep], j[keep]


def _group_labels(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Friends-of-friends labels: connected components of the close-pair graph.

    Labels are numbered in order of each group's first row.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def find_duplicates(
    df: pl.DataFrame | pl.LazyFrame,
    x: str = "ra",
    y: str = "dec",
    radius_arcsec: float = 1.0,
    group: str = "dup_group",
    size: str = "dup_n",
    n_jobs: int = -1,
    method: str = "kdtree",
) -> pl.DataFrame:
    """Label groups of sources that lie within ``radius_arcsec`` of each other.

    Groups are friends-of-friends: two sources share a group when they are
    linked by a chain of pairs each closer than ``radius_arcsec``. A chain
    A~B~C is one group even if A and C are further apart than the radius.
    Every source is in a group; isolated sources form groups of one, and so
    does every row with a null, NaN or infinite coordinate.

    Parameters
    ----------
    df
        Catalogue.
    x, y
        Longitude and latitude columns, in degrees.
    radius_arcsec
        Linking length, in arcseconds.
    group, size
        Names of the added columns: the group label (numbered in order of
        each group's first row) and the number of sources in the group.
    n_jobs
        Threads for the ``"balltree"`` query (the KD-tree pair search is
        single threaded).
    method
        ``"kdtree"`` (default) or ``"balltree"``. Both give the same groups.

    Returns
    -------
    polars.DataFrame
        ``df`` with ``group`` and ``size`` columns added, in the original row
        order.
    """
    if method not in ("kdtree", "balltree"):
        raise ValueError("method must be 'kdtree' or 'balltree'")
    df = df.lazy().collect()
    coords = df.select(x, y).to_numpy().astype(np.float64)
    # rows with a null, NaN or infinite coordinate stay in groups of their own
    rows = _finite_rows(coords)
    i, j = _close_pairs(coords[rows], radius_arcsec, n_jobs, method) if len(rows) else (rows, rows)
    i, j = rows[i], rows[j]
    labels = _group_labels(df.height, i, j)
    return df.with_columns(pl.Series(group, labels)).with_columns(
        pl.len().over(group).cast(pl.UInt32).alias(size)
    )


_STATS = ("mean", "median", "min", "max", "sum", "std", "var", "first", "last")


def _stat(col: str, stat: str) -> pl.Expr:
    if stat not in _STATS:
        raise ValueError(f"unknown statistic {stat!r}; use one of {', '.join(_STATS)}")
    return getattr(pl.col(col), stat)().alias(col)


def _centroid(x: str, y: str) -> list[pl.Expr]:
    """Group centroid on the sphere: mean unit vector, safe across RA = 0/360."""
    lon, lat = pl.col(x).radians(), pl.col(y).radians()
    cx = (lat.cos() * lon.cos()).mean()
    cy = (lat.cos() * lon.sin()).mean()
    cz = lat.sin().mean()
    return [
        wrap360(pl.arctan2(cy, cx).degrees()).alias(x),
        pl.arctan2(cz, (cx * cx + cy * cy).sqrt()).degrees().alias(y),
    ]


def remove_duplicates(
    df: pl.DataFrame | pl.LazyFrame,
    x: str = "ra",
    y: str = "dec",
    radius_arcsec: float = 1.0,
    priority: str | pl.Expr | None = None,
    descending: bool = False,
    agg: str | dict[str, str] | Sequence[pl.Expr] | None = None,
    size: str | None = None,
    n_jobs: int = -1,
    method: str = "kdtree",
) -> pl.DataFrame:
    """Collapse each group of close sources to a single row.

    Sources are grouped friends-of-friends with linking length
    ``radius_arcsec`` (see ``find_duplicates``), so pairs, triples,
    quadruples and chains (A~B, B~C, A and C further apart) each become one
    row. No two output rows are closer than ``radius_arcsec``.

    Parameters
    ----------
    df
        Catalogue.
    x, y
        Longitude and latitude columns, in degrees.
    radius_arcsec
        Linking length, in arcseconds.
    priority
        Column (or expression) that picks each group's representative: the
        row with the smallest value, or the largest with
        ``descending=True``. Default: the group's first row. Nulls rank last.
    descending
        Pick the largest ``priority`` instead of the smallest.
    agg
        ``None`` (default): return the representative row of each group,
        unchanged. Otherwise return statistics of each group:

        - a statistic name (``"mean"``, ``"median"``, ``"min"``, ``"max"``,
          ``"sum"``, ``"std"``, ``"var"``, ``"first"``, ``"last"``) applied
          to every numeric column;
        - a dict ``{column: statistic}`` for chosen columns;
        - a list of Polars expressions evaluated per group, e.g.
          ``[pl.col("g").median(), pl.col("g").std().alias("g_std")]``.

        Columns that ``agg`` does not set take the representative's value,
        except ``x`` and ``y``, which become the group centroid on the
        sphere (mean unit vector) unless ``agg`` sets them.
    size
        If given, add a column of this name with the number of sources
        merged into each row.
    n_jobs, method
        Passed to ``find_duplicates``.

    Returns
    -------
    polars.DataFrame
        One row per group, ordered by the representative's position in
        ``df``. With ``agg=None`` the columns are those of ``df`` (plus
        ``size``).

    Examples
    --------
    >>> clean = remove_duplicates(cat, radius_arcsec=1.0)
    >>> best = remove_duplicates(cat, priority="mag_err")            # keep most precise
    >>> merged = remove_duplicates(cat, agg="median", size="n_obs")  # median of each group
    >>> merged = remove_duplicates(
    ...     cat, agg=[pl.col("g").mean(), pl.col("g").std().alias("g_std")]
    ... )
    """
    grp, n_col, row = "__dup_group", "__dup_n", "__dup_row"
    lab = find_duplicates(
        df,
        x=x,
        y=y,
        radius_arcsec=radius_arcsec,
        group=grp,
        size=n_col,
        n_jobs=n_jobs,
        method=method,
    ).with_row_index(row)
    cols = [c for c in lab.columns if c not in (grp, n_col, row)]

    # representative first within each group
    if priority is None:
        ranked = lab
    else:
        key = pl.col(priority) if isinstance(priority, str) else priority
        ranked = lab.sort([key, pl.col(row)], descending=[descending, False], nulls_last=True)

    if agg is None:
        # only the (small) key columns are grouped; the rows are then taken in original order
        reps = ranked.group_by(grp, maintain_order=True).agg(pl.col(row).first())[row]
        out = lab.filter(pl.col(row).is_in(reps.implode())).select(cols + [n_col])
    else:
        if isinstance(agg, str):
            numeric = [c for c in cols if lab.schema[c].is_numeric()]
            exprs = [_stat(c, agg) for c in numeric if c not in (x, y)]
        elif isinstance(agg, dict):
            exprs = [_stat(c, s) for c, s in agg.items()]
        else:
            exprs = list(agg)
        set_by_agg = {e.meta.output_name() for e in exprs}
        # centroid for whichever of x, y the user did not aggregate
        exprs += [e for e in _centroid(x, y) if e.meta.output_name() not in set_by_agg]
        set_by_agg |= {x, y}
        rep = [pl.col(c).first() for c in cols if c not in set_by_agg]
        out = ranked.group_by(grp, maintain_order=True).agg(
            pl.col(row).first(), pl.col(n_col).first(), *rep, *exprs
        )
        extra = [e for e in out.columns if e not in cols and e not in (grp, n_col, row)]
        out = out.sort(row).select([c for c in cols if c in out.columns] + extra + [n_col])

    if size is None:
        return out.drop(n_col)
    return out.rename({n_col: size})
