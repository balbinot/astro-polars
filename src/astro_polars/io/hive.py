"""HEALPix hive-partitioned Parquet datasets.

Layout:

    root/
      _manifest.json                       # what has been converted (see Manifest)
      hpx32=1234/part-<tag>-1234.parquet   # one file per (input, partition)
      hpx32=1235/part-<tag>-1235.parquet

The partition key is a coarse nested HEALPix pixel (``hpx{nside}``, default
nside 32, about 3.4 deg²). It lives in the directory name only; Polars adds it
back as a column when reading with ``hive_partitioning=True``. Files usually
also carry a fine nested index (e.g. ``hpx4096``) for maps and fine filtering.

Each input (a survey tile, a chunk...) writes its own file in every partition
it touches, named by a ``tag``, so inputs never collide and one input can be
re-done by deleting its files. Writes go to a temporary file that is renamed
into place, so readers never see a partial file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ..healpix import cone_pixels, degrade, healpix_index, nside_to_order

__all__ = [
    "Manifest",
    "write_partitions",
    "remove_partitions",
    "scan_hive",
    "partition_nside",
    "partition_pixels",
]

_PART_RE = re.compile(r"^hpx(\d+)=(-?\d+)$")


class Manifest:
    """JSON record of converted input files, for incremental, restartable runs.

    A file is considered done when its name is in the manifest with the same
    size. Saving is atomic (temporary file + rename), so an interrupted run
    never leaves a corrupt manifest.
    """

    def __init__(self, path: str | os.PathLike, data: dict | None = None):
        self.path = Path(path)
        self.files: dict[str, dict] = dict(data or {})

    @classmethod
    def load(cls, path: str | os.PathLike) -> Manifest:
        """Read ``path`` if it exists, else start empty."""
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return cls(p, json.load(f))
        return cls(p)

    def save(self) -> None:
        """Write atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.files, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    @staticmethod
    def key(path: str | os.PathLike) -> str:
        return os.path.basename(os.fspath(path))

    @staticmethod
    def stamp(path: str | os.PathLike) -> dict:
        st = os.stat(path)
        return {"size": st.st_size, "mtime": int(st.st_mtime)}

    def is_done(self, path: str | os.PathLike) -> bool:
        """True if ``path`` was converted and its size has not changed."""
        rec = self.files.get(self.key(path))
        return rec is not None and rec.get("size") == self.stamp(path)["size"]

    def record(self, path: str | os.PathLike, **info) -> None:
        """Mark ``path`` as done, storing its size, mtime and ``info``."""
        self.files[self.key(path)] = {**self.stamp(path), **info}

    def __contains__(self, path) -> bool:
        return self.key(path) in self.files

    def __len__(self) -> int:
        return len(self.files)


def _partition_expr(
    df,
    nside: int,
    lon: str | None,
    lat: str | None,
    ipix: str | None,
    ipix_nside: int | None,
) -> pl.Expr:
    if ipix is not None:
        if ipix_nside is None:
            raise ValueError("ipix_nside is required with ipix")
        return degrade(ipix, ipix_nside, nside)
    if lon is None or lat is None:
        raise ValueError("give lon and lat, or ipix and ipix_nside")
    return healpix_index(lon, lat, nside, nest=True)


def write_partitions(
    df: pl.DataFrame | pl.LazyFrame,
    root: str | os.PathLike,
    nside: int = 32,
    tag: str = "part",
    *,
    lon: str | None = "ra",
    lat: str | None = "dec",
    ipix: str | None = None,
    ipix_nside: int | None = None,
    compression: str = "zstd",
) -> int:
    """Write ``df`` into ``root`` partitioned by nested HEALPix pixel at ``nside``.

    The partition index is computed from ``lon``/``lat`` (degrees), or, faster,
    degraded from an existing nested index column ``ipix`` at ``ipix_nside``.
    Files are named ``hpx{nside}={pix}/part-{tag}-{pix}.parquet``; an existing
    file with that name is replaced. Returns the number of files written.
    """
    nside_to_order(nside)
    key = f"hpx{nside}"
    df = df.lazy().collect() if isinstance(df, pl.LazyFrame) else df
    if key not in df.columns:
        df = df.with_columns(_partition_expr(df, nside, lon, lat, ipix, ipix_nside).alias(key))

    root = Path(root)
    n = 0
    for (pix,), part in df.partition_by(key, as_dict=True).items():
        pdir = root / f"{key}={pix}"
        pdir.mkdir(parents=True, exist_ok=True)
        fpath = pdir / f"part-{tag}-{pix}.parquet"
        tmp = fpath.with_name(fpath.name + ".tmp")
        part.drop(key).write_parquet(tmp, compression=compression)
        os.replace(tmp, fpath)
        n += 1
    return n


def remove_partitions(root: str | os.PathLike, nside: int, tag: str) -> int:
    """Delete every file written for ``tag`` (to redo one input). Returns the count."""
    stale = list(Path(root).glob(f"hpx{nside}=*/part-{tag}-*.parquet"))
    for s in stale:
        s.unlink()
    return len(stale)


def _partition_dirs(root: Path) -> list[tuple[int, int, Path]]:
    out = []
    if not root.is_dir():
        return out
    for d in root.iterdir():
        m = _PART_RE.match(d.name)
        if m and d.is_dir():
            out.append((int(m.group(1)), int(m.group(2)), d))
    return out


def partition_nside(root: str | os.PathLike) -> int:
    """The ``nside`` of a hive dataset, read from its directory names."""
    sides = {s for s, _, _ in _partition_dirs(Path(root))}
    if len(sides) != 1:
        raise ValueError(f"expected one hpx<nside>= partition level in {root}, found {sides}")
    return sides.pop()


def partition_pixels(root: str | os.PathLike) -> np.ndarray:
    """Sorted pixel indices of the partitions present in ``root``."""
    return np.sort(np.array([p for _, p, _ in _partition_dirs(Path(root))], dtype=np.int64))


def scan_hive(
    root: str | os.PathLike,
    *,
    cone: tuple[float, float, float] | None = None,
    pixels: Iterable[int] | None = None,
    **scan_kwargs,
) -> pl.LazyFrame:
    """Lazily read a HEALPix hive dataset, optionally only some partitions.

    Parameters
    ----------
    root
        Dataset directory.
    cone
        ``(lon, lat, radius)`` in degrees: read only partitions that overlap
        it. Rows are not filtered; add ``.sky.cone(...)`` for an exact cut.
    pixels
        Explicit partition pixels to read.
    **scan_kwargs
        Passed to ``pl.scan_parquet``.

    Returns
    -------
    polars.LazyFrame
        With the partition key ``hpx{nside}`` as an Int64 column.
    """
    root = Path(root)
    parts = _partition_dirs(root)
    if not parts:
        raise FileNotFoundError(f"no hpx<nside>=<pixel> partitions under {root}")
    nside = partition_nside(root)
    key = f"hpx{nside}"

    wanted = None
    if cone is not None:
        wanted = set(cone_pixels(*cone, nside=nside).tolist())
    if pixels is not None:
        p = set(int(x) for x in pixels)
        wanted = p if wanted is None else wanted & p

    dirs = sorted(d for _, pix, d in parts if wanted is None or pix in wanted)
    files = [str(f) for d in dirs for f in sorted(d.glob("*.parquet"))]
    if not files:
        # keep the schema: scan everything and select no rows
        files = [str(f) for _, _, d in parts[:1] for f in sorted(d.glob("*.parquet"))]
        return pl.scan_parquet(
            files, hive_partitioning=True, hive_schema={key: pl.Int64}, **scan_kwargs
        ).head(0)
    return pl.scan_parquet(
        files, hive_partitioning=True, hive_schema={key: pl.Int64}, **scan_kwargs
    )
