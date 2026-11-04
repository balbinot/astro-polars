"""Command-line entry point: ``astro-polars <command> ...``."""

from __future__ import annotations

import argparse
import sys


def _euclid_hive(args: argparse.Namespace) -> int:
    from .surveys import euclid

    try:
        summary = euclid.convert_tiles(args.indir, args.outdir, nside=args.nside, force=args.force)
    except (ValueError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(
        f"Read with: pl.scan_parquet('{args.outdir}/**/*.parquet', hive_partitioning=True)"
        " or astro_polars.io.scan_hive"
    )
    return 1 if summary["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astro-polars", description="astro-polars tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "euclid-hive",
        help="convert Euclid MER final-catalogue tiles to a HEALPix hive Parquet dataset",
    )
    p.add_argument("--indir", required=True, help="root of downloaded tiles (INDIR/<tile>/*.fits)")
    p.add_argument("--outdir", required=True, help="root of the hive-partitioned dataset")
    p.add_argument(
        "--nside", type=int, default=32, help="HEALPix nside of partitions (nested), default 32"
    )
    p.add_argument(
        "--force", action="store_true", help="reprocess all files, ignoring the manifest"
    )
    p.set_defaults(func=_euclid_hive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
