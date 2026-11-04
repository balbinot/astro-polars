import subprocess
import sys

import polars as pl

import astro_polars as ap


def test_version():
    assert isinstance(ap.__version__, str)


def test_namespaces_registered():
    df = pl.DataFrame({"x": [1.0]})
    for ns in ("sky", "viz"):
        assert hasattr(df, ns), ns
        assert hasattr(df.lazy(), ns), ns
    assert hasattr(pl.col("x"), "robust")


def test_import_is_light():
    """Optional and heavy dependencies are only imported when used."""
    code = (
        "import sys, astro_polars; "
        "heavy = {'matplotlib', 'skyproj', 'healpy', 'astropy', 'sklearn', 'hpgeom', 'pandas'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
