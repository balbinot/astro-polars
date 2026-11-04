import numpy as np
import polars as pl
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sky(rng):
    """10k random points uniform on the sphere."""
    n = 10_000
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return pl.DataFrame({"ra": ra, "dec": dec, "mag": rng.normal(20, 1, n)})
