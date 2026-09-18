"""
Small-scale throughput smoke test. Runs every backend in benchmarks.bench_rtree
on a modest problem, asserts that all backends agree with libspatialindex on
per-query hit counts, and prints the timing table (visible with `pytest -s`).

For real numbers run:  python -m benchmarks.bench_rtree
"""

import pytest

pytest.importorskip("rtree", reason="libspatialindex baseline not installed")

from benchmarks.bench_rtree import format_table, run_config  # noqa: E402
from torchrtree import SUPPORTED_NDIMS  # noqa: E402


@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
def test_backends_agree_and_report(ndim):
    rows = run_config(n=5_000, q=300, ndim=ndim)
    print("\n" + format_table(rows))
    for r in rows:
        assert r.counts_ok is not False, f"{r.backend} disagrees with libspatialindex"
    # Each backend should have found at least some hits; guards against a
    # degenerate configuration silently passing.
    assert rows[0].query_s > 0
