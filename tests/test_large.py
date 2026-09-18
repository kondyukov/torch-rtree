"""
Large-scale exactness checks against libspatialindex. Slow (about a minute):
run with `pytest -m slow`.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("rtree", reason="libspatialindex baseline not installed")

from benchmarks.bench_rtree import _lsi_build, _point_box_dist, make_boxes  # noqa: E402
from torchrtree import RTree  # noqa: E402

pytestmark = pytest.mark.slow
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_million_points_exact_ids_and_knn_vs_libspatialindex(ndim):
    n, q, k = 1_000_000, 2_000, 5
    side = {2: 0.03, 3: 0.08, 4: 0.15}[ndim]  # ~500-900 expected hits per query at each ndim
    mins, maxs = make_boxes(n, ndim, 0.0, seed=0, time_extent=1000.0)          # points
    qmins, qmaxs = make_boxes(q, ndim, side, seed=1, time_extent=1000.0)       # box queries
    lsi, _ = _lsi_build(mins, maxs)
    q64_lo, q64_hi = qmins.astype(np.float64), qmaxs.astype(np.float64)
    expected = [sorted(lsi.intersection((*q64_lo[j], *q64_hi[j]))) for j in range(q)]
    assert sum(map(len, expected)) > q, "degenerate configuration: almost no hits"

    knn_ref = np.full((q, k), -1, dtype=np.int64)
    for j in range(q):
        ids = list(lsi.nearest((*q64_lo[j], *q64_lo[j]), k))[:k]
        knn_ref[j, :len(ids)] = ids
    exp_d = np.sort(_point_box_dist(mins, maxs, qmins, knn_ref), axis=1)

    for device in DEVICES:
        tree = RTree(torch.from_numpy(mins).to(device), torch.from_numpy(maxs).to(device))
        res = tree.search(torch.from_numpy(qmins), torch.from_numpy(qmaxs), sort=True)
        assert res.to_list() == expected, f"hit sets differ on {device}"
        dist, idx = tree.nearest(torch.from_numpy(qmins), k=k)
        assert (idx >= 0).all()
        got_d = dist.cpu().numpy()
        assert np.allclose(got_d, exp_d, rtol=1e-5, atol=1e-6), f"kNN distances differ on {device}"
