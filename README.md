# torchrtree

A static, packed R-tree for axis-aligned boxes in 1 to 8 dimensions, built for **2D, 3D and 3D + time** indexing,
written in pure PyTorch. The same code runs on CPU and on GPU (CUDA or ROCm); no custom kernels, no compilation step.

- **Bulk load** 1M boxes in ~15 ms on a GPU, ~150 ms on a CPU core.
- **Batched queries**: box intersection / containment, point lookup, distance search, k-nearest, self-join.
- Every query path is verified against brute force and against libspatialindex in the test suite.

## Install

```bash
pip install -e .            # library only (needs torch)
pip install -e ".[dev]"     # + pytest, ruff, numpy, rtree (libspatialindex) for tests and benchmarks
```

## Quick start

```python
import torch
from torchrtree import RTree

mins = torch.rand(100_000, 3)                 # (N, ndim) lower corners
maxs = mins + 0.01 * torch.rand_like(mins)    # (N, ndim) upper corners
tree = RTree(mins, maxs)                      # CPU or GPU tensors, numpy arrays or lists

res = tree.search(qmins, qmaxs)               # boxes intersecting each query box
res.query_idx, res.box_idx                    # (P,) pairs grouped by query
res.counts, res.offsets                       # (Q,) hits per query, (Q+1,) CSR offsets
res[3]                                        # box ids for query 3
res.to_padded()                               # (Q, max_hits) padded with -1

tree.search(points)                           # boxes containing each point
tree.count(qmins, qmaxs)                      # hit counts only
tree.within_distance(points, distance=0.05)   # boxes within a distance
dist, idx = tree.nearest(points, k=5)         # k nearest boxes, scipy-style (dist, idx)
tree.self_join()                              # overlapping (i, j) pairs, i < j

tree.cuda(); tree.save("tree.pt"); RTree.load("tree.pt")
```

## API

| Call | Returns |
|---|---|
| `RTree(mins, maxs, fanout=8, curve="hilbert")` | tree over N boxes |
| `RTree(points)` / `RTree.from_bounds(b)` | tree over points / over an (N, 2·ndim) `[mins…, maxs…]` tensor |
| `tree.search(qmins, qmaxs=None, mode="intersects", sort=False)` | `QueryResult` |
| `tree.count(qmins, qmaxs=None, mode=...)` | `(Q,)` int64 |
| `tree.within_distance(qmins, qmaxs=None, distance=d, axis_scale=None)` | `QueryResult` |
| `tree.nearest(qmins, qmaxs=None, k=1, max_distance=None, axis_scale=None)` | `(dist (Q, k), idx (Q, k))`, padded `inf` / `-1` |
| `tree.self_join(mode=...)` | `QueryResult` of pairs with `query_idx < box_idx` |
| `tree.boxes()`, `tree.bounds`, `len(tree)` | indexed boxes in input order, root box, box count |
| `tree.to(dev)`, `.cpu()`, `.cuda()`, `.save(p)`, `RTree.load(p)` | device moves and persistence |

Every query method also takes `chunk_size` (bound memory by batching queries), `max_pairs` (raise instead of
allocating an oversized frontier) and `validate=False` (skip the NaN and min ≤ max checks, which each cost a
device sync). `build_rtree(mins, maxs, ...)` is a functional alias for the constructor.

**Semantics worth knowing**

- Intervals are closed: touching edges count as intersecting, matching libspatialindex.
- `mode="contains"` returns boxes lying entirely inside the query; `mode="within"` returns boxes that entirely contain it.
- Query bounds may be infinite, so `[t0, +inf)` expresses "everything after t0". Indexed boxes must be finite.
- The tree is static. Rebuilding is cheap, so the update path is "rebuild".
- Inputs may be `float32` or `float64`; integers are cast to `float64`. Use `float64` for raw epoch-second timestamps,
  since `float32` has ~128 s resolution at 1.7e9.
- Curve ordering normalises each axis independently, so a wide time axis needs no scaling. Distances in `nearest` and
  `within_distance` are Euclidean in raw units; pass `axis_scale` to weight axes, e.g. metres per second.
- Results are grouped by query in tree order; `sort=True` orders box ids within each query. Leaf order is stable, so
  results are identical on CPU and GPU.

## Performance

1M boxes, 10k queries, fan-out 8, Hilbert ordering, uniform data with a 1000x wider last axis. AMD Radeon RX 7900 XT
(ROCm), torch 2.9. The libspatialindex column is the `rtree` package driven one query at a time from Python. All torch
rows return exactly libspatialindex's hit sets.

Box intersection queries per second:

| ndim | libspatialindex | torch CPU | torch GPU |
|---|---|---|---|
| 2 | 7,100 | 62,600 | 719,000 |
| 3 | 25,300 | 409,000 | 1,970,000 |
| 4 | 25,100 | 628,000 | 1,710,000 |

k-nearest (k = 5) queries per second:

| ndim | libspatialindex | torch CPU | torch GPU |
|---|---|---|---|
| 2 | 25,100 | 15,900 | 566,000 |
| 3 | 16,100 | 28,000 | 472,000 |
| 4 | 3,200 | 26,300 | 208,000 |

Build time for 1M boxes: libspatialindex 2.6 to 3.3 s; torch CPU 0.14 s; torch GPU 0.015 s.

The 2D rows are output-bound: each 2D query returns thousands of hits at this box density.

**Tuning.** Fan-out 8 was fastest on both devices; 16 and 32 were slower to query in every configuration.
Hilbert ordering queries 20 to 50 percent faster than Morton and builds about 1.6x slower; pass `curve="morton"` when
build time dominates. Small query batches favour the CPU; the GPU pulls ahead from a few thousand queries per call.

Reproduce with:

```bash
python -m benchmarks.bench_rtree --n 1000000 --q 10000 --ndim 2 3 4 --knn 5
python -m benchmarks.bench_rtree --m 8 16 32 --curve morton hilbert --dist clustered
```

## How it works

Leaves are sorted along a Hilbert (or Morton) curve through their centres and packed `fanout` per node bottom-up,
giving a flat, uniform-depth tree stored as two tensors of node bounds plus a leaf permutation; child ranges are
arithmetic. Queries run level-synchronously: a flat frontier of (query, node) pairs is tested against its queries in one
tensor op and expanded to the children of survivors, once per level. The number of kernel batches is the tree depth
(7 for 1M boxes), independent of how many nodes any single query visits. `nearest` is an expanding cube search on top
of the box query, finishing a query once its k-th candidate lies inside the searched ball.

## Development

```bash
pytest            # accuracy against brute force + libspatialindex, all ndims, CPU and GPU if present
ruff check .
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.
