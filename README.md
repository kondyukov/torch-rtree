# torchrtree

A static, packed R-tree for axis-aligned boxes in **2D, 3D and 3D + time**, written in pure PyTorch.
The same code runs on CPU and on GPU (CUDA or ROCm); no custom kernels, no compilation step.

- **Bulk load** 1M boxes in ~10 ms on a GPU, ~150 ms on a CPU core.
- **Batched box queries** (intersects / contains / within), **point queries** and **k-nearest**.
- Every query path is verified against brute force and against libspatialindex in the test suite.

## Install

```
pip install -e .            # library only (needs torch)
pip install -e ".[dev]"     # + pytest, ruff, numpy, rtree (libspatialindex) for tests and benchmarks
```

## Quick start

```python
import torch
from torchrtree import build_rtree

mins = torch.rand(100_000, 3)          # (N, ndim) lower corners
maxs = mins + 0.01 * torch.rand_like(mins)
tree = build_rtree(mins, maxs)         # works on CPU or GPU tensors alike

# Boxes intersecting each query box: ragged (query_idx, box_idx) pairs, grouped by query.
q_idx, box_idx = tree.query_pairs(query_mins, query_maxs)

# Same, padded to (Q, max_results) with -1, plus per-query hit counts.
results, counts = tree.query(query_mins, query_maxs)

# Boxes containing points; k nearest boxes to points (Euclidean point-to-box distance).
q_idx, box_idx = tree.query_points(points)
idx, dist = tree.nearest(points, k=5)

tree.to("cuda")                        # move; tree.save(path) / RTree.load(path) to persist
```

## API

| Function / method | Returns |
|---|---|
| `build_rtree(mins, maxs, m=8, curve="hilbert")` | `RTree` |
| `tree.query_pairs(qmins, qmaxs, mode="intersects", chunk_size=None, max_pairs=None)` | `(query_idx, box_idx)` ragged, grouped by query |
| `tree.query(qmins, qmaxs, max_results=-1, mode=..., ...)` | `(results (Q, R) padded -1, counts (Q,))` |
| `tree.query_points(points, ...)` | pairs of boxes containing each point |
| `tree.nearest(points, k=1, ...)` | `(idx (Q, k), dist (Q, k))`, ascending distance, padded `-1` / `inf` |
| `tree.to(device)`, `tree.save(path)`, `RTree.load(path)` | device move and persistence |

Module-level equivalents exist for every method (`query_rtree_pairs`, `query_rtree`, `query_rtree_points`,
`query_rtree_nearest`).

**Semantics worth knowing**

- Intervals are closed: touching edges count as intersecting, matching libspatialindex.
- `mode="contains"` returns boxes lying entirely inside the query; `mode="within"` returns boxes that entirely contain it.
- The tree is static. Rebuilding is cheap, so the update path is "rebuild".
- `float32` and `float64` are supported. Use `float64` for raw epoch-second timestamps: `float32` has ~128 s resolution at 1.7e9.
- Space-filling-curve ordering normalises each axis independently, so a time axis with a far wider range than the spatial
  axes is handled without scaling. `nearest` uses Euclidean distance in raw units, so scale axes yourself for mixed units.
- `chunk_size` bounds memory by processing queries in batches; `max_pairs` raises instead of allocating an oversized frontier.

## Performance

1M boxes, 10k queries, fan-out 8, Hilbert ordering, uniform data with a 1000x wider last axis. AMD Radeon RX 7900 XT (ROCm),
torch 2.9. The libspatialindex column is the `rtree` package driven one query at a time from Python. All torch rows return
exactly libspatialindex's hit sets.

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

```
python -m benchmarks.bench_rtree --n 1000000 --q 10000 --ndim 2 3 4 --knn 5
python -m benchmarks.bench_rtree --m 8 16 32 --curve morton hilbert --dist clustered
```

## How it works

Leaves are sorted along a Hilbert (or Morton) curve through their centres and packed `m` per node bottom-up, giving a flat,
uniform-depth tree stored as a few tensors. Queries run level-synchronously: a flat frontier of (query, node) pairs is tested
against its queries in one tensor op and expanded to the children of survivors, once per level. The number of kernel batches is
the tree depth (7 for 1M boxes), independent of how many nodes any single query visits. `nearest` is an expanding cube search
on top of the box query, finishing a point once its k-th candidate lies inside the searched ball.

## Development

```
pytest            # accuracy against brute force + libspatialindex, all ndims, CPU and GPU if present
ruff check .
```
