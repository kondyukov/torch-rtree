# Changelog

## 0.2.0 (unreleased)

API reshaped around an `RTree` class and a `QueryResult` object.

- `RTree(mins, maxs)`, `RTree(points)` and `RTree.from_bounds(...)` replace the NamedTuple and `build_rtree`
  (kept as an alias). Inputs may be tensors, numpy arrays or lists; integers are cast to float64.
- `tree.search(...)` returns a `QueryResult` with `query_idx`, `box_idx`, `counts`, CSR `offsets`,
  per-query indexing, `to_padded()` and `to_list()`. It unpacks as `(query_idx, box_idx)`.
- `tree.nearest(...)` now returns `(dist, idx)` in scipy order and supports box queries, `max_distance`
  and `axis_scale`.
- New: `count`, `within_distance`, `self_join`, `boxes()`, `bounds`, `cpu()`, `cuda()`, a short `repr`,
  `sort=` for ordered results, `validate=False` to skip per-call checks, and ndim 1 to 8.
- Query bounds may be infinite (half-open ranges). Builds run under `no_grad` and store detached tensors.
- Leaf order is now stable, so results are identical across CPU and GPU.
- Child index tensors are no longer stored; the tree is about a third smaller. Saved-file format is 2.
- Nearest-neighbour search rounds its search cube outward by one ulp so float rounding cannot miss a box.

## 0.1.0

Initial pure-PyTorch implementation: Hilbert / Morton bulk load, level-synchronous frontier queries
(intersects, contains, within), point and k-nearest queries, ndim 2 to 4, float32 and float64.
