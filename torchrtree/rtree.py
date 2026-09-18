"""
R-Tree in pure PyTorch.

Construction packs leaves sorted along a space-filling curve (Morton or
Hilbert) into a flat, uniform-depth tree. Queries run level-synchronously: a
flat frontier of (query, node) pairs is tested and expanded once per level, so
the number of tensor-op batches is the tree depth, not the traversal length.

The tree is static. Rebuilding is cheap (about 10 ms per million boxes on a
GPU), so the update path is "rebuild".

Supported: ndim in {2, 3, 4} (2D, 3D, 3D+time); float32 and float64.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch import Tensor

SUPPORTED_NDIMS = (2, 3, 4)
SUPPORTED_DTYPES = (torch.float32, torch.float64)

Curve = Literal["morton", "hilbert"]
QueryMode = Literal["intersects", "contains", "within"]
_MODES = ("intersects", "contains", "within")


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _validate_boxes(
    mins: Tensor,
    maxs: Tensor,
    *,
    name: str,
    ndim: int | None = None,
    device: torch.device | None = None,
) -> None:
    if not (isinstance(mins, Tensor) and isinstance(maxs, Tensor)):
        raise TypeError(f"{name}: expected tensors, got {type(mins)} / {type(maxs)}")
    if mins.dim() != 2:
        raise ValueError(f"{name}: expected shape (N, ndim), got {tuple(mins.shape)}")
    if mins.shape != maxs.shape:
        raise ValueError(f"{name}: mins {tuple(mins.shape)} and maxs {tuple(maxs.shape)} differ")
    if mins.dtype != maxs.dtype:
        raise TypeError(f"{name}: mins dtype {mins.dtype} != maxs dtype {maxs.dtype}")
    if mins.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"{name}: dtype must be one of {SUPPORTED_DTYPES}, got {mins.dtype}")
    if mins.device != maxs.device:
        raise ValueError(f"{name}: mins on {mins.device} but maxs on {maxs.device}")
    if device is not None and mins.device != device:
        raise ValueError(f"{name}: expected tensors on {device}, got {mins.device}")
    d = mins.shape[1]
    if ndim is not None and d != ndim:
        raise ValueError(f"{name}: expected ndim={ndim}, got {d}")
    if d not in SUPPORTED_NDIMS:
        raise NotImplementedError(f"{name}: ndim must be one of {SUPPORTED_NDIMS}, got {d}")
    if mins.numel():
        if not (torch.isfinite(mins).all() and torch.isfinite(maxs).all()):
            raise ValueError(f"{name}: coordinates must be finite (no NaN / inf)")
        if not (mins <= maxs).all():
            raise ValueError(f"{name}: every min must be <= the corresponding max")


# -----------------------------------------------------------------------------
# Space-filling curves
# -----------------------------------------------------------------------------

def _sfc_bits(ndim: int) -> int:
    """Bits per axis so that ndim * bits <= 63 (key fits a signed int64)."""
    return 63 // ndim


def _quantize(centers: Tensor, g_min: Tensor, g_range: Tensor, bits: int) -> Tensor:
    """Per-axis normalise to [0, 2^bits - 1] and truncate to int64."""
    max_q = (1 << bits) - 1
    norm = ((centers - g_min) / g_range).to(torch.float64) * max_q
    return norm.to(torch.int64).clamp_(0, max_q)


def _interleave_bits(q: Tensor, bits: int, axis0_msb: bool) -> Tensor:
    """
    Interleave the low `bits` bits of each column of an (N, ndim) int64 tensor.
    Bit b of axis d lands at b * ndim + slot(d), where slot(d) is d (axis 0
    least significant, the Morton convention matching the CUDA reference) or
    ndim - 1 - d (axis 0 most significant, the Hilbert transpose convention).
    """
    ndim = q.shape[-1]
    code = torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
    for b in range(bits):
        for d in range(ndim):
            slot = (ndim - 1 - d) if axis0_msb else d
            code |= ((q[:, d] >> b) & 1) << (b * ndim + slot)
    return code


def _morton_codes(q: Tensor, bits: int) -> Tensor:
    return _interleave_bits(q, bits, axis0_msb=False)


def _hilbert_transpose(q: Tensor, bits: int) -> Tensor:
    """
    Axes -> transposed Hilbert coordinates (Skilling, "Programming the Hilbert
    curve", 2004), vectorised over rows. Input/output are (N, ndim) int64.
    """
    n = q.shape[1]
    X = [q[:, i].clone() for i in range(n)]
    M = 1 << (bits - 1)
    Q = M
    while Q > 1:  # inverse undo
        P = Q - 1
        for i in range(n):
            flag = (X[i] & Q) != 0
            # flag: invert X[0] by P. else: exchange the P bits of X[0] and X[i].
            t = torch.where(flag, torch.full_like(X[0], P), (X[0] ^ X[i]) & P)
            X[0] = X[0] ^ t
            if i > 0:
                X[i] = torch.where(flag, X[i], X[i] ^ t)
        Q >>= 1
    for i in range(1, n):  # gray encode
        X[i] = X[i] ^ X[i - 1]
    t = torch.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = torch.where((X[n - 1] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    return torch.stack([x ^ t for x in X], dim=1)


def _hilbert_codes(q: Tensor, bits: int) -> Tensor:
    return _interleave_bits(_hilbert_transpose(q, bits), bits, axis0_msb=True)


def _sfc_codes(centers: Tensor, g_min: Tensor, g_range: Tensor, curve: Curve = "hilbert") -> Tensor:
    """Map (N, ndim) centers to a 1-D sortable int64 key along the chosen curve."""
    ndim = centers.shape[-1]
    bits = _sfc_bits(ndim)
    q = _quantize(centers, g_min, g_range, bits)
    if curve == "morton":
        return _morton_codes(q, bits)
    if curve == "hilbert":
        return _hilbert_codes(q, bits)
    raise ValueError(f"curve must be 'morton' or 'hilbert', got {curve!r}")


# -----------------------------------------------------------------------------
# Tree container
# -----------------------------------------------------------------------------

class RTree(NamedTuple):
    mins: Tensor              # (total_nodes, ndim)
    maxs: Tensor              # (total_nodes, ndim)
    start_indices: Tensor     # (total_nodes,) child range start (internal nodes only)
    end_indices: Tensor       # (total_nodes,) child range end (inclusive)
    level_starts: Tensor      # (num_levels + 1,) cumulative level offsets
    leaf_order: Tensor        # (num_leaves,) original index of each sorted leaf
    leaf_start: int           # index of first leaf in flat layout
    num_leaves: int
    m: int                    # max children per internal node
    ndim: int

    # -- introspection -------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.mins.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mins.dtype

    @property
    def num_levels(self) -> int:
        return self.level_starts.numel() - 1

    @property
    def num_nodes(self) -> int:
        return self.mins.shape[0]

    # -- device / persistence ---------------------------------------------------
    def to(self, device) -> RTree:
        device = torch.device(device)
        if device == self.device:
            return self
        return self._replace(
            mins=self.mins.to(device), maxs=self.maxs.to(device),
            start_indices=self.start_indices.to(device), end_indices=self.end_indices.to(device),
            level_starts=self.level_starts.to(device), leaf_order=self.leaf_order.to(device),
        )

    def state_dict(self) -> dict:
        return {"format": 1, **self._asdict()}

    @classmethod
    def from_state_dict(cls, state: dict) -> RTree:
        if state.get("format") != 1:
            raise ValueError(f"unknown RTree state format {state.get('format')!r}")
        return cls(**{k: state[k] for k in cls._fields})

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, map_location=None) -> RTree:
        return cls.from_state_dict(torch.load(path, map_location=map_location))

    # -- queries (thin wrappers over the module functions) ------------------------
    def query(self, query_mins, query_maxs, max_results=-1, mode: QueryMode = "intersects", **kw):
        return query_rtree(self, query_mins, query_maxs, max_results, mode, **kw)

    def query_pairs(self, query_mins, query_maxs, mode: QueryMode = "intersects", **kw):
        return query_rtree_pairs(self, query_mins, query_maxs, mode, **kw)

    def query_points(self, points, **kw):
        return query_rtree_points(self, points, **kw)

    def nearest(self, points, k: int = 1, **kw):
        return query_rtree_nearest(self, points, k, **kw)


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

def build_rtree(mins: Tensor, maxs: Tensor, m: int = 8, curve: Curve = "hilbert") -> RTree:
    """
    Build a packed R-tree bottom-up by sorting leaves along a space-filling
    curve. `m` is the fan-out (children per internal node); `curve` selects
    the leaf ordering. "hilbert" (default) gives tighter nodes and 20-50%
    faster queries; "morton" builds about 1.6x faster.
    """
    _validate_boxes(mins, maxs, name="build_rtree")
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")
    n, ndim = mins.shape
    device, dtype = mins.device, mins.dtype

    # 1. Sort leaves by SFC key of their centers.
    if n > 0:
        centers = (mins + maxs) * 0.5
        g_min = mins.amin(dim=0)
        g_range = (maxs.amax(dim=0) - g_min).clamp_min(torch.finfo(dtype).tiny)
        order = torch.argsort(_sfc_codes(centers, g_min, g_range, curve))
    else:
        order = torch.empty(0, dtype=torch.int64, device=device)
    leaf_mins, leaf_maxs = mins[order], maxs[order]

    # 2. Compute level sizes (top-down: level 0 = root, last = leaves).
    sizes_bu = [n]
    while sizes_bu[-1] > 1:
        sizes_bu.append((sizes_bu[-1] + m - 1) // m)
    level_sizes = list(reversed(sizes_bu))

    level_starts = [0]
    for s in level_sizes:
        level_starts.append(level_starts[-1] + s)
    total = level_starts[-1]
    leaf_start = level_starts[-2]

    # 3. Allocate flat tree tensors.
    tree_mins = torch.empty((total, ndim), dtype=dtype, device=device)
    tree_maxs = torch.empty((total, ndim), dtype=dtype, device=device)
    start_idx = torch.full((total,), -1, dtype=torch.int64, device=device)
    end_idx = torch.full((total,), -1, dtype=torch.int64, device=device)

    # 4. Place sorted leaves at the bottom.
    tree_mins[leaf_start:leaf_start + n] = leaf_mins
    tree_maxs[leaf_start:leaf_start + n] = leaf_maxs

    # 5. Build internal levels bottom-up via reshape + amin/amax over groups.
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)
    for level in range(len(level_sizes) - 2, -1, -1):
        ls, sz = level_starts[level], level_sizes[level]
        cls_, csz = level_starts[level + 1], level_sizes[level + 1]

        cpn = min(-(-csz // sz), m)  # ceil(csz / sz), capped at m
        idx = torch.arange(sz, device=device)
        sc = idx * cpn
        ec = (sc + cpn - 1).clamp_max(csz - 1)
        start_idx[ls:ls + sz] = cls_ + sc
        end_idx[ls:ls + sz] = cls_ + ec

        c_mins = tree_mins[cls_:cls_ + csz]
        c_maxs = tree_maxs[cls_:cls_ + csz]
        pad = sz * cpn - csz
        if pad > 0:
            # Pad with +inf / -inf so they're identities under amin / amax.
            c_mins = torch.cat([c_mins, inf.expand(pad, ndim)], dim=0)
            c_maxs = torch.cat([c_maxs, (-inf).expand(pad, ndim)], dim=0)
        tree_mins[ls:ls + sz] = c_mins.view(sz, cpn, ndim).amin(dim=1)
        tree_maxs[ls:ls + sz] = c_maxs.view(sz, cpn, ndim).amax(dim=1)

    return RTree(
        mins=tree_mins,
        maxs=tree_maxs,
        start_indices=start_idx,
        end_indices=end_idx,
        level_starts=torch.tensor(level_starts, dtype=torch.int64, device=device),
        leaf_order=order,
        leaf_start=leaf_start,
        num_leaves=n,
        m=m,
        ndim=ndim,
    )


# -----------------------------------------------------------------------------
# Query — level-synchronous frontier expansion
# -----------------------------------------------------------------------------

def _node_test(tree: RTree, node: Tensor, qmins: Tensor, qmaxs: Tensor, q: Tensor,
               mode: str, leaf: bool) -> Tensor:
    """Predicate for (query q[i], node node[i]) pairs under the given mode."""
    nmin, nmax = tree.mins[node], tree.maxs[node]
    qmin, qmax = qmins[q], qmaxs[q]
    if mode == "within" or (mode == "contains" and leaf):
        # 'within': the query box lies inside the node/leaf box. This is a
        # valid pruning test at internal levels too, since ancestors contain
        # their leaves. 'contains' at the leaf: the leaf box lies inside the
        # query box.
        if mode == "contains":
            return (qmin <= nmin).all(dim=-1) & (nmax <= qmax).all(dim=-1)
        return (nmin <= qmin).all(dim=-1) & (qmax <= nmax).all(dim=-1)
    # closed-interval intersection (also the internal-node test for 'contains')
    return (nmin <= qmax).all(dim=-1) & (nmax >= qmin).all(dim=-1)


def _frontier_query(tree: RTree, qmins: Tensor, qmaxs: Tensor, mode: str,
                    max_pairs: int | None) -> tuple[Tensor, Tensor]:
    """Core traversal. Returns (query_idx, node_idx) with node_idx in tree space."""
    Q = qmins.shape[0]
    device = tree.device
    if Q == 0 or tree.num_leaves == 0:
        e = torch.empty(0, dtype=torch.int64, device=device)
        return e, e.clone()

    level_starts = tree.level_starts.tolist()
    num_levels = len(level_starts) - 1
    level_sizes = [level_starts[i + 1] - level_starts[i] for i in range(num_levels)]

    fq = torch.arange(Q, device=device)
    fn = torch.zeros(Q, dtype=torch.int64, device=device)  # root

    def guard(n_pairs: int):
        if max_pairs is not None and n_pairs > max_pairs:
            raise RuntimeError(
                f"query frontier reached {n_pairs} (query, node) pairs, above max_pairs={max_pairs}; "
                f"pass a larger max_pairs, a chunk_size, or narrow the queries"
            )

    for level in range(num_levels - 1):  # internal levels
        hit = _node_test(tree, fn, qmins, qmaxs, fq, mode, leaf=False)
        fq, fn = fq[hit], fn[hit]
        cpn = min(-(-level_sizes[level + 1] // level_sizes[level]), tree.m)
        start, end = tree.start_indices[fn], tree.end_indices[fn]
        child = start.unsqueeze(1) + torch.arange(cpn, device=device)   # (F, cpn)
        valid = child <= end.unsqueeze(1)
        fq = fq.unsqueeze(1).expand_as(child)[valid]
        fn = child[valid]
        guard(fn.numel())

    hit = _node_test(tree, fn, qmins, qmaxs, fq, mode, leaf=True)
    return fq[hit], fn[hit]


def _prepare_queries(tree: RTree, qmins: Tensor, qmaxs: Tensor, mode: str) -> tuple[Tensor, Tensor]:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    _validate_boxes(qmins, qmaxs, name="query", ndim=tree.ndim)
    return (qmins.to(device=tree.device, dtype=tree.dtype),
            qmaxs.to(device=tree.device, dtype=tree.dtype))


def query_rtree_pairs(
    tree: RTree,
    query_mins: Tensor,   # (Q, ndim)
    query_maxs: Tensor,   # (Q, ndim)
    mode: QueryMode = "intersects",
    chunk_size: int | None = None,
    max_pairs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Ragged query. Returns (query_idx, leaf_idx): int64 tensors of equal length
    P (total hits), grouped by ascending query index; leaf_idx are original
    input indices.

    mode:        "intersects" (boxes overlapping the query, closed intervals),
                 "contains"   (boxes lying entirely inside the query),
                 "within"     (boxes that entirely contain the query).
    chunk_size:  process at most this many queries at once to bound memory.
    max_pairs:   raise if any level's frontier exceeds this many pairs.
    """
    qmins, qmaxs = _prepare_queries(tree, query_mins, query_maxs, mode)
    Q = qmins.shape[0]
    if chunk_size is None or chunk_size >= Q:
        fq, fn = _frontier_query(tree, qmins, qmaxs, mode, max_pairs)
        return fq, tree.leaf_order[fn - tree.leaf_start]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qs, ls = [], []
    for s in range(0, Q, chunk_size):
        fq, fn = _frontier_query(tree, qmins[s:s + chunk_size], qmaxs[s:s + chunk_size], mode, max_pairs)
        qs.append(fq + s)
        ls.append(tree.leaf_order[fn - tree.leaf_start])
    return torch.cat(qs), torch.cat(ls)


def _group_positions(q_idx: Tensor, Q: int) -> tuple[Tensor, Tensor]:
    """For pairs grouped by query, return (counts (Q,), position within group (P,))."""
    counts = torch.bincount(q_idx, minlength=Q)
    group_start = counts.cumsum(0) - counts
    pos = torch.arange(q_idx.numel(), device=q_idx.device) - group_start[q_idx]
    return counts, pos


def query_rtree(
    tree: RTree,
    query_mins: Tensor,      # (Q, ndim)
    query_maxs: Tensor,      # (Q, ndim)
    max_results: int = -1,   # -1 → widest query's hit count (no truncation)
    mode: QueryMode = "intersects",
    chunk_size: int | None = None,
    max_pairs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Padded query, built on query_rtree_pairs.

    Returns:
        results       (Q, max_results) int64 — original leaf indices, padded -1
        result_counts (Q,) int64 — total hits per query (may exceed max_results)
    """
    Q = query_mins.shape[0]
    q_idx, leaf_idx = query_rtree_pairs(tree, query_mins, query_maxs, mode, chunk_size, max_pairs)
    counts, pos = _group_positions(q_idx, Q)
    if max_results <= 0:
        max_results = int(counts.max().item()) if Q > 0 else 0
    max_results = max(max_results, 1)
    keep = pos < max_results
    results = torch.full((Q, max_results), -1, dtype=torch.int64, device=tree.device)
    results[q_idx[keep], pos[keep]] = leaf_idx[keep]
    return results, counts


def query_rtree_points(tree: RTree, points: Tensor, **kw) -> tuple[Tensor, Tensor]:
    """Boxes containing each point. Same return convention as query_rtree_pairs."""
    return query_rtree_pairs(tree, points, points, "intersects", **kw)


# -----------------------------------------------------------------------------
# k-nearest boxes to points (Euclidean point-to-box distance, raw units)
# -----------------------------------------------------------------------------

def _point_box_distance(tree: RTree, node: Tensor, pts: Tensor) -> Tensor:
    below = (tree.mins[node] - pts).clamp_min(0)
    above = (pts - tree.maxs[node]).clamp_min(0)
    return (below * below + above * above).sum(dim=-1).sqrt()


def query_rtree_nearest(
    tree: RTree,
    points: Tensor,   # (Q, ndim)
    k: int = 1,
    chunk_size: int | None = None,
    max_pairs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    k nearest boxes to each point by Euclidean point-to-box distance (zero when
    the point lies inside the box). Distances mix axes in raw units, so scale
    axes yourself for mixed-unit (e.g. spatio-temporal) data.

    Implemented as an expanding box search: each point queries a cube of
    radius r, and is finished once it has >= k candidates whose k-th distance
    is <= r (the cube contains the ball of radius r, so nothing closer was
    missed). A point with >= k candidates but a k-th distance beyond r retries
    once with r = that distance; a point with fewer doubles r.

    Returns:
        idx  (Q, k) int64 — original box indices, ascending distance, -1 padded
        dist (Q, k)       — matching distances, +inf padded
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _validate_boxes(points, points, name="nearest", ndim=tree.ndim)
    pts_all = points.to(device=tree.device, dtype=tree.dtype)
    Q = pts_all.shape[0]
    device, dtype = tree.device, tree.dtype
    idx_out = torch.full((Q, k), -1, dtype=torch.int64, device=device)
    dist_out = torch.full((Q, k), float("inf"), dtype=dtype, device=device)
    if Q == 0 or tree.num_leaves == 0:
        return idx_out, dist_out
    k_eff = min(k, tree.num_leaves)

    root_lo, root_hi = tree.mins[0], tree.maxs[0]
    ranges = (root_hi - root_lo).double()
    extent = float(ranges.amax().item())
    # Initial radius: half the side of the cube that would hold ~k boxes if
    # they were spread uniformly over the root box. Using the root VOLUME
    # (not the widest axis) keeps this sane on anisotropic data such as a
    # time axis with a far wider range than the spatial axes.
    volume = float(ranges.clamp_min(extent * 1e-3).prod().item()) if extent > 0 else 0.0
    r0 = 0.5 * (k_eff / tree.num_leaves * volume) ** (1.0 / tree.ndim)
    r0 = max(r0, extent * 1e-6, 1e-30)
    # Beyond this radius every box is in range, so the search is exhaustive.
    r_max = float((torch.maximum((pts_all - root_lo).abs(), (pts_all - root_hi).abs()).amax()).item()) + 1.0

    pending = torch.arange(Q, device=device)
    # Points outside the root box start at their distance to it, so the first
    # cube already touches the tree instead of doubling up from nothing.
    root_node = torch.zeros(Q, dtype=torch.int64, device=device)
    d_root = _point_box_distance(tree, root_node, pts_all) * 1.001
    r = d_root.clamp_min(r0)

    while pending.numel() > 0:
        pts = pts_all[pending]
        rad = r[pending].unsqueeze(1)
        q, node = _frontier_query(tree, pts - rad, pts + rad, "intersects", max_pairs) \
            if chunk_size is None else _chunked_frontier(tree, pts - rad, pts + rad, chunk_size, max_pairs)
        d = _point_box_distance(tree, node, pts[q])

        # Sort pairs by (query, distance): sort by distance, then stable sort by query.
        d, perm = torch.sort(d)
        q, node = q[perm], node[perm]
        q, perm = torch.sort(q, stable=True)
        d, node = d[perm], node[perm]

        counts, pos = _group_positions(q, pts.shape[0])
        has_k = counts >= k_eff
        kth = torch.full((pts.shape[0],), float("inf"), dtype=dtype, device=device)
        sel = pos == (k_eff - 1)
        kth[q[sel]] = d[sel]
        done = (has_k & (kth <= r[pending])) | (r[pending] >= r_max)

        keep = done[q] & (pos < k_eff)
        idx_out[pending[q[keep]], pos[keep]] = tree.leaf_order[node[keep] - tree.leaf_start]
        dist_out[pending[q[keep]], pos[keep]] = d[keep]

        # Enough candidates but the k-th lies outside the ball: one more pass at
        # exactly that radius is guaranteed to finish. Otherwise double.
        r_cur = r[pending]
        r[pending] = torch.where(done, r_cur, torch.where(has_k, kth * (1 + 1e-6), r_cur * 2))
        pending = pending[~done]
    return idx_out, dist_out


def _chunked_frontier(tree, qmins, qmaxs, chunk_size, max_pairs):
    qs, ns = [], []
    for s in range(0, qmins.shape[0], chunk_size):
        lo, hi = qmins[s:s + chunk_size], qmaxs[s:s + chunk_size]
        fq, fn = _frontier_query(tree, lo, hi, "intersects", max_pairs)
        qs.append(fq + s)
        ns.append(fn)
    return torch.cat(qs), torch.cat(ns)
