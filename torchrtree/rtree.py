"""
R-Tree in pure PyTorch.

Construction packs leaves sorted along a space-filling curve (Hilbert or
Morton) into a flat, uniform-depth tree. Queries run level-synchronously: a
flat frontier of (query, node) pairs is tested and expanded once per level, so
the number of tensor-op batches is the tree depth, not the traversal length.

The tree is static. Rebuilding is cheap (about 10 ms per million boxes on a
GPU), so the update path is "rebuild".
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from ._curves import CURVES, Curve, sfc_codes

SUPPORTED_NDIMS: tuple[int, ...] = tuple(range(1, 9))
SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float64)

QueryMode = Literal["intersects", "contains", "within"]
_MODES: tuple[str, ...] = ("intersects", "contains", "within")
_STATE_FORMAT = 2


# -----------------------------------------------------------------------------
# Input conversion and validation
# -----------------------------------------------------------------------------

def _as_coords(x: Any, name: str) -> Tensor:
    """Accept tensors, numpy arrays and nested lists; map to a supported float dtype."""
    t = x if isinstance(x, Tensor) else torch.as_tensor(x)
    if t.dtype in SUPPORTED_DTYPES:
        return t
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.to(torch.float32)
    if not t.dtype.is_floating_point and not t.dtype.is_complex and t.dtype != torch.bool:
        return t.to(torch.float64)  # integers: exact up to 2^53
    raise TypeError(f"{name}: unsupported dtype {t.dtype}; use float32, float64 or an integer type")


def _as_boxes(mins: Any, maxs: Any | None, name: str, ndim: int | None = None) -> tuple[Tensor, Tensor]:
    """Normalise (mins, maxs) or points (maxs=None) to two same-shaped float tensors."""
    lo = _as_coords(mins, name)
    hi = lo if maxs is None else _as_coords(maxs, name)
    if lo.dim() == 1 and ndim is not None and lo.shape[0] == ndim:
        lo, hi = lo.unsqueeze(0), hi.unsqueeze(0)  # a single box / point
    if lo.dim() != 2:
        raise ValueError(f"{name}: expected shape (N, ndim), got {tuple(lo.shape)}")
    if lo.shape != hi.shape:
        raise ValueError(f"{name}: mins {tuple(lo.shape)} and maxs {tuple(hi.shape)} differ")
    if lo.device != hi.device:
        raise ValueError(f"{name}: mins on {lo.device} but maxs on {hi.device}")
    if lo.dtype != hi.dtype:
        dt = torch.promote_types(lo.dtype, hi.dtype)
        lo, hi = lo.to(dt), hi.to(dt)
    d = lo.shape[1]
    if ndim is not None and d != ndim:
        raise ValueError(f"{name}: expected ndim={ndim}, got {d}")
    if d not in SUPPORTED_NDIMS:
        raise ValueError(f"{name}: ndim must be in {SUPPORTED_NDIMS[0]}..{SUPPORTED_NDIMS[-1]}, got {d}")
    return lo, hi


def _check_values(lo: Tensor, hi: Tensor, name: str, allow_inf: bool) -> None:
    """Value checks (each is a host sync); skipped when a caller passes validate=False."""
    if lo.numel() == 0:
        return
    if allow_inf:
        if torch.isnan(lo).any() or torch.isnan(hi).any():
            raise ValueError(f"{name}: coordinates must not be NaN")
    elif not (torch.isfinite(lo).all() and torch.isfinite(hi).all()):
        raise ValueError(f"{name}: coordinates must be finite (no NaN / inf)")
    if not (lo <= hi).all():
        raise ValueError(f"{name}: every min must be <= the corresponding max")


# -----------------------------------------------------------------------------
# Query result
# -----------------------------------------------------------------------------

class QueryResult:
    """
    Ragged result of a box query: P (query, box) pairs grouped by ascending
    query index, plus per-query counts and CSR-style offsets.

    Unpacks as a pair: ``query_idx, box_idx = tree.search(...)``.
    ``result[q]`` gives the box indices for query q; ``to_padded()`` gives a
    dense (Q, max_results) tensor padded with -1.
    """

    __slots__ = ("query_idx", "box_idx", "counts", "offsets")

    def __init__(self, query_idx: Tensor, box_idx: Tensor, counts: Tensor):
        self.query_idx = query_idx
        self.box_idx = box_idx
        self.counts = counts
        self.offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    @property
    def num_queries(self) -> int:
        return self.counts.numel()

    @property
    def device(self) -> torch.device:
        return self.box_idx.device

    def __len__(self) -> int:
        return self.box_idx.numel()

    def __iter__(self):
        return iter((self.query_idx, self.box_idx))

    def __getitem__(self, q: int) -> Tensor:
        if not -self.num_queries <= q < self.num_queries:
            raise IndexError(f"query index {q} out of range for {self.num_queries} queries")
        q = q % self.num_queries
        s, e = self.offsets[q].item(), self.offsets[q + 1].item()
        return self.box_idx[s:e]

    def to_padded(self, max_results: int = -1, fill: int = -1) -> Tensor:
        """(Q, max_results) int64; -1 (default) sizes to the widest query. Extra hits are dropped."""
        Q = self.num_queries
        if max_results <= 0:
            max_results = int(self.counts.max().item()) if Q > 0 else 0
        max_results = max(max_results, 1)
        pos = torch.arange(len(self), device=self.device) - self.offsets[self.query_idx]
        keep = pos < max_results
        out = torch.full((Q, max_results), fill, dtype=torch.int64, device=self.device)
        out[self.query_idx[keep], pos[keep]] = self.box_idx[keep]
        return out

    def to_list(self) -> list[list[int]]:
        """Per-query Python lists (moves to host)."""
        boxes, offs = self.box_idx.tolist(), self.offsets.tolist()
        return [boxes[offs[i]:offs[i + 1]] for i in range(self.num_queries)]

    def to(self, device) -> QueryResult:
        return QueryResult(self.query_idx.to(device), self.box_idx.to(device), self.counts.to(device))

    def __repr__(self) -> str:
        return f"QueryResult(num_queries={self.num_queries}, num_pairs={len(self)}, device={self.device})"


# -----------------------------------------------------------------------------
# Tree
# -----------------------------------------------------------------------------

class RTree:
    """
    Packed, static R-tree over N axis-aligned boxes.

    RTree(mins, maxs)          index boxes given by (N, ndim) lower / upper corners
    RTree(points)              index points (zero-size boxes)
    RTree.from_bounds(b)       index boxes given as (N, 2 * ndim) [mins..., maxs...]

    Inputs may be tensors, numpy arrays or lists; integer coordinates are cast
    to float64. Everything is stored detached on the input device.
    """

    __slots__ = ("mins", "maxs", "leaf_order", "level_starts", "level_fanout",
                 "fanout", "curve", "ndim", "num_boxes", "leaf_start")

    def __init__(self, mins: Any, maxs: Any | None = None, *, fanout: int = 8, curve: Curve = "hilbert"):
        if fanout < 2:
            raise ValueError(f"fanout must be >= 2, got {fanout}")
        if curve not in CURVES:
            raise ValueError(f"curve must be one of {CURVES}, got {curve!r}")
        lo, hi = _as_boxes(mins, maxs, "RTree")
        _check_values(lo, hi, "RTree", allow_inf=False)
        with torch.no_grad():
            self._build(lo.detach(), hi.detach(), fanout, curve)

    # -- construction -----------------------------------------------------------
    def _build(self, mins: Tensor, maxs: Tensor, fanout: int, curve: str) -> None:
        n, ndim = mins.shape
        device, dtype = mins.device, mins.dtype

        # 1. Sort leaves by curve key of their centres (stable: reproducible across devices).
        if n > 0:
            centers = (mins + maxs) * 0.5
            g_min = mins.amin(dim=0)
            g_range = (maxs.amax(dim=0) - g_min).clamp_min(torch.finfo(dtype).tiny)
            order = torch.argsort(sfc_codes(centers, g_min, g_range, curve), stable=True)
        else:
            order = torch.empty(0, dtype=torch.int64, device=device)

        # 2. Level sizes, top-down (level 0 = root, last = leaves).
        sizes_bu = [n]
        while sizes_bu[-1] > 1:
            sizes_bu.append((sizes_bu[-1] + fanout - 1) // fanout)
        level_sizes = sizes_bu[::-1]
        level_starts = [0]
        for s in level_sizes:
            level_starts.append(level_starts[-1] + s)
        total = level_starts[-1]
        leaf_start = level_starts[-2]

        # 3. Flat node tensors; leaves at the bottom.
        tree_mins = torch.empty((total, ndim), dtype=dtype, device=device)
        tree_maxs = torch.empty((total, ndim), dtype=dtype, device=device)
        tree_mins[leaf_start:] = mins[order]
        tree_maxs[leaf_start:] = maxs[order]

        # 4. Internal levels bottom-up. Node i of a level owns children
        #    [i * cpn, i * cpn + cpn) of the level below (last node may be short),
        #    so child ranges are arithmetic and never stored.
        level_fanout = []
        inf = torch.tensor(float("inf"), dtype=dtype, device=device)
        for level in range(len(level_sizes) - 2, -1, -1):
            ls, sz = level_starts[level], level_sizes[level]
            cls_, csz = level_starts[level + 1], level_sizes[level + 1]
            cpn = min(-(-csz // sz), fanout)
            level_fanout.append(cpn)
            c_mins, c_maxs = tree_mins[cls_:cls_ + csz], tree_maxs[cls_:cls_ + csz]
            pad = sz * cpn - csz
            if pad > 0:  # +inf / -inf are identities under amin / amax
                c_mins = torch.cat([c_mins, inf.expand(pad, ndim)])
                c_maxs = torch.cat([c_maxs, (-inf).expand(pad, ndim)])
            tree_mins[ls:ls + sz] = c_mins.view(sz, cpn, ndim).amin(dim=1)
            tree_maxs[ls:ls + sz] = c_maxs.view(sz, cpn, ndim).amax(dim=1)

        self.mins, self.maxs, self.leaf_order = tree_mins, tree_maxs, order
        self.level_starts = tuple(level_starts)
        self.level_fanout = tuple(reversed(level_fanout))  # indexed by internal level, top-down
        self.fanout, self.curve, self.ndim = fanout, curve, ndim
        self.num_boxes, self.leaf_start = n, leaf_start

    @classmethod
    def from_bounds(cls, bounds: Any, **kw) -> RTree:
        """Build from an (N, 2 * ndim) tensor laid out as [min_0..min_d, max_0..max_d]."""
        b = _as_coords(bounds, "from_bounds")
        if b.dim() != 2 or b.shape[1] % 2:
            raise ValueError(f"from_bounds: expected shape (N, 2 * ndim), got {tuple(b.shape)}")
        d = b.shape[1] // 2
        return cls(b[:, :d], b[:, d:], **kw)

    @classmethod
    def _from_parts(cls, **parts) -> RTree:
        self = cls.__new__(cls)
        for k in cls.__slots__:
            setattr(self, k, parts[k])
        return self

    # -- introspection -------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.mins.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mins.dtype

    @property
    def num_levels(self) -> int:
        return len(self.level_starts) - 1

    @property
    def num_nodes(self) -> int:
        return self.mins.shape[0]

    @property
    def bounds(self) -> tuple[Tensor, Tensor] | None:
        """(mins, maxs) of the root box, or None for an empty tree."""
        return (self.mins[0], self.maxs[0]) if self.num_boxes else None

    def boxes(self) -> tuple[Tensor, Tensor]:
        """The indexed boxes in their original input order."""
        lo = torch.empty_like(self.mins[self.leaf_start:])
        hi = torch.empty_like(lo)
        lo[self.leaf_order] = self.mins[self.leaf_start:]
        hi[self.leaf_order] = self.maxs[self.leaf_start:]
        return lo, hi

    def __len__(self) -> int:
        return self.num_boxes

    def __repr__(self) -> str:
        return (f"RTree(num_boxes={self.num_boxes}, ndim={self.ndim}, fanout={self.fanout}, "
                f"curve={self.curve!r}, levels={self.num_levels}, dtype={self.dtype}, device={self.device})")

    # -- device / persistence -----------------------------------------------------
    def to(self, device) -> RTree:
        device = torch.device(device)
        same_index = device.index is None or device.index == self.device.index
        if device.type == self.device.type and same_index:
            return self
        parts = {k: getattr(self, k) for k in self.__slots__}
        for k in ("mins", "maxs", "leaf_order"):
            parts[k] = parts[k].to(device)
        return RTree._from_parts(**parts)

    def cpu(self) -> RTree:
        return self.to("cpu")

    def cuda(self, device=None) -> RTree:
        return self.to("cuda" if device is None else device)

    def state_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__slots__}
        d["level_starts"], d["level_fanout"] = list(d["level_starts"]), list(d["level_fanout"])
        return {"format": _STATE_FORMAT, **d}

    @classmethod
    def from_state_dict(cls, state: dict) -> RTree:
        if state.get("format") != _STATE_FORMAT:
            raise ValueError(f"unknown RTree state format {state.get('format')!r}")
        parts = {k: state[k] for k in cls.__slots__}
        parts["level_starts"] = tuple(parts["level_starts"])
        parts["level_fanout"] = tuple(parts["level_fanout"])
        return cls._from_parts(**parts)

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, map_location=None) -> RTree:
        return cls.from_state_dict(torch.load(path, map_location=map_location))

    # -- queries ----------------------------------------------------------------------
    def _queries(self, qmins: Any, qmaxs: Any | None, name: str,
                 validate: bool) -> tuple[Tensor, Tensor]:
        lo, hi = _as_boxes(qmins, qmaxs, name, ndim=self.ndim)
        if validate:
            _check_values(lo, hi, name, allow_inf=True)
        return lo.to(device=self.device, dtype=self.dtype), hi.to(device=self.device, dtype=self.dtype)

    def search(
        self,
        qmins: Any,
        qmaxs: Any | None = None,
        *,
        mode: QueryMode = "intersects",
        sort: bool = False,
        chunk_size: int | None = None,
        max_pairs: int | None = None,
        validate: bool = True,
    ) -> QueryResult:
        """
        Boxes matching each query box (or point, if `qmaxs` is omitted).

        mode:        "intersects" (closed intervals; touching counts),
                     "contains"   (indexed box lies entirely inside the query),
                     "within"     (indexed box entirely contains the query).
        sort:        sort box indices ascending within each query (default: tree order).
        chunk_size:  process at most this many queries at once to bound memory.
        max_pairs:   raise if any level's frontier exceeds this many pairs.
        validate:    check for NaN and min <= max (each check is a device sync).
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        lo, hi = self._queries(qmins, qmaxs, "search", validate)
        q_idx, node = _run_frontier(self, lo, hi, mode, chunk_size, max_pairs)
        box = self.leaf_order[node - self.leaf_start]
        if sort and q_idx.numel():
            perm = torch.argsort(q_idx * max(self.num_boxes, 1) + box)
            q_idx, box = q_idx[perm], box[perm]
        return QueryResult(q_idx, box, torch.bincount(q_idx, minlength=lo.shape[0]))

    def count(
        self,
        qmins: Any,
        qmaxs: Any | None = None,
        *,
        mode: QueryMode = "intersects",
        chunk_size: int | None = None,
        max_pairs: int | None = None,
        validate: bool = True,
    ) -> Tensor:
        """Number of matching boxes per query, as a (Q,) int64 tensor."""
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        lo, hi = self._queries(qmins, qmaxs, "count", validate)
        q_idx, _ = _run_frontier(self, lo, hi, mode, chunk_size, max_pairs)
        return torch.bincount(q_idx, minlength=lo.shape[0])

    def within_distance(
        self,
        qmins: Any,
        qmaxs: Any | None = None,
        *,
        distance: float,
        axis_scale: Any | None = None,
        sort: bool = False,
        chunk_size: int | None = None,
        max_pairs: int | None = None,
        validate: bool = True,
    ) -> QueryResult:
        """
        Boxes whose Euclidean box-to-box distance from each query box (or point)
        is <= `distance`. `axis_scale` (ndim,) multiplies per-axis gaps before
        the norm, e.g. to weight a time axis against spatial axes.
        """
        if distance < 0:
            raise ValueError(f"distance must be >= 0, got {distance}")
        lo, hi = self._queries(qmins, qmaxs, "within_distance", validate)
        scale = self._axis_scale(axis_scale)
        rad = (distance / scale).unsqueeze(0)
        q_idx, node = _run_frontier(self, _step_down(lo - rad), _step_up(hi + rad), "intersects",
                                    chunk_size, max_pairs)
        d = _box_distance(self, node, lo[q_idx], hi[q_idx], scale)
        keep = d <= distance
        q_idx, box = q_idx[keep], self.leaf_order[node[keep] - self.leaf_start]
        if sort and q_idx.numel():
            perm = torch.argsort(q_idx * max(self.num_boxes, 1) + box)
            q_idx, box = q_idx[perm], box[perm]
        return QueryResult(q_idx, box, torch.bincount(q_idx, minlength=lo.shape[0]))

    def nearest(
        self,
        qmins: Any,
        qmaxs: Any | None = None,
        *,
        k: int = 1,
        max_distance: float | None = None,
        axis_scale: Any | None = None,
        chunk_size: int | None = None,
        max_pairs: int | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        k nearest boxes to each query point (or box) by Euclidean box-to-box
        distance, which is zero when they overlap. Returns (dist, idx), each
        (Q, k), ascending; padded with +inf / -1 when fewer than k boxes exist
        or lie within `max_distance`. `axis_scale` weights axes as in
        `within_distance`.

        Implemented as an expanding search: each query searches a cube of
        radius r and is finished once it has >= k candidates whose k-th
        distance is <= r (the cube contains the ball, so nothing closer was
        missed). Queries with >= k candidates but a k-th distance beyond r
        retry once at exactly that distance; others double r.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_distance is not None and max_distance < 0:
            raise ValueError(f"max_distance must be >= 0, got {max_distance}")
        lo_all, hi_all = self._queries(qmins, qmaxs, "nearest", validate)
        Q, device, dtype = lo_all.shape[0], self.device, self.dtype
        idx_out = torch.full((Q, k), -1, dtype=torch.int64, device=device)
        dist_out = torch.full((Q, k), float("inf"), dtype=dtype, device=device)
        if Q == 0 or self.num_boxes == 0:
            return dist_out, idx_out
        k_eff = min(k, self.num_boxes)
        scale = self._axis_scale(axis_scale)

        root_lo, root_hi = self.mins[0], self.maxs[0]
        ranges = ((root_hi - root_lo) * scale).double()  # root extent in scaled units
        extent = float(ranges.amax().item())
        # Initial radius: half the side of the cube holding ~k boxes if they were
        # uniform over the root VOLUME (not its widest axis, which misbehaves on
        # anisotropic data such as a wide time axis).
        volume = float(ranges.clamp_min(extent * 1e-3).prod().item()) if extent > 0 else 0.0
        r0 = 0.5 * (k_eff / self.num_boxes * volume) ** (1.0 / self.ndim)
        r0 = max(r0, extent * 1e-6, 1e-30)
        # A cube of this radius covers the root box for every query, so the
        # search is exhaustive: r >= lo - root_lo and r >= root_hi - hi per axis.
        far = torch.maximum((lo_all - root_lo).abs(), (root_hi - hi_all).abs()) * scale
        r_max = float(far.amax().item()) * 1.001 + 1.0
        cap = r_max if max_distance is None else min(max_distance, r_max)

        root_node = torch.zeros(Q, dtype=torch.int64, device=device)
        r = (_box_distance(self, root_node, lo_all, hi_all, scale) * 1.001).clamp_min(r0).clamp_max(cap)
        pending = torch.arange(Q, device=device)

        while pending.numel() > 0:
            lo, hi = lo_all[pending], hi_all[pending]
            rad = (r[pending].unsqueeze(1) / scale)
            q, node = _run_frontier(self, _step_down(lo - rad), _step_up(hi + rad), "intersects",
                                    chunk_size, max_pairs)
            d = _box_distance(self, node, lo[q], hi[q], scale)

            # Order pairs by (query, distance): sort by distance, then stable sort by query.
            d, perm = torch.sort(d)
            q, node = q[perm], node[perm]
            q, perm = torch.sort(q, stable=True)
            d, node = d[perm], node[perm]

            counts = torch.bincount(q, minlength=pending.numel())
            pos = torch.arange(q.numel(), device=device) - (counts.cumsum(0) - counts)[q]
            has_k = counts >= k_eff
            kth = torch.full((pending.numel(),), float("inf"), dtype=dtype, device=device)
            sel = pos == (k_eff - 1)
            kth[q[sel]] = d[sel]
            r_cur = r[pending]
            done = (has_k & (kth <= r_cur)) | (r_cur >= cap)

            keep = done[q] & (pos < k_eff)
            if max_distance is not None:
                keep &= d <= max_distance
            idx_out[pending[q[keep]], pos[keep]] = self.leaf_order[node[keep] - self.leaf_start]
            dist_out[pending[q[keep]], pos[keep]] = d[keep]

            r_next = torch.where(has_k, kth * (1 + 1e-6), r_cur * 2).clamp_max(cap)
            r[pending] = torch.where(done, r_cur, r_next)
            pending = pending[~done]
        return dist_out, idx_out

    def self_join(self, *, mode: QueryMode = "intersects", sort: bool = False,
                  chunk_size: int | None = None, max_pairs: int | None = None) -> QueryResult:
        """
        Pairs (i, j) of indexed boxes with i < j that satisfy `mode` (box j
        relative to box i as the query). Each unordered pair is reported once.
        """
        lo, hi = self.boxes()
        res = self.search(lo, hi, mode=mode, sort=sort, chunk_size=chunk_size, max_pairs=max_pairs,
                          validate=False)
        keep = res.query_idx < res.box_idx
        q, b = res.query_idx[keep], res.box_idx[keep]
        return QueryResult(q, b, torch.bincount(q, minlength=self.num_boxes))

    def _axis_scale(self, axis_scale: Any | None) -> Tensor:
        if axis_scale is None:
            return torch.ones(self.ndim, dtype=self.dtype, device=self.device)
        s = torch.as_tensor(axis_scale, dtype=self.dtype, device=self.device).reshape(-1)
        if s.numel() != self.ndim:
            raise ValueError(f"axis_scale must have {self.ndim} entries, got {s.numel()}")
        if not (s > 0).all():
            raise ValueError("axis_scale entries must be > 0")
        return s


# -----------------------------------------------------------------------------
# Traversal
# -----------------------------------------------------------------------------

def _node_test(tree: RTree, node: Tensor, qmins: Tensor, qmaxs: Tensor, q: Tensor,
               mode: str, leaf: bool) -> Tensor:
    """Predicate for (query q[i], node node[i]) pairs under the given mode."""
    nmin, nmax = tree.mins[node], tree.maxs[node]
    qmin, qmax = qmins[q], qmaxs[q]
    if mode == "within":
        # Query inside node. Valid pruning at internal levels too, since
        # ancestors contain their leaves.
        return (nmin <= qmin).all(dim=-1) & (qmax <= nmax).all(dim=-1)
    if mode == "contains" and leaf:
        return (qmin <= nmin).all(dim=-1) & (nmax <= qmax).all(dim=-1)
    # Closed-interval intersection (also the internal-node test for 'contains').
    return (nmin <= qmax).all(dim=-1) & (nmax >= qmin).all(dim=-1)


def _frontier(tree: RTree, qmins: Tensor, qmaxs: Tensor, mode: str,
              max_pairs: int | None) -> tuple[Tensor, Tensor]:
    """Core traversal. Returns (query_idx, node_idx) with node_idx in tree space."""
    Q, device = qmins.shape[0], tree.device
    if Q == 0 or tree.num_boxes == 0:
        e = torch.empty(0, dtype=torch.int64, device=device)
        return e, e.clone()

    fq = torch.arange(Q, device=device)
    fn = torch.zeros(Q, dtype=torch.int64, device=device)  # root
    ls = tree.level_starts
    for level, cpn in enumerate(tree.level_fanout):  # internal levels, top-down
        hit = _node_test(tree, fn, qmins, qmaxs, fq, mode, leaf=False)
        fq, fn = fq[hit], fn[hit]
        first = ls[level + 1] + (fn - ls[level]) * cpn
        child = first.unsqueeze(1) + torch.arange(cpn, device=device)   # (F, cpn)
        valid = child < ls[level + 2]                                     # last node may be short
        fq = fq.unsqueeze(1).expand_as(child)[valid]
        fn = child[valid]
        if max_pairs is not None and fn.numel() > max_pairs:
            raise RuntimeError(
                f"query frontier reached {fn.numel()} (query, node) pairs, above max_pairs={max_pairs}; "
                f"pass a larger max_pairs, a chunk_size, or narrow the queries")

    hit = _node_test(tree, fn, qmins, qmaxs, fq, mode, leaf=True)
    return fq[hit], fn[hit]


def _run_frontier(tree: RTree, qmins: Tensor, qmaxs: Tensor, mode: str,
                  chunk_size: int | None, max_pairs: int | None) -> tuple[Tensor, Tensor]:
    Q = qmins.shape[0]
    if chunk_size is None or chunk_size >= Q:
        return _frontier(tree, qmins, qmaxs, mode, max_pairs)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qs, ns = [], []
    for s in range(0, Q, chunk_size):
        fq, fn = _frontier(tree, qmins[s:s + chunk_size], qmaxs[s:s + chunk_size], mode, max_pairs)
        qs.append(fq + s)
        ns.append(fn)
    return torch.cat(qs), torch.cat(ns)


def _box_distance(tree: RTree, node: Tensor, qmins: Tensor, qmaxs: Tensor, scale: Tensor) -> Tensor:
    """Euclidean gap between query boxes and tree nodes, per-axis gaps scaled."""
    gap = torch.maximum(tree.mins[node] - qmaxs, qmins - tree.maxs[node]).clamp_min(0) * scale
    return (gap * gap).sum(dim=-1).sqrt()


def _step_down(x: Tensor) -> Tensor:
    """Round a search bound outward by one ulp so float rounding cannot shrink the cube."""
    return torch.nextafter(x, torch.full_like(x, float("-inf")))


def _step_up(x: Tensor) -> Tensor:
    return torch.nextafter(x, torch.full_like(x, float("inf")))


# -----------------------------------------------------------------------------
# Functional alias
# -----------------------------------------------------------------------------

def build_rtree(mins: Any, maxs: Any | None = None, *, fanout: int = 8, curve: Curve = "hilbert") -> RTree:
    """Alias for RTree(mins, maxs, fanout=..., curve=...)."""
    return RTree(mins, maxs, fanout=fanout, curve=curve)
