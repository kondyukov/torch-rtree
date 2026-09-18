"""
Accuracy tests: every query path must match a brute-force tensor oracle, for
every supported ndim, dtype, curve and device.
"""

import numpy as np
import pytest
import torch

from torchrtree import SUPPORTED_DTYPES, QueryResult, RTree, build_rtree
from torchrtree._curves import hilbert_codes, sfc_bits, sfc_codes

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NDIMS = (1, 2, 3, 4)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _random_boxes(n, ndim, device="cpu", gen=None, extent=1.0, size=0.05, dtype=torch.float32):
    """Boxes with edge length ~ size on each axis; last axis scaled by `extent`
    to mimic a time axis in different units."""
    a = torch.rand((n, ndim), generator=gen)
    b = a + size * torch.rand((n, ndim), generator=gen)
    scale = torch.ones(ndim)
    scale[-1] = extent
    return (a * scale).to(device=device, dtype=dtype), (b * scale).to(device=device, dtype=dtype)


def _brute(mins, maxs, qmins, qmaxs, mode="intersects"):
    """(Q, N) boolean oracle."""
    m, M = mins.unsqueeze(0), maxs.unsqueeze(0)
    q, Qm = qmins.unsqueeze(1), qmaxs.unsqueeze(1)
    if mode == "intersects":
        return (m <= Qm).all(-1) & (M >= q).all(-1)
    if mode == "contains":
        return (q <= m).all(-1) & (M <= Qm).all(-1)
    if mode == "within":
        return (m <= q).all(-1) & (Qm <= M).all(-1)
    raise ValueError(mode)


def _brute_dist(mins, maxs, qmins, qmaxs, scale=None):
    """(Q, N) box-to-box Euclidean distance with optional per-axis scale."""
    below, above = mins.unsqueeze(0) - qmaxs.unsqueeze(1), qmins.unsqueeze(1) - maxs.unsqueeze(0)
    gap = torch.maximum(below, above).clamp_min(0)
    if scale is not None:
        gap = gap * scale
    return (gap * gap).sum(-1).sqrt()


def _pairs(res):
    q, b = res
    return set(zip(q.tolist(), b.tolist(), strict=True))


def _expected_pairs(expected):
    eq, el = torch.nonzero(expected, as_tuple=True)
    return set(zip(eq.tolist(), el.tolist(), strict=True))


# -----------------------------------------------------------------------------
# box search
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", NDIMS)
@pytest.mark.parametrize("n", [1, 7, 8, 9, 500, 3000])
def test_intersects_matches_brute_force(device, ndim, n):
    gen = torch.Generator().manual_seed(1234 + n + ndim)
    mins, maxs = _random_boxes(n, ndim, device, gen, extent=1000.0)
    tree = RTree(mins, maxs)
    qmins, qmaxs = _random_boxes(64, ndim, device, gen, extent=1000.0, size=0.2)
    expected = _brute(mins, maxs, qmins, qmaxs)

    res = tree.search(qmins, qmaxs)
    assert isinstance(res, QueryResult)
    assert res.counts.tolist() == expected.sum(-1).tolist()
    assert (res.query_idx[1:] >= res.query_idx[:-1]).all()
    assert _pairs(res) == _expected_pairs(expected)
    assert torch.equal(res.offsets[1:] - res.offsets[:-1], res.counts)
    assert torch.equal(tree.count(qmins, qmaxs), res.counts)

    # Padded view and per-query access agree with the pairs.
    padded = res.to_padded()
    for qi, row in enumerate(padded.tolist()):
        got = {r for r in row if r >= 0}
        assert got == set(res[qi].tolist()) == set(torch.nonzero(expected[qi]).flatten().tolist())

    # sort=True orders box ids within each query.
    srt = tree.search(qmins, qmaxs, sort=True)
    assert _pairs(srt) == _pairs(res)
    for qi in range(64):
        ids = srt[qi].tolist()
        assert ids == sorted(ids)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["contains", "within"])
@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_contains_and_within_modes(device, mode, ndim):
    gen = torch.Generator().manual_seed(99 + ndim)
    mins, maxs = _random_boxes(2000, ndim, device, gen, size=0.1)
    tree = RTree(mins, maxs)
    size = 0.3 if mode == "contains" else 0.005
    qmins, qmaxs = _random_boxes(100, ndim, device, gen, size=size)
    expected = _brute(mins, maxs, qmins, qmaxs, mode)
    assert expected.any(), "degenerate test configuration"
    assert _pairs(tree.search(qmins, qmaxs, mode=mode)) == _expected_pairs(expected)
    assert torch.equal(tree.count(qmins, qmaxs, mode=mode), expected.sum(-1))


@pytest.mark.parametrize("ndim", (1, 2, 3, 4, 5))
def test_point_queries_and_point_index(ndim):
    gen = torch.Generator().manual_seed(5)
    mins, maxs = _random_boxes(2000, ndim, gen=gen, size=0.1)
    tree = RTree(mins, maxs)
    pts = torch.rand((200, ndim), generator=gen)
    expected = _brute(mins, maxs, pts, pts)
    assert _pairs(tree.search(pts)) == _expected_pairs(expected)

    # Index of points, queried by boxes.
    ptree = RTree(pts)
    assert ptree.num_boxes == 200
    qmins, qmaxs = _random_boxes(50, ndim, gen=gen, size=0.3)
    assert _pairs(ptree.search(qmins, qmaxs)) == _expected_pairs(_brute(pts, pts, qmins, qmaxs))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("curve", ["morton", "hilbert"])
def test_dtypes_and_curves(dtype, curve):
    gen = torch.Generator().manual_seed(21)
    mins, maxs = _random_boxes(1500, 3, gen=gen, dtype=dtype)
    tree = RTree(mins, maxs, curve=curve)
    assert tree.dtype == dtype and tree.curve == curve
    qmins, qmaxs = _random_boxes(50, 3, gen=gen, size=0.2, dtype=dtype)
    assert _pairs(tree.search(qmins, qmaxs)) == _expected_pairs(_brute(mins, maxs, qmins, qmaxs))


def test_float64_keeps_timestamp_resolution():
    t0 = 1_700_000_000.0
    mins = torch.tensor([[0.0, 0.0, 0.0, t0 + i] for i in range(10)], dtype=torch.float64)
    maxs = mins + torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float64)
    tree = RTree(mins, maxs)
    q = torch.tensor([[0.0, 0.0, 0.0, t0 + 3.2]], dtype=torch.float64)
    _, box = tree.search(q, q + torch.tensor([1.0, 1.0, 1.0, 0.1], dtype=torch.float64))
    assert box.tolist() == [3]


def test_infinite_query_bounds_are_half_open_ranges():
    gen = torch.Generator().manual_seed(4)
    mins, maxs = _random_boxes(500, 4, gen=gen, extent=100.0)
    tree = RTree(mins, maxs)
    qmin = torch.tensor([[-float("inf"), -float("inf"), -float("inf"), 50.0]])
    qmax = torch.tensor([[float("inf"), float("inf"), float("inf"), float("inf")]])
    _, box = tree.search(qmin, qmax)
    expected = torch.nonzero(maxs[:, 3] >= 50.0).flatten().tolist()
    assert sorted(box.tolist()) == expected
    with pytest.raises(ValueError, match="finite"):
        RTree(qmin, qmax)  # data boxes must stay finite


# -----------------------------------------------------------------------------
# inputs
# -----------------------------------------------------------------------------

def test_accepts_numpy_lists_ints_and_single_boxes():
    rng = np.random.default_rng(0)
    lo = rng.integers(0, 1000, (300, 2))
    hi = lo + rng.integers(0, 50, (300, 2))
    tree = RTree(lo, hi)  # numpy int64
    assert tree.dtype == torch.float64
    t = torch.tensor([[100, 100], [300, 300]])
    res = tree.search(t[0], t[1])  # a single box as two 1-D vectors
    assert res.num_queries == 1
    lo64, hi64 = torch.from_numpy(lo).double(), torch.from_numpy(hi).double()
    expected = _brute(lo64, hi64, t[:1].double(), t[1:].double())
    assert set(res[0].tolist()) == set(torch.nonzero(expected[0]).flatten().tolist())
    assert res.to_list() == [res[0].tolist()]
    assert _pairs(tree.search([[100, 100]], [[300, 300]])) == _pairs(res)  # nested lists
    assert _pairs(tree.search(np.array([[100, 100]]), np.array([[300, 300]]))) == _pairs(res)
    assert RTree(lo.astype(np.float16), hi.astype(np.float16)).dtype == torch.float32
    with pytest.raises(TypeError, match="dtype"):
        RTree(lo > 5, hi > 5)


def test_from_bounds_and_boxes_roundtrip():
    gen = torch.Generator().manual_seed(2)
    mins, maxs = _random_boxes(400, 3, gen=gen)
    tree = RTree.from_bounds(torch.cat([mins, maxs], dim=1))
    lo, hi = tree.boxes()
    assert torch.equal(lo, mins) and torch.equal(hi, maxs)
    b_lo, b_hi = tree.bounds
    assert torch.equal(b_lo, mins.amin(0)) and torch.equal(b_hi, maxs.amax(0))
    with pytest.raises(ValueError, match="2 \\* ndim"):
        RTree.from_bounds(torch.rand(5, 3))


def test_build_detaches_and_does_not_mutate_inputs():
    mins = torch.rand(100, 2, requires_grad=True)
    maxs = mins + 0.1
    copy = mins.detach().clone()
    tree = RTree(mins, maxs)
    assert not tree.mins.requires_grad and tree.mins.grad_fn is None
    assert torch.equal(mins.detach(), copy)
    res = tree.search(mins, maxs)  # queries requiring grad are fine too
    assert len(res) >= 100


def test_identical_and_duplicate_boxes():
    same = torch.zeros(50, 3)
    tree = RTree(same, same + 1)  # zero-range curve normalisation
    res = tree.search(torch.tensor([[0.5, 0.5, 0.5]]))
    assert sorted(res[0].tolist()) == list(range(50))
    dup = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    tree = RTree(dup, dup + 0.5)
    assert sorted(tree.search(torch.tensor([[0.1, 0.1]]))[0].tolist()) == [0, 1]


@pytest.mark.parametrize("ndim", [0, 9])
def test_unsupported_ndim(ndim):
    boxes = torch.rand((10, ndim))
    with pytest.raises(ValueError, match="ndim"):
        RTree(boxes, boxes + 0.1)


def test_reproducible_leaf_order_across_devices():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    gen = torch.Generator().manual_seed(9)
    mins, maxs = _random_boxes(5000, 2, gen=gen)
    a = RTree(mins, maxs)
    b = RTree(mins.cuda(), maxs.cuda())
    assert torch.equal(a.leaf_order, b.leaf_order.cpu())
    qmins, qmaxs = _random_boxes(30, 2, gen=gen, size=0.2)
    ra, rb = a.search(qmins, qmaxs), b.search(qmins, qmaxs)
    assert torch.equal(ra.box_idx, rb.box_idx.cpu())  # same order, not just same set


# -----------------------------------------------------------------------------
# chunking / memory guard / edge cases
# -----------------------------------------------------------------------------

def test_chunk_size_is_transparent():
    gen = torch.Generator().manual_seed(8)
    mins, maxs = _random_boxes(3000, 2, gen=gen)
    tree = RTree(mins, maxs)
    qmins, qmaxs = _random_boxes(257, 2, gen=gen, size=0.2)
    full = tree.search(qmins, qmaxs)
    for cs in (1, 100, 256, 257, 10_000):
        chunked = tree.search(qmins, qmaxs, chunk_size=cs)
        assert torch.equal(full.query_idx, chunked.query_idx) and torch.equal(full.box_idx, chunked.box_idx)
    with pytest.raises(ValueError, match="chunk_size"):
        tree.search(qmins, qmaxs, chunk_size=0)


@pytest.mark.parametrize("device", DEVICES)
def test_auto_batching_is_transparent(device):
    from torchrtree.rtree import _FIRST_CHUNK

    gen = torch.Generator().manual_seed(13)
    mins, maxs = _random_boxes(20_000, 3, device, gen, size=0.02)
    qmins, qmaxs = _random_boxes(3 * _FIRST_CHUNK, 3, device, gen, size=0.05)  # forces several batches
    pts = torch.rand((3 * _FIRST_CHUNK, 3), generator=gen).to(device)

    big = RTree(mins, maxs)                       # default budget: likely one or two batches
    tiny = RTree(mins, maxs, pairs_budget=5_000)  # far below one batch's peak: many batches
    assert tiny.pairs_budget == 5_000
    single = big.search(qmins, qmaxs, chunk_size=qmins.shape[0])
    for tree in (big, tiny):
        res = tree.search(qmins, qmaxs)
        assert torch.equal(res.query_idx, single.query_idx) and torch.equal(res.box_idx, single.box_idx)
        assert torch.equal(tree.count(qmins, qmaxs), single.counts)
        d, i = tree.nearest(pts, k=3)
        d1, i1 = big.nearest(pts, k=3, chunk_size=pts.shape[0])
        assert torch.equal(d, d1) and torch.equal(i, i1)
        wd = tree.within_distance(pts, distance=0.05)
        wd1 = big.within_distance(pts, distance=0.05, chunk_size=pts.shape[0])
        assert torch.equal(wd.box_idx, wd1.box_idx)
    with pytest.raises(ValueError, match="pairs_budget"):
        RTree(mins, maxs, pairs_budget=0)
    tiny.pairs_budget = 1  # runtime override still clamps to the minimum batch and works
    assert torch.equal(tiny.search(qmins, qmaxs).box_idx, single.box_idx)


def test_max_pairs_guard():
    gen = torch.Generator().manual_seed(8)
    mins, maxs = _random_boxes(3000, 2, gen=gen)
    tree = RTree(mins, maxs)
    everything = (torch.zeros((100, 2)), torch.ones((100, 2)) * 2)
    with pytest.raises(RuntimeError, match="max_pairs"):
        tree.search(*everything, max_pairs=1000)
    assert len(tree.search(*everything, max_pairs=10_000_000)) == 100 * 3000


def test_padded_truncation():
    gen = torch.Generator().manual_seed(3)
    mins, maxs = _random_boxes(200, 2, gen=gen)
    res = RTree(mins, maxs).search(torch.zeros((1, 2)), torch.ones((1, 2)) * 2)
    padded = res.to_padded(max_results=10)
    assert padded.shape == (1, 10) and (padded >= 0).all()
    assert res.counts.item() == 200


def test_empty_tree_and_empty_queries():
    empty = torch.empty((0, 3))
    tree = RTree(empty, empty)
    assert tree.num_boxes == 0 and tree.num_nodes == 0 and tree.bounds is None and len(tree) == 0
    far = torch.zeros((5, 3))
    res = tree.search(far, far + 1)
    assert len(res) == 0 and res.num_queries == 5 and (res.counts == 0).all()
    assert res.to_padded().shape == (5, 1)
    dist, idx = tree.nearest(far, k=2)
    assert (idx == -1).all() and torch.isinf(dist).all()

    gen = torch.Generator().manual_seed(11)
    mins, maxs = _random_boxes(100, 3, gen=gen)
    tree = RTree(mins, maxs)
    assert len(tree.search(empty, empty)) == 0
    res = tree.search(torch.full((5, 3), 10.0), torch.full((5, 3), 11.0))
    assert len(res) == 0 and (res.counts == 0).all()
    with pytest.raises(IndexError):
        res[5]


@pytest.mark.parametrize("ndim", NDIMS)
@pytest.mark.parametrize("fanout", [2, 8, 16])
def test_tree_invariants(ndim, fanout):
    gen = torch.Generator().manual_seed(7)
    mins, maxs = _random_boxes(1000, ndim, gen=gen)
    tree = RTree(mins, maxs, fanout=fanout)
    assert sorted(tree.leaf_order.tolist()) == list(range(1000))
    assert torch.equal(tree.mins[tree.leaf_start:], mins[tree.leaf_order])
    assert len(tree.level_fanout) == tree.num_levels - 1
    ls = tree.level_starts
    for level, cpn in enumerate(tree.level_fanout):
        assert 2 <= cpn <= fanout
        for i in range(ls[level], ls[level + 1]):
            s = ls[level + 1] + (i - ls[level]) * cpn
            e = min(s + cpn, ls[level + 2])
            assert s < e, "every internal node has at least one child"
            assert (tree.mins[i] <= tree.mins[s:e]).all() and (tree.maxs[i] >= tree.maxs[s:e]).all()


# -----------------------------------------------------------------------------
# validation
# -----------------------------------------------------------------------------

def test_validation_errors():
    good = torch.rand((10, 2))
    with pytest.raises(ValueError, match="min must be <="):
        RTree(good + 1, good)
    bad = good.clone()
    bad[3, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        RTree(bad, bad + 1)
    with pytest.raises(ValueError, match="shape"):
        RTree(torch.rand(10), torch.rand(10))
    with pytest.raises(ValueError, match="differ"):
        RTree(good, torch.rand(11, 2))
    with pytest.raises(ValueError, match="fanout"):
        RTree(good, good + 1, fanout=1)
    with pytest.raises(ValueError, match="curve"):
        RTree(good, good + 1, curve="peano")
    tree = RTree(good, good + 1)
    with pytest.raises(ValueError, match="ndim=2"):
        tree.search(torch.rand((3, 3)), torch.rand((3, 3)) + 1)
    with pytest.raises(ValueError, match="mode"):
        tree.search(good, good + 1, mode="touches")
    with pytest.raises(ValueError, match="NaN"):
        tree.search(bad, bad + 1)
    tree.search(bad, bad + 1, validate=False)  # caller opts out of the checks
    with pytest.raises(ValueError, match="k must be"):
        tree.nearest(good, k=0)
    with pytest.raises(ValueError, match="axis_scale"):
        tree.nearest(good, axis_scale=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="distance"):
        tree.within_distance(good, distance=-1.0)


# -----------------------------------------------------------------------------
# nearest / within_distance / self_join
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", (1, 2, 3, 4))
@pytest.mark.parametrize("k", [1, 5])
def test_nearest_matches_brute_force(device, ndim, k):
    gen = torch.Generator().manual_seed(31 + ndim + k)
    mins, maxs = _random_boxes(2000, ndim, device, gen, size=0.01)
    tree = RTree(mins, maxs)
    inside = torch.rand((100, ndim), generator=gen)
    outside = torch.rand((10, ndim), generator=gen) * 4 - 2
    pts = torch.cat([inside, outside]).to(device)
    dist, idx = tree.nearest(pts, k=k)
    exp_d = torch.topk(_brute_dist(mins, maxs, pts, pts), k, dim=1, largest=False).values
    assert torch.allclose(dist, exp_d, atol=1e-6)
    assert (idx >= 0).all()
    # Ties may pick different boxes; the distances at the returned boxes must still match.
    got = _brute_dist(mins, maxs, pts, pts).gather(1, idx)
    assert torch.allclose(got, exp_d, atol=1e-6)


def test_nearest_with_box_queries_scale_and_max_distance():
    gen = torch.Generator().manual_seed(12)
    mins, maxs = _random_boxes(1500, 3, gen=gen, size=0.01, extent=100.0)
    tree = RTree(mins, maxs)
    qmins, qmaxs = _random_boxes(60, 3, gen=gen, size=0.02, extent=100.0)
    scale = torch.tensor([1.0, 1.0, 0.01])  # bring the wide axis back to ~unit range
    dist, idx = tree.nearest(qmins, qmaxs, k=3, axis_scale=scale)
    full = _brute_dist(mins, maxs, qmins, qmaxs, scale)
    exp = torch.topk(full, 3, dim=1, largest=False).values
    assert torch.allclose(dist, exp, atol=1e-5)
    assert torch.allclose(full.gather(1, idx), exp, atol=1e-5)

    md = float(exp[:, 1].median())
    dist, idx = tree.nearest(qmins, qmaxs, k=3, axis_scale=scale, max_distance=md)
    within = exp <= md
    assert torch.equal(idx >= 0, within)
    assert torch.allclose(dist[within], exp[within], atol=1e-5) and torch.isinf(dist[~within]).all()


def test_nearest_k_exceeds_n_and_chunking():
    gen = torch.Generator().manual_seed(2)
    mins, maxs = _random_boxes(3, 2, gen=gen)
    tree = RTree(mins, maxs)
    pts = torch.rand((4, 2), generator=gen)
    dist, idx = tree.nearest(pts, k=5)
    assert (idx[:, 3:] == -1).all() and torch.isinf(dist[:, 3:]).all() and (idx[:, :3] >= 0).all()
    exp = torch.topk(_brute_dist(mins, maxs, pts, pts), 3, dim=1, largest=False).values
    assert torch.allclose(dist[:, :3], exp, atol=1e-6)

    mins, maxs = _random_boxes(500, 2, gen=gen)
    tree = RTree(mins, maxs)
    pts = torch.rand((77, 2), generator=gen)
    a, b = tree.nearest(pts, k=3), tree.nearest(pts, k=3, chunk_size=10)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("device", DEVICES)
def test_within_distance(device):
    gen = torch.Generator().manual_seed(17)
    mins, maxs = _random_boxes(2000, 3, device, gen, size=0.02)
    tree = RTree(mins, maxs)
    pts = torch.rand((80, 3), generator=gen).to(device)
    qmins, qmaxs = _random_boxes(40, 3, device, gen, size=0.05)
    for lo, hi in ((pts, None), (qmins, qmaxs)):
        hi_ = lo if hi is None else hi
        for d in (0.0, 0.03, 0.2):
            res = tree.within_distance(lo, hi, distance=d, sort=True)
            expected = _brute_dist(mins, maxs, lo, hi_) <= d
            assert _pairs(res) == _expected_pairs(expected)
    scale = torch.tensor([1.0, 5.0, 1.0], device=device)
    res = tree.within_distance(pts, distance=0.1, axis_scale=scale)
    assert _pairs(res) == _expected_pairs(_brute_dist(mins, maxs, pts, pts, scale) <= 0.1)


def test_self_join():
    gen = torch.Generator().manual_seed(23)
    mins, maxs = _random_boxes(600, 2, gen=gen, size=0.05)
    tree = RTree(mins, maxs)
    res = tree.self_join(sort=True)
    expected = _brute(mins, maxs, mins, maxs)
    exp_pairs = {(i, j) for i, j in _expected_pairs(expected) if i < j}
    assert _pairs(res) == exp_pairs and res.num_queries == 600
    assert (res.query_idx < res.box_idx).all()


# -----------------------------------------------------------------------------
# container: repr / to / save / load
# -----------------------------------------------------------------------------

def test_repr_is_short_and_save_load_roundtrip(tmp_path):
    gen = torch.Generator().manual_seed(1)
    mins, maxs = _random_boxes(300, 2, gen=gen)
    tree = RTree(mins, maxs)
    text = repr(tree)
    assert text.startswith("RTree(num_boxes=300") and len(text) < 200
    assert repr(tree.search(mins[:3], maxs[:3])).startswith("QueryResult(num_queries=3")

    path = tmp_path / "tree.pt"
    tree.save(path)
    loaded = RTree.load(path)
    assert repr(loaded) == text
    assert torch.equal(tree.mins, loaded.mins) and torch.equal(tree.leaf_order, loaded.leaf_order)
    assert tree.level_starts == loaded.level_starts and tree.level_fanout == loaded.level_fanout
    qmins, qmaxs = _random_boxes(20, 2, gen=gen, size=0.2)
    assert torch.equal(tree.search(qmins, qmaxs).box_idx, loaded.search(qmins, qmaxs).box_idx)
    with pytest.raises(ValueError, match="format"):
        RTree.from_state_dict({"format": 99})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_to_device_roundtrip():
    gen = torch.Generator().manual_seed(1)
    mins, maxs = _random_boxes(300, 3, gen=gen)
    tree = RTree(mins, maxs)
    gpu = tree.cuda()
    assert gpu.device.type == "cuda" and tree.cpu() is tree and gpu.cuda() is gpu
    qmins, qmaxs = _random_boxes(20, 3, gen=gen, size=0.2)
    a, b = tree.search(qmins, qmaxs), gpu.search(qmins, qmaxs)  # CPU queries move to the tree's device
    assert b.device.type == "cuda"
    assert _pairs(a) == _pairs(b.to("cpu"))
    assert torch.equal(gpu.boxes()[0].cpu(), mins)


def test_build_rtree_alias():
    gen = torch.Generator().manual_seed(1)
    mins, maxs = _random_boxes(50, 2, gen=gen)
    tree = build_rtree(mins, maxs, fanout=4, curve="morton")
    assert isinstance(tree, RTree) and tree.fanout == 4 and tree.curve == "morton"


# -----------------------------------------------------------------------------
# curves
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", (1, 2, 3, 4, 8))
def test_sfc_keys_fit_int64_and_separate_points(ndim):
    max_q = (1 << sfc_bits(ndim)) - 1  # the quantiser scales [0, 1] by this
    n = min(256, max_q)
    centers = torch.zeros((n, ndim), dtype=torch.float64)
    centers[:, 0] = (torch.arange(n, dtype=torch.float64) + 0.5) / max_q  # cell midpoints
    for curve in ("morton", "hilbert"):
        codes = sfc_codes(centers, torch.zeros(ndim), torch.ones(ndim), curve)
        assert codes.dtype == torch.int64 and (codes >= 0).all()
        assert len(set(codes.tolist())) == n
    codes = sfc_codes(centers, torch.zeros(ndim), torch.ones(ndim), "morton")
    assert (codes[1:] >= codes[:-1]).all()  # monotone along one axis is a Morton property


@pytest.mark.parametrize("ndim", (2, 3, 4))
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_hilbert_is_a_continuous_space_filling_curve(ndim, bits):
    side = 1 << bits
    grid = torch.cartesian_prod(*[torch.arange(side)] * ndim).reshape(-1, ndim)
    keys = hilbert_codes(grid, bits)
    assert sorted(keys.tolist()) == list(range(side ** ndim))
    walk = grid[torch.argsort(keys)]
    assert ((walk[1:] - walk[:-1]).abs().sum(-1) == 1).all()
