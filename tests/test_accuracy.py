"""
Accuracy tests: every query path must match a brute-force tensor oracle, for
every supported ndim, dtype, curve and device.
"""

import pytest
import torch

from torchrtree import (
    SUPPORTED_DTYPES,
    SUPPORTED_NDIMS,
    RTree,
    build_rtree,
    query_rtree,
    query_rtree_nearest,
    query_rtree_pairs,
    query_rtree_points,
)
from torchrtree.rtree import _hilbert_codes, _sfc_codes

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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


def _pairs_to_set(q_idx, leaf_idx):
    return set(zip(q_idx.tolist(), leaf_idx.tolist(), strict=True))


def _expected_pairs(expected):
    eq, el = torch.nonzero(expected, as_tuple=True)
    return set(zip(eq.tolist(), el.tolist(), strict=True))


# -----------------------------------------------------------------------------
# box queries
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
@pytest.mark.parametrize("n", [1, 7, 8, 9, 500, 3000])
def test_intersects_matches_brute_force(device, ndim, n):
    gen = torch.Generator().manual_seed(1234 + n + ndim)
    mins, maxs = _random_boxes(n, ndim, device, gen, extent=1000.0)
    tree = build_rtree(mins, maxs, m=8)

    qmins, qmaxs = _random_boxes(64, ndim, device, gen, extent=1000.0, size=0.2)
    expected = _brute(mins, maxs, qmins, qmaxs)

    results, counts = query_rtree(tree, qmins, qmaxs)
    assert counts.tolist() == expected.sum(-1).tolist()
    for row, c, exp_row in zip(results.tolist(), counts.tolist(), expected, strict=True):
        got = {r for r in row if r >= 0}
        assert got == set(torch.nonzero(exp_row).flatten().tolist())
        assert len(got) == c

    q_idx, leaf_idx = query_rtree_pairs(tree, qmins, qmaxs)
    assert q_idx.shape == leaf_idx.shape
    assert (q_idx[1:] >= q_idx[:-1]).all()
    assert _pairs_to_set(q_idx, leaf_idx) == _expected_pairs(expected)


@pytest.mark.parametrize("mode", ["contains", "within"])
@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
def test_contains_and_within_modes(mode, ndim):
    gen = torch.Generator().manual_seed(99 + ndim)
    mins, maxs = _random_boxes(2000, ndim, gen=gen, size=0.1)
    tree = build_rtree(mins, maxs)
    # 'contains' wants big queries; 'within' wants tiny ones so some boxes contain them.
    size = 0.3 if mode == "contains" else 0.005
    qmins, qmaxs = _random_boxes(100, ndim, gen=gen, size=size)
    expected = _brute(mins, maxs, qmins, qmaxs, mode)
    assert expected.any(), "degenerate test configuration"
    q_idx, leaf_idx = query_rtree_pairs(tree, qmins, qmaxs, mode=mode)
    assert _pairs_to_set(q_idx, leaf_idx) == _expected_pairs(expected)
    assert _pairs_to_set(*tree.query_pairs(qmins, qmaxs, mode=mode)) == _expected_pairs(expected)


@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
def test_point_queries(ndim):
    gen = torch.Generator().manual_seed(5)
    mins, maxs = _random_boxes(2000, ndim, gen=gen, size=0.1)
    tree = build_rtree(mins, maxs)
    pts = torch.rand((200, ndim), generator=gen)
    expected = _brute(mins, maxs, pts, pts)
    assert _pairs_to_set(*query_rtree_points(tree, pts)) == _expected_pairs(expected)
    assert _pairs_to_set(*tree.query_points(pts)) == _expected_pairs(expected)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("curve", ["morton", "hilbert"])
def test_dtypes_and_curves(dtype, curve):
    gen = torch.Generator().manual_seed(21)
    mins, maxs = _random_boxes(1500, 3, gen=gen, dtype=dtype)
    tree = build_rtree(mins, maxs, curve=curve)
    assert tree.dtype == dtype
    qmins, qmaxs = _random_boxes(50, 3, gen=gen, size=0.2, dtype=dtype)
    expected = _brute(mins, maxs, qmins, qmaxs)
    assert _pairs_to_set(*query_rtree_pairs(tree, qmins, qmaxs)) == _expected_pairs(expected)


def test_float64_keeps_timestamp_resolution():
    # Unix-epoch seconds in float32 have ~128 s resolution; float64 must not merge these.
    t0 = 1_700_000_000.0
    mins = torch.tensor([[0.0, 0.0, 0.0, t0 + i] for i in range(10)], dtype=torch.float64)
    maxs = mins + torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float64)
    tree = build_rtree(mins, maxs)
    q = torch.tensor([[0.0, 0.0, 0.0, t0 + 3.2]], dtype=torch.float64)
    q_idx, leaf_idx = query_rtree_pairs(tree, q, q + torch.tensor([1.0, 1.0, 1.0, 0.1], dtype=torch.float64))
    assert leaf_idx.tolist() == [3]


# -----------------------------------------------------------------------------
# chunking / memory guard / edge cases
# -----------------------------------------------------------------------------

def test_chunk_size_is_transparent():
    gen = torch.Generator().manual_seed(8)
    mins, maxs = _random_boxes(3000, 2, gen=gen)
    tree = build_rtree(mins, maxs)
    qmins, qmaxs = _random_boxes(257, 2, gen=gen, size=0.2)
    full = query_rtree_pairs(tree, qmins, qmaxs)
    for cs in (1, 100, 256, 257, 10_000):
        chunked = query_rtree_pairs(tree, qmins, qmaxs, chunk_size=cs)
        assert torch.equal(full[0], chunked[0]) and torch.equal(full[1], chunked[1])
    r1, c1 = query_rtree(tree, qmins, qmaxs)
    r2, c2 = query_rtree(tree, qmins, qmaxs, chunk_size=50)
    assert torch.equal(r1, r2) and torch.equal(c1, c2)


def test_max_pairs_guard():
    gen = torch.Generator().manual_seed(8)
    mins, maxs = _random_boxes(3000, 2, gen=gen)
    tree = build_rtree(mins, maxs)
    everything = (torch.zeros((100, 2)), torch.ones((100, 2)) * 2)
    with pytest.raises(RuntimeError, match="max_pairs"):
        query_rtree_pairs(tree, *everything, max_pairs=1000)
    q_idx, _ = query_rtree_pairs(tree, *everything, max_pairs=10_000_000)
    assert q_idx.numel() == 100 * 3000


def test_max_results_truncation():
    gen = torch.Generator().manual_seed(3)
    mins, maxs = _random_boxes(200, 2, gen=gen)
    tree = build_rtree(mins, maxs)
    results, counts = query_rtree(tree, torch.zeros((1, 2)), torch.ones((1, 2)) * 2, max_results=10)
    assert results.shape == (1, 10) and (results >= 0).all()
    assert counts.item() == 200  # count is total hits, not truncated


def test_empty_tree_and_empty_queries():
    empty = torch.empty((0, 3))
    tree = build_rtree(empty, empty)
    assert tree.num_leaves == 0 and tree.num_nodes == 0
    far = torch.zeros((5, 3))
    results, counts = query_rtree(tree, far, far + 1)
    assert results.shape == (5, 1) and (results == -1).all() and (counts == 0).all()
    q_idx, leaf_idx = query_rtree_pairs(tree, far, far + 1)
    assert q_idx.numel() == 0 == leaf_idx.numel()
    idx, dist = query_rtree_nearest(tree, far, k=2)
    assert (idx == -1).all() and torch.isinf(dist).all()

    gen = torch.Generator().manual_seed(11)
    mins, maxs = _random_boxes(100, 3, gen=gen)
    tree = build_rtree(mins, maxs)
    q_idx, leaf_idx = query_rtree_pairs(tree, empty, empty)
    assert q_idx.numel() == 0
    results, counts = query_rtree(tree, torch.full((5, 3), 10.0), torch.full((5, 3), 11.0))
    assert results.shape == (5, 1) and (results == -1).all() and (counts == 0).all()


@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
def test_tree_invariants(ndim):
    gen = torch.Generator().manual_seed(7)
    mins, maxs = _random_boxes(1000, ndim, gen=gen)
    tree = build_rtree(mins, maxs, m=8)
    assert sorted(tree.leaf_order.tolist()) == list(range(1000))
    leaves = slice(tree.leaf_start, tree.leaf_start + tree.num_leaves)
    assert torch.equal(tree.mins[leaves], mins[tree.leaf_order])
    assert torch.equal(tree.maxs[leaves], maxs[tree.leaf_order])
    for node in range(tree.leaf_start):
        s, e = tree.start_indices[node].item(), tree.end_indices[node].item()
        assert 0 <= s <= e < tree.num_nodes
        assert e - s + 1 <= tree.m
        assert (tree.mins[node] <= tree.mins[s : e + 1]).all()
        assert (tree.maxs[node] >= tree.maxs[s : e + 1]).all()


# -----------------------------------------------------------------------------
# validation
# -----------------------------------------------------------------------------

def test_validation_errors():
    good = torch.rand((10, 2))
    with pytest.raises(NotImplementedError):
        build_rtree(torch.rand((10, 5)), torch.rand((10, 5)) + 1)
    with pytest.raises(ValueError, match="min must be <="):
        build_rtree(good + 1, good)
    bad = good.clone()
    bad[3, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_rtree(bad, bad + 1)
    with pytest.raises(ValueError, match="shape"):
        build_rtree(torch.rand(10), torch.rand(10))
    with pytest.raises(TypeError, match="dtype"):
        build_rtree(good, (good + 1).double())
    with pytest.raises(TypeError, match="dtype must be"):
        build_rtree(good.half(), (good + 1).half())
    with pytest.raises(ValueError, match="m must be"):
        build_rtree(good, good + 1, m=1)
    with pytest.raises(ValueError, match="curve"):
        build_rtree(good, good + 1, curve="peano")
    tree = build_rtree(good, good + 1)
    with pytest.raises(ValueError, match="ndim=2"):
        query_rtree_pairs(tree, torch.rand((3, 3)), torch.rand((3, 3)) + 1)
    with pytest.raises(ValueError, match="mode"):
        query_rtree_pairs(tree, good, good + 1, mode="touches")
    with pytest.raises(ValueError, match="k must be"):
        query_rtree_nearest(tree, good, k=0)


# -----------------------------------------------------------------------------
# nearest
# -----------------------------------------------------------------------------

def _brute_nearest(mins, maxs, pts, k):
    below = (mins.unsqueeze(0) - pts.unsqueeze(1)).clamp_min(0)
    above = (pts.unsqueeze(1) - maxs.unsqueeze(0)).clamp_min(0)
    d = (below * below + above * above).sum(-1).sqrt()  # (Q, N)
    return torch.topk(d, k, dim=1, largest=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
@pytest.mark.parametrize("k", [1, 5])
def test_nearest_matches_brute_force(device, ndim, k):
    gen = torch.Generator().manual_seed(31 + ndim + k)
    mins, maxs = _random_boxes(2000, ndim, device, gen, size=0.01)
    tree = build_rtree(mins, maxs)
    # Points inside the extent and some well outside it.
    inside = torch.rand((100, ndim), generator=gen)
    outside = torch.rand((10, ndim), generator=gen) * 4 - 2
    pts = torch.cat([inside, outside]).to(device)
    idx, dist = query_rtree_nearest(tree, pts, k=k)
    exp_d, exp_i = _brute_nearest(mins, maxs, pts, k)
    assert torch.allclose(dist, exp_d, atol=1e-6)
    # Indices may legitimately differ on exact ties; compare distances at the
    # returned indices instead.
    assert (idx >= 0).all()
    assert torch.allclose(_point_box_dist(mins, maxs, pts, idx), exp_d, atol=1e-6)


def _point_box_dist(mins, maxs, pts, idx):
    m, M = mins[idx], maxs[idx]  # (Q, k, ndim)
    p = pts.unsqueeze(1)
    below = (m - p).clamp_min(0)
    above = (p - M).clamp_min(0)
    return (below * below + above * above).sum(-1).sqrt()


def test_nearest_k_exceeds_n_and_chunking():
    gen = torch.Generator().manual_seed(2)
    mins, maxs = _random_boxes(3, 2, gen=gen)
    tree = build_rtree(mins, maxs)
    pts = torch.rand((4, 2), generator=gen)
    idx, dist = query_rtree_nearest(tree, pts, k=5)
    assert (idx[:, 3:] == -1).all() and torch.isinf(dist[:, 3:]).all()
    assert (idx[:, :3] >= 0).all()
    exp_d, _ = _brute_nearest(mins, maxs, pts, 3)
    assert torch.allclose(dist[:, :3], exp_d, atol=1e-6)

    mins, maxs = _random_boxes(500, 2, gen=gen)
    tree = build_rtree(mins, maxs)
    pts = torch.rand((77, 2), generator=gen)
    a = tree.nearest(pts, k=3)
    b = tree.nearest(pts, k=3, chunk_size=10)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# -----------------------------------------------------------------------------
# container: to / save / load
# -----------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    gen = torch.Generator().manual_seed(1)
    mins, maxs = _random_boxes(300, 2, gen=gen)
    tree = build_rtree(mins, maxs)
    path = tmp_path / "tree.pt"
    tree.save(path)
    loaded = RTree.load(path)
    for a, b in zip(tree, loaded, strict=True):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        else:
            assert a == b
    qmins, qmaxs = _random_boxes(20, 2, gen=gen, size=0.2)
    assert torch.equal(tree.query(qmins, qmaxs)[0], loaded.query(qmins, qmaxs)[0])
    with pytest.raises(ValueError, match="format"):
        RTree.from_state_dict({"format": 99})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_to_device_roundtrip():
    gen = torch.Generator().manual_seed(1)
    mins, maxs = _random_boxes(300, 3, gen=gen)
    tree = build_rtree(mins, maxs)
    gpu = tree.to("cuda")
    assert gpu.device.type == "cuda" and tree.to("cpu") is tree
    qmins, qmaxs = _random_boxes(20, 3, gen=gen, size=0.2)
    a = tree.query_pairs(qmins, qmaxs)
    b = gpu.query_pairs(qmins, qmaxs)  # CPU queries are moved to the tree's device
    assert b[0].device.type == "cuda"
    assert _pairs_to_set(*a) == _pairs_to_set(b[0].cpu(), b[1].cpu())


# -----------------------------------------------------------------------------
# curves
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
def test_sfc_key_fits_int64_and_is_monotone_on_axis0(ndim):
    n = 256
    centers = torch.zeros((n, ndim))
    centers[:, 0] = torch.linspace(0, 1, n)
    for curve in ("morton", "hilbert"):
        codes = _sfc_codes(centers, torch.zeros(ndim), torch.ones(ndim), curve)
        assert codes.dtype == torch.int64 and (codes >= 0).all()
        assert len(set(codes.tolist())) == n, "distinct centres must get distinct keys"
    # Monotone along one axis is a Morton property (Hilbert folds back on itself).
    codes = _sfc_codes(centers, torch.zeros(ndim), torch.ones(ndim), "morton")
    assert (codes[1:] >= codes[:-1]).all()


@pytest.mark.parametrize("ndim", SUPPORTED_NDIMS)
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_hilbert_is_a_continuous_space_filling_curve(ndim, bits):
    # Every cell of the 2^bits grid must get a unique key, and consecutive keys
    # must be grid neighbours (Manhattan distance 1) — the Hilbert property.
    side = 1 << bits
    grid = torch.cartesian_prod(*[torch.arange(side)] * ndim).reshape(-1, ndim)
    keys = _hilbert_codes(grid, bits)
    assert sorted(keys.tolist()) == list(range(side ** ndim))
    walk = grid[torch.argsort(keys)]
    assert ((walk[1:] - walk[:-1]).abs().sum(-1) == 1).all()
