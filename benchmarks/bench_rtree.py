"""
Throughput benchmark for the torch R-tree against CPU baselines.

Backends
  libspatialindex  rtree.index.Index (bulk-loaded via generator), one
                   `intersection()` call per query - the conventional CPU path.
  torch-cpu        torchrtree on the CPU device.
  torch-cuda       torchrtree on the GPU (CUDA / ROCm), if available.
  brute-cpu/cuda   chunked all-pairs box test - a tree-free reference point.

Every backend's per-query hit counts are cross-checked against libspatialindex
so a speedup is never reported for a wrong answer. With --knn, k-nearest
queries are benchmarked too and checked by distance.

Usage
  python -m benchmarks.bench_rtree                          # default sweep
  python -m benchmarks.bench_rtree --n 100000 1000000 --q 10000 --ndim 2 3 4
  python -m benchmarks.bench_rtree --m 8 16 32 --curve morton hilbert
  python -m benchmarks.bench_rtree --dist clustered --knn 5 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import torch

from torchrtree import RTree

try:
    from rtree import index as _lsi_index
except ImportError:  # pragma: no cover
    _lsi_index = None


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------

def make_boxes(n: int, ndim: int, size: float, seed: int, time_extent: float = 1.0,
               dist: str = "uniform"):
    """
    Random axis-aligned boxes. Each edge is U(0, size). The last axis is scaled
    by `time_extent` so ndim=4 mimics a time axis in different units.

    dist="uniform":   centres uniform in the unit cube.
    dist="clustered": centres drawn from 50 Gaussian blobs (sigma 0.02) - the
                      shape real spatial data usually has.
    Returns float32 numpy (mins, maxs).
    """
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        lo = rng.random((n, ndim), dtype=np.float32)
    elif dist == "clustered":
        centres = rng.random((50, ndim))
        which = rng.integers(0, 50, n)
        lo = (centres[which] + rng.normal(0.0, 0.02, (n, ndim))).clip(0, 1).astype(np.float32)
    else:
        raise ValueError(f"unknown dist {dist!r}")
    hi = lo + (size * rng.random((n, ndim), dtype=np.float32)).astype(np.float32)
    scale = np.ones(ndim, dtype=np.float32)
    scale[-1] = time_extent
    return lo * scale, hi * scale


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------

class _Timer:
    def __init__(self, device: str | None = None):
        self.device = device

    def __enter__(self):
        if self.device and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0


@dataclass
class Result:
    backend: str
    kind: str           # "box" or "knn"
    ndim: int
    n: int
    q: int
    m: int
    curve: str
    dist: str
    build_s: float
    query_s: float
    counts_ok: bool | None  # None for the reference backend

    @property
    def queries_per_s(self) -> float:
        return self.q / self.query_s if self.query_s > 0 else float("inf")


# -----------------------------------------------------------------------------
# Box-query backends. Each returns (build_s, query_s, counts: np.int64[Q]).
# -----------------------------------------------------------------------------

def _lsi_build(mins, maxs):
    if _lsi_index is None:
        raise RuntimeError("the `rtree` package (libspatialindex) is not installed")
    n, ndim = mins.shape
    props = _lsi_index.Property()
    props.dimension = ndim
    # libspatialindex stores doubles; float32 -> float64 is exact so hit sets
    # are directly comparable with the torch backends.
    lo, hi = mins.astype(np.float64), maxs.astype(np.float64)

    def stream():
        for i in range(n):
            yield (i, (*lo[i], *hi[i]), None)

    with _Timer() as tb:
        idx = _lsi_index.Index(stream(), properties=props)
    return idx, tb.seconds


def run_libspatialindex(mins, maxs, qmins, qmaxs):
    idx, build_s = _lsi_build(mins, maxs)
    qlo, qhi = qmins.astype(np.float64), qmaxs.astype(np.float64)
    counts = np.empty(qmins.shape[0], dtype=np.int64)
    with _Timer() as tq:
        for j in range(qmins.shape[0]):
            counts[j] = len(list(idx.intersection((*qlo[j], *qhi[j]))))
    return build_s, tq.seconds, counts


def _to(device, *arrays):
    return [torch.from_numpy(a).to(device) for a in arrays]


def run_torch_tree(mins, maxs, qmins, qmaxs, device: str, m: int, curve: str):
    t_mins, t_maxs, t_qmins, t_qmaxs = _to(device, mins, maxs, qmins, qmaxs)
    # Untimed warm-up at full size: first-call kernel compilation and allocator
    # growth otherwise land in the timed build.
    RTree(t_mins, t_maxs, fanout=m, curve=curve).search(t_qmins, t_qmaxs)

    with _Timer(device) as tb:
        tree = RTree(t_mins, t_maxs, fanout=m, curve=curve)
    with _Timer(device) as tq:
        res = tree.search(t_qmins, t_qmaxs)
    return tb.seconds, tq.seconds, res.counts.cpu().numpy()


def run_brute_force(mins, maxs, qmins, qmaxs, device: str, chunk: int = 512):
    t_mins, t_maxs, t_qmins, t_qmaxs = _to(device, mins, maxs, qmins, qmaxs)
    out = []
    with _Timer(device) as tq:
        for s in range(0, qmins.shape[0], chunk):
            ql, qh = t_qmins[s:s + chunk], t_qmaxs[s:s + chunk]
            hit = (t_mins.unsqueeze(0) <= qh.unsqueeze(1)).all(-1) & (
                t_maxs.unsqueeze(0) >= ql.unsqueeze(1)
            ).all(-1)
            out.append(hit.sum(-1))
    return 0.0, tq.seconds, torch.cat(out).cpu().numpy()


# -----------------------------------------------------------------------------
# k-NN backends. Each returns (query_s, dists: float64[Q, k]) sorted ascending.
# -----------------------------------------------------------------------------

def _point_box_dist(mins, maxs, pts, idx):
    m, M = mins[idx], maxs[idx]                        # (Q, k, ndim)
    p = pts[:, None, :]
    below = np.clip(m - p, 0, None)
    above = np.clip(p - M, 0, None)
    return np.sqrt((below * below + above * above).sum(-1))


def run_libspatialindex_knn(idx, mins, maxs, pts, k):
    out = np.full((pts.shape[0], k), -1, dtype=np.int64)
    p64 = pts.astype(np.float64)
    with _Timer() as tq:
        for j in range(pts.shape[0]):
            ids = list(idx.nearest((*p64[j], *p64[j]), k))[:k]  # may return extra on ties
            out[j, :len(ids)] = ids
    return tq.seconds, np.sort(_point_box_dist(mins, maxs, pts, out), axis=1)


def run_torch_knn(mins, maxs, pts, device, m, curve, k):
    t_mins, t_maxs, t_pts = _to(device, mins, maxs, pts)
    tree = RTree(t_mins, t_maxs, fanout=m, curve=curve)
    tree.nearest(t_pts, k=k)  # warm-up
    with _Timer(device) as tq:
        dist, _ = tree.nearest(t_pts, k=k)
    return tq.seconds, dist.cpu().numpy().astype(np.float64)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def available_devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def run_config(
    n: int,
    q: int,
    ndim: int,
    box_size: float = 0.01,
    query_size: float = 0.05,
    seed: int = 0,
    devices: Iterable[str] | None = None,
    brute: bool = True,
    m: int = 8,
    curve: str = "hilbert",
    dist: str = "uniform",
    knn: int = 0,
) -> list[Result]:
    devices = list(devices) if devices is not None else available_devices()
    mins, maxs = make_boxes(n, ndim, box_size, seed, time_extent=1000.0, dist=dist)
    qmins, qmaxs = make_boxes(q, ndim, query_size, seed + 1, time_extent=1000.0, dist=dist)
    tag = dict(ndim=ndim, n=n, q=q, m=m, curve=curve, dist=dist)

    results: list[Result] = []
    b, s, ref = run_libspatialindex(mins, maxs, qmins, qmaxs)
    results.append(Result("libspatialindex", "box", **tag, build_s=b, query_s=s, counts_ok=None))

    for dev in devices:
        b, s, c = run_torch_tree(mins, maxs, qmins, qmaxs, dev, m, curve)
        results.append(Result(f"torch-{dev}", "box", **tag, build_s=b, query_s=s,
                              counts_ok=bool(np.array_equal(c, ref))))
        if brute:
            b, s, c = run_brute_force(mins, maxs, qmins, qmaxs, dev)
            results.append(Result(f"brute-{dev}", "box", **tag, build_s=b, query_s=s,
                                  counts_ok=bool(np.array_equal(c, ref))))

    if knn > 0:
        pts = qmins  # query box corners double as query points
        lsi, _ = _lsi_build(mins, maxs)
        s, ref_d = run_libspatialindex_knn(lsi, mins, maxs, pts, knn)
        results.append(Result("libspatialindex", "knn", **tag, build_s=0.0, query_s=s, counts_ok=None))
        for dev in devices:
            s, d = run_torch_knn(mins, maxs, pts, dev, m, curve, knn)
            results.append(Result(f"torch-{dev}", "knn", **tag, build_s=0.0, query_s=s,
                                  counts_ok=bool(np.allclose(d, ref_d, rtol=1e-5, atol=1e-6))))
    return results


def format_table(rows: list[Result]) -> str:
    hdr = (f"{'backend':<16}{'kind':<5}{'ndim':>5}{'N':>9}{'Q':>7}{'m':>4}{'curve':>9}{'dist':>10}"
           f"{'build s':>10}{'query s':>10}{'q/s':>12}{'ok':>5}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        ok = "ref" if r.counts_ok is None else ("yes" if r.counts_ok else "NO")
        lines.append(
            f"{r.backend:<16}{r.kind:<5}{r.ndim:>5}{r.n:>9}{r.q:>7}{r.m:>4}{r.curve:>9}{r.dist:>10}"
            f"{r.build_s:>10.4f}{r.query_s:>10.4f}{r.queries_per_s:>12.0f}{ok:>5}"
        )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000], help="number of indexed boxes")
    p.add_argument("--q", type=int, nargs="+", default=[1_000], help="number of queries")
    p.add_argument("--ndim", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--m", type=int, nargs="+", default=[8], help="fan-out values to sweep")
    p.add_argument("--curve", nargs="+", default=["hilbert"], choices=["morton", "hilbert"])
    p.add_argument("--dist", nargs="+", default=["uniform"], choices=["uniform", "clustered"])
    p.add_argument("--knn", type=int, default=0, help="also benchmark k-nearest with this k")
    p.add_argument("--box-size", type=float, default=0.01)
    p.add_argument("--query-size", type=float, default=0.05)
    p.add_argument("--no-brute", action="store_true", help="skip the brute-force reference")
    p.add_argument("--devices", nargs="+", default=None, help="torch devices (default: cpu [+ cuda])")
    p.add_argument("--csv", type=str, default=None, help="append rows to this CSV file")
    args = p.parse_args(argv)

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    all_rows: list[Result] = []
    for dist in args.dist:
        for curve in args.curve:
            for m in args.m:
                for ndim in args.ndim:
                    for n in args.n:
                        for q in args.q:
                            rows = run_config(n, q, ndim, args.box_size, args.query_size,
                                              devices=args.devices, brute=not args.no_brute,
                                              m=m, curve=curve, dist=dist, knn=args.knn)
                            all_rows.extend(rows)
                            print(format_table(rows), "\n", flush=True)

    bad = [r for r in all_rows if r.counts_ok is False]
    if args.csv:
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()) + ["queries_per_s"])
            if f.tell() == 0:
                w.writeheader()
            for r in all_rows:
                w.writerow({**asdict(r), "queries_per_s": r.queries_per_s})
    if bad:
        print("MISMATCH against libspatialindex:",
              [(r.backend, r.kind, r.ndim, r.n) for r in bad], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
