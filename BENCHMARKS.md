# Benchmarks

All numbers below were measured on one machine:

- GPU: **AMD Radeon RX 7900 XT** (20 GB), ROCm 7.2, torch 2.9.1+rocm7.2.1
- CPU: single process, torch CPU backend, 61 GB RAM
- Python 3.12; libspatialindex via the `rtree` 1.4.1 package, bulk-loaded through a Python generator and
  queried one call at a time from Python (the conventional CPU path)

Every torch row is validated against libspatialindex: box queries by exact per-query hit sets, k-nearest by
distance. A speedup is never reported for a wrong answer. GPU timings include a device sync. Reproduce any table
with `python -m benchmarks.bench_rtree ...`; the CSVs land next to the script.

Throughput is queries per second for one batched call. The GPU only pulls ahead of the CPU once a call carries
enough queries to fill it (a few thousand); with 1,000 queries per call both devices are latency-bound on the
per-level sync and land close together.

## Scaling: 1M, 10M and 100M points

Fan-out 8, Hilbert ordering, uniform data with a 1000x wider last axis, k = 5. Query box side and query count were
reduced as N grew to keep 2D output volume sane: 1M uses 10k queries of side 0.05 over boxes of side 0.01; 10M uses
1k queries of side 0.02 over points; 100M uses 1k queries of side 0.01 over points.

Build time:

| ndim | N | libspatialindex | torch CPU | torch GPU |
|---|---|---|---|---|
| 2 | 1M | 2.6 s | 0.11 s | 0.014 s |
| 2 | 10M | 27.7 s | 3.6 s | 0.25 s |
| 2 | 100M | 311 s | 45 s | 2.5 s |
| 3 | 100M | 359 s | 47 s | 3.0 s |
| 4 | 100M | 538 s | 47 s | 11.8 s |

Box intersection, queries per second:

| ndim | N | queries / call | libspatialindex | torch CPU | torch GPU |
|---|---|---|---|---|---|
| 2 | 1M | 10k | 6,800 | 61,800 | 665,000 |
| 3 | 1M | 10k | 25,600 | 470,000 | 2,009,000 |
| 4 | 1M | 10k | 26,400 | 825,000 | 1,868,000 |
| 2 | 10M | 1k | 6,200 | 98,000 | 196,000 |
| 3 | 10M | 1k | 26,900 | 255,000 | 244,000 |
| 4 | 10M | 1k | 21,900 | 289,000 | 197,000 |
| 2 | 100M | 1k | 2,500 | 29,400 | 138,000 |
| 3 | 100M | 1k | 15,000 | 212,000 | 180,000 |
| 4 | 100M | 1k | 15,100 | 256,000 | 181,000 |

k-nearest (k = 5), queries per second:

| ndim | N | queries / call | libspatialindex | torch CPU | torch GPU |
|---|---|---|---|---|---|
| 2 | 1M | 10k | 26,500 | 17,400 | 556,000 |
| 3 | 1M | 10k | 15,800 | 31,500 | 419,000 |
| 4 | 1M | 10k | 3,300 | 29,300 | 212,000 |
| 2 | 10M | 1k | 13,700 | 75,000 | 59,700 |
| 3 | 10M | 1k | 1,800 | 25,800 | 43,100 |
| 4 | 10M | 1k | 270 | 8,900 | 29,000 |
| 2 | 100M | 1k | 12,600 | 70,400 | 38,100 |
| 3 | 100M | 1k | 394 | 22,900 | 64,500 |
| 4 | 100M | 1k | 93 | 7,800 | 24,000 |

Observations:

- Per-query cost is flat from 1M to 100M. The tree gains one level per 8x growth; the 2D drop at 100M is output
  volume (each query returns ~10x more points).
- The 4D build at 100M ran at the edge of the 20 GB card. Curve keys are now computed in fixed-size chunks so the
  peak no longer grows with N; the 11.8 s figure predates that change.
- libspatialindex k-nearest collapses with dimension at scale (93 queries/s in 4D at 100M); the expanding-cube
  search stays within a factor of three across dimensions.

## Large query batches and automatic batching

Queries are batched automatically to `tree.pairs_budget` (default 2^24 pairs). A first batch of 16k queries
measures the peak (query, node) pairs per query; later batches are sized to the budget. Measured on the GPU with a
1M-box 3D tree, box queries of side 0.02, `validate=False`; "single" forces one batch via `chunk_size=Q`.

| queries / call | hits | auto-batched | single batch | overhead | k-nearest (k = 5) |
|---|---|---|---|---|---|
| 100k | 331k | 16 ms (6.1M q/s) | 10 ms | 1.6x (6 ms absolute) | 135 ms |
| 500k | 1.66M | 52 ms (9.7M q/s) | 47 ms | 9% | 690 ms |
| 2M | 6.64M | 191 ms (10.5M q/s) | 182 ms | 5% | 2.85 s |

The fixed cost is the small first batch, which is latency-bound; it vanishes as calls grow. On the CPU the batched
form was equal or faster at every size (better cache locality). k-nearest also batches its distance and sort stage:
before that change the 2M-query call peaked at 23 GB and took 11 s; it now peaks at 1.8 GB.

## Fan-out and curve (1M boxes, 10k queries)

Box intersection, torch GPU, queries per second:

| ndim | m=8 Morton | m=8 Hilbert | m=16 Hilbert | m=32 Hilbert |
|---|---|---|---|---|
| 2 | 763,000 | 719,000 | 817,000 | 798,000 |
| 3 | 1,552,000 | 1,967,000 | 2,051,000 | 1,885,000 |
| 4 | 1,192,000 | 1,712,000 | 1,658,000 | 1,341,000 |

Box intersection, torch CPU, queries per second:

| ndim | m=8 Morton | m=8 Hilbert | m=16 Hilbert | m=32 Hilbert |
|---|---|---|---|---|
| 2 | 60,900 | 62,600 | 61,900 | 57,200 |
| 3 | 303,000 | 409,000 | 287,000 | 209,000 |
| 4 | 302,000 | 628,000 | 486,000 | 299,000 |

Hilbert ordering queries 20 to 50 percent faster than Morton in 3D and 4D and builds about 1.6x slower
(0.14 s vs 0.085 s per 1M on CPU; 0.015 s vs 0.009 s on GPU). Fan-out 8 is the best or within noise of the best
everywhere on CPU; 16 is marginally ahead on GPU in 2D and 3D. The defaults are fan-out 8 and Hilbert.

## Clustered data (100k boxes, 10k queries)

Centres drawn from 50 Gaussian blobs (sigma 0.02) instead of uniform. Queries per second:

| ndim | kind | libspatialindex | torch CPU | torch GPU |
|---|---|---|---|---|
| 2 | box | 31,100 | 652,000 | 1,708,000 |
| 3 | box | 73,100 | 1,735,000 | 2,204,000 |
| 4 | box | 93,200 | 2,826,000 | 1,885,000 |
| 2 | knn | 26,200 | 45,500 | 379,000 |
| 3 | knn | 5,800 | 38,800 | 497,000 |
| 4 | knn | 2,600 | 24,600 | 222,000 |

## Curve key width

The Morton / Hilbert key uses floor(63 / ndim) bits per axis (31 in 2D, 15 in 4D). Measured on 1M boxes, shrinking
the width had no effect on query time until about 6 bits per axis, so wider (128-bit) keys were not pursued.

| ndim | bits per axis | GPU query time (10k queries) |
|---|---|---|
| 4 | 15 | 7.9 ms |
| 4 | 10 | 7.9 ms |
| 4 | 6 | 7.6 ms |
| 4 | 3 | 14.0 ms |
| 2 | 31 | 13.9 ms |
| 2 | 10 | 12.7 ms |
