# Benchmarks

All numbers below come from one sequential run (2026-09-18, about an hour) on one machine:

- GPU: **AMD Radeon RX 7900 XT** (20 GB), ROCm 7.2, torch 2.9.1+rocm7.2.1
- CPU: single process, torch CPU backend, 61 GB RAM
- Python 3.12; libspatialindex via the `rtree` 1.4.1 package, bulk-loaded through a Python generator and
  queried one call at a time from Python (the conventional CPU path)

Validation: with libspatialindex as the reference, every torch row's per-query hit sets were compared exactly and
k-nearest by distance. For the large query batches (100k and 1M queries per call) libspatialindex would take hours,
so torch CPU is the reference there and the GPU is validated against it the same way. No row in this file
disagreed with its reference. GPU timings include a device sync. Reproduce any table with
`python -m benchmarks.bench_rtree ...`; the exact commands are at the end.

Throughput is queries per second for one batched call, k = 5 for k-nearest. Fan-out 8 and Hilbert ordering
unless stated. Data is uniform with a 1000x wider last axis (a time axis).

## Scaling: 1M boxes, 10M and 100M points

Query box side shrinks with N to keep the 2D output volume sane: 1M uses boxes of side 0.01 and queries of side
0.05; 10M uses points and queries of side 0.02; 100M uses points and queries of side 0.01.

Build time:

| ndim | N | libspatialindex | torch CPU | torch GPU |
| --- | --- | --- | --- | --- |
| 2 | 1M | 2.6 s | 0.12 s | 0.014 s |
| 3 | 1M | 2.8 s | 0.10 s | 0.015 s |
| 4 | 1M | 3.2 s | 0.11 s | 0.014 s |
| 2 | 10M | 27 s | 2.2 s | 0.16 s |
| 3 | 10M | 29 s | 2.5 s | 0.19 s |
| 4 | 10M | 35 s | 2.6 s | 0.20 s |
| 2 | 100M | 309 s | 26 s | 1.8 s |
| 3 | 100M | 351 s | 30 s | 2.0 s |
| 4 | 100M | 478 s | 30 s | 2.4 s |

Box intersection, 1k queries per call at 10M and 100M, 10k at 1M:

| ndim | N | libspatialindex | torch CPU | torch GPU |
| --- | --- | --- | --- | --- |
| 2 | 1M | 6,300 | 65,100 | 657,000 |
| 3 | 1M | 25,800 | 424,000 | 2,095,000 |
| 4 | 1M | 26,500 | 1,091,000 | 1,753,000 |
| 2 | 10M | 5,800 | 64,800 | 198,000 |
| 3 | 10M | 26,800 | 259,000 | 249,000 |
| 4 | 10M | 23,400 | 407,000 | 212,000 |
| 2 | 100M | 2,700 | 31,700 | 151,000 |
| 3 | 100M | 18,100 | 198,000 | 176,000 |
| 4 | 100M | 16,100 | 376,000 | 199,000 |

k-nearest, same batch sizes:

| ndim | N | libspatialindex | torch CPU | torch GPU |
| --- | --- | --- | --- | --- |
| 2 | 1M | 26,300 | 13,900 | 534,000 |
| 3 | 1M | 16,500 | 27,400 | 396,000 |
| 4 | 1M | 3,200 | 32,100 | 209,000 |
| 2 | 10M | 14,400 | 113,000 | 73,500 |
| 3 | 10M | 1,900 | 30,500 | 43,300 |
| 4 | 10M | 290 | 9,700 | 33,800 |
| 2 | 100M | 13,200 | 98,200 | 44,700 |
| 3 | 100M | 1,100 | 23,100 | 61,700 |
| 4 | 100M | 120 | 8,500 | 23,100 |

With only 1k queries per call both torch devices are latency-bound on the per-level sync, which is why the GPU
does not separate from the CPU in the 10M and 100M rows. The next section shows what happens with bigger calls.

## Large query batches: 100k and 1M queries per call

Queries are batched automatically to `tree.pairs_budget` (default 2^24 pairs): a first batch of 16k queries
measures the peak (query, node) pairs per query, and later batches are sized to the budget. Query box side: 0.005
at 1M boxes, 0.003 at 10M points, 0.001 at 100M points. GPU validated against torch CPU by exact hit sets.

Box intersection, queries per second:

| ndim | N | 100k / call, CPU | 100k / call, GPU | 1M / call, CPU | 1M / call, GPU |
| --- | --- | --- | --- | --- | --- |
| 2 | 1M | 494,000 | 4,215,000 | 443,000 | 5,565,000 |
| 3 | 1M | 1,649,000 | 8,976,000 | 1,258,000 | 13,535,000 |
| 4 | 1M | 1,612,000 | 5,086,000 | 1,198,000 | 6,097,000 |
| 2 | 10M | 883,000 | 6,804,000 | 766,000 | 9,846,000 |
| 3 | 10M | 1,381,000 | 8,277,000 | 1,235,000 | 14,516,000 |
| 4 | 10M | 1,375,000 | 4,551,000 | 1,017,000 | 5,714,000 |
| 2 | 100M | 587,000 | 5,919,000 | 615,000 | 8,731,000 |
| 3 | 100M | 973,000 | 6,066,000 | 874,000 | 12,262,000 |
| 4 | 100M | 974,000 | 3,237,000 | 886,000 | 3,922,000 |

k-nearest, queries per second:

| ndim | N | 100k / call, CPU | 100k / call, GPU | 1M / call, CPU | 1M / call, GPU |
| --- | --- | --- | --- | --- | --- |
| 2 | 1M | 13,700 | 564,000 | 13,600 | 583,000 |
| 3 | 1M | 26,200 | 696,000 | 25,300 | 732,000 |
| 4 | 1M | 27,800 | 233,000 | 26,700 | 240,000 |
| 2 | 10M | 110,000 | 1,213,000 | 105,000 | 1,760,000 |
| 3 | 10M | 30,700 | 376,000 | 26,000 | 381,000 |
| 4 | 10M | 10,000 | 53,000 | 9,200 | 54,700 |
| 2 | 100M | 101,000 | 1,115,000 | 92,800 | 1,637,000 |
| 3 | 100M | 22,900 | 331,000 | 22,400 | 346,000 |
| 4 | 100M | 8,300 | 28,800 | 7,200 | 33,000 |

Observations:

- A 1M-query call against 100M points answers 12M box queries per second in 3D on the GPU; the same call takes
  1.1 s on the CPU. Per-query cost is flat from 1M to 100M points at equal batch size.
- The GPU needs large calls to shine: from 1k to 1M queries per call its box throughput rises 40 to 70x, the CPU's
  about 4x. k-nearest on the GPU gains 10 to 25x.
- 4D k-nearest is the slowest path (the expanding cube has to grow through the wide time axis); it is still
  60 to 270x libspatialindex at 100M.
- Auto-batching overhead against a single hand-sized batch was measured at 5% (2M queries) to 9% (500k) on the GPU,
  and the batched form was equal or faster on the CPU. Peak GPU memory of a 2M-query k-nearest call is 1.8 GB.

## Fan-out and curve (1M boxes, 10k queries)

Box intersection, torch GPU, queries per second:

| ndim | m=8 Morton | m=8 Hilbert | m=16 Morton | m=16 Hilbert | m=32 Morton | m=32 Hilbert |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 562,000 | 652,000 | 548,000 | 698,000 | 521,000 | 740,000 |
| 3 | 1,366,000 | 1,867,000 | 1,454,000 | 1,979,000 | 1,133,000 | 1,809,000 |
| 4 | 1,160,000 | 1,739,000 | 967,000 | 1,788,000 | 687,000 | 1,306,000 |

Box intersection, torch CPU, queries per second:

| ndim | m=8 Morton | m=8 Hilbert | m=16 Morton | m=16 Hilbert | m=32 Morton | m=32 Hilbert |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 55,100 | 63,900 | 63,300 | 80,400 | 49,600 | 66,800 |
| 3 | 254,000 | 653,000 | 187,000 | 368,000 | 142,000 | 334,000 |
| 4 | 359,000 | 656,000 | 347,000 | 517,000 | 157,000 | 344,000 |

Build time per 1M boxes: Morton 0.08 s CPU / 0.008 s GPU; Hilbert 0.11 s CPU / 0.014 s GPU.

Hilbert queries 15 to 160 percent faster than Morton at the same fan-out and builds about 1.5x slower. Fan-out 8
is best on the CPU in 3D and 4D by a wide margin; on the GPU 16 is within a few percent of 8 and 32 is slower.
The defaults are fan-out 8 and Hilbert; pass `curve="morton"` when build time dominates.

## Clustered data (100k boxes, 10k queries)

Centres drawn from 50 Gaussian blobs (sigma 0.02) instead of uniform. Queries per second:

| ndim | kind | libspatialindex | torch CPU | torch GPU |
| --- | --- | --- | --- | --- |
| 2 | box | 30,400 | 433,000 | 1,313,000 |
| 3 | box | 73,700 | 1,770,000 | 2,370,000 |
| 4 | box | 93,000 | 3,444,000 | 1,968,000 |
| 2 | knn | 26,100 | 38,300 | 464,000 |
| 3 | knn | 5,800 | 35,400 | 492,000 |
| 4 | knn | 2,600 | 23,400 | 224,000 |

## Curve key width

The Morton / Hilbert key uses floor(63 / ndim) bits per axis (31 in 2D, 15 in 4D). Measured on 1M boxes,
shrinking the width had no effect on query time until about 6 bits per axis, so wider (128-bit) keys were not
pursued.

| ndim | bits per axis | GPU query time (10k queries) |
| --- | --- | --- |
| 4 | 15 | 7.9 ms |
| 4 | 10 | 7.9 ms |
| 4 | 6 | 7.6 ms |
| 4 | 3 | 14.0 ms |
| 2 | 31 | 13.9 ms |
| 2 | 10 | 12.7 ms |

## Commands

```bash
python -m benchmarks.bench_rtree --n 1000000 --q 10000 --ndim 2 3 4 --no-brute --knn 5
python -m benchmarks.bench_rtree --n 1000000 --q 10000 --ndim 2 3 4 --m 8 16 32 --curve morton hilbert --no-brute --knn 5
python -m benchmarks.bench_rtree --n 100000 --q 10000 --ndim 2 3 4 --dist clustered --no-brute --knn 5
python -m benchmarks.bench_rtree --n 1000000 --q 100000 1000000 --ndim 2 3 4 --query-size 0.005 --no-brute --no-ref --knn 5
python -m benchmarks.bench_rtree --n 10000000 --q 1000 --ndim 2 3 4 --box-size 0 --query-size 0.02 --no-brute --knn 5
python -m benchmarks.bench_rtree --n 10000000 --q 100000 1000000 --ndim 2 3 4 --box-size 0 --query-size 0.003 --no-brute --no-ref --knn 5
python -m benchmarks.bench_rtree --n 100000000 --q 1000 --ndim 2 3 4 --box-size 0 --query-size 0.01 --no-brute --knn 5
python -m benchmarks.bench_rtree --n 100000000 --q 100000 1000000 --ndim 2 3 4 --box-size 0 --query-size 0.001 --no-brute --no-ref --knn 5
```
