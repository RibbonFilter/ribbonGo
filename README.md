# ribbonGo
[![GoDoc](https://godoc.org/github.com/RibbonGo/ribbonGo?status.svg)](https://godoc.org/github.com/RibbonGo/ribbonGo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance **Ribbon filter** implementation in Go — the space-efficient probabilistic data structure that is **practically smaller than Bloom and Xor filters**.

Based on the paper:

> **"Ribbon filter: practically smaller than Bloom and Xor"**
> Peter C. Dillinger & Stefan Walzer, 2021
> ([arXiv:2103.02515](https://arxiv.org/abs/2103.02515))

---

## What is a Ribbon Filter?

A **Ribbon filter** is a static, space-efficient probabilistic data structure for approximate set membership queries — the same problem Bloom filters solve, but using significantly less space.

Given a set of keys, a Ribbon filter can answer *"is this key in the set?"* with:

- **No false negatives** — if a key is in the set, the filter always says yes.
- **Configurable false positive rate** — non-member keys have a tunable probability of being incorrectly reported as members (FPR ≈ 2<sup>−r</sup> for *r* result bits).
- **Near-optimal space** — approaches the information-theoretic lower bound, using only ~1–5% more space than the minimum possible.

### Ribbon vs Bloom vs Xor

| Property | Bloom | Xor | **Ribbon** |
|----------|-------|-----|------------|
| Space overhead | ~44% over optimal | ~23% over optimal | **~1–5%** over optimal |
| Construction | Fast, incremental | Fast, static | Moderate, static |
| Supports deletion | No (without counting) | No | No |
| FPR at 1% target (bits/key) | 9.6 | 9.8 | **≈ 8.0** |

Ribbon filters are ideal for **read-heavy, write-once** workloads — LSM-tree engines (RocksDB, LevelDB), static lookup tables, networking data planes, and any scenario where a set is built once and queried millions of times.

---

## Quick Start

### Installation

```bash
go get github.com/RibbonGo/ribbonGo
```

Requires **Go 1.25+**.

### Usage

```go
package main

import (
    "fmt"
    "log"

    "github.com/RibbonGo/ribbonGo"
)

func main() {
    // Create a filter with default settings (w=128, r=7, FPR ≈ 0.78%)
    f := ribbon.New()

    // Build the filter from a set of keys
    keys := []string{"apple", "banana", "cherry", "date", "elderberry"}
    if err := f.Build(keys); err != nil {
        log.Fatal(err)
    }

    // Query membership
    fmt.Println(f.Contains("banana"))    // true  (always correct for members)
    fmt.Println(f.Contains("fig"))       // false (probably — FPR ≈ 0.78%)
}
```

### Custom Configuration

```go
// Trade space for faster construction with a narrower ribbon width
f := ribbon.NewWithConfig(ribbon.Config{
    CoeffBits:           64,    // w=64: balanced speed/space
    ResultBits:          8,     // r=8: FPR ≈ 0.39%
    FirstCoeffAlwaysOne: true,  // deterministic pivot (faster)
    MaxSeeds:            256,   // hash seed retries before failure
})

if err := f.Build(keys); err != nil {
    // err is ribbon.ErrConstructionFailed if banding fails
    log.Fatal(err)
}
```

---

## API Reference

The public API is intentionally minimal — a single type with four functions.

### Types

| Type | Description |
|------|-------------|
| `Ribbon` | The main filter type. Create → Build → Contains. |
| `Config` | Construction parameters (ribbon width, result bits, etc.). |
| `ErrConstructionFailed` | Sentinel error returned when banding fails for all seed retries. |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `New` | `func New() *Ribbon` | Create a filter with defaults: w=128, r=7, fcao=true, maxSeeds=256 |
| `NewWithConfig` | `func NewWithConfig(cfg Config) *Ribbon` | Create a filter with custom parameters. Panics on invalid config. |
| `Build` | `func (r *Ribbon) Build(keys []string) error` | Construct the filter from a set of unique keys. May be called multiple times. |
| `Contains` | `func (r *Ribbon) Contains(key string) bool` | Test membership. Zero allocations, safe for concurrent use after Build. |

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `CoeffBits` | `uint32` | 128 | Ribbon width *w* ∈ {32, 64, 128}. Larger → more compact, slower to build. |
| `ResultBits` | `uint` | 7 | Fingerprint bits *r* ∈ [1, 8]. FPR ≈ 2<sup>−r</sup>. |
| `FirstCoeffAlwaysOne` | `bool` | true | Force bit 0 of coefficient rows to 1 for deterministic pivoting. |
| `MaxSeeds` | `uint32` | 256 | Maximum hash seed retries before returning `ErrConstructionFailed`. |

---

## Features

- **Paper-faithful implementation** — every design decision traces to a specific section of Dillinger & Walzer (2021), with `§N` citations throughout the code
- **Configurable ribbon width** — w ∈ {32, 64, 128} to trade construction speed for space efficiency
- **Configurable result bits** — r bits per key for FPR ≈ 2<sup>−r</sup>
- **Dynamic slot computation** — overhead ratio grows logarithmically with *n*, ported from RocksDB's empirical tables (`ribbon_config.cc`)
- **firstCoeffAlwaysOne optimisation** — deterministic pivot from paper §4, skipping leading-zero scan
- **SoA memory layout** — Struct-of-Arrays for maximum cache utilisation (8 coefficients per cache line at w=64)
- **Width-specialised inner loops** — pure `uint64` path for w≤64, separate lo/hi ops for w=128
- **Software-pipelined prefetching** — `AddRange` hides L2/L3 latency by prefetching the next key's slot
- **Zero allocations on hot paths** — no heap escapes during construction or query
- **Two-phase hash pipeline** — hash keys once, remix per seed attempt; uses [XXH3](https://github.com/zeebo/xxh3) (64-bit)
- **Research-friendly** — all paper parameters exposed; reference (`slow*`) implementations included for cross-validation

---

## Benchmarks

**Apple M3 Pro · Go 1.25 · ARM64 · r = 7 result bits · firstCoeffAlwaysOne = true**

All benchmarks follow the methodology of Dillinger & Walzer (2021), testing at both *n* = 10⁶ and *n* = 10⁸ keys.

### Build Performance

Construction throughput and space efficiency at scale.

| *n* | Width | ns/key | bits/key | Overhead |
|-----|-------|--------|----------|----------|
| 10⁶ | w=32 | 56.89 | 10.54 | 31.81% |
| 10⁶ | w=64 | 64.56 | 8.959 | 11.99% |
| 10⁶ | w=128 | 106.3 | 8.380 | 4.749% |
| 10⁸ | w=32 | 355.3 | 11.62 | 45.30% |
| 10⁸ | w=64 | 266.2 | 9.406 | 17.58% |
| 10⁸ | w=128 | 384.7 | 8.585 | 7.314% |

> At *n* = 10⁶, `w=128` achieves **8.38 bits/key** — only 4.7% above the information-theoretic minimum (7 bits for r=7). At *n* = 10⁸, it remains under 8.6 bits/key with just 7.3% overhead.

### Query Performance

Lookup latency per key (*n* = 10⁶).

| Width | Positive (ns/op) | Negative (ns/op) |
|-------|-------------------|-------------------|
| w=32 | 37.25 | 36.85 |
| w=64 | 53.70 | 52.49 |
| w=128 | 88.66 | 84.73 |

> Query time scales linearly with ribbon width, as expected — the inner loop performs a dot product over *w* bits. Positive and negative queries have nearly identical cost.

### Space Efficiency

Bits per key at both scales, with packed (information-theoretic) comparison.

| *n* | Width | bits/key | packed bits/key | Overhead |
|-----|-------|----------|-----------------|----------|
| 10⁶ | w=32 | 10.54 | 9.227 | 31.81% |
| 10⁶ | w=64 | 8.959 | 7.839 | 11.99% |
| 10⁶ | w=128 | 8.380 | 7.332 | 4.749% |
| 10⁸ | w=32 | 11.62 | 10.17 | 45.30% |
| 10⁸ | w=64 | 9.406 | 8.231 | 17.58% |
| 10⁸ | w=128 | 8.585 | 7.512 | 7.314% |

> **w=128** at *n* = 10⁶ uses only **8.38 bits/key** — compare with Bloom filters at **9.6 bits/key** for the same FPR. That's a **12.7% space saving** over Bloom.

### Run Benchmarks Yourself

```bash
# Full benchmark suite
go test -bench=. -benchmem -count=3 ./...

# Paper-aligned benchmarks at n=10⁶ and n=10⁸
go test -run=^$ -bench='BenchmarkRibbon' -benchtime=3s -count=1

# Specific benchmark
go test -run=^$ -bench='BenchmarkRibbonBuild/w=128/n=1000000' -benchtime=3s
```

---

## Architecture

The implementation follows the paper's full algorithmic pipeline:

```
Key → [Hash] → [Bander] → [Solver] → [Filter/Query]
       §2         §2,§4       §2           §2
```

### Pipeline Layers

| Layer | File | Description |
|-------|------|-------------|
| **uint128** | `uint128.go` | 128-bit integer type for coefficient rows when w=128. |
| **Hash** | `hash.go` | Two-phase pipeline: hash each key once with XXH3, then cheaply remix per seed to derive `(start, coeffRow, result)` triples. |
| **Bander** | `bander.go` | On-the-fly Gaussian elimination over GF(2). Converts hashed equations into an upper-triangular banded matrix. Hottest code path during construction. |
| **Solver** | `solver.go` | Back-substitution: solves the upper-triangular system to produce the compact solution vector encoding the filter. |
| **Filter** | `filter.go` | Query evaluation: one hash, one dot product over GF(2). Also provides false-positive rate estimation. |
| **Builder** | `builder.go` | Orchestrates the full pipeline: hashing → banding → solving → filter construction. Includes RocksDB-style dynamic slot computation. |
| **Public API** | `ribbon.go` | Sole public surface: `Ribbon`, `Config`, `New()`, `NewWithConfig()`, `Build()`, `Contains()`. |

### Key Optimisations

**SoA layout** — Coefficient data is stored in parallel arrays (`coeffLo []uint64`, `coeffHi []uint64`, `result []uint8`) instead of an array of structs. For w≤64, `coeffHi` is `nil` — zero memory, zero operations. This doubles the number of coefficients per cache line compared to AoS layout.

**Width specialisation** — `Add()` dispatches to `addW64()` (pure `uint64`, ~10 ARM64 instructions per elimination step) or `addW128()` (separate lo/hi `uint64` ops). The generic `uint128.rsh()` branch dispatch is avoided entirely.

**Software-pipelined prefetching** — `AddRange()` touches `coeffLo[nextKey.start]` while processing the current key, pulling the next cache line into L1. This mirrors RocksDB's `BandingAddRange` approach and yields 20–36% throughput improvement at scale.

**Dynamic overhead ratio** — The number of slots *m* is computed using RocksDB-style empirical tables where overhead grows logarithmically with *n*. This ensures reliable banding at all scales, unlike a fixed overhead formula that breaks down at large *n*.

**Two-phase hashing** — Keys are hashed once with XXH3. On seed retry, only a cheap remix is applied to derive new `(start, coeff, result)` triples — no re-hashing of the original key data.

---

## How It Works

At a high level, a Ribbon filter encodes set membership as a system of linear equations over GF(2):

1. **Hash** each key to a triple: starting position *s*, a *w*-bit coefficient row *c*, and an *r*-bit result *r*.
2. **Band** the equations into an upper-triangular matrix via Gaussian elimination (the "banding" step).
3. **Solve** the triangular system by back-substitution, producing a compact solution vector *Z*.
4. **Query** a candidate key by computing its triple, then checking if `c · Z[s..s+w] == r`.

The "ribbon" name refers to the banded structure of the coefficient matrix — each equation touches only *w* consecutive columns starting at position *s*, forming a ribbon-like diagonal band.

### Why is it smaller than Bloom?

A Bloom filter wastes space because its bit-setting positions are independent — you can't pack information as tightly. A Ribbon filter encodes key membership as *equations*, allowing the solver to pack nearly *r* bits of information per key into the solution vector. The information-theoretic minimum is *r* bits/key, and Ribbon achieves within 1–5% of this.

---

## Testing

23 test functions across the public API, plus extensive internal tests — all passing.

Tests cover:

- **Correctness** — single insertion, no false negatives, false-positive rate validation, collision chains, 128-bit boundary crossing, redundant/contradictory equations
- **All configurations** — w ∈ {32, 64, 128} × firstCoeffAlwaysOne ∈ {true, false} × r ∈ {1, 4, 7, 8}
- **Scale** — builds at *n* = 100,000 verified against expected FPR
- **Edge cases** — empty input, single key, nil/empty filter, rebuild semantics, invalid config panics
- **Cross-validation** — optimised `Add()` vs reference `slowAdd()`, `AddRange()` vs `Add()`-loop — all verified slot-by-slot

```bash
go test -v -count=1 ./...
```

---

## Project Structure

```
ribbon/
├── ribbon.go              # Public API (Ribbon, Config, New, Build, Contains)
├── builder.go             # Pipeline orchestration, slot computation
├── bander.go              # Gaussian elimination (banding)
├── hash.go                # XXH3-based two-phase hash pipeline
├── solver.go              # Back-substitution solver
├── filter.go              # Query evaluation and FPR estimation
├── uint128.go             # 128-bit integer type
├── ribbon_test.go         # Public API tests (23 functions)
├── ribbon_bench_test.go   # Paper-aligned benchmarks
├── filter_test.go         # Internal filter/pipeline tests
├── filter_bench_test.go   # Internal benchmarks
├── bander_test.go         # Bander unit tests
├── bander_bench_test.go   # Bander benchmarks
├── hash_test.go           # Hash layer tests
├── hash_bench_test.go     # Hash benchmarks
├── solver_test.go         # Solver tests
├── solver_bench_test.go   # Solver benchmarks
├── uint128_test.go        # uint128 tests
├── go.mod
├── go.sum
├── LICENSE
└── README.md
```

---

## Contributing

Contributions are welcome! This project follows the paper's design closely — please read the relevant paper section before modifying any algorithm.

### Getting Started

```bash
git clone https://github.com/RibbonGo/ribbonGo.git
cd ribbonGo
go test ./...
```

### Guidelines

- **Paper-first** — every design decision should cite a specific section (`§N`) of Dillinger & Walzer (2021)
- **RocksDB cross-references** — use `[RocksDB: FunctionName in file.h]` format for implementation parallels
- **All 6 configs** — tests must cover w ∈ {32, 64, 128} × firstCoeffAlwaysOne ∈ {true, false}
- **Reference implementations** — include `slow*` variants for cross-validation of optimised code
- **Zero allocations** — hot paths must not escape to heap; verify with `go test -bench=X -benchmem`
- **Naming** — unexported everything (`standardBander`, not `StandardBander`); constants use `kCamelCase`

---

## References

- Dillinger, P. C. & Walzer, S. (2021). *Ribbon filter: practically smaller than Bloom and Xor*. [arXiv:2103.02515](https://arxiv.org/abs/2103.02515)
- RocksDB Ribbon filter: [ribbon_impl.h](https://github.com/facebook/rocksdb/blob/main/util/ribbon_impl.h), [ribbon_alg.h](https://github.com/facebook/rocksdb/blob/main/util/ribbon_alg.h), [ribbon_config.cc](https://github.com/facebook/rocksdb/blob/main/util/ribbon_config.cc)
- XXH3 hash function (Go): [github.com/zeebo/xxh3](https://github.com/zeebo/xxh3)

## License

[MIT](LICENSE) © 2026 RibbonGo
