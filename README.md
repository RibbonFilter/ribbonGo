# ribbonGo

[![GoDoc](https://godoc.org/github.com/RibbonFilter/ribbonGo?status.svg)](https://godoc.org/github.com/RibbonFilter/ribbonGo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/RibbonFilter/ribbonGo)](https://goreportcard.com/report/github.com/RibbonFilter/ribbonGo)
[![Coverage Status](https://coveralls.io/repos/github/RibbonFilter/ribbonGo/badge.svg?branch=main)](https://coveralls.io/github/RibbonFilter/ribbonGo?branch=main)

A high-performance **Ribbon filter** in Go — a static, space-efficient
probabilistic data structure for approximate set membership that is
**practically smaller than Bloom and Xor filters**.

Based on:

> **"Ribbon filter: practically smaller than Bloom and Xor"**
> Peter C. Dillinger & Stefan Walzer, 2021 — [arXiv:2103.02515](https://arxiv.org/abs/2103.02515)

---

## Table of Contents

- [What is a Ribbon Filter?](#what-is-a-ribbon-filter)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

## What is a Ribbon Filter?

A **Ribbon filter** answers *"is this key in the set?"* — the same problem
Bloom filters solve — while using significantly less space. Given a fixed set
of keys, it provides:

- **No false negatives** — a member key always returns `true`.
- **Configurable false-positive rate** — a non-member returns `true` with
  probability ≈ 2<sup>−r</sup> for *r* result bits.
- **Near-optimal space** — within ~1–5% of the information-theoretic minimum.

### Ribbon vs Bloom vs Xor

| Property | Bloom | Xor | **Ribbon** |
|----------|-------|-----|------------|
| Space overhead | ~44% over optimal | ~23% over optimal | **~1–5% over optimal** |
| Construction | Fast, incremental | Fast, static | Moderate, static |
| Supports deletion | No (without counting) | No | No |
| Bits/key at 1% FPR | 9.6 | 9.8 | **≈ 8.0** |

Ribbon filters suit **read-heavy, write-once** workloads: LSM-tree engines
(RocksDB, LevelDB), static lookup tables, networking data planes — any case
where a set is built once and queried many times.

---

## Installation

```bash
go get github.com/RibbonFilter/ribbonGo
```

Requires **Go 1.24+**.

---

## Quick Start

The package name is `ribbonGo`, so the qualifier for all exported symbols is
`ribbonGo.`.

```go
package main

import (
	"fmt"
	"log"

	"github.com/RibbonFilter/ribbonGo"
)

func main() {
	// Build a filter from a set of unique keys with default settings
	// (w=128, r=7, FPR ≈ 0.78%).
	keys := []string{"apple", "banana", "cherry", "date", "elderberry"}

	f, err := ribbonGo.NewFromKeys(keys)
	if err != nil {
		log.Fatal(err)
	}

	// Query membership.
	fmt.Println(f.Contains("banana")) // true  (always correct for members)
	fmt.Println(f.Contains("fig"))    // false (probably — FPR ≈ 0.78%)
}
```

### Building Explicitly

`New` + `Build` separates construction from key insertion, and `Build` may be
called again to rebuild against a different key set.

```go
f := ribbonGo.New()
if err := f.Build(keys); err != nil {
	log.Fatal(err)
}
```

### Custom Configuration

```go
f, err := ribbonGo.NewFromKeysWithConfig(ribbonGo.Config{
	CoeffBits:           64,   // w=64: balanced speed/space
	ResultBits:          8,    // r=8: FPR ≈ 0.39%
	FirstCoeffAlwaysOne: true, // deterministic pivot (faster)
	MaxSeeds:            256,  // hash-seed retries before failure
}, keys)
if err != nil {
	// err is ribbonGo.ErrConstructionFailed if banding never succeeds.
	log.Fatal(err)
}
```

> **Note:** keys must be **unique**. Duplicate keys produce identical
> equations regardless of the hash seed and cause guaranteed construction
> failure. De-duplicate the input first if duplicates may be present.

---

## API Reference

The public surface is intentionally small: one type, its config, and a
sentinel error.

### Types

| Type | Description |
|------|-------------|
| `Ribbon` | The main filter type. Create → Build → Contains. |
| `Config` | Construction parameters (ribbon width, result bits, etc.). |
| `ErrConstructionFailed` | Sentinel error returned when banding fails after all seed retries. |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `New` | `func New() *Ribbon` | Create a filter with defaults: w=128, r=7, fcao=true, maxSeeds=256. |
| `NewWithConfig` | `func NewWithConfig(cfg Config) *Ribbon` | Create a filter with custom parameters. Panics on invalid config. |
| `NewFromKeys` | `func NewFromKeys(keys []string) (*Ribbon, error)` | Create with defaults and build in one step. |
| `NewFromKeysWithConfig` | `func NewFromKeysWithConfig(cfg Config, keys []string) (*Ribbon, error)` | Create with custom config and build in one step. |
| `Build` | `func (r *Ribbon) Build(keys []string) error` | Construct the filter from a set of unique keys. May be called repeatedly. |
| `Contains` | `func (r *Ribbon) Contains(key string) bool` | Test membership. Zero allocations; safe for concurrent use after Build. |

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `CoeffBits` | `uint32` | 128 | Ribbon width *w* ∈ {32, 64, 128}. Larger → more compact, slower to build. |
| `ResultBits` | `uint` | 7 | Fingerprint bits *r* ∈ [1, 8]. FPR ≈ 2<sup>−r</sup>. |
| `FirstCoeffAlwaysOne` | `bool` | true | Force bit 0 of each coefficient row to 1 for deterministic pivoting. |
| `MaxSeeds` | `uint32` | 256 | Maximum hash-seed retries before returning `ErrConstructionFailed`. `0` uses the default. |

Full documentation: [pkg.go.dev/github.com/RibbonFilter/ribbonGo](https://pkg.go.dev/github.com/RibbonFilter/ribbonGo).

---

## How It Works

A Ribbon filter encodes set membership as a system of linear equations over
GF(2):

1. **Hash** each key to a triple: a starting position *s*, a *w*-bit
   coefficient row *c*, and an *r*-bit result *r*.
2. **Band** the equations into an upper-triangular matrix via Gaussian
   elimination — the "banding" step.
3. **Solve** the triangular system by back-substitution, producing a compact
   solution vector *Z*.
4. **Query** a candidate key by recomputing its triple and checking whether
   `c · Z[s..s+w] == r`.

The name refers to the banded structure of the coefficient matrix: each
equation touches only *w* consecutive columns starting at position *s*,
forming a ribbon-like diagonal band.

### Why is it smaller than Bloom?

A Bloom filter sets independent bit positions, so it cannot pack information
tightly. A Ribbon filter encodes membership as *equations*, letting the solver
store nearly *r* bits of information per key in the solution vector. The
information-theoretic minimum is *r* bits/key, and Ribbon comes within 1–5% of
it.

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
| **Bander** | `bander.go` | On-the-fly Gaussian elimination over GF(2). Converts hashed equations into an upper-triangular banded matrix. Hottest construction path. |
| **Solver** | `solver.go` | Back-substitution writing the solution directly into the Interleaved Column-Major Layout (paper §5.2). |
| **Filter** | `filter.go` | Query evaluation: one hash, then a per-result-column GF(2) dot product over the ICML words, short-circuiting on the first mismatch. |
| **Builder** | `builder.go` | Orchestrates hashing → banding → solving → filter construction, including RocksDB-style dynamic slot computation. |
| **Public API** | `ribbon.go` | Sole public surface: `Ribbon`, `Config`, constructors, `Build`, `Contains`. |

### Key Optimisations

- **Interleaved Column-Major Layout (ICML)** — the solution is stored as
  *w*-bit words grouped into blocks of *r* words (RocksDB's
  `InterleavedSolutionStorage`, paper §5.2), not one byte per slot. Queries
  decode *r* result columns via XOR-fold + POPCNT parity and short-circuit on
  the first mismatch. Storage is width-specialised: `[]uint64` for w ∈ {64,
  128}, `[]uint32` for w=32.
- **SoA construction layout** — coefficient data lives in parallel arrays
  (`coeffLo`, `coeffHi`, `result`). For w≤64, `coeffHi` is `nil`, doubling
  coefficients per cache line versus an array-of-structs.
- **Width specialisation** — `add()` dispatches to `addW64()` (pure `uint64`)
  or `addW128()` (separate lo/hi ops), avoiding generic branch dispatch.
- **Software-pipelined prefetching** — `addRange()` prefetches the next key's
  cache line while processing the current one (20–36% throughput at scale).
- **Dynamic overhead ratio** — slot count *m* uses RocksDB-style empirical
  tables where overhead grows logarithmically with *n*, keeping banding
  reliable at all scales.
- **Two-phase hashing** — keys are hashed once with [XXH3](https://github.com/zeebo/xxh3);
  seed retries apply only a cheap remix, never re-hashing key data.

---

## Benchmarks

All benchmarks follow the methodology of Dillinger & Walzer (2021), at both
*n* = 10⁶ and *n* = 10⁸ keys, with r = 7 result bits and
firstCoeffAlwaysOne = true.

> The *Build* and *Space* tables were measured on an Apple M3 Pro (ARM64); the
> *Query* table was re-measured on an x86 Xeon Platinum host after the ICML
> migration. Space/correctness figures (bits/key, FPR) are CPU-independent;
> absolute ns/op are **not** comparable across the two hosts.

### Build Performance

| *n* | Width | ns/key | bits/key | Overhead |
|-----|-------|--------|----------|----------|
| 10⁶ | w=32 | 56.89 | 9.227 | 31.81% |
| 10⁶ | w=64 | 64.56 | 7.840 | 11.99% |
| 10⁶ | w=128 | 106.3 | 7.334 | 4.749% |
| 10⁸ | w=32 | 355.3 | 10.17 | 45.30% |
| 10⁸ | w=64 | 266.2 | 8.231 | 17.58% |
| 10⁸ | w=128 | 384.7 | 7.512 | 7.314% |

> `bits/key` is the **actual ICML physical storage** (paper §5.2): the *r*
> result columns packed at *r* bits per slot (`r × numSlots / n`). At *n* = 10⁶,
> w=128 achieves **7.33 bits/key** — only 4.7% above the 7-bit minimum for r=7.

### Query Performance

Lookup latency per key at *n* = 10⁶, on an **x86 Xeon Platinum** host.

| Width | Positive (ns/op) | Negative (ns/op) |
|-------|-------------------|-------------------|
| w=32 | 32.3 | 26.9 |
| w=64 | 31.5 | 26.3 |
| w=128 | 43.2 | 31.7 |

> With ICML each query decodes the *r* result columns one at a time (XOR-fold +
> POPCNT parity), so cost is roughly flat in *w* for w ≤ 64 and only rises at
> w=128 (two 64-bit halves per column). Negative queries **short-circuit** on
> the first mismatched column, so they are consistently faster than positive
> queries.

### Space Efficiency

| *n* | Width | bits/key | Overhead |
|-----|-------|----------|----------|
| 10⁶ | w=32 | 9.227 | 31.81% |
| 10⁶ | w=64 | 7.840 | 11.99% |
| 10⁶ | w=128 | 7.334 | 4.749% |
| 10⁸ | w=32 | 10.17 | 45.30% |
| 10⁸ | w=64 | 8.231 | 17.58% |
| 10⁸ | w=128 | 7.512 | 7.314% |

> w=128 at *n* = 10⁶ uses only **7.33 bits/key** vs Bloom's **9.6 bits/key** at
> the same FPR — a **23.6% space saving**.

### Running the Benchmarks

```bash
# Full benchmark suite
go test -bench=. -benchmem -count=3 ./...

# Paper-aligned benchmarks at n=10⁶ and n=10⁸
go test -run=^$ -bench='BenchmarkRibbon' -benchtime=3s -count=1

# A specific benchmark
go test -run=^$ -bench='BenchmarkRibbonBuild/w=128/n=1000000' -benchtime=3s
```

---

## Testing

The suite covers correctness, every configuration, scale, and cross-validation
of the optimised code against reference implementations.

- **Correctness** — single insertion, no false negatives, FPR validation,
  collision chains, 128-bit boundary crossing, redundant/contradictory
  equations.
- **All configurations** — w ∈ {32, 64, 128} × firstCoeffAlwaysOne ∈ {true,
  false} × r ∈ {1, 4, 7, 8}.
- **Scale** — builds at *n* = 100,000 verified against expected FPR.
- **Edge cases** — empty input, single key, nil/empty filter, rebuild
  semantics, invalid-config panics.
- **Cross-validation** — optimised `add()` vs reference `slowadd()`, and
  `addRange()` vs an `add()`-loop, verified slot-by-slot.

```bash
go test -v -count=1 ./...
```

---

## Contributing

Contributions are welcome. This project tracks the paper's design closely —
please read the relevant paper section before modifying any algorithm.

### Getting Started

```bash
git clone https://github.com/RibbonFilter/ribbonGo.git
cd ribbonGo
go test ./...
```

### Guidelines

- **Paper-first** — every design decision should cite a specific section
  (`§N`) of Dillinger & Walzer (2021).
- **RocksDB cross-references** — use the `[RocksDB: FunctionName in file.h]`
  format for implementation parallels.
- **All configs** — tests must cover w ∈ {32, 64, 128} × firstCoeffAlwaysOne
  ∈ {true, false}.
- **Reference implementations** — include `slow*` variants for
  cross-validation of optimised code.
- **Zero allocations** — hot paths must not escape to the heap; verify with
  `go test -bench=X -benchmem`.
- **Naming** — unexported everything internal (`standardBander`, not
  `StandardBander`); constants use `kCamelCase`.

---

## References

- Dillinger, P. C. & Walzer, S. (2021). *Ribbon filter: practically smaller
  than Bloom and Xor*. [arXiv:2103.02515](https://arxiv.org/abs/2103.02515)
- RocksDB Ribbon filter:
  [ribbon_impl.h](https://github.com/facebook/rocksdb/blob/main/util/ribbon_impl.h),
  [ribbon_alg.h](https://github.com/facebook/rocksdb/blob/main/util/ribbon_alg.h),
  [ribbon_config.cc](https://github.com/facebook/rocksdb/blob/main/util/ribbon_config.cc)
- XXH3 hash function (Go): [github.com/zeebo/xxh3](https://github.com/zeebo/xxh3)

## License

[MIT](LICENSE) © 2026 RibbonFilter
