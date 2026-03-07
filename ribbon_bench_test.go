package ribbonGo

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"testing"
)

// =============================================================================
// Paper-Aligned Benchmarks — Dillinger & Walzer (2021), §6
//
// The paper's Figure 7 evaluates Ribbon filter performance at two key
// set sizes that highlight the transition from cache-resident to
// memory-latency-bound queries:
//
//   n = 10^6  — filter (~1 MB) fits comfortably in L2/L3 cache
//   n = 10^8  — filter (~100 MB) exceeds all cache levels
//
// Configurations (paper §4, "Standard Ribbon"):
//
//   w ∈ {32, 64, 128}     — ribbon width (coefficient bits)
//   r = 7                  — result bits (FPR ≈ 2^{-7} ≈ 0.78%)
//   firstCoeffAlwaysOne    — true (the standard, faster variant)
//
// Paper §4 overhead ratios: m/n ≈ 1 + 2.3/w
//   w=32:  ~7.2% overhead, fastest per-key construction
//   w=64:  ~3.6% overhead, balanced speed/space
//   w=128: ~1.8% overhead, most compact (recommended)
//
// Space overhead note:
//
//   Our implementation uses the "simple" solution format — one uint8
//   per slot regardless of r (like RocksDB's InMemSimpleSolution).
//   This means:
//     bits/key (actual)  = 8 × numSlots / n
//     bits/key (packed)  = r × numSlots / n  (paper's interleaved format)
//     bits/key (optimal) = r = 7              (information-theoretic min)
//
//   For r=7, our storage is 8/7 ≈ 14.3% larger than the interleaved
//   format. An interleaved solution format would close this gap.
//
// Memory requirements:
//   n=10^6:  ~50 MB peak during construction
//   n=10^8:  ~8 GB peak during construction (skipped with -short)
//
// Run paper benchmarks (n=10^6 only, fast):
//   go test -bench="BenchmarkPaper" -benchmem -short
//
// Run paper benchmarks (both scales, requires ~8 GB RAM):
//   go test -bench="BenchmarkPaper" -benchmem -benchtime=3s
//
// Run a single iteration at full scale:
//   go test -bench="BenchmarkPaper" -benchmem -benchtime=1x
// =============================================================================

// =============================================================================
// HELPERS
// =============================================================================

// makePaperKeys generates n unique keys for benchmarking.
//
// Keys are compact 8-byte binary strings (little-endian uint64) for
// minimal memory footprint (~24 bytes per key including header) and
// fast generation. The binary format also ensures uniform-length keys,
// eliminating key-length variance from the benchmark.
func makePaperKeys(n int) []string {
	keys := make([]string, n)
	buf := make([]byte, 8)
	for i := range keys {
		binary.LittleEndian.PutUint64(buf, uint64(i))
		keys[i] = string(buf)
	}
	return keys
}

// makePaperNonMemberKeys generates n non-member probe keys guaranteed
// to be distinct from makePaperKeys output by using a high offset.
func makePaperNonMemberKeys(n int) []string {
	const offset = 1 << 40 // well above any realistic test key count
	keys := make([]string, n)
	buf := make([]byte, 8)
	for i := range keys {
		binary.LittleEndian.PutUint64(buf, uint64(i+offset))
		keys[i] = string(buf)
	}
	return keys
}

// intLog10 returns floor(log10(n)) for benchmark name labels.
func intLog10(n int) int {
	e := 0
	for n >= 10 {
		n /= 10
		e++
	}
	return e
}

// paperSizes returns the key set sizes to benchmark.
// n=10^8 is skipped in short mode due to memory requirements (~8 GB).
func paperSizes(b *testing.B) []int {
	if testing.Short() {
		return []int{1_000_000}
	}
	return []int{1_000_000, 100_000_000}
}

// =============================================================================
// BUILD — construction throughput
//
// Paper §6: "Construction time" column in the comparison table.
//
// Measures the full Build() pipeline through the public API:
//   string→[]byte conversion → XXH3 hashing (Phase 1) →
//   seed-retry loop { derive (Phase 2) → banding → back-substitution } →
//   filter packaging
//
// Reports:
//   ns/op       — total time to build a filter from n keys
//   ns/key      — amortized per-key construction cost
//   bits/key    — actual storage per key (simple format: 8 × numSlots / n)
//   overhead%   — slot overhead: (numSlots/n − 1) × 100
//   B/op        — total bytes allocated per build
//   allocs/op   — total heap allocations per build
// =============================================================================

func BenchmarkPaper_Build(b *testing.B) {
	sizes := paperSizes(b)

	for _, w := range []uint32{32, 64, 128} {
		for _, n := range sizes {
			name := fmt.Sprintf("w=%d/n=1e%d", w, intLog10(n))
			b.Run(name, func(b *testing.B) {
				keys := makePaperKeys(n)
				cfg := Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: true,
				}

				var rb *Ribbon
				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					rb = NewWithConfig(cfg)
					if err := rb.Build(keys); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()

				// Report paper-aligned custom metrics from the last build.
				if rb != nil && rb.f != nil {
					nf := float64(n)
					numSlots := float64(rb.f.numSlots)
					bitsPerKey := (numSlots * 8) / nf
					overheadPct := ((numSlots / nf) - 1) * 100
					nsPerKey := float64(b.Elapsed().Nanoseconds()) / (float64(b.N) * nf)

					b.ReportMetric(nsPerKey, "ns/key")
					b.ReportMetric(bitsPerKey, "bits/key")
					b.ReportMetric(overheadPct, "overhead%")
				}
			})
		}
	}
}

// =============================================================================
// QUERY — membership query throughput (true positives)
//
// Paper §6: "Positive query time" column.
//
// The filter is built once during setup (not timed). The benchmark loop
// measures only the Contains() call:
//   XXH3([]byte(key)) → derive(hash) → GF(2) dot product → compare
//
// Probe strategy:
//   8192 evenly-spaced member keys are sampled from the build set.
//   XXH3 maps these to pseudo-random positions in the solution vector,
//   giving a realistic random-access pattern that exercises cache
//   behavior faithfully.
//
// At n=10^6, the filter (~1 MB) is L2-resident → fast queries.
// At n=10^8, the filter (~100 MB) is DRAM-resident → queries are
// memory-latency-bound, revealing the paper's key insight about
// Ribbon's cache-friendly contiguous access pattern.
// =============================================================================

func BenchmarkPaper_Query_Positive(b *testing.B) {
	sizes := paperSizes(b)

	for _, w := range []uint32{32, 64, 128} {
		for _, n := range sizes {
			name := fmt.Sprintf("w=%d/n=1e%d", w, intLog10(n))
			b.Run(name, func(b *testing.B) {
				// Setup: build filter (not timed).
				keys := makePaperKeys(n)
				rb := NewWithConfig(Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: true,
				})
				if err := rb.Build(keys); err != nil {
					b.Fatal(err)
				}

				// Sample 8192 evenly-spaced member keys for probing.
				// These are spread across the full key range so queries
				// exercise the entire filter address space.
				const probeCount = 8192
				probes := make([]string, probeCount)
				step := n / probeCount
				if step == 0 {
					step = 1
				}
				for i := range probes {
					probes[i] = keys[(i*step)%n]
				}

				// Release the full key set to reduce memory pressure
				// during the timed query loop.
				keys = nil
				runtime.GC()

				mask := probeCount - 1
				b.ResetTimer()
				b.ReportAllocs()
				var sink bool
				for i := 0; i < b.N; i++ {
					sink = rb.Contains(probes[i&mask])
				}
				_ = sink
			})
		}
	}
}

// =============================================================================
// QUERY — membership query throughput (true negatives / non-members)
//
// Paper §6: "Negative query time" column.
//
// Same setup as positive queries, but probes are keys NOT in the build
// set. The cost should be virtually identical to positive queries
// because the full GF(2) dot product is always computed before the
// result comparison — there is no early-exit branch.
//
// Identical true-positive vs true-negative cost confirms there is no
// timing side-channel: an adversary cannot distinguish member from
// non-member queries by observing latency.
// =============================================================================

func BenchmarkPaper_Query_Negative(b *testing.B) {
	sizes := paperSizes(b)

	for _, w := range []uint32{32, 64, 128} {
		for _, n := range sizes {
			name := fmt.Sprintf("w=%d/n=1e%d", w, intLog10(n))
			b.Run(name, func(b *testing.B) {
				// Setup: build filter (not timed).
				keys := makePaperKeys(n)
				rb := NewWithConfig(Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: true,
				})
				if err := rb.Build(keys); err != nil {
					b.Fatal(err)
				}

				// Release keys immediately — not needed for negative queries.
				keys = nil
				runtime.GC()

				// Generate non-member probe keys.
				const probeCount = 8192
				probes := makePaperNonMemberKeys(probeCount)
				mask := probeCount - 1

				b.ResetTimer()
				b.ReportAllocs()
				var sink bool
				for i := 0; i < b.N; i++ {
					sink = rb.Contains(probes[i&mask])
				}
				_ = sink
			})
		}
	}
}

// =============================================================================
// SPACE — overhead summary (complements Build metrics)
//
// This benchmark reports detailed space metrics for every configuration
// without timing construction, making it fast to run even at n=10^8.
//
// Reported metrics:
//   bits/key         — actual storage: 8 × numSlots / n (our simple format)
//   packed-bits/key  — theoretical with packed r-bit storage: r × numSlots / n
//   overhead%        — slot overhead: (numSlots/n − 1) × 100
//
// Paper §4 theoretical bits/key (interleaved format):
//   w=32:  r × (1 + 2.3/32)  = 7 × 1.072 ≈ 7.50
//   w=64:  r × (1 + 2.3/64)  = 7 × 1.036 ≈ 7.25
//   w=128: r × (1 + 2.3/128) = 7 × 1.018 ≈ 7.13
//
// Information-theoretic minimum: log₂(1/ε) = r = 7 bits/key.
// =============================================================================

func BenchmarkPaper_Space(b *testing.B) {
	sizes := paperSizes(b)

	for _, w := range []uint32{32, 64, 128} {
		for _, n := range sizes {
			name := fmt.Sprintf("w=%d/n=1e%d", w, intLog10(n))
			b.Run(name, func(b *testing.B) {
				keys := makePaperKeys(n)
				rb := NewWithConfig(Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: true,
				})
				if err := rb.Build(keys); err != nil {
					b.Fatal(err)
				}
				keys = nil

				nf := float64(n)
				numSlots := float64(rb.f.numSlots)

				// The benchmark loop is trivial — space is a static
				// metric. We run a single Contains to give the framework
				// a non-zero timing anchor for the custom metrics.
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					rb.Contains("probe")
				}
				b.StopTimer()

				b.ReportMetric((numSlots*8)/nf, "bits/key")
				b.ReportMetric((numSlots*7)/nf, "packed-bits/key")
				b.ReportMetric(((numSlots/nf)-1)*100, "overhead%")
			})
		}
	}
}
