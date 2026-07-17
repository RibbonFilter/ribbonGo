package ribbonGo

import (
	"fmt"
	"testing"

	"github.com/zeebo/xxh3"
)

// =============================================================================
// Benchmarks — Filter construction and query performance
//
// The Filter is the user-facing API for the Ribbon filter. The two
// performance-critical operations are:
//
//   1. Build / BuildFromHashes — one-time construction cost.
//   2. Contains / ContainsHash — per-query cost (the hot path).
//
// Contains dominates runtime in production. It performs:
//   (a) XXH3 hash of the key bytes (Phase 1).
//   (b) derive() — rehash + fastRange64 + coeff + result (Phase 2).
//   (c) GF(2) dot product — skip-zero iteration over set coefficient
//       bits, XORing solution bytes.
//   (d) Comparison: computed result == expected result.
//
// We benchmark:
//   1. Contains for true positives (member keys).
//   2. Contains for true negatives (non-member keys).
//   3. ContainsHash (pre-hashed, removes XXH3 from the measured path).
//   4. Build throughput (keys/sec for construction).
//   5. Build by width (construction cost scaling with w).
//
// Each is measured across ribbon widths (32/64/128) to surface
// width-dependent costs in the dot-product loop.
// =============================================================================

// benchBuildFilter constructs a Filter with the given parameters.
// Returns the filter and the original keys (for true-positive queries).
func benchBuildFilter(w uint32, numKeys int) (*filter, []string) {
	keys := make([]string, numKeys)
	for i := range keys {
		keys[i] = fmt.Sprintf("bench_filter_key_%d", i)
	}

	cfg := Config{
		CoeffBits:           w,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
	}
	f, err := buildFilter(keys, cfg)
	if err != nil {
		panic(fmt.Sprintf("benchBuildFilter: Build failed: %v", err))
	}
	return f, keys
}

// benchBuildFilterHashes constructs a Filter and returns pre-computed hashes
// for both member and non-member keys. Used by ContainsHash benchmarks.
func benchBuildFilterHashes(w uint32, numKeys int) (*filter, []uint64, []uint64) {
	keys := make([]string, numKeys)
	for i := range keys {
		keys[i] = fmt.Sprintf("bench_hash_key_%d", i)
	}

	cfg := Config{
		CoeffBits:           w,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
	}
	f, err := buildFilter(keys, cfg)
	if err != nil {
		panic(fmt.Sprintf("benchBuildFilterHashes: Build failed: %v", err))
	}

	// Pre-compute member hashes.
	memberHashes := make([]uint64, numKeys)
	for i, key := range keys {
		memberHashes[i] = xxh3.HashString(key)
	}

	// Pre-compute non-member hashes.
	nonMemberHashes := make([]uint64, numKeys)
	for i := range nonMemberHashes {
		nonMemberHashes[i] = xxh3.HashString(fmt.Sprintf("bench_nonmember_%d", i))
	}

	return f, memberHashes, nonMemberHashes
}

// ---------------------------------------------------------------------------
// 1. Contains — true positive queries (member keys)
//
// Measures the full per-query cost: XXH3 hash + derive + dot product +
// comparison. All queries return true (the key is in the filter).
//
// The dominant costs are:
//   (a) XXH3: ~10–20 ns depending on key length.
//   (b) derive: ~0.4 ns (inlined, branchless).
//   (c) dot product: ~w/2 byte reads + XORs (skip-zero iteration).
//
// For short keys, XXH3 dominates. For long keys, XXH3 cost grows
// linearly while the filter query cost stays constant.
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   Keys      ns/op   B/op   allocs/op
//   ─────   ──────    ─────   ────   ─────────
//   w=32    1,000     25.27   0      0
//   w=32    10,000    35.16   0      0
//   w=64    1,000     49.80   0      0
//   w=64    10,000    51.97   0      0
//   w=64    100,000   53.42   0      0
//   w=128   1,000     78.70   0      0
//   w=128   10,000    84.49   0      0
//   w=128   100,000   84.75   0      0
//
// Key observations:
//   • Zero allocations — the entire Contains path (XXH3 + derive + dot
//     product + comparison) runs without touching the heap.
//   • w=32: 25–35 ns/query. The faster time at n=1K reflects better L1
//     cache residency of the small solution array.
//   • w=64: ~50–53 ns/query. Cost is stable across key counts, indicating
//     the solution array fits comfortably in L2 for up to 100K keys.
//   • w=128: ~79–85 ns/query. ~1.6× the w=64 cost, consistent with
//     iterating ~64 vs ~32 set bits in the coefficient row.
//   • XXH3 accounts for ~25 ns of the total (compare ContainsHash below).
// ---------------------------------------------------------------------------

func BenchmarkContains_TruePositive(b *testing.B) {
	for _, w := range []uint32{32, 64, 128} {
		// w=32 has higher per-key overhead (~7.2%) and does not scale well
		// to very large key sets. Limit to 10K keys for w=32.
		sizes := []int{1000, 10000, 100000}
		if w == 32 {
			sizes = []int{1000, 10000}
		}
		for _, numKeys := range sizes {
			name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
			b.Run(name, func(b *testing.B) {
				f, keys := benchBuildFilter(w, numKeys)
				n := len(keys)

				b.ResetTimer()
				b.ReportAllocs()
				var sink bool
				for i := 0; i < b.N; i++ {
					sink = f.contains(keys[i%n])
				}
				_ = sink
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 2. Contains — true negative queries (non-member keys)
//
// Measures the same per-query path, but all queries return false. With the
// ICML layout (paper §5.2) the query decodes the r result columns one at a
// time and SHORT-CIRCUITS on the first mismatched column (D3), so true
// negatives are markedly faster than true positives. This Contains path also
// pays XXH3 (~10 ns), which narrows the relative gap versus ContainsHash.
//
// Reference results (x86 Xeon Platinum, Go 1.25.x, n=10000): true-negative
// cost is below true-positive cost because a non-member mismatches the first
// result column with probability ~1/2 and exits early.
//
// Key observations:
//   • True negatives short-circuit on the first mismatched result column, so
//     adversarial workloads (all non-member queries) are cheaper than member
//     queries; do not rely on this path for constant-time behaviour.
// ---------------------------------------------------------------------------

func BenchmarkContains_TrueNegative(b *testing.B) {
	for _, w := range []uint32{32, 64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 10000
			f, _ := benchBuildFilter(w, numKeys)

			// Pre-generate non-member keys.
			probes := make([]string, 4096)
			for i := range probes {
				probes[i] = fmt.Sprintf("bench_negative_%d", i)
			}
			mask := len(probes) - 1 // power of 2 for cheap modulo

			b.ResetTimer()
			b.ReportAllocs()
			var sink bool
			for i := 0; i < b.N; i++ {
				sink = f.contains(probes[i&mask])
			}
			_ = sink
		})
	}
}

// ---------------------------------------------------------------------------
// 3. ContainsHash — pre-hashed query (isolates derive + dot product)
//
// Removes the XXH3 Phase 1 hash from the measured path. Measures
// only the Phase 2 (derive) + dot product + comparison cost.
//
// This is the pure "filter query" cost, useful for comparing against
// other filter implementations that separate hashing from querying.
//
// Reference results (x86 Xeon Platinum, Go 1.25.x, ICML, n=10000):
//
//   Width   TP ns/op   TN ns/op   B/op   allocs/op
//   ─────   ────────   ────────   ────   ─────────
//   w=32    ~20.4      ~8.9       0      0
//   w=64    ~19.2      ~8.4       0      0
//   w=128   ~34.6      ~10.7      0      0
//
// Key observations:
//   • ICML decodes the r result columns one at a time (r POPCNT-based column
//     parities) and short-circuits on the first mismatch (D3).
//   • True positives confirm all r columns; the cost is roughly flat in w
//     (r column parities, not ~w/2 byte XORs), so w=64 no longer costs ~2×
//     w=32. w=128 costs more only because each column parity folds two
//     64-bit halves.
//   • True negatives exit on the first mismatched column (prob ~1/2 at
//     column 0), so TN ≪ TP — the opposite of the old row-major layout,
//     where the full dot product was always computed.
// ---------------------------------------------------------------------------

func BenchmarkContainsHash_TruePositive(b *testing.B) {
	for _, w := range []uint32{32, 64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 10000
			f, memberHashes, _ := benchBuildFilterHashes(w, numKeys)
			mask := numKeys - 1

			b.ResetTimer()
			b.ReportAllocs()
			var sink bool
			for i := 0; i < b.N; i++ {
				sink = f.containsHash(memberHashes[i&mask])
			}
			_ = sink
		})
	}
}

func BenchmarkContainsHash_TrueNegative(b *testing.B) {
	for _, w := range []uint32{32, 64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 10000
			f, _, nonMemberHashes := benchBuildFilterHashes(w, numKeys)
			mask := numKeys - 1

			b.ResetTimer()
			b.ReportAllocs()
			var sink bool
			for i := 0; i < b.N; i++ {
				sink = f.containsHash(nonMemberHashes[i&mask])
			}
			_ = sink
		})
	}
}

// ---------------------------------------------------------------------------
// 4. Build — construction throughput
//
// Measures the full construction pipeline: Phase 1 hashing + Phase 2
// derive + banding (with retry loop) + back-substitution + filter
// packaging. Reports ns/op which includes the cost of all retries.
//
// The dominant cost is banding (O(numKeys) per seed attempt). With a
// typical overhead ratio (1.05 for w=128), 1–3 seeds suffice.
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   Keys      ns/op        B/op        allocs/op
//   ─────   ──────    ──────────   ─────────   ─────────
//   w=64    1,000       107,618      54,160    8
//   w=64    10,000      332,135     532,496    8
//   w=64    100,000   3,820,237   5,226,640    8
//   w=128   1,000       157,540      60,816    9
//   w=128   10,000      698,448     622,608    9
//   w=128   100,000   6,744,833   5,996,688    9
//
// Key observations:
//   • w=64/n=1K requires 2 seed attempts (seed 0 fails banding), doubling
//     its cost to ~107 µs vs ~33 ns/key amortised at larger n.
//   • Scales linearly with key count at n≥10K: ~33.2 ns/key (w=64) and
//     ~69.8 ns/key (w=128) at n=10K; ~38.2 ns/key and ~67.4 ns/key at
//     n=100K.
//   • Only 8–9 allocations regardless of scale: hasher (1), key hashes (1),
//     bander arrays (2–3 for coeffLo/coeffHi/result), hashResults (1),
//     solution struct (1), solution data (1).
//   • Memory: ~5.2 B/key at 100K (w=64). Dominated by the bander's SoA
//     arrays (coeffLo + result) and the solution data.
//   • w=128 is ~1.8× slower than w=64 at n=100K, reflecting the wider
//     coefficient rows in both banding and back-substitution.
//   • Construction of a 100K-key filter takes ~3.8 ms (w=64) or ~6.7 ms
//     (w=128) — fast enough for online use cases.
// ---------------------------------------------------------------------------

func BenchmarkBuildFilter(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		for _, numKeys := range []int{1000, 10000, 100000} {
			name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
			b.Run(name, func(b *testing.B) {
				keys := make([]string, numKeys)
				for i := range keys {
					keys[i] = fmt.Sprintf("bench_build_key_%d", i)
				}

				cfg := Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: true,
				}

				b.ResetTimer()
				b.ReportAllocs()
				var sink *filter
				for i := 0; i < b.N; i++ {
					var err error
					sink, err = buildFilter(keys, cfg)
					if err != nil {
						b.Fatal(err)
					}
				}
				_ = sink
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 5. BuildFromHashes — construction from pre-hashed keys
//
// Isolates the Phase 2 + banding + solving cost by removing the XXH3
// Phase 1 hash from the measured path.
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   Keys      ns/op        B/op        allocs/op
//   ─────   ──────    ──────────   ─────────   ─────────
//   w=64    1,000        47,821      45,968    7
//   w=64    10,000      289,695     450,577    7
//   w=64    100,000   3,660,172   4,423,833    7
//   w=128   1,000       129,142      52,624    8
//   w=128   10,000      820,365     540,689    8
//   w=128   100,000  10,842,208   5,193,881    8
//
// Key observations:
//   • Saves ~60 µs at n=1K vs Build (48 vs 108 µs for w=64) by skipping
//     Phase 1 hashing. The gap narrows at large n where banding dominates.
//   • One fewer allocation than Build (7 vs 8 for w=64): the key-hash
//     slice is caller-provided, not internally allocated.
//   • w=128/n=100K shows ~10.8 ms — higher than Build's 6.7 ms — likely
//     due to seed retry variance (more retries in this particular run).
//     Amortised cost is comparable.
//   • Memory at 100K: ~4.4 MB (w=64), ~5.2 MB (w=128) — same order as
//     Build, dominated by bander SoA arrays and solution data.
// ---------------------------------------------------------------------------

func BenchmarkBuildFromHashes(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		for _, numKeys := range []int{1000, 10000, 100000} {
			name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
			b.Run(name, func(b *testing.B) {
				hashes := make([]uint64, numKeys)
				for i := range hashes {
					hashes[i] = xxh3.HashString(fmt.Sprintf("bench_build_hash_%d", i))
				}

				cfg := Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: true,
				}

				b.ResetTimer()
				b.ReportAllocs()
				var sink *filter
				for i := 0; i < b.N; i++ {
					var err error
					sink, err = buildFromHashes(hashes, cfg)
					if err != nil {
						b.Fatal(err)
					}
				}
				_ = sink
			})
		}
	}
}
