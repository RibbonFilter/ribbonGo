package ribbon

import (
	"fmt"
	"testing"
)

// =============================================================================
// Benchmarks — banding (Gaussian elimination) hot path
//
// The Add() method is the innermost loop of filter construction. For each
// key, it performs on-the-fly Gaussian elimination over GF(2): finding the
// pivot (TrailingZeros), probing the slot, and XOR-reducing on collision.
//
// The benchmark pre-computes hashResult values so that the measured time
// is strictly the Gaussian elimination logic — no hashing overhead.
//
// We benchmark:
//   1. Add()        — per-key amortised cost of Gaussian elimination.
//   2. Add() with high load — measures collision chain impact.
//
// Each is measured across ribbon widths (32/64/128), with and without
// firstCoeffAlwaysOne, to surface any width-dependent or optimisation cost.
// =============================================================================

// benchHashResults generates N pre-computed hashResult values for benchmarking.
// Uses deterministic keys and seed 0.
func benchHashResults(w, numStarts uint32, fcao bool, n int) []hashResult {
	h := newStandardHasher(w, numStarts, 7, fcao)
	h.setOrdinalSeed(0)
	results := make([]hashResult, n)
	for i := range results {
		kh := h.keyHash([]byte(fmt.Sprintf("bench_bander_key_%d", i)))
		results[i] = h.derive(kh)
	}
	return results
}

// ---------------------------------------------------------------------------
// 1. Add — per-key amortised Gaussian elimination cost
//
// Inserts keys into a well-sized bander (numStarts >> numKeys) to minimise
// collision chains and measure the common-case single-probe fast path.
// The bander is reset every cycle to keep conditions consistent.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Width   fcao     ns/op   allocs/op
//   ─────   ─────    ─────   ─────────
//   w=32    true      4.99   0
//   w=32    false     5.71   0
//   w=64    true      4.99   0
//   w=64    false     5.73   0
//   w=128   true      6.67   0
//   w=128   false     7.33   0
//
// Key observations:
//   • Zero allocations — all values are stack-local or slice-indexed.
//   • SoA layout + width-specialised addW64/addW128: w≤64 now operates
//     purely on uint64 with no uint128 overhead — ~5.0 ns/key vs ~5.3
//     in v1 AoS (~6% faster).
//   • fcao gap narrowed dramatically for w≤64: ~14% (5.0→5.7) vs ~40%
//     in v1. The uint64 fast path has far less work to skip.
//   • w=128 is now measurably slower than w≤64 (6.7 vs 5.0 ns, +34%)
//     because addW128 requires two uint64 operations per step. In v1
//     all widths were ~equal because all used the same uint128 path.
//   • ~5.0 ns/key (w≤64, fcao=true) translates to ~200M insertions/sec.
// ---------------------------------------------------------------------------

func BenchmarkAdd(b *testing.B) {
	const numKeys = 4096 // power-of-2 for cheap masking

	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			b.Run(name, func(b *testing.B) {
				numStarts := uint32(10000)
				numSlots := numStarts + w - 1
				hashes := benchHashResults(w, numStarts, fcao, numKeys)
				bd := newStandardBander(numSlots, w, fcao)

				b.ResetTimer()
				b.ReportAllocs()
				var sink bool
				for i := 0; i < b.N; i++ {
					idx := i & (numKeys - 1)
					if idx == 0 {
						bd.reset()
					}
					sink = bd.Add(hashes[idx])
				}
				_ = sink
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 2. Add under high load — collision chain impact
//
// Uses tighter sizing (numStarts ≈ 1.1 * numKeys) so that collision chains
// are more frequent. This benchmarks the worst-case inner loop (multiple
// XOR iterations per Add call).
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Width   ns/op   allocs/op
//   ─────   ─────   ─────────
//   w=64    12.44   0
//   w=128   16.39   0
//
// Key observations:
//   • ~45% faster than v1 AoS (12.4 vs 22.8 for w=64, 16.4 vs 23.0 for
//     w=128). The SoA layout's improved cache density dominates here:
//     collision chains probe multiple sequential slots, and 8 uint64
//     coefficients per cache line (vs ~2.67 bandingSlot structs) means
//     far fewer L1 misses during chain traversal.
//   • Still zero allocations — the inner XOR loop is entirely stack-local.
//   • w=64 is now faster than w=128 (12.4 vs 16.4 ns, ~24%) because
//     addW64 avoids all hi-half operations during the XOR chain.
// ---------------------------------------------------------------------------

func BenchmarkAddHighLoad(b *testing.B) {
	const numKeys = 4096

	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			// Tight sizing: ~10% overhead.
			n := numKeys // avoid constant folding
			numStarts := uint32(float64(n)*1.1) + 1
			numSlots := numStarts + w - 1
			hashes := benchHashResults(w, numStarts, true, numKeys)
			bd := newStandardBander(numSlots, w, true)

			b.ResetTimer()
			b.ReportAllocs()
			var sink bool
			for i := 0; i < b.N; i++ {
				idx := i & (numKeys - 1)
				if idx == 0 {
					bd.reset()
				}
				sink = bd.Add(hashes[idx])
			}
			_ = sink
		})
	}
}

// ---------------------------------------------------------------------------
// 3. Full banding pass — throughput benchmark
//
// Simulates a complete banding pass: insert all N keys, measure wall-clock
// time and report keys/op throughput. This is the most realistic benchmark
// as it captures the actual mix of fast-path hits and collision chains.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Width   ns/op      keys/op   ~keys/sec     allocs/op
//   ─────   ────────   ───────   ──────────    ─────────
//   w=64     70,865    10,000    ~141M         0
//   w=128   111,909    10,000    ~89.4M        0
//
// Key observations:
//   • ~2× throughput improvement over v1 AoS (141M vs 68.4M keys/sec
//     for w=64; 89.4M vs 67.4M for w=128).
//   • ~7.1 ns/key amortised for w=64 over a full pass — the mix of
//     fast-path hits and collision chains. The SoA cache advantage
//     compounds over the full pass as the matrix fills.
//   • ~141M keys/sec (w=64) means a 1M-key filter's banding pass
//     takes ~7.1 ms per seed attempt (excluding derive() cost).
//   • w=64 is ~37% faster than w=128, reflecting the addW64 advantage.
// ---------------------------------------------------------------------------

func BenchmarkBandingPassThroughput(b *testing.B) {
	const numKeys = 10000

	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			// Generous sizing so the pass almost always succeeds.
			numStarts := uint32(float64(numKeys) * 1.2)
			numSlots := numStarts + w - 1
			hashes := benchHashResults(w, numStarts, true, numKeys)
			bd := newStandardBander(numSlots, w, true)

			b.ResetTimer()
			b.ReportAllocs()
			var sink bool
			for i := 0; i < b.N; i++ {
				bd.reset()
				for _, hr := range hashes {
					sink = bd.Add(hr)
				}
			}
			_ = sink
			b.ReportMetric(float64(numKeys), "keys/op")
		})
	}
}

// ---------------------------------------------------------------------------
// 4. Banding pass comparison — Add-loop vs AddRange (prefetching)
//
// Compares a plain Add-loop against AddRange (with software-pipelined
// prefetching) for full banding passes at two scales:
//
//   • 10K keys (~80KB coefficient array for w≤64) — fits in L1.
//     Prefetching should be a no-op; both methods equivalent.
//
//   • 100K keys (~800KB coefficient array for w≤64) — exceeds L1,
//     lives in L2. Prefetching converts random L2 misses into L1 hits,
//     showing a measurable throughput improvement.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Keys     Width   Method      ns/op        keys/op   ~keys/sec     allocs/op
//   ──────   ─────   ──────      ──────────   ───────   ──────────    ─────────
//   10K      w=64    Add-loop       71,574    10,000    ~140M         0
//   10K      w=64    AddRange       52,725    10,000    ~190M         0
//   10K      w=128   Add-loop      110,234    10,000    ~90.7M        0
//   10K      w=128   AddRange       94,640    10,000    ~106M         0
//   100K     w=64    Add-loop    1,835,023    100,000   ~54.5M        0
//   100K     w=64    AddRange    1,466,061    100,000   ~68.2M        0
//   100K     w=128   Add-loop    2,182,444    100,000   ~45.8M        0
//   100K     w=128   AddRange    1,954,879    100,000   ~51.2M        0
//
// Key observations:
//   • 10K/w=64: AddRange is ~26% faster (52.7 vs 71.6 µs). Even though
//     the 80KB coefficient array fits in L1, the prefetch eliminates
//     the overhead of method-call dispatch per key (AddRange inlines
//     the full elimination loop and amortises slice header loads).
//   • 100K/w=64: AddRange is ~20% faster (1.47 vs 1.84 ms). At 800KB,
//     the coefficient array spills to L2 — prefetching converts random
//     L2 misses into L1 hits on the common first-probe path.
//   • w=128 benefits less (~14–11%) because the elimination loop does
//     more work per key (two uint64 ops per step), leaving less time
//     for the prefetch to complete before the next key's access.
//   • Zero allocations in all cases.
// ---------------------------------------------------------------------------

func BenchmarkBandingPassComparison(b *testing.B) {
	for _, numKeys := range []int{10000, 100000} {
		for _, w := range []uint32{64, 128} {
			n := numKeys // avoid constant folding with float64(numKeys)
			numStarts := uint32(float64(n)*1.2) + 1
			numSlots := numStarts + w - 1
			hashes := benchHashResults(w, numStarts, true, numKeys)

			b.Run(fmt.Sprintf("keys=%d/w=%d/Add-loop", numKeys, w), func(b *testing.B) {
				bd := newStandardBander(numSlots, w, true)
				b.ResetTimer()
				b.ReportAllocs()
				var sink bool
				for i := 0; i < b.N; i++ {
					bd.reset()
					for _, hr := range hashes {
						sink = bd.Add(hr)
					}
				}
				_ = sink
				b.ReportMetric(float64(numKeys), "keys/op")
			})

			b.Run(fmt.Sprintf("keys=%d/w=%d/AddRange", numKeys, w), func(b *testing.B) {
				bd := newStandardBander(numSlots, w, true)
				b.ResetTimer()
				b.ReportAllocs()
				var sink bool
				for i := 0; i < b.N; i++ {
					bd.reset()
					sink = bd.AddRange(hashes)
				}
				_ = sink
				b.ReportMetric(float64(numKeys), "keys/op")
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 5. Reset — cost of clearing the slot array
//
// Measures the cost of clear(slots) for various slot array sizes.
// This is called once per seed retry, so it should be fast relative to
// the full banding pass.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Slots      ns/op     allocs/op
//   ──────     ───────   ─────────
//   1,000      162       0
//   10,000     1,514     0
//   100,000    15,353    0
//
// Key observations:
//   • Scales linearly with slot count — clear() compiles to memset.
//   • ~25% faster than v1 AoS reset: SoA clears three flat arrays
//     (coeffLo + coeffHi + result) instead of one array of 24-byte
//     structs with 7 bytes padding per element. Less total memory
//     to zero (17 bytes/slot for w=128, 9 bytes/slot for w≤64
//     vs 24 bytes/slot for AoS).
//   • 10K slots: 1.5 µs — negligible vs a banding pass (~71 µs).
//   • 100K slots: 15 µs — still <22% of a 100K-key pass.
// ---------------------------------------------------------------------------

func BenchmarkReset(b *testing.B) {
	for _, numSlots := range []uint32{1000, 10000, 100000} {
		name := fmt.Sprintf("slots=%d", numSlots)
		b.Run(name, func(b *testing.B) {
			bd := newStandardBander(numSlots, 128, true)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				bd.reset()
			}
		})
	}
}
