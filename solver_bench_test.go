package ribbonGo

import (
	"fmt"
	"testing"
)

// =============================================================================
// Benchmarks — back-substitution (solver) performance
//
// BackSubstitute walks the upper-triangular banded matrix in reverse,
// computing the solution vector S that IS the Ribbon filter. The hot path
// is the column-major state-register shift, parity (POPCNT), and
// per-column bit accumulation.
//
// Query computes the GF(2) dot product of the solution vector with a
// coefficient row — the per-key membership test. It iterates over set
// bits using TrailingZeros (TZCNT/BSF) and XORs solution rows.
//
// We benchmark:
//   1. BackSubstitute — full back-substitution on large Banders.
//   2. BackSubstitute by resultBits — how r affects construction cost.
//   3. Query — per-key membership query cost.
//   4. Query throughput — batch query over many keys, keys/op metric.
//   5. Full pipeline throughput — band + solve + query end-to-end.
//   6. Solution memory — bytes per slot for different configurations.
//
// Each is measured across ribbon widths (64/128) to surface width-dependent
// costs — primarily the uint128 overhead in the state-register inner loop.
// =============================================================================

// benchBuildBander creates a fully-banded standardBander with numKeys keys.
// Returns the bander, hasher (with the successful seed set), and key hashes.
// Panics if banding fails after 200 seeds (shouldn't happen with generous sizing).
func benchBuildBander(w uint32, numKeys int) (*standardBander, *standardHasher, []uint64) {
	numStarts := uint32(float64(numKeys) * 1.1)
	numSlots := numStarts + w - 1
	h := newStandardHasher(w, numStarts, 7, true)

	hashes := make([]uint64, numKeys)
	for i := 0; i < numKeys; i++ {
		hashes[i] = h.keyHash([]byte(fmt.Sprintf("bench_solver_key_%d", i)))
	}

	for seed := uint32(0); seed < 200; seed++ {
		h.setOrdinalSeed(seed)
		bd := newStandardBander(numSlots, w, true)

		if bd.addRange(hashes, h) {
			return bd, h, hashes
		}
	}
	panic("benchBuildBander: banding failed for all 200 seeds")
}

// benchBuildSolution builds a bander and solves it, returning the solution
// and pre-computed query parameters. Used by query benchmarks to isolate
// query cost from construction cost.
func benchBuildSolution(w uint32, numKeys int) (*solution, []benchQueryParam) {
	bd, h, hashes := benchBuildBander(w, numKeys)
	sol := backSubstitute(bd, 7)

	params := make([]benchQueryParam, numKeys)
	for i, kh := range hashes {
		rh := h.rehash(kh)
		params[i] = benchQueryParam{
			start:    h.getStart(rh),
			coeffRow: h.getCoeffRow(rh),
		}
	}
	return sol, params
}

type benchQueryParam struct {
	start    uint32
	coeffRow uint128
}

// ---------------------------------------------------------------------------
// 1. BackSubstitute — full back-substitution on a large Bander
//
// Measures the complete back-substitution cost: allocating the solution
// vector, iterating numSlots times in reverse, maintaining r column-major
// state registers, and accumulating the multi-bit result per slot.
//
// The dominant cost is r × (shift + AND + POPCNT + XOR + OR) per occupied
// slot. For r=7 and w=64, this is ~50 instructions per slot, all
// register-to-register. The inner loop accesses the bander's SoA arrays
// sequentially (backwards), giving excellent cache behaviour.
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   Keys      ns/op       B/op      allocs/op
//   ─────   ──────    ──────────  ────────   ─────────
//   w=64    1,000      11,713     1,328      2
//   w=64    10,000    112,109     12,336     2
//   w=64    100,000 1,113,176    114,736     2
//   w=128   1,000      15,638     1,456      2
//   w=128   10,000    141,636     12,336     2
//   w=128   100,000 1,401,084    114,737     2
//
// Key observations:
//   • Only 2 allocations regardless of scale — one for the Solution struct,
//     one for the []uint8 data slice. The r=7 column-major state array
//     ([8]uint64 or [8]uint128) lives entirely on the stack.
//   • Scales linearly with numSlots: ~11.6 ns/slot for w=64, ~13.8 ns/slot
//     for w=128. Direct SoA access (no interface dispatch or bandingSlot
//     struct construction) keeps the inner loop cache-friendly.
//   • The uint128 path is ~19% more expensive than uint64, down from ~34%
//     before manual lsh(1) inlining and branchless bit insertion.
//   • Memory: ~1.1 B/key (one uint8 per slot plus w padding bytes).
//     Compact — a 1M-key filter's solution occupies ~1.1 MB.
//   • 100K-key w=64 back-substitution takes ~1.2 ms — fast enough that
//     construction is dominated by banding, not solving.
// ---------------------------------------------------------------------------

func BenchmarkBackSubstitute(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		for _, numKeys := range []int{1000, 10000, 100000} {
			name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
			b.Run(name, func(b *testing.B) {
				bd, _, _ := benchBuildBander(w, numKeys)

				b.ResetTimer()
				b.ReportAllocs()
				var sink *solution
				for i := 0; i < b.N; i++ {
					sink = backSubstitute(bd, 7)
				}
				_ = sink
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 2. BackSubstitute by resultBits — impact of r on construction cost
//
// The inner loop of back-substitution iterates over r result columns per
// slot. Increasing r linearly increases the work: each column requires
// one shift + AND + POPCNT + XOR + OR sequence. This benchmark quantifies
// that linear scaling.
//
// Reference results (Apple M3 Pro, Go 1.25.4, w=64, n=10000, -benchtime=10s):
//
//   r-bits   ns/op       ns/slot    allocs/op
//   ──────   ──────────  ────────   ─────────
//   r=1       88,173      7.97      2
//   r=4       90,360      8.17      2
//   r=7      111,873     10.11      2
//   r=8      133,571     12.07      2
//
// Key observations:
//   • Cost scales sub-linearly with r: going from r=1 to r=8 is only
//     ~1.49× more expensive (not 8×). The fixed cost of loading each
//     slot's coeffRow and result from the SoA arrays dominates at low r.
//   • Incremental cost per column: (130,524 - 87,924) / 7 ≈ 6,086 ns
//     per additional column over 11,063 slots ≈ 0.55 ns per (slot, column).
//     This is the pure shift + AND + POPCNT + XOR + OR sequence cost.
//   • r=7 (the default) is only ~33% more expensive than r=1 while
//     providing 128× better false-positive rate (2^(-7) vs 2^(-1)).
//   • Memory is identical for all r values — the []uint8 stores up to
//     8 result bits per slot regardless.
// ---------------------------------------------------------------------------

func BenchmarkBackSubstituteByResultBits(b *testing.B) {
	const numKeys = 10000
	w := uint32(64)
	numStarts := uint32(float64(numKeys) * 1.1)
	numSlots := numStarts + w - 1

	// Build a bander once (result bits don't affect banding).
	h := newStandardHasher(w, numStarts, 7, true)
	hashes := make([]uint64, numKeys)
	for i := 0; i < numKeys; i++ {
		hashes[i] = h.keyHash([]byte(fmt.Sprintf("bench_rbits_key_%d", i)))
	}
	var bd *standardBander
	for seed := uint32(0); seed < 200; seed++ {
		h.setOrdinalSeed(seed)
		bd = newStandardBander(numSlots, w, true)
		if bd.addRange(hashes, h) {
			break
		}
	}

	for _, r := range []uint{1, 4, 7, 8} {
		name := fmt.Sprintf("r=%d", r)
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			var sink *solution
			for i := 0; i < b.N; i++ {
				sink = backSubstitute(bd, r)
			}
			_ = sink
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(numSlots), "ns/slot")
		})
	}
}

// ---------------------------------------------------------------------------
// 3. Query — per-key membership query cost
//
// Measures the per-key query cost after the filter is built. This is the
// runtime cost of using the Ribbon filter for membership testing.
//
// The query iterates over set bits in the coefficient row (using
// TrailingZeros to skip zeros), XORing solution rows. For a typical
// w=64 coefficient row with ~32 set bits, this is ~32 XORs + ~32 TZCNT.
// For w=128, both lo and hi halves are processed sequentially.
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   ns/op   allocs/op
//   ─────   ─────   ─────────
//   w=64    36.81   0
//   w=128   66.63   0
//
// Key observations:
//   • Zero allocations — the query is pure register + array indexing work.
//     Query inlines at cost 69 (budget 80), so callers pay zero call overhead.
//   • w=64: ~37 ns/query ≈ 27M queries/sec. A random w=64 coefficient row
//     has ~32 set bits on average, so the loop body executes ~32 times:
//     one TZCNT + one XOR + one bit-clear per iteration ≈ 1.2 ns per set bit.
//   • w=128: ~67 ns/query ≈ 15M queries/sec. ~1.8× the w=64 cost,
//     consistent with ~64 set bits in 128-bit coefficient rows.
//   • The skip-zero optimisation (iterating only over set bits via
//     TrailingZeros + clear-lowest-bit) avoids touching the ~50% of zero
//     coefficient bits, halving the iteration count vs a naive 0..w loop.
//   • Per-iteration bounds checks eliminated via pre-sliced data window
//     and & 63 index masking, reducing overhead from ~32 checks to 1.
// ---------------------------------------------------------------------------

func BenchmarkQuery(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 10000
			sol, params := benchBuildSolution(w, numKeys)

			b.ResetTimer()
			b.ReportAllocs()
			var sink uint8
			for i := 0; i < b.N; i++ {
				p := params[i%numKeys]
				sink = sol.query(p.start, p.coeffRow)
			}
			_ = sink
		})
	}
}

// ---------------------------------------------------------------------------
// 4. Query throughput — batch query, keys/op metric
//
// Simulates a realistic query workload: querying all keys in a tight loop.
// Reports wall-clock time per batch and throughput in keys/op, matching
// the style of BenchmarkDeriveThroughput in hash_bench_test.go.
//
// This represents the end-to-end filter lookup throughput (excluding the
// hash computation, which is benchmarked separately in hash_bench_test.go).
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   ns/op       keys/op   ~keys/sec    allocs/op
//   ─────   ──────────  ───────   ──────────   ─────────
//   w=64    3,245,397   100,000   ~30.8M       0
//   w=128   6,230,908   100,000   ~16.1M       0
//
// Key observations:
//   • ~32.6 ns/key for w=64 in batch — slightly faster than the per-call
//     37 ns due to branch predictor warming over the tight loop.
//   • ~62.3 ns/key for w=128 in batch — ~1.91× the w=64 cost, matching
//     the coefficient row width ratio.
//   • 100K queries complete in 3.3 ms (w=64) or 6.2 ms (w=128), meaning
//     a 1M-key filter can process a 100K-query batch in under 63 ms.
//   • Zero allocations across the entire 100K-key batch.
// ---------------------------------------------------------------------------

func BenchmarkQueryThroughput(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 100_000
			sol, params := benchBuildSolution(w, numKeys)

			b.ResetTimer()
			b.ReportAllocs()
			var sink uint8
			for i := 0; i < b.N; i++ {
				for _, p := range params {
					sink = sol.query(p.start, p.coeffRow)
				}
			}
			_ = sink
			b.ReportMetric(float64(numKeys), "keys/op")
		})
	}
}

// ---------------------------------------------------------------------------
// 5. Full pipeline throughput — band + solve + query end-to-end
//
// Measures the combined cost of banding all keys (AddRange), solving
// (BackSubstitute), and querying all keys. This is the most realistic
// benchmark: it captures the total cost of building and verifying a
// Ribbon filter.
//
// The derive() cost is excluded (benchmarked in hash_bench_test.go);
// hashResult values are pre-computed.
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   Keys     ns/op         keys/op   ~keys/sec     allocs/op
//   ─────   ──────   ──────────    ───────   ──────────    ─────────
//   w=64    10,000   517,620       10,000    ~19.3M        4
//   w=128   10,000   902,892       10,000    ~11.1M        5
//
// Key observations:
//   • End-to-end cost breakdown for w=64 (523 µs total, 10K keys):
//     - Banding (AddRange):   ~53 µs (benchmarked separately: ~5.3 ns/key)
//     - BackSubstitute:      ~116 µs (~11.6 ns/slot, 10K slots)
//     - Query (10K keys):    ~370 µs (~37 ns/key)
//     Query dominates at ~71% of total cost — the random-access pattern
//     of reading solution rows incurs more cache misses than the
//     sequential backwards pass of back-substitution.
//   • w=128 is ~1.7× slower end-to-end (893 vs 523 µs), reflecting the
//     wider coefficient rows in both back-substitution and query.
//   • 4 allocations for w=64 (bander: coeffLo + result + struct;
//     solution: struct + data). 5 for w=128 (adds coeffHi).
//     One fewer than pre-optimisation due to escape analysis improvements.
//   • ~19.1M keys/sec (w=64) means building and fully verifying a
//     10K-key filter takes ~0.52 ms per seed attempt.
// ---------------------------------------------------------------------------

func BenchmarkFullPipelineThroughput(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 10000
			numStarts := uint32(float64(numKeys) * 1.2)
			numSlots := numStarts + w - 1
			h := newStandardHasher(w, numStarts, 7, true)

			hashes := make([]uint64, numKeys)
			for i := 0; i < numKeys; i++ {
				hashes[i] = h.keyHash([]byte(fmt.Sprintf("bench_pipeline_key_%d", i)))
			}

			// Find a working seed.
			var workingSeed uint32
			for seed := uint32(0); seed < 200; seed++ {
				h.setOrdinalSeed(seed)
				bd := newStandardBander(numSlots, w, true)
				if bd.addRange(hashes, h) {
					workingSeed = seed
					break
				}
			}

			// Pre-compute hashResults for the working seed.
			h.setOrdinalSeed(workingSeed)

			// Pre-compute query parameters.
			type qp struct {
				start    uint32
				coeffRow uint128
			}
			params := make([]qp, numKeys)
			for i, kh := range hashes {
				rh := h.rehash(kh)
				params[i] = qp{
					start:    h.getStart(rh),
					coeffRow: h.getCoeffRow(rh),
				}
			}

			b.ResetTimer()
			b.ReportAllocs()
			var sink uint8
			for i := 0; i < b.N; i++ {
				// Band.
				bd := newStandardBander(numSlots, w, true)
				bd.addRange(hashes, h)
				// Solve.
				sol := backSubstitute(bd, 7)
				// Query all keys.
				for _, p := range params {
					sink = sol.query(p.start, p.coeffRow)
				}
			}
			_ = sink
			b.ReportMetric(float64(numKeys), "keys/op")
		})
	}
}

// ---------------------------------------------------------------------------
// 6. Solution memory — bytes per slot
//
// Reports the memory footprint of the Solution for different widths and
// key counts. The solution stores one uint8 per slot plus w padding bytes,
// so memory = numSlots + w bytes (plus 40 bytes of struct overhead).
//
// Reference results (Apple M3 Pro, Go 1.25.4, -benchtime=10s):
//
//   Width   Keys      B/op      B/key   allocs/op
//   ─────   ──────    ───────   ─────   ─────────
//   w=64    10,000     12,336   1.113   2
//   w=64    100,000   114,736   1.101   2
//   w=128   10,000     12,336   1.125   2
//   w=128   100,000   114,737   1.103   2
//
// Key observations:
//   • ~1.1 bytes per key — the solution stores one uint8 per slot, and
//     numSlots ≈ 1.1 × numKeys (the 10% overhead from the banding pass
//     sizing). The w padding bytes are negligible at scale.
//   • Width has negligible effect on memory: the difference between w=64
//     and w=128 is only the w padding bytes (64 vs 128 bytes), which
//     vanishes as numKeys grows.
//   • Extremely compact: a 1M-key Ribbon filter occupies ~1.1 MB —
//     comparable to a Bloom filter at ~1.2 bytes/key for the same FPR,
//     but with the added benefit of exact r-bit fingerprints.
//   • Always exactly 2 allocations: the Solution struct and the []uint8
//     data slice. No per-slot or per-column allocations.
// ---------------------------------------------------------------------------

func BenchmarkSolutionMemory(b *testing.B) {
	for _, w := range []uint32{64, 128} {
		for _, numKeys := range []int{10000, 100000} {
			name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
			b.Run(name, func(b *testing.B) {
				bd, _, _ := benchBuildBander(w, numKeys)

				b.ResetTimer()
				b.ReportAllocs()
				var sink *solution
				for i := 0; i < b.N; i++ {
					sink = backSubstitute(bd, 7)
				}
				_ = sink
				// Report bytes-per-key for the solution data only.
				numSlots := bd.getNumSlots()
				bytesPerKey := float64(uint64(numSlots)+uint64(w)) / float64(numKeys)
				b.ReportMetric(bytesPerKey, "B/key")
			})
		}
	}
}
