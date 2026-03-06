package ribbon

import (
	"fmt"
	"testing"
)

// =============================================================================
// Benchmarks — hash pipeline hot path
//
// The derive() function is the innermost loop of both banding (construction)
// and seed-retry. Every key passes through it once per seed attempt:
//
//   for _, h := range storedHashes { hr := hasher.derive(h) }
//
// Therefore its throughput directly determines construction speed.
//
// We benchmark:
//   1. derive()       — full pipeline (rehash → start + coeff + result)
//   2. rehash()       — seed remixing in isolation
//   3. getStart()     — fastRange64 mapping
//   4. getCoeffRow()  — coefficient derivation (width-dependent)
//   5. getResultRow() — fingerprint extraction
//   6. keyHash()      — Phase 1 (XXH3), for reference
//
// Each is measured across ribbon widths (32/64/128), with and without
// firstCoeffAlwaysOne, to surface any width-dependent cost.
// =============================================================================

// precomputedHashes generates N deterministic 64-bit hashes to feed into
// Phase 2 benchmarks, removing XXH3 from the measured path.
func precomputedHashes(h *standardHasher, n int) []uint64 {
	hashes := make([]uint64, n)
	for i := range hashes {
		hashes[i] = h.keyHash([]byte(fmt.Sprintf("bench_key_%d", i)))
	}
	return hashes
}

// ---------------------------------------------------------------------------
// 1. derive — full pipeline
//
// Full Phase 2 pipeline: rehash → getStart + getCoeffRow + getResultRow.
// This is the per-key cost on every seed attempt during banding.
//
// derive() is hand-inlined and branchless (no switch, no if). The compiler
// inlines it into the caller (cost 67, budget 80), so the benchmark loop
// runs the entire pipeline without a function-call boundary.
//
// Optimisations applied:
//   (a) Single h*kCoeffAndResultFactor multiply shared by coeff and result.
//   (b) Per-width switch replaced by pre-computed masks (coeffLoMask,
//       coeffHiMask, coeffXor) — branchless coefficient derivation.
//   (c) forceFirstCoeff branch replaced by pre-computed coeffOrMask.
//   (d) resultBits shift replaced by pre-computed resultMask.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Width   fcao     ns/op   allocs/op
//   ─────   ─────    ─────   ─────────
//   w=32    true      0.39   0
//   w=32    false     0.39   0
//   w=64    true      0.39   0
//   w=64    false     0.39   0
//   w=128   true      0.39   0
//   w=128   false     0.39   0
//
// Key observations:
//   • ~0.39 ns/key with zero allocations — fully inlined, branchless.
//   • All widths identical (masks absorb the width difference).
//   • 6.9× faster than the pre-optimisation baseline of ~2.7 ns/key.
// ---------------------------------------------------------------------------

func BenchmarkDerive(b *testing.B) {
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			b.Run(name, func(b *testing.B) {
				h := newStandardHasher(w, 10000, 7, fcao)
				h.setOrdinalSeed(0)
				hashes := precomputedHashes(h, 4096)
				mask := len(hashes) - 1 // power-of-2 for cheap modulo

				b.ResetTimer()
				b.ReportAllocs()
				var sink hashResult
				for i := 0; i < b.N; i++ {
					sink = h.derive(hashes[i&mask])
				}
				_ = sink
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 2. rehash — seed remixing
//
// Isolated cost of (hash ^ rawSeed) * kRehashFactor.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   ns/op   allocs/op
//   ─────   ─────────
//   0.39    0
//
// Key observation: single multiply — sub-nanosecond, fully inlined.
// ---------------------------------------------------------------------------

func BenchmarkRehash(b *testing.B) {
	h := newStandardHasher(128, 10000, 7, true)
	h.setOrdinalSeed(0)
	hashes := precomputedHashes(h, 4096)
	mask := len(hashes) - 1

	b.ResetTimer()
	b.ReportAllocs()
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink = h.rehash(hashes[i&mask])
	}
	_ = sink
}

// ---------------------------------------------------------------------------
// 3. getStart — fastRange64
//
// Maps a 64-bit hash uniformly into [0, numStarts) via 128-bit multiply.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   ns/op   allocs/op
//   ─────   ─────────
//   0.40    0
//
// Key observation: single math/bits.Mul64 — sub-nanosecond, fully inlined.
// ---------------------------------------------------------------------------

func BenchmarkGetStart(b *testing.B) {
	h := newStandardHasher(128, 10000, 7, true)
	h.setOrdinalSeed(0)
	hashes := precomputedHashes(h, 4096)
	mask := len(hashes) - 1

	// Pre-rehash so we measure getStart in isolation
	rehashed := make([]uint64, len(hashes))
	for i, kh := range hashes {
		rehashed[i] = h.rehash(kh)
	}

	b.ResetTimer()
	b.ReportAllocs()
	var sink uint32
	for i := 0; i < b.N; i++ {
		sink = h.getStart(rehashed[i&mask])
	}
	_ = sink
}

// ---------------------------------------------------------------------------
// 4. getCoeffRow — width-dependent coefficient derivation
//
// Derives the w-bit coefficient row from a rehashed hash. Contains a
// 3-way switch on w and an optional |=1 for forceFirstCoeff.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Width   fcao     ns/op   allocs/op
//   ─────   ─────    ─────   ─────────
//   w=32    true      0.39   0
//   w=32    false     0.39   0
//   w=64    true      0.39   0
//   w=64    false     0.39   0
//   w=128   true      0.39   0
//   w=128   false     0.39   0
//
// Key observations:
//   • All widths identical — the switch is branch-predicted perfectly
//     because the width is fixed for the hasher's lifetime.
//   • fcao adds no measurable cost.
// ---------------------------------------------------------------------------

func BenchmarkGetCoeffRow(b *testing.B) {
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			b.Run(name, func(b *testing.B) {
				h := newStandardHasher(w, 10000, 7, fcao)
				h.setOrdinalSeed(0)
				hashes := precomputedHashes(h, 4096)
				mask := len(hashes) - 1

				rehashed := make([]uint64, len(hashes))
				for i, kh := range hashes {
					rehashed[i] = h.rehash(kh)
				}

				b.ResetTimer()
				b.ReportAllocs()
				var sink uint128
				for i := 0; i < b.N; i++ {
					sink = h.getCoeffRow(rehashed[i&mask])
				}
				_ = sink
			})
		}
	}
}

// ---------------------------------------------------------------------------
// 5. getResultRow — fingerprint extraction
//
// Derives the r-bit fingerprint: multiply → byte-swap → mask.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   r-bits   ns/op   allocs/op
//   ──────   ─────   ─────────
//   r=1      0.39    0
//   r=7      0.40    0
//   r=8      0.39    0
//
// Key observation: identical cost regardless of r — the mask is a
// constant after construction, so there is no branch on r.
// ---------------------------------------------------------------------------

func BenchmarkGetResultRow(b *testing.B) {
	for _, r := range []uint{1, 7, 8} {
		name := fmt.Sprintf("r=%d", r)
		b.Run(name, func(b *testing.B) {
			h := newStandardHasher(128, 10000, r, true)
			h.setOrdinalSeed(0)
			hashes := precomputedHashes(h, 4096)
			mask := len(hashes) - 1

			rehashed := make([]uint64, len(hashes))
			for i, kh := range hashes {
				rehashed[i] = h.rehash(kh)
			}

			b.ResetTimer()
			b.ReportAllocs()
			var sink uint8
			for i := 0; i < b.N; i++ {
				sink = h.getResultRow(rehashed[i&mask])
			}
			_ = sink
		})
	}
}

// ---------------------------------------------------------------------------
// 6. keyHash (Phase 1) — XXH3, for reference baseline
//
// Phase 1 cost: XXH3_64bits over the raw key bytes. Called once per key
// (amortised across all seed attempts).
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Key size   ns/op    MB/s       allocs/op
//   ────────   ─────    ────────   ─────────
//   8 B        2.33     3,438      0
//   32 B       3.24     9,888      0
//   128 B      8.36     15,313     0
//   1024 B     49.66    20,622     0
//
// Key observations:
//   • For typical filter keys (8–32 B), keyHash is now the dominant cost
//     (~6× more expensive than derive at 0.39 ns).
//   • XXH3 saturates memory bandwidth around 20 GB/s for large keys.
//   • Total per-key cost (Phase 1 + Phase 2) ≈ 2.7 ns for 8 B keys.
// ---------------------------------------------------------------------------

func BenchmarkKeyHash(b *testing.B) {
	for _, size := range []int{8, 32, 128, 1024} {
		name := fmt.Sprintf("keySize=%d", size)
		b.Run(name, func(b *testing.B) {
			h := newStandardHasher(128, 10000, 7, true)
			key := make([]byte, size)
			for i := range key {
				key[i] = byte(i)
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(size))
			var sink uint64
			for i := 0; i < b.N; i++ {
				sink = h.keyHash(key)
			}
			_ = sink
		})
	}
}

// ---------------------------------------------------------------------------
// 7. Throughput — derive() in a realistic batch loop
//
// Simulates a full seed-attempt pass over 100K pre-hashed keys.
// Reports wall-clock time per pass and throughput in keys/op.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Width   ns/op     keys/op   ~keys/sec     allocs/op
//   ─────   ───────   ───────   ──────────    ─────────
//   w=64    39,142    100,000   ~2.56B        0
//   w=128   38,730    100,000   ~2.58B        0
//
// Key observations:
//   • ~0.39 ns/key in a tight loop — matches per-call BenchmarkDerive.
//   • ~2.5 billion keys/sec means a 1M-key filter's Phase 2 takes ~0.39 ms
//     per seed attempt.
//   • 7× faster than pre-optimisation baseline (~271 µs → ~39 µs).
//   • w=64 and w=128 are indistinguishable at batch scale.
// ---------------------------------------------------------------------------

func BenchmarkDeriveThroughput(b *testing.B) {
	// Simulates a full seed-attempt pass over N keys.
	// Reports throughput in keys/sec.
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		b.Run(name, func(b *testing.B) {
			const numKeys = 100_000
			h := newStandardHasher(w, numKeys, 7, true)
			h.setOrdinalSeed(0)
			hashes := precomputedHashes(h, numKeys)

			b.ResetTimer()
			b.ReportAllocs()
			var sink hashResult
			for i := 0; i < b.N; i++ {
				for _, kh := range hashes {
					sink = h.derive(kh)
				}
			}
			_ = sink
			b.ReportMetric(float64(numKeys), "keys/op")
		})
	}
}

// ---------------------------------------------------------------------------
// 8. Seed conversion — ordinalSeedToRaw / rawSeedToOrdinal
//
// Bijective mixing between small ordinal seeds and 64-bit raw seeds.
// Called once per seed attempt (not per key), so not on the hot path.
//
// Reference results (Apple M3 Pro, Go 1.25, -benchtime=2s):
//
//   Direction        ns/op   allocs/op
//   ─────────        ─────   ─────────
//   ordinalToRaw     0.39    0
//   rawToOrdinal     0.39    0
//
// Key observation: multiply + XOR — sub-nanosecond, negligible.
// ---------------------------------------------------------------------------

func BenchmarkSeedConversion(b *testing.B) {
	b.Run("ordinalToRaw", func(b *testing.B) {
		var sink uint64
		for i := 0; i < b.N; i++ {
			sink = ordinalSeedToRaw(uint32(i))
		}
		_ = sink
	})
	b.Run("rawToOrdinal", func(b *testing.B) {
		var sink uint32
		for i := 0; i < b.N; i++ {
			sink = rawSeedToOrdinal(uint64(i))
		}
		_ = sink
	})
}
