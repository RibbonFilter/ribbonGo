package ribbon

import "math/bits"

// =============================================================================
// FILTER — the read-only, thread-safe Ribbon filter
// =============================================================================

// filter is the immutable, query-ready Ribbon filter structure.
//
// After construction (via buildFilter or buildFromHashes), the filter
// supports fast, zero-allocation membership queries through contains and
// containsHash.
//
// Paper §2 & §3: the filter stores the solution vector S computed by
// back-substitution, plus the hash seed and configuration needed to
// reproduce the (start, coeffRow, result) triple for any key. A query
// hashes the key, derives the triple, computes the GF(2) dot product of
// the coefficient row against the solution window, and compares the
// computed result to the expected fingerprint.
//
// Memory layout:
//
// The solution vector is stored as one uint8 per slot (row-major), padded
// to numSlots + 128 bytes. The extra 128 bytes ensure that containsHash
// can use a single bounds-check elimination proof (`_ = data[127]`) for
// all ribbon widths (w = 32, 64, 128), avoiding per-iteration bounds
// checks in the dot-product loop.
//
// The standardHasher is stored by value (not pointer) so that its ~96
// bytes of pre-computed masks sit adjacent to the slice header in memory,
// improving cache locality on the hot query path.
//
// Thread safety: filter is immutable after construction. contains and
// containsHash may be called concurrently from multiple goroutines
// without synchronisation.
//
// [RocksDB: InMemSimpleSolution + SimpleFilterQuery in ribbon_impl.h / ribbon_alg.h]
type filter struct {
	// data holds the solution vector: one r-bit result row (uint8) per
	// slot, padded to numSlots + 128 bytes for bounds-check elimination.
	// Padding bytes are zero, matching empty (unoccupied) solution slots.
	data []uint8

	// hasher is the concrete standardHasher configured with the successful
	// seed and all per-width pre-computed masks (coeffLoMask, coeffHiMask,
	// coeffXor, coeffOrMask, resultMask). Stored by value for two reasons:
	//   (a) cache locality: the hasher's fields are in the same allocation
	//       as the filter, likely on the same cache line.
	//   (b) devirtualisation: calling derive() on a concrete struct (not an
	//       interface) allows the compiler to inline the method (cost 67 <
	//       budget 80), eliminating call overhead on the hot query path.
	hasher standardHasher

	// seed is the ordinal seed that succeeded during construction.
	// Stored for serialisation: the filter can be reconstructed from
	// (data[:numSlots], seed, config). Not used in the query path.
	seed uint32

	// numSlots is the total number of slots in the solution vector.
	// Equal to numStarts + coeffBits - 1. Used by accessors and
	// serialisation; not accessed in the query hot path.
	numSlots uint32
}

// =============================================================================
// CONTAINS — the membership query hot path
// =============================================================================

// contains tests whether key is a member of the set used to build this
// filter. Returns true if the key is probably in the set (with false-positive
// probability ≈ 2^(-r)), or false if the key is definitely not in the set.
//
// Paper §2: "To query whether x is a member, compute (s(x), c(x), r(x))
// and check whether c(x) · S[s(x)..s(x)+w-1] = r(x) over GF(2)."
//
// Performance: this method is the most frequently called code in the
// entire library. It is designed for:
//   - Zero heap allocations: hashResult is a value type (stack-allocated),
//     and derive() is inlineable (cost 67 < budget 80).
//   - Minimal branching: the only branch is the numStarts==0 guard
//     (always-false for non-empty filters, perfectly predicted).
//   - Bounds-check elimination: a single `_ = data[127]` proof at the
//     top of containsHash eliminates all per-iteration bounds checks.
//   - Skip-zero iteration: the dot-product loop iterates only over set
//     bits using TZCNT + clear-lowest-bit, halving iteration count vs
//     a naive 0..w loop (~w/2 iterations for random coefficient rows).
//
// [RocksDB: SimpleFilterQuery in ribbon_alg.h]
func (f *filter) contains(key []byte) bool {
	if f.hasher.numStarts == 0 {
		return false
	}
	h := f.hasher.keyHash(key)
	return f.containsHash(h)
}

// containsHash is the inlined query core. It performs the full Phase 2
// derive + GF(2) dot product in one tight sequence.
//
// Algorithm:
//
//  1. derive(h) → (start, coeffRow, expectedResult)
//     derive() is inlineable (cost 67) and computes:
//     - rehash: (h ^ rawSeed) * kRehashFactor
//     - start:  fastRange64(rehashed, numStarts)   [high bits]
//     - coeff:  multiply → mask/xor/or             [branchless]
//     - result: multiply → bswap → mask            [byte-swapped]
//
//  2. Dot product: iterate over set bits in coeffRow, XOR the
//     corresponding solution bytes into a running result.
//     Uses TrailingZeros64 (TZCNT/BSF) + clear-lowest-bit (BLSR)
//     to visit only the ~w/2 set bits in a random coefficient row.
//
//  3. Compare: return (computed == expected).
//
// Bounds-check elimination (BCE):
//
// A single `_ = data[127]` proof guarantees all per-iteration accesses
// are in-bounds. The filter's data is padded to numSlots + 128 bytes,
// and the maximum start is numStarts - 1 = numSlots - w, so
// data[start:] has length >= w + 128 >= 160 for all w in {32, 64, 128}.
//
// The `& 63` masks on TrailingZeros64 results provide a compile-time
// proof that lo-loop indices are in [0, 63] and hi-loop indices are
// in [64, 127], both <= 127, eliminating bounds checks entirely.
//
// Returns false for empty filters (numStarts == 0).
//
// [RocksDB: SimpleQueryHelper in ribbon_alg.h]
func (f *filter) containsHash(h uint64) bool {
	if f.hasher.numStarts == 0 {
		return false
	}

	hr := f.hasher.derive(h)

	// Slice the solution window starting at the key's start position.
	// The BCE proof ensures data[0..127] are all in-bounds.
	data := f.data[hr.start:]
	_ = data[127] // BCE proof: all subsequent accesses are within [0, 127].

	// GF(2) dot product: XOR solution bytes at each set coefficient bit.
	var result uint8

	// Process the lower 64 bits of the coefficient row (positions 0..63).
	lo := hr.coeffRow.lo
	for lo != 0 {
		result ^= data[bits.TrailingZeros64(lo)&63]
		lo &= lo - 1 // clear lowest set bit (BLSR)
	}

	// Process the upper 64 bits (positions 64..127).
	// For w <= 64, coeffRow.hi is always 0, making this a zero-iteration loop.
	hi := hr.coeffRow.hi
	for hi != 0 {
		result ^= data[64+bits.TrailingZeros64(hi)&63]
		hi &= hi - 1
	}

	return result == hr.result
}

// =============================================================================
// HELPERS — filter metadata for inspection and testing
// =============================================================================

// fpRate returns the theoretical false-positive rate: 2^(-r).
//
// Paper §3: "the false-positive probability is 2^(-r), where r is the
// number of result bits."
//
// Returns 0.0 for empty filters (no false positives when there are no keys).
func (f *filter) fpRate() float64 {
	if f.hasher.numStarts == 0 {
		return 0.0
	}
	return 1.0 / float64(uint64(1)<<f.hasher.resultBits)
}
