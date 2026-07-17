package ribbonGo

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
// Memory layout (ICML — Interleaved Column-Major Layout, paper §5.2):
//
// The solution is stored in RocksDB's InterleavedSolutionStorage format:
// memory is divided into w-bit logical words, grouped into blocks of r
// words, where block b is the column-major transpose of solution rows
// [b·w, b·w + w). A query at start s reads column j from the two logical
// words that straddle the block boundary at s (block s/w and s/w+1),
// combines them with a shift by s%w, and takes the GF(2) dot product with
// the coefficient row.
//
// One trailing zero block is allocated beyond the numBlocks data blocks so
// the query may read word(blockIdx+1, j) unconditionally (no branch on the
// last block) while remaining in bounds under a single width-specific BCE
// proof at the top of containsHash. This replaces the old row-major
// "+128-byte" padding: bounds are proved per width from blockIdx and r.
//
// Physical storage is width-specific and selected by enc:
//   - w=64:  data64, logical word L → data64[L].
//   - w=128: data64, L → (data64[2L+1] : data64[2L]).
//   - w=32:  data32 (unpacked []uint32), L → data32[L]. Chosen over a
//     packed two-lanes-per-uint64 encoding by the Task 4 benchmark.
// Only one backing slice is populated per encoding.
//
// The standardHasher is stored by value (not pointer) so that its ~96
// bytes of pre-computed masks sit adjacent to the slice header in memory,
// improving cache locality on the hot query path.
//
// Thread safety: filter is immutable after construction. contains and
// containsHash may be called concurrently from multiple goroutines
// without synchronisation.
//
// [RocksDB: InterleavedSolutionStorage + InterleavedFilterQuery in ribbon_impl.h / ribbon_alg.h]
type filter struct {
	// data64 holds the ICML solution words (see icmlSolution) for w∈{64,128};
	// nil for w=32. Only one backing slice is used per encoding.
	data64 []uint64

	// data32 holds the ICML solution words for w=32 (unpacked); nil otherwise.
	data32 []uint32

	// numBlocks is numSlots / w — the number of ICML data blocks (excluding
	// the trailing zero block).
	numBlocks uint32

	// enc selects the physical ICML encoding for this filter's width, copied
	// from the icmlSolution so the query path maps logical→physical without
	// re-deriving it from the width.
	enc icmlEncoding

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
// entire library. The query decodes the ICML solution (paper §5.2) and is
// designed for:
//   - Zero heap allocations: hashResult is a value type (stack-allocated),
//     and derive() is inlineable (cost 67 < budget 80).
//   - Minimal branching: the numStarts==0 guard and, in containsHash, one
//     branch on the ICML encoding (w=32 vs w≤64 vs w=128).
//   - Bounds-check elimination: a single width-specific `_ = store[maxIdx]`
//     proof at the top of containsHash eliminates all per-column bounds
//     checks (the trailing zero block keeps the two-block read in bounds).
//   - Column short-circuit: containsHash compares result columns one at a
//     time and returns false on the first mismatch, so non-members exit
//     early (D3).
//
// [RocksDB: InterleavedFilterQuery in ribbon_alg.h]
func (f *filter) contains(key string) bool {
	if f.hasher.numStarts == 0 {
		return false
	}
	h := f.hasher.keyHashString(key)
	return f.containsHash(h)
}

// containsHash is the inlined ICML query core. It performs the full Phase 2
// derive + per-column GF(2) dot product with short-circuiting.
//
// Algorithm (paper §5.2, InterleavedFilterQuery):
//
//  1. derive(h) → (start, coeffRow, expectedResult).
//
//  2. blockIdx = start/w, offset = start%w. Rows [start, start+w) span
//     ICML block blockIdx (its upper w-offset rows) and block blockIdx+1
//     (its lower offset rows). For each result column j, read the two
//     logical words word(blockIdx, j) and word(blockIdx+1, j), combine
//     them with a shift by offset to reconstruct the w-bit column slice,
//     then bit_j = parity(slice & coeffRow).
//
//  3. Short-circuit (D3): compare bit_j to (expectedResult >> j) & 1 and
//     return false immediately on the first mismatch. A non-member fails
//     column 0 with probability 1/2, so negatives return ~2× faster on
//     average than the full r-column scan. Returns true after all r
//     columns match. (icmlSolution.icmlQuery is the non-short-circuit
//     oracle counterpart.)
//
// Bounds-check elimination (BCE):
//
// A single width-specific `_ = store[maxIdx]` proof at the top guarantees
// all per-column reads are in-bounds. The two-block read touches at most
// logical word (blockIdx+2)·r - 1; the trailing zero block ensures this is
// always allocated, so maxIdx is derived from blockIdx, r, and the
// width-specific logical→physical mapping — no per-column bounds checks and
// no branch on blockIdx == numBlocks-1.
//
// Returns false for empty filters (numStarts == 0).
//
// [RocksDB: InterleavedFilterQuery in ribbon_alg.h]
func (f *filter) containsHash(h uint64) bool {
	if f.hasher.numStarts == 0 {
		return false
	}

	hr := f.hasher.derive(h)
	w := f.hasher.coeffBits
	r := uint32(f.hasher.resultBits)

	blockIdx := hr.start / w
	offset := hr.start % w

	// Logical word index of word(blockIdx+1, r-1) — the highest logical word
	// the two-block read can touch this query.
	maxL := (blockIdx+2)*r - 1

	if w == 128 {
		return f.containsHash128(hr, blockIdx, offset, r, maxL)
	}
	return f.containsHash64(hr, w, blockIdx, offset, r, maxL)
}

// containsHash64 is the ICML short-circuit query for ribbon width w ≤ 64.
func (f *filter) containsHash64(hr hashResult, w, blockIdx, offset, r, maxL uint32) bool {
	c := hr.coeffRow.lo
	base0 := blockIdx * r
	base1 := (blockIdx + 1) * r

	if f.enc == encW32 {
		data := f.data32
		_ = data[maxL] // BCE proof: all reads below are in [0, maxL].
		mask := (uint64(1) << w) - 1 // w == 32 here
		for j := uint32(0); j < r; j++ {
			lo0 := uint64(data[base0+j])
			var slice uint64
			if offset == 0 {
				slice = lo0
			} else {
				lo1 := uint64(data[base1+j])
				slice = ((lo0 >> offset) | (lo1 << (w - offset))) & mask
			}
			bit := bits.OnesCount64(slice&c) & 1
			if bit != int((hr.result>>j)&1) {
				return false
			}
		}
		return true
	}

	// encW64: logical word L → data64[L]; full 64-bit slices (no masking).
	data := f.data64
	_ = data[maxL] // BCE proof: all reads below are in [0, maxL].
	for j := uint32(0); j < r; j++ {
		lo0 := data[base0+j]
		var slice uint64
		if offset == 0 {
			slice = lo0
		} else {
			lo1 := data[base1+j]
			slice = (lo0 >> offset) | (lo1 << (w - offset))
		}
		bit := bits.OnesCount64(slice&c) & 1
		if bit != int((hr.result>>j)&1) {
			return false
		}
	}
	return true
}

// containsHash128 is the ICML short-circuit query for ribbon width w = 128.
func (f *filter) containsHash128(hr hashResult, blockIdx, offset, r, maxL uint32) bool {
	data := f.data64
	_ = data[2*maxL+1] // BCE proof: highest physical (hi) index.

	cLo := hr.coeffRow.lo
	cHi := hr.coeffRow.hi
	base0 := blockIdx * r
	base1 := (blockIdx + 1) * r

	for j := uint32(0); j < r; j++ {
		l0 := base0 + j
		lo0 := data[2*l0]
		hi0 := data[2*l0+1]

		var sLo, sHi uint64
		if offset == 0 {
			sLo, sHi = lo0, hi0
		} else {
			l1 := base1 + j
			lo1 := data[2*l1]
			hi1 := data[2*l1+1]
			w0 := uint128{hi: hi0, lo: lo0}.rsh(uint(offset))
			w1 := uint128{hi: hi1, lo: lo1}.lsh(uint(128 - offset))
			s := w0.or(w1)
			sLo, sHi = s.lo, s.hi
		}
		bit := bits.OnesCount64((sLo&cLo)^(sHi&cHi)) & 1
		if bit != int((hr.result>>j)&1) {
			return false
		}
	}
	return true
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
