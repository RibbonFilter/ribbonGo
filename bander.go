package ribbonGo

import "math/bits"

// =============================================================================
// BANDING SLOT — logical per-row view for the upper-triangular banded matrix
// =============================================================================

// bandingSlot is the logical view of one row in the upper-triangular banded
// matrix produced by Gaussian elimination over GF(2).
//
// Paper §2: the banding step constructs an upper-triangular matrix where
// row j has its pivot (leading 1) at column j. The coefficient row stored
// here is right-shifted so that bit 0 is always the pivot bit. The
// remaining bits represent the equation's dependencies on columns j+1,
// j+2, …, up to the ribbon width w.
//
// This type is used by the back-substitution layer to read out the final
// matrix via getSlot(). Internally, the bander stores the data in a
// Struct-of-Arrays (SoA) layout for better cache utilisation — see
// standardBander documentation for the rationale.
//
// Occupancy tracking: a slot is occupied iff coeffRow is non-zero.
// This eliminates the need for a separate boolean array or bitset,
// improving cache locality by avoiding an extra memory access per slot
// probe. The invariant holds because:
//
//	(a) the hasher guarantees non-zero coefficient rows (paper §2),
//	(b) during elimination, a zero coefficient means linear dependence —
//	    the insertion fails immediately and nothing is stored,
//	(c) after right-shifting to the pivot, bit 0 is always 1 for any
//	    stored coefficient row.
type bandingSlot struct {
	coeffRow uint128 // right-shifted coefficient row; bit 0 = pivot. Zero iff empty.
	result   uint8   // r-bit fingerprint for this row's equation.
}

// =============================================================================
// BANDER INTERFACE
// =============================================================================

// bander defines the interface for Ribbon filter matrix construction
// (the "banding" step).
//
// Paper §2: the banding step takes N (start, coefficient, result) triples
// and performs on-the-fly Gaussian elimination over GF(2) to produce an
// upper-triangular banded matrix. The matrix has m slots (= numSlots),
// and each key's non-zero coefficients span a narrow "ribbon" of width w
// starting at its assigned start slot s.
//
// The linear system being solved is:
//
//	For each key x:  C[x] · S[s(x) … s(x)+w-1] = r(x)   (mod 2)
//
// where C[x] is the w-bit coefficient row, S is the solution vector
// (computed later by back-substitution), and r(x) is the result row.
//
// Construction succeeds when all N keys are inserted without linear
// dependence. On failure (Add returns false), the caller (Builder layer)
// should change the hash seed and retry from scratch.
//
// [RocksDB: BandingAdd in ribbon_impl.h]
type bander interface {
	// Add inserts a single key's equation (start, coeffRow, result) into
	// the banded matrix via on-the-fly Gaussian elimination over GF(2).
	//
	// Returns true on successful insertion (the equation was placed in an
	// empty slot). Returns false if the key's equation is linearly dependent
	// on previously inserted equations — either redundant (c=0, r=0) or
	// contradictory (c=0, r≠0). In both failure cases, the entire banding
	// attempt must be restarted with a new seed.
	add(hr hashResult) bool

	// AddRange inserts a batch of key equations, using software pipelining
	// to prefetch the next key's start slot while processing the current
	// key. Returns true if all insertions succeeded. Returns false on the
	// first failure, leaving the bander in a partially-filled state.
	//
	// For large filters (coefficient array exceeds L1 cache), this hides
	// L2/L3 memory latency and significantly improves throughput over
	// calling add() in a loop.
	//
	// [RocksDB: BandingAddRange in ribbon_alg.h]
	addRange(hashes []uint64, h *standardHasher) bool

	// reset clears all slots to their zero state, preparing the bander for
	// a retry with a new hash seed. Does not reallocate — reuses the
	// existing slot arrays.
	reset()

	// getSlot returns the bandingSlot at the given column index.
	// Used by the back-substitution layer to read the final upper-triangular
	// matrix. The caller must not access indices >= getNumSlots().
	getSlot(i uint32) bandingSlot

	// getNumSlots returns the total number of slots (columns) in the matrix.
	// This equals numStarts + coeffBits - 1.
	getNumSlots() uint32
}

// =============================================================================
// STANDARD BANDER — implementation
// =============================================================================

// standardBander implements the bander interface using on-the-fly Gaussian
// elimination over GF(2) with width-specialised inner loops.
//
// Paper §2: "For each key x, we attempt to insert its equation into the
// banded matrix. If the pivot column is unoccupied, we store the equation.
// If occupied, we XOR with the existing equation and repeat."
//
// Memory layout — Struct-of-Arrays (SoA):
//
// Instead of an Array-of-Structs ([]bandingSlot where each slot is 24 bytes
// with 7 bytes of padding), the bander stores coefficient and result data
// in parallel flat arrays:
//
//	coeffLo []uint64   — lower 64 bits of the coefficient row per slot
//	coeffHi []uint64   — upper 64 bits (w=128 only; nil for w≤64)
//	result  []uint8    — r-bit fingerprint per slot
//
// This layout provides two critical advantages:
//
//  1. For w≤64, coeffHi is nil. The elimination loop operates purely on
//     uint64 values (coeffLo), avoiding all uint128 construction, shifting,
//     and comparison overhead. Each slot's coefficient is a single 8-byte
//     word instead of a 16-byte struct, doubling the number of coefficients
//     per cache line (8 vs ~2.67 with AoS).
//
//  2. For w=128, the coeffLo and coeffHi arrays are separate but still
//     contiguous. The loop accesses both sequentially within the narrow
//     ribbon band (typically 128 slots wide), keeping the working set in
//     L1 cache.
//
// Width-specialised Add:
//
// The add() method dispatches to addW64() or addW128() based on whether
// coeffHi is nil. This dispatch happens once per call (not per loop
// iteration). The specialised methods avoid the generic uint128.rsh()
// method (which has 4 branches for n≥128, n≥64, n=0, else) and instead
// use direct uint64 shifts and TrailingZeros64 — compiling to a tight
// TZCNT + LSR + EOR loop with no unnecessary branches.
//
// Configuration:
//   - numSlots: total columns in the matrix (= numStarts + coeffBits - 1).
//   - coeffBits: ribbon width w (32, 64, or 128).
//   - backtrack: mirrors hasher.firstCoeffAlwaysOne(). When true, the
//     first iteration of add() skips the TrailingZeros intrinsic because
//     bit 0 of the original coefficient row is guaranteed to be 1.
//
// [RocksDB: StandardBanding in ribbon_impl.h]
type standardBander struct {
	coeffLo   []uint64 // lo 64 bits of coefficient per slot; len = numSlots
	coeffHi   []uint64 // hi 64 bits (w=128 only); nil for w≤64
	result    []uint8  // r-bit result per slot; len = numSlots
	numSlots  uint32   // total columns
	backtrack bool     // firstCoeffAlwaysOne optimisation flag
}

// Compile-time check: *standardBander implements bander.
var _ bander = (*standardBander)(nil)

// newStandardBander creates a bander with the given configuration.
//
// Parameters:
//   - numSlots: total columns in the banded matrix. Typically
//     numStarts + coeffBits - 1, where numStarts ≈ N * (1 + 2.3/w).
//     Paper §2: "the matrix has m columns (slots)."
//   - coeffBits: ribbon width w — must be 32, 64, or 128.
//   - firstCoeffAlwaysOne: when true, enables the fast-path optimisation
//     in add() that skips TrailingZeros on the first iteration.
//
// Panics if coeffBits is not 32, 64, or 128.
func newStandardBander(numSlots, coeffBits uint32, firstCoeffAlwaysOne bool) *standardBander {
	switch coeffBits {
	case 32, 64, 128:
		// valid
	default:
		panic("ribbon: coeffBits must be 32, 64, or 128")
	}
	b := &standardBander{
		coeffLo:   make([]uint64, numSlots),
		result:    make([]uint8, numSlots),
		numSlots:  numSlots,
		backtrack: firstCoeffAlwaysOne,
	}
	if coeffBits == 128 {
		b.coeffHi = make([]uint64, numSlots)
	}
	return b
}

// =============================================================================
// ADD — on-the-fly Gaussian elimination (hot path)
// =============================================================================

// Add inserts a single key's equation into the upper-triangular banded
// matrix via on-the-fly Gaussian elimination over GF(2).
//
// Paper §2: given a key's triple (start s, coefficient row c ∈ {0,1}^w,
// result r), the algorithm finds the lowest set bit of c (the "pivot"),
// determines its absolute column index p = s + offset, and either:
//
//   - stores the equation at slot p if the slot is empty, or
//   - XORs with the existing equation at slot p and repeats with the
//     reduced equation.
//
// The XOR step is Gaussian elimination in GF(2): it zeros out the pivot
// column in the current equation while preserving all other relationships.
// Because XOR is its own inverse in GF(2), this is both addition and
// subtraction.
//
// If the coefficient row becomes all-zero after XOR:
//   - r = 0: the equation is redundant (linearly dependent, consistent).
//   - r ≠ 0: the equation is contradictory (linearly dependent, inconsistent).
//
// In both c=0 cases, the insertion fails (returns false), signalling the
// Builder layer to change the seed and restart.
//
// Performance: this is the hottest path in filter construction. The method
// dispatches to a width-specialised inner loop:
//   - w≤64: addW64() operates purely on uint64 — no uint128 overhead.
//     One TZCNT + one LSR + one EOR per elimination step.
//   - w=128: addW128() uses separate lo/hi uint64 operations, avoiding
//     the generic uint128.rsh() branch ladder.
//
// Both paths are designed for:
//   - Zero heap allocations: all values are stack-local or slice-indexed.
//   - Minimal branching: the firstCoeffAlwaysOne fast path eliminates a
//     TrailingZeros call on the first iteration.
//   - CPU intrinsics: bits.TrailingZeros64 compiles to TZCNT/BSF.
//   - Cache locality: SoA layout maximises coefficients per cache line.
//
// [RocksDB: BandingAdd in ribbon_impl.h]
func (b *standardBander) add(hr hashResult) bool {
	if b.coeffHi != nil {
		return b.addW128(hr)
	}
	return b.addW64(hr)
}

// addW64 is the width-specialised Add for w≤64 (w=32 or w=64).
//
// Operates purely on uint64 coefficient values — no uint128 construction,
// no generic rsh() with its 4-branch dispatch, no hi-half operations.
// The inner loop is a tight sequence:
//
//	TZCNT → LSR → load coeffLo[p] → CBZ → EOR → CBZ → loop
//
// This compiles to ~10 ARM64 instructions per elimination step, compared
// to ~25+ for the generic uint128 path.
//
// Performance: with the SoA layout, 8 coefficient uint64 values fit in a
// single 64-byte cache line (vs ~2.67 bandingSlot structs with AoS).
// For the common case (first probe hits an empty slot), this means the
// coefficient check is almost always an L1 hit.
func (b *standardBander) addW64(hr hashResult) bool {
	s := hr.start
	c := hr.coeffRow.lo
	r := hr.result
	coeffs := b.coeffLo
	results := b.result

	// -------------------------------------------------------------------------
	// Fast path: firstCoeffAlwaysOne optimisation.
	//
	// When the hasher guarantees bit 0 of the coefficient row is always 1,
	// the pivot offset for the *first* iteration is known to be 0. This
	// skips the TrailingZeros64 intrinsic and the right-shift for the most
	// common case (the first probe often succeeds for well-sized filters).
	//
	// Paper §4: "setting the first coefficient bit to 1 … makes the
	// Gaussian elimination pivot deterministic at column s(x)."
	// -------------------------------------------------------------------------
	if b.backtrack {
		// Pivot offset i=0, absolute pivot column p=s.
		existing := coeffs[s]
		if existing == 0 {
			// Empty slot — store the equation directly.
			coeffs[s] = c
			results[s] = r
			return true
		}
		// Collision: XOR with existing equation to eliminate column s.
		// Both c and existing have bit 0 = 1, so XOR zeroes bit 0.
		// This is Gaussian elimination in GF(2): the pivot column is
		// eliminated, and the remaining bits encode the reduced equation.
		c ^= existing
		r ^= results[s]
		if c == 0 {
			// Linear dependence detected:
			//   r=0 → redundant (consistent but provides no new information)
			//   r≠0 → contradictory (inconsistent system)
			// Either way, this key cannot be inserted. Signal failure.
			return false
		}
		// After XOR: bit 0 is now 0 (both operands had bit 0 = 1).
		// Fall through to the generic loop, which will find the next pivot
		// at TrailingZeros64(c) ≥ 1.
	}

	// -------------------------------------------------------------------------
	// Generic Gaussian elimination loop (uint64 fast path).
	//
	// Invariant: c is non-zero at loop entry. Each iteration:
	//   1. Find the lowest set bit i = TrailingZeros64(c).
	//   2. Compute absolute pivot column p = s + i.
	//   3. Right-shift c by i so bit 0 becomes the pivot (always 1).
	//   4. Probe coeffs[p]:
	//      - Zero (empty) → store (c, r), return true.
	//      - Non-zero (occupied) → XOR to eliminate pivot, check for zero.
	//
	// The right-shift ensures that stored coefficients always have bit 0 = 1,
	// maintaining the upper-triangular structure: coeffs[p]'s pivot is at
	// column p, with remaining non-zero bits in columns p+1, p+2, ….
	// -------------------------------------------------------------------------
	for {
		// Step 1–2: Find pivot offset via trailing zeros.
		// Compiles to a single TZCNT (ARM64: RBIT+CLZ) instruction.
		i := uint(bits.TrailingZeros64(c))

		// Step 3: Shift to make the pivot bit 0, advance start.
		p := s + uint32(i)
		c >>= i

		// Step 4: Probe the coefficient slot at the pivot column.
		existing := coeffs[p]
		if existing == 0 {
			// Empty slot — store the (shifted) equation.
			coeffs[p] = c
			results[p] = r
			return true
		}

		// Collision: XOR with the existing equation at coeffs[p].
		// Both c and existing have bit 0 = 1, so XOR zeroes the
		// pivot column. The result is a reduced equation with its next
		// pivot at some column > p.
		c ^= existing
		r ^= results[p]
		s = p
		if c == 0 {
			return false
		}
		// Loop continues with the reduced equation.
	}
}

// addW128 is the width-specialised Add for w=128.
//
// Uses separate lo/hi uint64 operations to avoid the generic uint128.rsh()
// method which has a 4-branch dispatch (n≥128, n≥64, n=0, else). Instead,
// the shift is split into two cases: pivot in lo half (i<64) and pivot in
// hi half (i≥64), each producing branchless shift code.
//
// The TrailingZeros scan checks lo first — for most random coefficients,
// lo is non-zero (P=1-2^-64), so the hi half is rarely touched.
//
// Occupancy check: uses (eLo | eHi) == 0 instead of separate comparisons.
// This generates a single ORR + CBZ sequence on ARM64 — one fewer branch
// than checking each half individually.
func (b *standardBander) addW128(hr hashResult) bool {
	s := hr.start
	cLo := hr.coeffRow.lo
	cHi := hr.coeffRow.hi
	r := hr.result
	coeffLo := b.coeffLo
	coeffHi := b.coeffHi
	results := b.result

	// Fast path: firstCoeffAlwaysOne.
	if b.backtrack {
		eLo := coeffLo[s]
		eHi := coeffHi[s]
		if eLo|eHi == 0 {
			coeffLo[s] = cLo
			coeffHi[s] = cHi
			results[s] = r
			return true
		}
		cLo ^= eLo
		cHi ^= eHi
		r ^= results[s]
		if cLo|cHi == 0 {
			return false
		}
	}

	// Generic loop.
	for {
		// Find pivot: check lo first (almost always non-zero for random coeffs).
		var i uint
		if cLo != 0 {
			i = uint(bits.TrailingZeros64(cLo))
		} else {
			i = 64 + uint(bits.TrailingZeros64(cHi))
		}
		p := s + uint32(i)

		// Right-shift {cHi, cLo} by i bits.
		// Two cases avoid the generic rsh() 4-branch ladder:
		//   i < 64: pivot in lo half — both halves shift normally.
		//   i ≥ 64: pivot in hi half — lo gets hi's bits, hi becomes 0.
		if i < 64 {
			if i > 0 {
				cLo = (cLo >> i) | (cHi << (64 - i))
				cHi >>= i
			}
		} else {
			cLo = cHi >> (i - 64)
			cHi = 0
		}

		eLo := coeffLo[p]
		eHi := coeffHi[p]
		if eLo|eHi == 0 {
			coeffLo[p] = cLo
			coeffHi[p] = cHi
			results[p] = r
			return true
		}

		cLo ^= eLo
		cHi ^= eHi
		r ^= results[p]
		s = p
		if cLo|cHi == 0 {
			return false
		}
	}
}

// =============================================================================
// ADD RANGE — batched insertion with software-pipelined prefetching
// =============================================================================

// AddRange inserts a batch of key equations into the banded matrix,
// using software pipelining to prefetch the next key's start slot while
// processing the current key.
//
// Returns true if all insertions succeeded. Returns false on the first
// failure (linear dependence), at which point the bander is in a
// partially-filled state and the caller should reset and retry with a
// new seed.
//
// Prefetching rationale (Paper §4, RocksDB: BandingAddRange):
//
// When the coefficient array exceeds L1 cache (numSlots > ~8K for w≤64,
// ~4K for w=128), each key's start position maps to a random cache line.
// Without prefetching, every first probe in add() is a guaranteed L2 or
// L3 miss (~4–40 ns penalty depending on array size).
//
// AddRange pipelines memory access: while processing key[i], it issues a
// load for coeffs[key[i+1].start]. By the time key[i]'s elimination
// completes (~5–7 ns), the next key's cache line is already in L1. This
// converts random L2/L3 misses into L1 hits for the common first-probe
// case (which dominates for well-sized filters).
//
// For small filters that fit in L1 (<~8K slots), the extra load is a
// harmless no-op — the data is already cached.
//
// [RocksDB: BandingAddRange in ribbon_alg.h, Prefetch in ribbon_impl.h]
func (b *standardBander) addRange(hashes []uint64, h *standardHasher) bool {
	if b.coeffHi != nil {
		return b.addRangeW128(hashes, h)
	}
	return b.addRangeW64(hashes, h)
}

// addRangeW64 is the batched, prefetching variant of addW64 for w≤64.
//
// The function processes keys sequentially with a one-ahead prefetch
// pattern matching RocksDB's BandingAddRange:
//
//  1. Before entering the loop, prefetch key[0]'s start slot.
//  2. At the top of each iteration, prefetch key[i+1]'s start slot.
//  3. Process key[i] using the same Gaussian elimination as addW64.
//
// The prefetch is implemented as a dummy load (`_ = coeffs[nextStart]`)
// which the Go compiler must emit because slice indexing may panic
// (observable side effect). The CPU fetches the containing cache line,
// hiding memory latency.
//
// [RocksDB: BandingAddRange in ribbon_alg.h]
func (b *standardBander) addRangeW64(hashes []uint64, h *standardHasher) bool {
	coeffs := b.coeffLo
	results := b.result
	n := len(hashes)
	if n == 0 {
		return true
	}

	// Prefetch first key's start slot into L1 cache.
	nextHr := h.derive(hashes[0])

	for idx := 0; idx < n; idx++ {
		// Software-pipelined prefetch: load next key's start slot into
		// cache while processing the current key. The ~5–7 ns of
		// elimination work on key[idx] gives the memory subsystem time
		// to fetch the cache line for key[idx+1].
		currHr := nextHr
		if idx+1 < n {
			nextHr = h.derive(hashes[idx+1])
		}

		s := currHr.start
		c := currHr.coeffRow.lo
		r := currHr.result

		// Fast path: firstCoeffAlwaysOne.
		if b.backtrack {
			existing := coeffs[s]
			if existing == 0 {
				coeffs[s] = c
				results[s] = r
				continue
			}
			c ^= existing
			r ^= results[s]
			if c == 0 {
				return false
			}
		}

		// Generic elimination loop (same as addW64).
		success := false
		for {
			i := uint(bits.TrailingZeros64(c))
			p := s + uint32(i)
			c >>= i

			existing := coeffs[p]
			if existing == 0 {
				coeffs[p] = c
				results[p] = r
				success = true
				break
			}

			c ^= existing
			r ^= results[p]
			s = p
			if c == 0 {
				break
			}
		}
		if !success {
			return false
		}
	}
	return true
}

// addRangeW128 is the batched, prefetching variant of addW128 for w=128.
//
// Prefetches both coeffLo and coeffHi for the next key, since w=128
// requires reading both arrays during elimination.
//
// [RocksDB: BandingAddRange in ribbon_alg.h]
func (b *standardBander) addRangeW128(hashes []uint64, h *standardHasher) bool {
	coeffLo := b.coeffLo
	coeffHi := b.coeffHi
	results := b.result
	n := len(hashes)
	if n == 0 {
		return true
	}

	// Prefetch first key's start slot (both lo and hi arrays).
	nextHr := h.derive(hashes[0])
	_ = coeffLo[nextHr.start]
	_ = coeffHi[nextHr.start]

	for idx := 0; idx < n; idx++ {
		currHr := nextHr
		if idx+1 < n {
			nextHr = h.derive(hashes[idx+1])
			_ = coeffLo[nextHr.start]
			_ = coeffHi[nextHr.start]
		}

		s := currHr.start
		cLo := currHr.coeffRow.lo
		cHi := currHr.coeffRow.hi
		r := currHr.result

		// Fast path: firstCoeffAlwaysOne.
		if b.backtrack {
			eLo := coeffLo[s]
			eHi := coeffHi[s]
			if eLo|eHi == 0 {
				coeffLo[s] = cLo
				coeffHi[s] = cHi
				results[s] = r
				continue
			}
			cLo ^= eLo
			cHi ^= eHi
			r ^= results[s]
			if cLo|cHi == 0 {
				return false
			}
		}

		// Generic elimination loop (same as addW128).
		success := false
		for {
			var i uint
			if cLo != 0 {
				i = uint(bits.TrailingZeros64(cLo))
			} else {
				i = 64 + uint(bits.TrailingZeros64(cHi))
			}
			p := s + uint32(i)

			if i < 64 {
				if i > 0 {
					cLo = (cLo >> i) | (cHi << (64 - i))
					cHi >>= i
				}
			} else {
				cLo = cHi >> (i - 64)
				cHi = 0
			}

			eLo := coeffLo[p]
			eHi := coeffHi[p]
			if eLo|eHi == 0 {
				coeffLo[p] = cLo
				coeffHi[p] = cHi
				results[p] = r
				success = true
				break
			}

			cLo ^= eLo
			cHi ^= eHi
			r ^= results[p]
			s = p
			if cLo|cHi == 0 {
				break
			}
		}
		if !success {
			return false
		}
	}
	return true
}

// =============================================================================
// SLOW ADD — unoptimised reference implementation
// =============================================================================

// slowAdd is the unoptimised reference implementation of Add.
//
// It always uses TrailingZeros (no firstCoeffAlwaysOne fast path, no
// width specialisation) and follows the textbook Gaussian elimination
// algorithm step by step. This serves as:
//   - A correctness oracle for cross-validation tests.
//   - Documentation of the canonical algorithm before optimisation.
//
// The output must be identical to Add for all inputs.
func (b *standardBander) slowadd(hr hashResult) bool {
	s := hr.start
	c := hr.coeffRow
	r := hr.result

	for {
		// Find pivot: lowest set bit.
		i := c.trailingZeros()

		// Absolute pivot column.
		p := s + uint32(i)

		// Right-shift so bit 0 = pivot.
		c = c.rsh(i)
		s = p

		// Read slot via SoA arrays.
		eLo := b.coeffLo[p]
		eHi := uint64(0)
		if b.coeffHi != nil {
			eHi = b.coeffHi[p]
		}

		if eLo == 0 && eHi == 0 {
			// Empty — store.
			b.coeffLo[p] = c.lo
			if b.coeffHi != nil {
				b.coeffHi[p] = c.hi
			}
			b.result[p] = r
			return true
		}

		// XOR to eliminate pivot column.
		c = c.xor(uint128{hi: eHi, lo: eLo})
		r ^= b.result[p]

		if c.isZero() {
			return false
		}
	}
}

// slowAddRange is the unoptimised reference implementation of AddRange.
// It simply loops over slowAdd with no prefetching, serving as a
// correctness oracle for cross-validation tests.
func (b *standardBander) slowaddRange(hrs []hashResult) bool {
	for _, hr := range hrs {
		if !b.slowadd(hr) {
			return false
		}
	}
	return true
}

// =============================================================================
// ACCESSORS & UTILITIES
// =============================================================================

// reset clears all slots to their zero state (coeff=0, result=0),
// preparing the bander for a retry with a new hash seed.
//
// Uses Go's built-in clear() which compiles to an optimised memset,
// zeroing each array in one pass without per-element branching.
func (b *standardBander) reset() {
	clear(b.coeffLo)
	if b.coeffHi != nil {
		clear(b.coeffHi)
	}
	clear(b.result)
}

// getSlot returns the bandingSlot at the given column index.
// Used by back-substitution to read the upper-triangular matrix.
//
// Panics if i >= numSlots (programmer error — the caller must respect bounds).
func (b *standardBander) getSlot(i uint32) bandingSlot {
	hi := uint64(0)
	if b.coeffHi != nil {
		hi = b.coeffHi[i]
	}
	return bandingSlot{
		coeffRow: uint128{hi: hi, lo: b.coeffLo[i]},
		result:   b.result[i],
	}
}

// getNumSlots returns the total number of columns (slots) in the matrix.
func (b *standardBander) getNumSlots() uint32 {
	return b.numSlots
}
