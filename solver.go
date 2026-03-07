package ribbon

import "math/bits"

// =============================================================================
// SOLUTION — the in-memory Ribbon filter (solution vector)
// =============================================================================

// solution holds the solved solution rows produced by back-substitution.
//
// After the Bander has constructed an upper-triangular banded matrix via
// on-the-fly Gaussian elimination (paper §2), back-substitution walks the
// matrix in reverse to compute the solution vector S. This vector IS the
// Ribbon filter — each query hashes a key to a (start, coefficients)
// pair, loads w consecutive solution rows, and computes the dot product
// over GF(2) per result column.
//
// Memory layout:
//
// The solution is stored as one result row (uint8) per slot — matching
// RocksDB's SimpleSolutionStorage. Each S[i] is a multi-bit value where
// bit j represents the solution for result column j at slot i.
//
// This is a row-major layout: data[slot_index] holds all r result
// bits for that slot. Queries iterate over w slots (the ribbon width),
// computing a dot product per result column simultaneously via branchless
// masking.
//
// Paper §3: the query for key x computes:
//
//	result = ⊕_{k=0}^{w-1} (c[k] ? S[s(x)+k] : 0)
//
// where c[k] is coefficient bit k, and S[i] is the r-bit solution row.
// A true member satisfies result == getResultRow(hash(key)).
//
// [RocksDB: InMemSimpleSolution in ribbon_impl.h]
type solution struct {
	data       []uint8 // one result row per slot; len = numSlots + w (padded)
	numSlots   uint32  // total number of slots from the bander
	coeffBits  uint32  // ribbon width w (32, 64, or 128)
	resultBits uint    // number of fingerprint bits r (e.g. 7)
}

// load returns the solution row at the given slot index.
// Used by the query path and tests. Returns the full r-bit result row.
func (s *solution) load(i uint32) uint8 {
	return s.data[i]
}

// =============================================================================
// PARITY — GF(2) dot product helpers
// =============================================================================

// parity64 returns the parity (XOR-sum) of the set bits in val.
// Returns 0 if the number of set bits is even, 1 if odd.
//
// Over GF(2), the dot product of two bit-vectors a and b is:
//
//	parity(a & b)
//
// Uses math/bits.OnesCount64, which compiles to a POPCNT instruction
// on x86-64 and a sequence of shifts on ARM64. The & 1 extracts parity.
func parity64(val uint64) int {
	return bits.OnesCount64(val) & 1
}

// parity128 returns the parity of the set bits in a 128-bit value.
// Reduces to a single parity64 by XOR-folding the two halves.
//
// Proof: parity(hi ^ lo) = parity(hi) ^ parity(lo), which equals
// the parity of the full 128-bit concatenation {hi, lo}.
func parity128(val uint128) int {
	return parity64(val.hi ^ val.lo)
}

// =============================================================================
// BACK-SUBSTITUTION — solving the upper-triangular system
// =============================================================================

// backSubstitute solves the upper-triangular banded linear system produced
// by the Bander, computing the solution vector S that will serve as the
// Ribbon filter's in-memory representation.
//
// The resultBits parameter r specifies how many fingerprint bits are stored
// per key (e.g. r=7 for FPR ≈ 2^(-7) ≈ 0.78%).
//
// Paper §2 & RocksDB SimpleBackSubst (ribbon_alg.h):
//
// The Bander has produced an m×m upper-triangular banded matrix where each
// occupied row i has:
//   - coeffRow c:  a w-bit coefficient row with c[0] = 1 (the pivot).
//   - result   r:  the r-bit fingerprint for that equation.
//
// Back-substitution walks backwards from i = m-1 down to i = 0, solving
// for each result column j independently.
//
// RocksDB tracks a column-major state buffer: state[j] is a CoeffRow-sized
// shift register holding the most recent w solution bits for column j.
// For each slot i and each result column j:
//
//	tmp = state[j] << 1
//	bit = parity(tmp & c) ^ ((r >> j) & 1)
//	tmp |= bit
//	state[j] = tmp
//	S[i].bit_j = bit
//
// This is equivalent to the scalar formulation:
//
//	S[i].bit_j = ((r >> j) & 1) ⊕ parity(S[i+1..i+w-1].bit_j & c[1..w-1])
//
// The state shift register elegantly avoids "unaligned reads" from the
// solution array: instead of reading back from memory, we maintain a
// sliding window of the last w bits per column in a register. The
// left-shift makes room for the new bit at position 0.
//
// Width-specialised inner loops:
//   - w≤64:  state[j] is uint64. Shift + AND + POPCNT is register-only.
//   - w=128: state[j] is uint128. Uses the uint128.lsh/and methods.
//
// Memory allocation: the solution ([]uint8) is allocated once. The state
// array is stack-allocated (r entries of CoeffRow size). The inner loop
// is allocation-free.
//
// [RocksDB: SimpleBackSubst in ribbon_alg.h]
func backSubstitute(sb *standardBander, resultBits uint) *solution {
	numSlots := sb.numSlots
	if numSlots == 0 {
		return &solution{
			data:       nil,
			numSlots:   0,
			coeffBits:  0,
			resultBits: resultBits,
		}
	}

	// Determine ribbon width from the concrete bander.
	coeffBits := uint32(64)
	if sb.coeffHi != nil {
		coeffBits = 128
	}

	// Allocate one result row per slot, plus w padding slots (zero-valued)
	// so that query can read w consecutive entries starting at any valid
	// start position without bounds checking.
	sol := &solution{
		data:       make([]uint8, uint64(numSlots)+uint64(coeffBits)),
		numSlots:   numSlots,
		coeffBits:  coeffBits,
		resultBits: resultBits,
	}

	// Dispatch to width-specialised back-substitution.
	if coeffBits <= 64 {
		backSubst64(sol, sb, numSlots, resultBits)
	} else {
		backSubst128(sol, sb, numSlots, resultBits)
	}

	return sol
}

// backSubst64 performs back-substitution for ribbon width w ≤ 64.
//
// Uses a column-major state buffer (one uint64 per result column) that
// acts as a shift register of the last w solution bits for each column.
// This avoids all unaligned reads from the solution array during
// construction — the sliding window is maintained entirely in registers
// (for small r, like r=7, the state fits in 7 general-purpose registers).
//
// For each slot i (reverse order) and each result column j:
//  1. Left-shift state[j] by 1 to make room for the new bit at position 0.
//  2. Compute: bit = parity(shifted_state & coeff) ⊕ result_bit_j.
//  3. OR the new bit into position 0 of the shifted state.
//  4. Store the updated state and accumulate the bit into the solution row.
//
// The left-shift naturally discards bits older than w positions because
// we're using a uint64 for w≤64. Bits that "fall off" the top are
// irrelevant — they correspond to slots more than w positions ahead,
// which can never appear in any coefficient row's reach.
//
// Uses the well-tested getSlot() accessor rather than direct array access.
// The compiler inlines getSlot (cost 32) and dead-code-eliminates the
// coeffHi nil branch since it's unreachable in the w≤64 path.
func backSubst64(sol *solution, sb *standardBander, numSlots uint32, resultBits uint) {
	// Clamp resultBits to the maximum supported by uint8 result rows.
	// This lets the compiler prove j < 8 for state[j] bounds elimination.
	if resultBits > 8 {
		resultBits = 8
	}

	data := sol.data[:numSlots]

	// Column-major state: state[j] holds the last w bits of column j.
	var state [8]uint64

	for i := int64(numSlots) - 1; i >= 0; i-- {
		slot := sb.getSlot(uint32(i))
		c := slot.coeffRow.lo
		r := slot.result

		var sr uint8

		for j := uint(0); j < resultBits; j++ {
			tmp := state[j] << 1

			bit := parity64(tmp&c) ^ int((r>>j)&1)
			tmp |= uint64(bit)

			state[j] = tmp
			sr |= uint8(bit) << j
		}

		data[i] = sr
	}
}

// backSubst128 performs back-substitution for ribbon width w = 128.
//
// Identical algorithm to backSubst64 but uses uint128 for the state
// registers and coefficient rows. Each shift register is 128 bits wide,
// matching the ribbon width.
//
// Uses getSlot() for data access (clean, tested accessor) but manually
// inlines the constant-1 shift and parity128 in the inner loop. The
// uint128.lsh() method has a 4-branch dispatch (n>=128, n>=64, n==0, else)
// that the compiler can't constant-fold after inlining, costing ~27%
// extra per slot. For the constant n=1, we expand to 2 shifts + 1 OR.
func backSubst128(sol *solution, sb *standardBander, numSlots uint32, resultBits uint) {
	if resultBits > 8 {
		resultBits = 8
	}

	data := sol.data[:numSlots]

	var state [8]uint128

	for i := int64(numSlots) - 1; i >= 0; i-- {
		slot := sb.getSlot(uint32(i))
		cLo := slot.coeffRow.lo
		cHi := slot.coeffRow.hi
		r := slot.result

		var sr uint8

		for j := uint(0); j < resultBits; j++ {
			// Manual lsh(1): avoids the 4-branch dispatch in uint128.lsh().
			sj := state[j]
			tmpLo := sj.lo << 1
			tmpHi := (sj.hi << 1) | (sj.lo >> 63)

			// Inline parity128(tmp.and(c)): 2 ANDs + 1 XOR + POPCNT.
			bit := bits.OnesCount64((tmpHi&cHi)^(tmpLo&cLo))&1 ^ int((r>>j)&1)

			tmpLo |= uint64(bit)

			state[j] = uint128{hi: tmpHi, lo: tmpLo}
			sr |= uint8(bit) << j
		}

		data[i] = sr
	}
}

// =============================================================================
// QUERY — verifying membership against the solution vector
// =============================================================================

// query computes the GF(2) dot product of the solution vector S with the
// coefficient row c, starting at position `start`. Returns the r-bit
// result row for comparison against the expected fingerprint.
//
// For a key that was in the original set:
//
//	query(start, coeffRow) == expectedResult
//
// For a key NOT in the set, this holds with probability 2^(-r).
//
// This is the "filter query" operation: hash the key to get
// (start, coeffRow, expectedResult), compute query(start, coeffRow),
// and compare with expectedResult.
//
// The query iterates over set bits in the coefficient row using
// TrailingZeros + clear-lowest-bit (TZCNT + BLSR on x86, RBIT+CLZ on
// ARM64). For each set bit k, it XORs S[start+k] into the result.
//
// Optimisations:
//   - Single function body: processes lo and hi halves sequentially,
//     eliminating the dispatch overhead of separate query64/query128
//     methods. For w=64, coeffRow.hi is always 0, so the second loop
//     is a zero-iteration no-op.
//   - Pre-sliced data: s.data[start:] is sliced once at the top,
//     converting per-iteration bounds checks (start + lsb < len) into
//     a single slice bounds check. The solution's w-padding bytes
//     guarantee start + w ≤ len(s.data).
//   - Skip-zero iteration: only set bits are visited via TrailingZeros
//   - clear-lowest-bit, halving the iteration count vs a naive 0..w
//     loop (~32 iterations for w=64 random coefficients instead of 64).
//
// [RocksDB: SimpleQueryHelper in ribbon_alg.h]
func (s *solution) query(start uint32, coeffRow uint128) uint8 {
	var result uint8
	// Single bounds check for the entire query window.
	data := s.data[start:]
	_ = data[127] // BCE proof: len(data) >= 128, eliminates per-iteration checks below.

	// Process the lower 64 bits (positions 0..63).
	lo := coeffRow.lo
	for lo != 0 {
		result ^= data[bits.TrailingZeros64(lo)&63]
		lo &= lo - 1
	}

	// Process the upper 64 bits (positions 64..127).
	// For w<=64, coeffRow.hi is always 0, making this a zero-iteration loop.
	hi := coeffRow.hi
	for hi != 0 {
		result ^= data[64+bits.TrailingZeros64(hi)&63]
		hi &= hi - 1
	}

	return result
}
