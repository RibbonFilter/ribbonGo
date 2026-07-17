package ribbonGo

import (
	"fmt"
	"testing"
)

// =============================================================================
// TEST HELPERS
// =============================================================================

// makeBanderFromSlots creates a standardBander with pre-set slot data,
// bypassing the normal add() path. This lets us construct hand-crafted
// upper-triangular matrices with known mathematical solutions.
func makeBanderFromSlots(numSlots, coeffBits uint32, slots []bandingSlot) *standardBander {
	b := newStandardBander(numSlots, coeffBits, true)
	for i, s := range slots {
		b.coeffLo[i] = s.coeffRow.lo
		if b.coeffHi != nil {
			b.coeffHi[i] = s.coeffRow.hi
		}
		b.result[i] = s.result
	}
	return b
}

// =============================================================================
// HAND-CRAFTED MATRIX TESTS — known mathematical solutions
//
// These tests use resultBits=1 for simplicity: each result is a single
// bit, and each S[i] is either 0x00 or 0x01. This makes the math
// easy to verify by hand.
// =============================================================================

func TestBackSubstitute_TrivialSingleSlot(t *testing.T) {
	// Simplest case: 1 slot with coefficient = 1, result = 1.
	// Equation: 1 · S[0] = 1  →  S[0] = 1.
	slots := make([]bandingSlot, 64)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 1}, result: 1}
	// All other slots are empty (coeffRow = 0), so S[1..63] = 0.
	b := makeBanderFromSlots(64, 64, slots)

	sol := backSubstitute(b, 1)

	if sol.load(0) != 1 {
		t.Errorf("S[0] = %d, want 1", sol.load(0))
	}
	// All other rows should be 0.
	for i := uint32(1); i < 64; i++ {
		if sol.load(i) != 0 {
			t.Errorf("S[%d] = %d, want 0", i, sol.load(i))
		}
	}
}

func TestBackSubstitute_TrivialSingleSlotZeroResult(t *testing.T) {
	// Equation: 1 · S[0] = 0  →  S[0] = 0.
	slots := make([]bandingSlot, 64)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 1}, result: 0}
	b := makeBanderFromSlots(64, 64, slots)

	sol := backSubstitute(b, 1)

	if sol.load(0) != 0 {
		t.Errorf("S[0] = %d, want 0", sol.load(0))
	}
}

func TestBackSubstitute_TwoSlots_WithDependency(t *testing.T) {
	// 2-slot system (numSlots=65, w=64):
	//
	// Slot 0: coeff = 0b11 (bits 0 and 1 set), result = 1
	//   Equation: S[0] ⊕ S[1] = 1
	//
	// Slot 1: coeff = 0b1 (bit 0 set), result = 0
	//   Equation: S[1] = 0
	//
	// Back-substitution (reverse order):
	//   i=1: S[1] = 0 (from equation: 1·S[1] = 0)
	//   i=0: S[0] = 1 ⊕ parity(state & 0b11)
	//     state[0] after i=1: shifted=0, bit=0, state=0
	//     At i=0: tmp = state<<1 = 0
	//       bit = parity(0 & 0b11) ^ (1 & 1) = 0 ^ 1 = 1
	//       state = 0 | 1 = 1
	//       S[0] = 1  ✓
	numSlots := uint32(65) // 2 starts + 64 - 1
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 0b11}, result: 1}
	slots[1] = bandingSlot{coeffRow: uint128{lo: 0b1}, result: 0}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	if sol.load(0) != 1 {
		t.Errorf("S[0] = %d, want 1", sol.load(0))
	}
	if sol.load(1) != 0 {
		t.Errorf("S[1] = %d, want 0", sol.load(1))
	}
}

func TestBackSubstitute_ThreeSlots_ChainDependency(t *testing.T) {
	// 3-slot chain:
	//
	// Slot 0: coeff = 0b111 (bits 0,1,2), result = 0
	//   Equation: S[0] ⊕ S[1] ⊕ S[2] = 0
	//
	// Slot 1: coeff = 0b11 (bits 0,1), result = 1
	//   Equation: S[1] ⊕ S[2] = 1
	//
	// Slot 2: coeff = 0b1 (bit 0), result = 1
	//   Equation: S[2] = 1
	//
	// Back-substitution (column j=0, single result bit):
	//   i=2: tmp=0, bit=parity(0 & 1) ^ 1 = 1, state=1       → S[2]=1
	//   i=1: tmp=1<<1=2, bit=parity(2 & 3) ^ 1 = parity(2)^1 = 1^1 = 0
	//     state=2|0=2  → S[1]=0
	//   i=0: tmp=2<<1=4, bit=parity(4 & 7) ^ 0 = parity(4)^0 = 1
	//     state=4|1=5  → S[0]=1
	numSlots := uint32(66) // 3 starts + 64 - 1
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 0b111}, result: 0}
	slots[1] = bandingSlot{coeffRow: uint128{lo: 0b11}, result: 1}
	slots[2] = bandingSlot{coeffRow: uint128{lo: 0b1}, result: 1}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	want := []uint8{1, 0, 1}
	for i, w := range want {
		if sol.load(uint32(i)) != w {
			t.Errorf("S[%d] = %d, want %d", i, sol.load(uint32(i)), w)
		}
	}
}

func TestBackSubstitute_AllEmpty(t *testing.T) {
	// All slots empty → all free variables → S = all zeros.
	numSlots := uint32(128)
	slots := make([]bandingSlot, numSlots)
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	for i := uint32(0); i < numSlots; i++ {
		if sol.load(i) != 0 {
			t.Errorf("S[%d] = %d, want 0 (all-empty matrix)", i, sol.load(i))
		}
	}
}

func TestBackSubstitute_AllOccupied_Identity(t *testing.T) {
	// Every slot occupied with coefficient = 1 (identity-like).
	// Equation i: S[i] = result[i].
	// The solution is simply the result vector.
	numSlots := uint32(70)
	slots := make([]bandingSlot, numSlots)
	for i := uint32(0); i < numSlots; i++ {
		// Alternate results: 0, 1, 0, 1, …
		slots[i] = bandingSlot{coeffRow: uint128{lo: 1}, result: uint8(i & 1)}
	}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	for i := uint32(0); i < numSlots; i++ {
		want := uint8(i & 1)
		if sol.load(i) != want {
			t.Errorf("S[%d] = %d, want %d", i, sol.load(i), want)
		}
	}
}

func TestBackSubstitute_SparseMatrix(t *testing.T) {
	// Occupied slots interspersed with empty slots.
	// Slots 0,2,4 are occupied; slots 1,3,5+ are empty (free → 0).
	//
	// Slot 4: coeff = 1, result = 1  →  S[4] = 1
	// Slot 3: empty                  →  S[3] = 0
	// Slot 2: coeff = 0b101, result = 0
	//   Back-subst state after slots 4,3:
	//     After i=4: state = 1 (bit 0 = S[4] = 1)
	//     After i=3: state = 1<<1 = 2 (shifted, bit = parity(2&0)^0 = 0)
	//       state = 2 | 0 = 2
	//     At i=2: tmp = 2<<1 = 4
	//       bit = parity(4 & 0b101) ^ (0 & 1) = parity(4 & 5) ^ 0
	//            = parity(4) ^ 0 = 1
	//       state = 4 | 1 = 5     → S[2] = 1
	// Slot 1: empty → S[1] = 0
	//     After i=1: tmp = 5<<1 = 10, bit = parity(10&0)^0 = 0, state=10
	// Slot 0: coeff = 0b10101, result = 1
	//     At i=0: tmp = 10<<1 = 20
	//       bit = parity(20 & 0b10101) ^ (1 & 1)
	//            = parity(20 & 21) ^ 1 = parity(20) ^ 1 = 1 ^ 1 = 0
	//       → S[0] = 0
	//
	// Wait, let me re-check. 20 = 0b10100, 0b10101 = 21.
	// 20 & 21 = 0b10100 = 20. parity(20) = parity(0b10100) = 2 bits → 0.
	// bit = 0 ^ 1 = 1. → S[0] = 1.
	numSlots := uint32(68) // 5 starts + 64 - 1
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 0b10101}, result: 1}
	slots[2] = bandingSlot{coeffRow: uint128{lo: 0b101}, result: 0}
	slots[4] = bandingSlot{coeffRow: uint128{lo: 0b1}, result: 1}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	want := map[uint32]uint8{0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
	for idx, w := range want {
		if sol.load(idx) != w {
			t.Errorf("S[%d] = %d, want %d", idx, sol.load(idx), w)
		}
	}
}

// =============================================================================
// MULTI-BIT RESULT TESTS — verifying multi-column back-substitution
// =============================================================================

func TestBackSubstitute_MultiBitResult(t *testing.T) {
	// 2-bit result: each slot stores a 2-bit value.
	// resultBits = 2.
	//
	// Slot 0: coeff = 0b11, result = 0b10 (decimal 2)
	//   Equations (per column):
	//     col 0: S[0].bit0 ⊕ S[1].bit0 = 0
	//     col 1: S[0].bit1 ⊕ S[1].bit1 = 1
	//
	// Slot 1: coeff = 0b1, result = 0b11 (decimal 3)
	//   Equations:
	//     col 0: S[1].bit0 = 1
	//     col 1: S[1].bit1 = 1
	//
	// Back-substitution:
	//   i=1: For each column j:
	//     j=0: tmp=0, bit=parity(0&1)^(3>>0&1)=0^1=1 → S[1].bit0=1
	//     j=1: tmp=0, bit=parity(0&1)^(3>>1&1)=0^1=1 → S[1].bit1=1
	//     S[1] = 0b11 = 3
	//   i=0:
	//     j=0: tmp=1<<1=2, bit=parity(2&3)^(2>>0&1)=parity(2)^0=1^0=1 → S[0].bit0=1
	//     j=1: tmp=1<<1=2, bit=parity(2&3)^(2>>1&1)=parity(2)^1=1^1=0 → S[0].bit1=0
	//     S[0] = 0b01 = 1
	//
	// Verify: slot 0: S[0]⊕S[1] per column:
	//   col 0: 1⊕1 = 0 = result bit 0 ✓
	//   col 1: 0⊕1 = 1 = result bit 1 ✓
	numSlots := uint32(65)
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 0b11}, result: 2}
	slots[1] = bandingSlot{coeffRow: uint128{lo: 0b1}, result: 3}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 2)

	if sol.load(0) != 1 {
		t.Errorf("S[0] = %d, want 1", sol.load(0))
	}
	if sol.load(1) != 3 {
		t.Errorf("S[1] = %d, want 3", sol.load(1))
	}
}

func TestBackSubstitute_MultiBitResult_7Bits(t *testing.T) {
	// 7-bit result (the typical case): single slot.
	// Slot 0: coeff = 1, result = 0x5A (0b1011010 = 90)
	// Since it's just the identity: S[0] = result = 90.
	numSlots := uint32(64)
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 1}, result: 0x5A}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 7)

	if sol.load(0) != 0x5A {
		t.Errorf("S[0] = 0x%02x, want 0x5A", sol.load(0))
	}
}

// =============================================================================
// BOUNDARY ALIGNMENT TESTS
// =============================================================================

func TestBackSubstitute_StatePropagation_AcrossEmptySlots(t *testing.T) {
	// Verifies that the state shift register correctly propagates
	// through empty slots (where the state still shifts but no bit
	// is extracted from the result).
	//
	// Slot 127: coeff = 1, result = 1  →  S[127] = 1
	// Slots 64-126: all empty
	// Slot 63: coeff = 0b11 (bits 0,1), result = 0
	//   The state for column 0 at slot 63 has been shifted 64 times
	//   since slot 127 (through 64 empty slots). For w=64, the bit
	//   from slot 127 has shifted to position 64, which falls off
	//   the 64-bit register. So state = 0 at slot 63.
	//   bit = parity(0 & 0b11) ^ 0 = 0  → S[63] = 0
	numSlots := uint32(192)
	slots := make([]bandingSlot, numSlots)
	slots[127] = bandingSlot{coeffRow: uint128{lo: 1}, result: 1}
	slots[63] = bandingSlot{coeffRow: uint128{lo: 0b11}, result: 0}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	if sol.load(127) != 1 {
		t.Errorf("S[127] = %d, want 1", sol.load(127))
	}
	if sol.load(63) != 0 {
		t.Errorf("S[63] = %d, want 0 (bit fell off the 64-bit window)", sol.load(63))
	}
}

func TestBackSubstitute_StatePropagation_WithinWindow(t *testing.T) {
	// Slot 64: coeff = 1, result = 1  → S[64] = 1
	// Slot 60: coeff = 0b10001 (bits 0 and 4), result = 0
	//   At slot 60, the state has been shifted 4 times since slot 64.
	//   state has bit 4 = S[64] = 1 (within the 64-bit window).
	//   tmp = state<<1 (now bit 5)
	//   bit = parity(tmp & 0b10001) ^ 0
	//       = parity(tmp & 0b10001)
	//   tmp has bit 5 set. 0b10001 has bits 0 and 4. AND = 0. → bit = 0^0 = 0.
	//
	//   Hmm, let me trace more carefully.
	//   After i=64: state[0] = 1 (bit 0 = 1).
	//   i=63: empty. tmp=1<<1=2, bit=parity(2&0)^0=0, state=2.
	//   i=62: empty. tmp=2<<1=4, bit=0, state=4.
	//   i=61: empty. tmp=4<<1=8, bit=0, state=8.
	//   i=60: tmp=8<<1=16=0b10000. bit=parity(16 & 0b10001)^0
	//        = parity(0b10000) = 1. → S[60] = 1.
	numSlots := uint32(192)
	slots := make([]bandingSlot, numSlots)
	slots[64] = bandingSlot{coeffRow: uint128{lo: 1}, result: 1}
	slots[60] = bandingSlot{coeffRow: uint128{lo: 0b10001}, result: 0}
	b := makeBanderFromSlots(numSlots, 64, slots)

	sol := backSubstitute(b, 1)

	if sol.load(64) != 1 {
		t.Errorf("S[64] = %d, want 1", sol.load(64))
	}
	if sol.load(60) != 1 {
		t.Errorf("S[60] = %d, want 1 (state propagation within window)", sol.load(60))
	}
}

// =============================================================================
// w=128 TESTS
// =============================================================================

func TestBackSubstitute_W128_SingleSlot(t *testing.T) {
	numSlots := uint32(128)
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 1}, result: 1}
	b := makeBanderFromSlots(numSlots, 128, slots)

	sol := backSubstitute(b, 1)

	if sol.load(0) != 1 {
		t.Errorf("S[0] = %d, want 1", sol.load(0))
	}
	for i := uint32(1); i < numSlots; i++ {
		if sol.load(i) != 0 {
			t.Errorf("S[%d] = %d, want 0", i, sol.load(i))
		}
	}
}

func TestBackSubstitute_W128_CrossWordDependency(t *testing.T) {
	// w=128 with coefficient spanning the lo half.
	numSlots := uint32(129) // 2 starts + 128 - 1
	slots := make([]bandingSlot, numSlots)
	slots[0] = bandingSlot{coeffRow: uint128{lo: 0b11}, result: 0}
	slots[1] = bandingSlot{coeffRow: uint128{lo: 0b1}, result: 1}
	b := makeBanderFromSlots(numSlots, 128, slots)

	sol := backSubstitute(b, 1)

	if sol.load(0) != 1 {
		t.Errorf("S[0] = %d, want 1", sol.load(0))
	}
	if sol.load(1) != 1 {
		t.Errorf("S[1] = %d, want 1", sol.load(1))
	}
}

func TestBackSubstitute_W128_HiHalfCoefficient(t *testing.T) {
	// w=128 with a coefficient that has a bit set in the hi half.
	// Slot 65: coeff = 1, result = 1  →  S[65] = 1
	// Slot 0: coeff = {hi: 2, lo: 1}
	//   bit 0 is the pivot (lo bit 0). bit 65 overall = hi bit 1 = value 2.
	//   After i=65, the state for column 0 has been shifted 65 times.
	//   At i=0: tmp = state<<1. The S[65]=1 bit is now at position 66.
	//   We need parity(tmp & coeff). coeff bit 65 = hi bit 1. tmp bit 65 = ?
	//
	//   Actually with the state-register approach, let me trace:
	//   After i=65: state=1 (bit 0).
	//   i=64..1 (all empty): state shifts left each time.
	//   After i=1: state has bit 64 set = {hi: 1, lo: 0}.
	//   At i=0: tmp = state<<1 = {hi: 2, lo: 0} (bit 65).
	//     bit = parity(tmp & coeff) ^ result
	//         = parity({hi:2, lo:0} & {hi:2, lo:1}) ^ 0
	//         = parity({hi:2, lo:0}) ^ 0 = 1 ^ 0 = 1
	//     → S[0] = 1  ✓
	numSlots := uint32(200)
	slots := make([]bandingSlot, numSlots)
	slots[65] = bandingSlot{coeffRow: uint128{lo: 1}, result: 1}
	slots[0] = bandingSlot{coeffRow: uint128{hi: 2, lo: 1}, result: 0}
	b := makeBanderFromSlots(numSlots, 128, slots)

	sol := backSubstitute(b, 1)

	if sol.load(65) != 1 {
		t.Errorf("S[65] = %d, want 1", sol.load(65))
	}
	if sol.load(0) != 1 {
		t.Errorf("S[0] = %d, want 1 (hi-half coefficient dependency)", sol.load(0))
	}
}

// =============================================================================
// PARITY TESTS
// =============================================================================

func TestParity64(t *testing.T) {
	tests := []struct {
		val  uint64
		want int
	}{
		{0, 0},
		{1, 1},
		{3, 0},    // 2 bits → even
		{7, 1},    // 3 bits → odd
		{0xFF, 0}, // 8 bits → even
		{0xFFFF, 0},
		{0xFFFFFFFFFFFFFFFF, 0}, // 64 bits → even
		{0xFFFFFFFFFFFFFFFE, 1}, // 63 bits → odd
	}
	for _, tt := range tests {
		got := parity64(tt.val)
		if got != tt.want {
			t.Errorf("parity64(0x%x) = %d, want %d", tt.val, got, tt.want)
		}
	}
}

func TestParity128(t *testing.T) {
	tests := []struct {
		val  uint128
		want int
	}{
		{uint128{lo: 0, hi: 0}, 0},
		{uint128{lo: 1, hi: 0}, 1},
		{uint128{lo: 0, hi: 1}, 1},
		{uint128{lo: 1, hi: 1}, 0},       // 2 bits → even
		{uint128{lo: 3, hi: 0}, 0},       // 2 bits → even
		{uint128{lo: 0xFF, hi: 0xFF}, 0}, // 16 bits → even
	}
	for _, tt := range tests {
		got := parity128(tt.val)
		if got != tt.want {
			t.Errorf("parity128({lo:0x%x, hi:0x%x}) = %d, want %d",
				tt.val.lo, tt.val.hi, got, tt.want)
		}
	}
}

// =============================================================================
// QUERY TESTS — verifying the dot product against the solution vector
// =============================================================================

func TestQuery_Simple(t *testing.T) {
	// Build a solution manually and verify queries.
	// 2 slots, w=64, resultBits=7.
	// S[0] = 0x5A, S[1] = 0x3C.
	sol := &solution{
		data:       make([]uint8, 128), // numSlots + w padding
		numSlots:   2,
		coeffBits:  64,
		resultBits: 7,
	}
	sol.data[0] = 0x5A
	sol.data[1] = 0x3C

	// Query with coeff = 0b01 (only bit 0): result = S[0] = 0x5A.
	got := sol.query(0, uint128{lo: 1})
	if got != 0x5A {
		t.Errorf("Query(0, 0b01) = 0x%02x, want 0x5A", got)
	}

	// Query with coeff = 0b10 (only bit 1): result = S[1] = 0x3C.
	got = sol.query(0, uint128{lo: 2})
	if got != 0x3C {
		t.Errorf("Query(0, 0b10) = 0x%02x, want 0x3C", got)
	}

	// Query with coeff = 0b11 (both bits): result = S[0] ⊕ S[1] = 0x5A ^ 0x3C = 0x66.
	got = sol.query(0, uint128{lo: 3})
	if got != 0x66 {
		t.Errorf("Query(0, 0b11) = 0x%02x, want 0x66", got)
	}
}

func TestQuery128_Simple(t *testing.T) {
	// w=128 query.
	sol := &solution{
		data:       make([]uint8, 256),
		numSlots:   128,
		coeffBits:  128,
		resultBits: 7,
	}
	sol.data[0] = 0xAA
	sol.data[65] = 0x55

	// Coeff with bits 0 and 65 set: {hi: 2, lo: 1}.
	got := sol.query(0, uint128{hi: 2, lo: 1})
	if got != 0xAA^0x55 {
		t.Errorf("Query(0, {hi:2,lo:1}) = 0x%02x, want 0x%02x", got, uint8(0xAA^0x55))
	}
}

// =============================================================================
// INTEGRATION TESTS — full pipeline: hash → band → solve → query
// =============================================================================

func TestBackSubstitute_FullPipeline(t *testing.T) {
	// Full integration test: hash keys, band them, solve, and verify that
	// querying every original key returns the correct result.
	for _, w := range []uint32{64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				const numKeys = 2000
				const resultBits = 7
				numStarts := uint32(float64(numKeys) * 1.1) // generous overhead
				numSlots := numStarts + w - 1
				h := newStandardHasher(w, numStarts, resultBits, fcao)

				// Try seeds until banding succeeds.
				var bd *standardBander
				var hashes []uint64
				var seed uint32
				for seed = 0; seed < 100; seed++ {
					h.setOrdinalSeed(seed)
					bd = newStandardBander(numSlots, w, fcao)
					hashes = make([]uint64, numKeys)
					allOk := true
					for i := 0; i < numKeys; i++ {
						kh := h.keyHash([]byte(fmt.Sprintf("solver_test_key_%d", i)))
						hashes[i] = kh
						hr := h.derive(kh)
						if !bd.add(hr) {
							allOk = false
							break
						}
					}
					if allOk {
						t.Logf("banding succeeded with seed=%d", seed)
						break
					}
				}
				if seed >= 100 {
					t.Fatal("banding failed for all seeds")
				}

				// Back-substitute.
				sol := backSubstitute(bd, resultBits)

				// Verify: for every key, Query must return the expected result.
				for i := 0; i < numKeys; i++ {
					rh := h.rehash(hashes[i])
					start := h.getStart(rh)
					coeffRow := h.getCoeffRow(rh)
					expectedResult := h.getResultRow(rh)

					gotResult := sol.query(start, coeffRow)
					if gotResult != expectedResult {
						t.Fatalf("key %d: Query returned %d, want %d (start=%d)",
							i, gotResult, expectedResult, start)
					}
				}
				t.Logf("all %d keys verified correctly", numKeys)
			})
		}
	}
}

func TestBackSubstitute_FullPipeline_LargeScale(t *testing.T) {
	// Larger-scale test: 10k keys, w=128, verify all queries.
	if testing.Short() {
		t.Skip("skipping large-scale test in short mode")
	}

	const numKeys = 10000
	const resultBits = 7
	w := uint32(128)
	numStarts := uint32(float64(numKeys) * 1.05)
	numSlots := numStarts + w - 1
	h := newStandardHasher(w, numStarts, resultBits, true)

	var bd *standardBander
	var hashes []uint64
	var seed uint32
	for seed = 0; seed < 200; seed++ {
		h.setOrdinalSeed(seed)
		bd = newStandardBander(numSlots, w, true)
		hashes = make([]uint64, numKeys)
		allOk := true
		for i := 0; i < numKeys; i++ {
			kh := h.keyHash([]byte(fmt.Sprintf("large_key_%d", i)))
			hashes[i] = kh
			hr := h.derive(kh)
			if !bd.add(hr) {
				allOk = false
				break
			}
		}
		if allOk {
			break
		}
	}
	if seed >= 200 {
		t.Fatal("banding failed for all seeds")
	}

	sol := backSubstitute(bd, resultBits)

	// Verify all keys.
	for i := 0; i < numKeys; i++ {
		rh := h.rehash(hashes[i])
		start := h.getStart(rh)
		coeffRow := h.getCoeffRow(rh)
		expectedResult := h.getResultRow(rh)

		gotResult := sol.query(start, coeffRow)
		if gotResult != expectedResult {
			t.Fatalf("key %d: Query returned %d, want %d", i, gotResult, expectedResult)
		}
	}
}

func TestBackSubstitute_FalsePositiveRate(t *testing.T) {
	// Build a filter with known keys, then test that random non-member
	// keys have an FP rate close to 2^(-r) = 2^(-7) ≈ 0.78%.
	if testing.Short() {
		t.Skip("skipping FP rate test in short mode")
	}

	const numKeys = 5000
	const resultBits = 7
	w := uint32(128)
	numStarts := uint32(float64(numKeys) * 1.05)
	numSlots := numStarts + w - 1
	h := newStandardHasher(w, numStarts, resultBits, true)

	var bd *standardBander
	var seed uint32
	for seed = 0; seed < 200; seed++ {
		h.setOrdinalSeed(seed)
		bd = newStandardBander(numSlots, w, true)
		allOk := true
		for i := 0; i < numKeys; i++ {
			kh := h.keyHash([]byte(fmt.Sprintf("fp_key_%d", i)))
			hr := h.derive(kh)
			if !bd.add(hr) {
				allOk = false
				break
			}
		}
		if allOk {
			break
		}
	}
	if seed >= 200 {
		t.Fatal("banding failed for all seeds")
	}

	sol := backSubstitute(bd, resultBits)

	// Test non-member keys.
	const numNonMembers = 100000
	fps := 0
	for i := 0; i < numNonMembers; i++ {
		kh := h.keyHash([]byte(fmt.Sprintf("non_member_%d", i)))
		rh := h.rehash(kh)
		start := h.getStart(rh)
		coeffRow := h.getCoeffRow(rh)
		expectedResult := h.getResultRow(rh)

		gotResult := sol.query(start, coeffRow)
		if gotResult == expectedResult {
			fps++
		}
	}

	fpRate := float64(fps) / float64(numNonMembers)
	expectedRate := 1.0 / float64(uint64(1)<<resultBits) // 2^(-7) ≈ 0.0078
	t.Logf("FP rate: %.4f%% (%d / %d), expected ≈ %.4f%%",
		fpRate*100, fps, numNonMembers, expectedRate*100)

	// Allow a generous margin: within 3x of expected.
	if fpRate > expectedRate*3 {
		t.Errorf("FP rate %.4f%% is much higher than expected %.4f%%",
			fpRate*100, expectedRate*100)
	}
	if fpRate < expectedRate*0.3 {
		t.Errorf("FP rate %.4f%% is suspiciously low (expected ≈ %.4f%%)",
			fpRate*100, expectedRate*100)
	}
}

func TestBackSubstitute_ZeroSlots(t *testing.T) {
	// Edge case: numSlots == 0.
	b := newStandardBander(0, 64, true)
	sol := backSubstitute(b, 7)
	if sol.numSlots != 0 {
		t.Errorf("numSlots = %d, want 0", sol.numSlots)
	}
}

// =============================================================================
// EQUATION VERIFICATION HELPER
// =============================================================================

func TestBackSubstitute_VerifyEquations(t *testing.T) {
	// Build a small system, solve it, then directly verify that every
	// occupied equation holds across ALL result bit columns.
	//
	// For each occupied slot i with coeffRow c and result r:
	//   For each result column j:
	//     ⊕_{k=0}^{w-1} c[k] · S[i+k].bit_j  ==  (r >> j) & 1
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			const numKeys = 500
			const resultBits = 7
			numStarts := uint32(float64(numKeys) * 1.2)
			numSlots := numStarts + w - 1
			h := newStandardHasher(w, numStarts, resultBits, true)

			var bd *standardBander
			for seed := uint32(0); seed < 100; seed++ {
				h.setOrdinalSeed(seed)
				bd = newStandardBander(numSlots, w, true)
				allOk := true
				for i := 0; i < numKeys; i++ {
					kh := h.keyHash([]byte(fmt.Sprintf("verify_key_%d", i)))
					hr := h.derive(kh)
					if !bd.add(hr) {
						allOk = false
						break
					}
				}
				if allOk {
					break
				}
			}

			sol := backSubstitute(bd, resultBits)

			// For every occupied slot, verify all result bit columns.
			for i := uint32(0); i < numSlots; i++ {
				slot := bd.getSlot(i)
				if slot.coeffRow.isZero() {
					continue
				}

				// Use Query to verify: Query(i, coeffRow) should equal result.
				got := sol.query(i, slot.coeffRow)
				if got != slot.result {
					t.Fatalf("equation at slot %d failed: Query=%d, result=%d",
						i, got, slot.result)
				}
			}
		})
	}
}

// =============================================================================
// ICML (Interleaved Column-Major Layout) — Task 2 tests
// =============================================================================

// allICMLConfigs returns the full ICML correctness matrix: w∈{32,64,128} ×
// firstCoeffAlwaysOne∈{true,false} × r∈{1,4,7,8}. Every ICML correctness/oracle
// test iterates this (unlike allConfigs, which fixes r=7).
func allICMLConfigs() []Config {
	var cfgs []Config
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			for _, r := range []uint{1, 4, 7, 8} {
				cfgs = append(cfgs, Config{
					CoeffBits:           w,
					ResultBits:          r,
					FirstCoeffAlwaysOne: fcao,
				})
			}
		}
	}
	return cfgs
}

func icmlConfigName(cfg Config) string {
	return fmt.Sprintf("w=%d/fcao=%v/r=%d", cfg.CoeffBits, cfg.FirstCoeffAlwaysOne, cfg.ResultBits)
}

// buildBandedSystem builds a solved banded system for the given config by
// finding a seed that bands numKeys keys, rounding numSlots up to a multiple
// of w. Returns the bander, hasher, numSlots and numStarts.
func buildBandedSystem(t *testing.T, cfg Config, numKeys int, prefix string) (*standardBander, *standardHasher, uint32, uint32) {
	t.Helper()
	w := cfg.CoeffBits
	numStarts := uint32(float64(numKeys) * 1.4)
	if numStarts < 1 {
		numStarts = 1
	}
	numSlots := numStarts + w - 1
	numSlots = roundUpToMultiple(numSlots, w)
	numStarts = numSlots - w + 1

	h := newStandardHasher(w, numStarts, cfg.ResultBits, cfg.FirstCoeffAlwaysOne)
	var bd *standardBander
	for seed := uint32(0); seed < 200; seed++ {
		h.setOrdinalSeed(seed)
		bd = newStandardBander(numSlots, w, cfg.FirstCoeffAlwaysOne)
		ok := true
		for i := 0; i < numKeys; i++ {
			kh := h.keyHash([]byte(fmt.Sprintf("%s_%d", prefix, i)))
			if !bd.add(h.derive(kh)) {
				ok = false
				break
			}
		}
		if ok {
			return bd, h, numSlots, numStarts
		}
	}
	t.Fatalf("could not band system for %s", icmlConfigName(cfg))
	return nil, nil, 0, 0
}

func TestBackSubstICML_MatchesRowMajor(t *testing.T) {
	for _, cfg := range allICMLConfigs() {
		t.Run(icmlConfigName(cfg), func(t *testing.T) {
			w := cfg.CoeffBits
			bd, _, numSlots, _ := buildBandedSystem(t, cfg, 300, "icml_bs")

			rowMajor := backSubstitute(bd, cfg.ResultBits)
			icml := backSubstituteICML(bd, w, cfg.ResultBits)

			if icml.numBlocks != numSlots/w {
				t.Fatalf("numBlocks=%d, want %d", icml.numBlocks, numSlots/w)
			}

			r := uint32(cfg.ResultBits)
			// Slot-by-slot: decode every bit from ICML and compare to row-major.
			for i := uint32(0); i < numSlots; i++ {
				b := i / w
				off := i % w
				rm := rowMajor.load(i)
				for j := uint32(0); j < r; j++ {
					lo, hi := icml.icmlGetWord(b*r + j)
					var got uint64
					if off < 64 {
						got = (lo >> off) & 1
					} else {
						got = (hi >> (off - 64)) & 1
					}
					want := uint64(rm>>j) & 1
					if got != want {
						t.Fatalf("slot %d col %d: ICML bit=%d, row-major bit=%d",
							i, j, got, want)
					}
				}
			}
		})
	}
}

func TestICMLQuery_VerifyEquations(t *testing.T) {
	for _, cfg := range allICMLConfigs() {
		t.Run(icmlConfigName(cfg), func(t *testing.T) {
			w := cfg.CoeffBits
			bd, _, numSlots, _ := buildBandedSystem(t, cfg, 300, "icml_eq")
			icml := backSubstituteICML(bd, w, cfg.ResultBits)

			// Also cross-check against the w=32-safe row-major oracle.
			rowMajor := backSubstitute(bd, cfg.ResultBits)

			for i := uint32(0); i < numSlots; i++ {
				slot := bd.getSlot(i)
				if slot.coeffRow.isZero() {
					continue
				}
				got := icml.icmlQuery(i, slot.coeffRow)
				if got != slot.result {
					t.Fatalf("icmlQuery(slot %d)=%d, want result=%d", i, got, slot.result)
				}
				oracle := slowRowMajorQuery(rowMajor, i, slot.coeffRow, w)
				if oracle != slot.result {
					t.Fatalf("slowRowMajorQuery(slot %d)=%d, want result=%d", i, oracle, slot.result)
				}
			}
		})
	}
}

func TestBackSubstICML_ZeroSlots(t *testing.T) {
	for _, w := range []uint32{32, 64, 128} {
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			bd := newStandardBander(0, w, true)
			sol := backSubstituteICML(bd, w, 7)
			if sol.numBlocks != 0 {
				t.Errorf("numBlocks=%d, want 0", sol.numBlocks)
			}
		})
	}
}

func TestBackSubstICML_SingleSlot(t *testing.T) {
	// One occupied slot at index 0: 1·S[0]=1 → S[0]=1, all else 0.
	for _, w := range []uint32{32, 64, 128} {
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			numSlots := w * 2
			slots := make([]bandingSlot, numSlots)
			slots[0] = bandingSlot{coeffRow: uint128{lo: 1}, result: 1}
			bd := makeBanderFromSlots(numSlots, w, slots)
			sol := backSubstituteICML(bd, w, 1)

			// word(0,0) bit 0 == 1, everything else 0.
			lo, _ := sol.icmlGetWord(0)
			if lo&1 != 1 {
				t.Errorf("S[0] bit = %d, want 1", lo&1)
			}
			if lo&^uint64(1) != 0 {
				t.Errorf("word(0,0) high bits set: %#x", lo)
			}
			lo1, hi1 := sol.icmlGetWord(uint32(1))
			if lo1 != 0 || hi1 != 0 {
				t.Errorf("word(1,0) not zero: lo=%#x hi=%#x", lo1, hi1)
			}
		})
	}
}

func TestICMLColumnSlice_EdgeCases(t *testing.T) {
	// Deterministic (not random): construct a known 2-data-block solution with
	// a single column (r=1), set word(0,0)=A, word(1,0)=B, word(2,0)=0 (the
	// trailing zero block), and assert icmlColumnSlice combines them correctly
	// at offset 0 (block boundary), offset 1, offset w-1 (crossing into the
	// next block), the last valid start, and a read touching the trailing zero
	// block — for w=32, w=64, w=128.
	type wcase struct {
		name string
		enc  icmlEncoding
		w    uint32
	}
	cases := []wcase{
		{"w=32", encW32, 32},
		{"w=64", encW64, 64},
		{"w=128", encW128, 128},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := tc.w
			const numBlocks = 2
			const r = uint(1)
			sol := newICMLSolution(numBlocks, w, r)
			if sol.enc != tc.enc {
				t.Fatalf("newICMLSolution(w=%d) enc = %v, want %v", w, sol.enc, tc.enc)
			}

			// Choose w-bit patterns for A and B (block 0 and block 1, column 0).
			var aLo, aHi, bLo, bHi uint64
			switch w {
			case 32:
				aLo = 0xA5A5A5A5
				bLo = 0x3C3C3C3C
			case 64:
				aLo = 0xA5A5A5A5F0F0F0F0
				bLo = 0x123456789ABCDEF0
			case 128:
				aLo = 0xA5A5A5A5F0F0F0F0
				aHi = 0x0F0F0F0F5A5A5A5A
				bLo = 0x123456789ABCDEF0
				bHi = 0x0FEDCBA987654321
			}
			sol.icmlSetWord(0, aLo, aHi) // word(0,0)
			sol.icmlSetWord(1, bLo, bHi) // word(1,0)
			// word(2,0) left zero (trailing block).

			// Reference: the logical 2w-bit sequence is [A (rows 0..w-1)] then
			// [B (rows w..2w-1)] then [0 (rows 2w..3w-1)]. icmlColumnSlice(start)
			// returns the w-bit window rows [start, start+w).
			wantSlice := func(start uint32) (uint64, uint64) {
				// Build a 384-bit logical bit array conceptually; extract via
				// per-bit reads to avoid width juggling.
				bitAt := func(pos uint32) uint64 {
					// pos in [0, 3w).
					blk := pos / w
					off := pos % w
					var lo, hi uint64
					switch blk {
					case 0:
						lo, hi = aLo, aHi
					case 1:
						lo, hi = bLo, bHi
					default:
						return 0
					}
					if off < 64 {
						return (lo >> off) & 1
					}
					return (hi >> (off - 64)) & 1
				}
				var lo, hi uint64
				for k := uint32(0); k < w; k++ {
					b := bitAt(start + k)
					if k < 64 {
						lo |= b << k
					} else {
						hi |= b << (k - 64)
					}
				}
				return lo, hi
			}

			starts := []uint32{0, 1, w - 1, w /* last valid start, offset 0 */, w + 1 /* touches trailing zero */}
			for _, start := range starts {
				gotLo, gotHi := sol.icmlColumnSlice(start, 0)
				wLo, wHi := wantSlice(start)
				if gotLo != wLo || gotHi != wHi {
					t.Errorf("start=%d: got (%#x,%#x), want (%#x,%#x)",
						start, gotHi, gotLo, wHi, wLo)
				}
			}
		})
	}
}

func TestICMLGetSetWord_RoundTrip(t *testing.T) {
	// Verify icmlSetWord/icmlGetWord round-trip and that packed lanes do not
	// clobber their pair.
	for _, tc := range []struct {
		name string
		enc  icmlEncoding
		w    uint32
	}{
		{"w=32", encW32, 32},
		{"w=64", encW64, 64},
		{"w=128", encW128, 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sol := newICMLSolution(4, tc.w, 3)
			if sol.enc != tc.enc {
				t.Fatalf("newICMLSolution(w=%d) enc = %v, want %v", tc.w, sol.enc, tc.enc)
			}
			logicalWords := (4 + 1) * uint32(3)
			// Write distinct patterns to every logical word.
			for L := uint32(0); L < logicalWords; L++ {
				lo := uint64(L)*0x1000001 + 0x9E3779B97F4A7C15
				var hi uint64
				if tc.w == 128 {
					hi = uint64(L)*0xDEADBEEF + 0x123456789
				}
				if tc.w == 32 {
					lo &= 0xffffffff
				}
				sol.icmlSetWord(L, lo, hi)
			}
			// Read back — every word must match its masked write.
			for L := uint32(0); L < logicalWords; L++ {
				lo := uint64(L)*0x1000001 + 0x9E3779B97F4A7C15
				var hi uint64
				if tc.w == 128 {
					hi = uint64(L)*0xDEADBEEF + 0x123456789
				}
				if tc.w == 32 {
					lo &= 0xffffffff
				}
				gotLo, gotHi := sol.icmlGetWord(L)
				if gotLo != lo || gotHi != hi {
					t.Errorf("L=%d: got (%#x,%#x), want (%#x,%#x)", L, gotHi, gotLo, hi, lo)
				}
			}
		})
	}
}
