package ribbonGo

import (
	"fmt"
	"testing"
)

// =============================================================================
// TEST HELPERS
// =============================================================================

// newTestBander creates a standardBander with common test defaults.
// Uses numSlots = numStarts + coeffBits - 1, with generous over-provisioning
// so that typical random insertions succeed reliably.
func newTestBander(coeffBits uint32, firstCoeffAlwaysOne bool) *standardBander {
	numStarts := uint32(10000)
	numSlots := numStarts + coeffBits - 1
	return newStandardBander(numSlots, coeffBits, firstCoeffAlwaysOne)
}

// newTestHasherForBander creates a standardHasher matching a bander's config.
func newTestHasherForBander(coeffBits uint32, numStarts uint32, firstCoeffAlwaysOne bool) *standardHasher {
	return newStandardHasher(coeffBits, numStarts, 7, firstCoeffAlwaysOne)
}

// precomputedHashResults generates N deterministic hashResult values
// using the given hasher. Removes Phase 1 (XXH3) and Phase 2 (derive)
// from the code under test — isolates banding logic.
func precomputedHashResults(h *standardHasher, n int) []hashResult {
	results := make([]hashResult, n)
	for i := range results {
		kh := h.keyHash([]byte(fmt.Sprintf("bander_key_%d", i)))
		results[i] = h.derive(kh)
	}
	return results
}

// =============================================================================
// CONSTRUCTOR TESTS
// =============================================================================

func TestNewStandardBander(t *testing.T) {
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				numSlots := uint32(1000) + w - 1
				bd := newStandardBander(numSlots, w, fcao)

				if bd.getNumSlots() != numSlots {
					t.Errorf("numSlots = %d, want %d", bd.getNumSlots(), numSlots)
				}
				if uint32(len(bd.coeffLo)) != numSlots {
					t.Errorf("len(coeffLo) = %d, want %d", len(bd.coeffLo), numSlots)
				}

				// All slots should be empty (zero-valued).
				for i := uint32(0); i < numSlots; i++ {
					slot := bd.getSlot(i)
					if !slot.coeffRow.isZero() || slot.result != 0 {
						t.Fatalf("slot[%d] not empty after construction", i)
					}
				}
			})
		}
	}
}

func TestNewStandardBander_InvalidCoeffBits(t *testing.T) {
	for _, w := range []uint32{0, 16, 48, 96, 256} {
		w := w
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("newStandardBander with coeffBits=%d should panic", w)
				}
			}()
			newStandardBander(100, w, true)
		})
	}
}

// =============================================================================
// ADD — basic insertion tests
// =============================================================================

func TestAdd_SingleInsertion(t *testing.T) {
	// Insert a single key into each configuration. Must always succeed.
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				bd := newTestBander(w, fcao)
				numStarts := uint32(10000)
				h := newTestHasherForBander(w, numStarts, fcao)
				h.setOrdinalSeed(0)

				kh := h.keyHash([]byte("single_key"))
				hr := h.derive(kh)

				ok := bd.Add(hr)
				if !ok {
					t.Fatal("Add returned false for single key insertion")
				}

				// Verify the pivot slot is occupied.
				// The pivot column depends on firstCoeffAlwaysOne:
				// if true, pivot = start (bit 0 is always 1).
				if fcao {
					slot := bd.getSlot(hr.start)
					if slot.coeffRow.isZero() {
						t.Error("slot at start position is empty after insertion")
					}
				}
			})
		}
	}
}

func TestAdd_MultipleNonConflicting(t *testing.T) {
	// Insert many keys with a well-sized filter. Most should succeed.
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				numStarts := uint32(10000)
				numSlots := numStarts + w - 1
				bd := newStandardBander(numSlots, w, fcao)
				h := newTestHasherForBander(w, numStarts, fcao)
				h.setOrdinalSeed(0)

				const numKeys = 500
				successes := 0
				for i := 0; i < numKeys; i++ {
					kh := h.keyHash([]byte(fmt.Sprintf("key_%d", i)))
					hr := h.derive(kh)
					if bd.Add(hr) {
						successes++
					}
				}

				// With 10000 slots and only 500 keys, virtually all
				// should succeed (failure rate < 0.1%).
				if successes < numKeys-5 {
					t.Errorf("too many failures: %d/%d succeeded", successes, numKeys)
				}
				t.Logf("%d/%d insertions succeeded", successes, numKeys)
			})
		}
	}
}

// =============================================================================
// ADD — linear dependence tests (c=0 failure states)
// =============================================================================

func TestAdd_RedundantEquation(t *testing.T) {
	// Insert the same hashResult twice. The second insertion should fail
	// with c=0, r=0 (redundant — the equation is already represented).
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				bd := newTestBander(w, fcao)
				numStarts := uint32(10000)
				h := newTestHasherForBander(w, numStarts, fcao)
				h.setOrdinalSeed(0)

				kh := h.keyHash([]byte("duplicate"))
				hr := h.derive(kh)

				// First insertion succeeds.
				if !bd.Add(hr) {
					t.Fatal("first Add failed")
				}

				// Second insertion of the same equation: c XOR c = 0, r XOR r = 0.
				ok := bd.Add(hr)
				if ok {
					t.Error("second Add of same hashResult should return false (redundant)")
				}
			})
		}
	}
}

func TestAdd_ContradictoryEquation(t *testing.T) {
	// Craft two equations with the same start and coeffRow but different
	// result. This forces c=0, r≠0 (contradictory — no solution exists).
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				bd := newTestBander(w, fcao)

				// Choose a coefficient with bit 0 = 1 (to work with both
				// fcao=true and fcao=false).
				var coeff uint128
				switch w {
				case 32:
					coeff = uint128{lo: 0xDEADBEEF & 0xFFFFFFFF}
				case 64:
					coeff = uint128{lo: 0xDEADBEEFCAFEBABE}
				case 128:
					coeff = uint128{hi: 0x0123456789ABCDEF, lo: 0xDEADBEEFCAFEBABE}
				}
				// Ensure bit 0 is set (required for valid equation).
				coeff.lo |= 1

				hr1 := hashResult{start: 100, coeffRow: coeff, result: 42}
				hr2 := hashResult{start: 100, coeffRow: coeff, result: 99}

				if !bd.Add(hr1) {
					t.Fatal("first Add failed")
				}

				ok := bd.Add(hr2)
				if ok {
					t.Error("contradictory equation should return false")
				}
			})
		}
	}
}

func TestAdd_MultiStepCollisionChain(t *testing.T) {
	// Craft equations that force a multi-step collision chain during
	// elimination, eventually resulting in linear dependence (c=0).
	//
	// Setup for w=64, fcao=true:
	//   eq1: start=10, coeff=0b...0001 (pivot at col 10)
	//   eq2: start=10, coeff=0b...0011 (pivot at col 10 → XOR with eq1 → coeff=0b...0010, pivot at col 11)
	//   eq3: start=10, coeff=0b...0011, same result as eq2 → same chain → c=0, r=0
	for _, fcao := range []bool{true, false} {
		name := fmt.Sprintf("fcao=%v", fcao)
		t.Run(name, func(t *testing.T) {
			bd := newStandardBander(100, 64, fcao)

			eq1 := hashResult{
				start:    10,
				coeffRow: uint128{lo: 0b0001},
				result:   5,
			}
			eq2 := hashResult{
				start:    10,
				coeffRow: uint128{lo: 0b0011},
				result:   7,
			}
			// eq3 is identical to eq2 — same start, coeff, result.
			// Chain: pivot at 10 → XOR with eq1 → coeff=0b0010 → pivot at 11
			//        → XOR with eq2's remainder → c=0, r=0.
			eq3 := hashResult{
				start:    10,
				coeffRow: uint128{lo: 0b0011},
				result:   7,
			}

			if !bd.Add(eq1) {
				t.Fatal("eq1 Add failed")
			}
			if !bd.Add(eq2) {
				t.Fatal("eq2 Add failed")
			}

			ok := bd.Add(eq3)
			if ok {
				t.Error("eq3 should fail (redundant after multi-step chain)")
			}
		})
	}
}

func TestAdd_ChainResolves(t *testing.T) {
	// A collision chain that resolves successfully — two keys collide at
	// the same start but the XOR produces a non-zero coefficient with a
	// different pivot, which finds an empty slot.
	for _, fcao := range []bool{true, false} {
		name := fmt.Sprintf("fcao=%v", fcao)
		t.Run(name, func(t *testing.T) {
			bd := newStandardBander(100, 64, fcao)

			eq1 := hashResult{
				start:    10,
				coeffRow: uint128{lo: 0b0101}, // bits 0 and 2 set
				result:   3,
			}
			eq2 := hashResult{
				start:    10,
				coeffRow: uint128{lo: 0b0111}, // bits 0, 1, and 2 set
				result:   5,
			}

			if !bd.Add(eq1) {
				t.Fatal("eq1 Add failed")
			}
			// eq2 collides with eq1 at col 10. XOR: 0b0101 ^ 0b0111 = 0b0010.
			// New pivot at col 11 (bit 1). Shift right by 1 → coeff=0b0001.
			// Slot 11 is empty → store successfully.
			if !bd.Add(eq2) {
				t.Fatal("eq2 should succeed after resolving collision chain")
			}

			// Verify both slots are occupied.
			if bd.getSlot(10).coeffRow.isZero() {
				t.Error("slot 10 should be occupied")
			}
			if bd.getSlot(11).coeffRow.isZero() {
				t.Error("slot 11 should be occupied")
			}
		})
	}
}

// =============================================================================
// ADD — w=128 specific tests
// =============================================================================

func TestAdd_128BitCoefficients(t *testing.T) {
	// Test with coefficients that span both lo and hi halves of uint128.
	for _, fcao := range []bool{true, false} {
		name := fmt.Sprintf("fcao=%v", fcao)
		t.Run(name, func(t *testing.T) {
			bd := newStandardBander(200, 128, fcao)

			// Coefficient with bits in both lo and hi.
			eq1 := hashResult{
				start:    5,
				coeffRow: uint128{hi: 0xABCD, lo: 0x1},
				result:   10,
			}

			if !bd.Add(eq1) {
				t.Fatal("128-bit coefficient Add failed")
			}

			// Verify the slot at the pivot column is occupied.
			if bd.getSlot(5).coeffRow.isZero() {
				t.Error("slot 5 should be occupied")
			}
		})
	}
}

func TestAdd_128BitCollisionInHiHalf(t *testing.T) {
	// Force a collision where the pivot resolves into the hi half
	// (TrailingZeros must cross the 64-bit boundary).
	bd := newStandardBander(200, 128, true)

	// eq1: pivot at col 5 (bit 0 = 1, start=5).
	eq1 := hashResult{
		start:    5,
		coeffRow: uint128{hi: 0, lo: 1},
		result:   3,
	}
	// eq2: same start, coeff has bit 0=1 and a bit in hi.
	// XOR with eq1: lo becomes 0, hi retains the set bit.
	// This forces the pivot into the hi half (offset ≥ 64).
	eq2 := hashResult{
		start:    5,
		coeffRow: uint128{hi: 1, lo: 1},
		result:   7,
	}

	if !bd.Add(eq1) {
		t.Fatal("eq1 failed")
	}
	// eq2 XOR eq1: coeff = {hi:1, lo:0}. TrailingZeros = 64.
	// Pivot at col 5 + 64 = 69. Should succeed (slot 69 is empty).
	if !bd.Add(eq2) {
		t.Fatal("eq2 should succeed (pivot at col 69 in hi half)")
	}
	if bd.getSlot(69).coeffRow.isZero() {
		t.Error("slot 69 should be occupied after hi-half pivot resolution")
	}
}

// =============================================================================
// CROSS-VALIDATION: Add vs slowAdd
// =============================================================================

func TestAdd_MatchesSlowAdd(t *testing.T) {
	// Verify that the optimised Add() produces the exact same outcome
	// (success/failure) and the exact same slot contents as slowAdd()
	// for a large number of keys across all 6 configurations.
	//
	// This catches any divergence introduced by the firstCoeffAlwaysOne
	// fast path.
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				numStarts := uint32(10000)
				numSlots := numStarts + w - 1
				h := newTestHasherForBander(w, numStarts, fcao)
				h.setOrdinalSeed(0)

				const numKeys = 5000
				hashes := precomputedHashResults(h, numKeys)

				// Run Add on one bander, slowAdd on another.
				bdFast := newStandardBander(numSlots, w, fcao)
				bdSlow := newStandardBander(numSlots, w, fcao)

				for i, hr := range hashes {
					gotFast := bdFast.Add(hr)
					gotSlow := bdSlow.slowAdd(hr)

					if gotFast != gotSlow {
						t.Fatalf("key %d: Add()=%v, slowAdd()=%v (start=%d)",
							i, gotFast, gotSlow, hr.start)
					}
				}

				// Verify all slots are identical.
				for i := uint32(0); i < numSlots; i++ {
					sf := bdFast.getSlot(i)
					ss := bdSlow.getSlot(i)
					if sf.coeffRow != ss.coeffRow || sf.result != ss.result {
						t.Fatalf("slot %d differs: fast={coeff:%+v, r:%d} slow={coeff:%+v, r:%d}",
							i, sf.coeffRow, sf.result, ss.coeffRow, ss.result)
					}
				}
			})
		}
	}
}

// =============================================================================
// ADD RANGE — batched insertion with prefetching
// =============================================================================

func TestAddRange_MatchesAdd(t *testing.T) {
	// Verify that AddRange produces the exact same slot state as calling
	// Add() in a loop for the same inputs.
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				numStarts := uint32(10000)
				numSlots := numStarts + w - 1
				h := newTestHasherForBander(w, numStarts, fcao)
				h.setOrdinalSeed(0)

				const numKeys = 5000
				hashes := precomputedHashResults(h, numKeys)

				// Run Add() in a loop.
				bdAdd := newStandardBander(numSlots, w, fcao)
				addOk := true
				for _, hr := range hashes {
					if !bdAdd.Add(hr) {
						addOk = false
						break
					}
				}

				// Run AddRange() on the full batch.
				bdRange := newStandardBander(numSlots, w, fcao)
				rangeOk := bdRange.AddRange(hashes)

				if addOk != rangeOk {
					t.Fatalf("Add-loop=%v, AddRange=%v", addOk, rangeOk)
				}

				// All slots must be identical.
				for i := uint32(0); i < numSlots; i++ {
					sa := bdAdd.getSlot(i)
					sr := bdRange.getSlot(i)
					if sa.coeffRow != sr.coeffRow || sa.result != sr.result {
						t.Fatalf("slot %d differs: Add={coeff:%+v, r:%d} AddRange={coeff:%+v, r:%d}",
							i, sa.coeffRow, sa.result, sr.coeffRow, sr.result)
					}
				}
			})
		}
	}
}

func TestAddRange_MatchesSlowAddRange(t *testing.T) {
	// Cross-validate AddRange against slowAddRange across all 6 configs.
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				numStarts := uint32(10000)
				numSlots := numStarts + w - 1
				h := newTestHasherForBander(w, numStarts, fcao)
				h.setOrdinalSeed(0)

				const numKeys = 5000
				hashes := precomputedHashResults(h, numKeys)

				bdFast := newStandardBander(numSlots, w, fcao)
				bdSlow := newStandardBander(numSlots, w, fcao)

				gotFast := bdFast.AddRange(hashes)
				gotSlow := bdSlow.slowAddRange(hashes)

				if gotFast != gotSlow {
					t.Fatalf("AddRange()=%v, slowAddRange()=%v", gotFast, gotSlow)
				}

				for i := uint32(0); i < numSlots; i++ {
					sf := bdFast.getSlot(i)
					ss := bdSlow.getSlot(i)
					if sf.coeffRow != ss.coeffRow || sf.result != ss.result {
						t.Fatalf("slot %d differs: fast={coeff:%+v, r:%d} slow={coeff:%+v, r:%d}",
							i, sf.coeffRow, sf.result, ss.coeffRow, ss.result)
					}
				}
			})
		}
	}
}

func TestAddRange_Empty(t *testing.T) {
	bd := newStandardBander(100, 64, true)
	if !bd.AddRange(nil) {
		t.Error("AddRange(nil) should return true")
	}
	if !bd.AddRange([]hashResult{}) {
		t.Error("AddRange(empty) should return true")
	}
}

func TestAddRange_StopsOnFailure(t *testing.T) {
	// Verify that AddRange returns false on failure, and that
	// the state matches calling Add() key-by-key with the same
	// stop-on-first-failure semantics.
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			// Tight sizing to force some failures.
			const numKeys = 2000
			numStarts := uint32(float64(numKeys) * 1.02)
			numSlots := numStarts + w - 1
			h := newStandardHasher(w, numStarts, 7, true)

			// Try seeds until we find one that fails mid-batch.
			for seed := uint32(0); seed < 100; seed++ {
				h.setOrdinalSeed(seed)
				hashes := precomputedHashResults(h, numKeys)

				bdRange := newStandardBander(numSlots, w, true)
				rangeOk := bdRange.AddRange(hashes)

				// Replay with Add-loop using same stop-on-failure.
				bdLoop := newStandardBander(numSlots, w, true)
				loopOk := true
				for _, hr := range hashes {
					if !bdLoop.Add(hr) {
						loopOk = false
						break
					}
				}

				if rangeOk != loopOk {
					t.Fatalf("seed=%d: AddRange()=%v, Add-loop=%v", seed, rangeOk, loopOk)
				}

				// Verify identical state.
				for i := uint32(0); i < numSlots; i++ {
					sr := bdRange.getSlot(i)
					sl := bdLoop.getSlot(i)
					if sr.coeffRow != sl.coeffRow || sr.result != sl.result {
						t.Fatalf("seed=%d slot %d differs", seed, i)
					}
				}

				if !rangeOk {
					t.Logf("seed=%d: AddRange correctly returned false", seed)
					return // test passed — found a failure case
				}
			}
			t.Log("all seeds succeeded (no failure case found)")
		})
	}
}

// =============================================================================
// RESET
// =============================================================================

func TestReset(t *testing.T) {
	for _, w := range []uint32{32, 64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			bd := newTestBander(w, true)
			numStarts := uint32(10000)
			h := newTestHasherForBander(w, numStarts, true)
			h.setOrdinalSeed(0)

			// Insert some keys.
			for i := 0; i < 100; i++ {
				kh := h.keyHash([]byte(fmt.Sprintf("reset_key_%d", i)))
				hr := h.derive(kh)
				bd.Add(hr)
			}

			// Verify at least some slots are occupied.
			occupied := 0
			for i := uint32(0); i < bd.getNumSlots(); i++ {
				if !bd.getSlot(i).coeffRow.isZero() {
					occupied++
				}
			}
			if occupied == 0 {
				t.Fatal("no slots occupied after insertions")
			}

			// Reset and verify all slots are empty.
			bd.reset()
			for i := uint32(0); i < bd.getNumSlots(); i++ {
				slot := bd.getSlot(i)
				if !slot.coeffRow.isZero() || slot.result != 0 {
					t.Fatalf("slot[%d] not empty after reset", i)
				}
			}

			// Re-insertion after reset should succeed.
			kh := h.keyHash([]byte("after_reset"))
			hr := h.derive(kh)
			if !bd.Add(hr) {
				t.Error("Add failed after reset")
			}
		})
	}
}

// =============================================================================
// SEED RETRY — verifies banding can succeed after seed change
// =============================================================================

func TestAdd_SeedRetry(t *testing.T) {
	// Load the filter heavily enough that some seeds might fail, then
	// verify that retrying with a new seed can succeed.
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			// Tight sizing: numStarts ≈ 1.05 * numKeys.
			const numKeys = 2000
			numStarts := uint32(float64(numKeys) * 1.05)
			numSlots := numStarts + w - 1
			h := newStandardHasher(w, numStarts, 7, true)

			succeeded := false
			for seed := uint32(0); seed < 100; seed++ {
				h.setOrdinalSeed(seed)
				bd := newStandardBander(numSlots, w, true)

				allOk := true
				for i := 0; i < numKeys; i++ {
					kh := h.keyHash([]byte(fmt.Sprintf("retry_key_%d", i)))
					hr := h.derive(kh)
					if !bd.Add(hr) {
						allOk = false
						break
					}
				}

				if allOk {
					succeeded = true
					t.Logf("banding succeeded with seed=%d", seed)
					break
				}
			}

			if !succeeded {
				t.Error("banding failed for all 100 seeds — unexpected")
			}
		})
	}
}

// =============================================================================
// EDGE CASES
// =============================================================================

func TestAdd_SingleSlot(t *testing.T) {
	// Minimum viable bander: numSlots = coeffBits (numStarts = 1).
	// Only one possible start position.
	for _, w := range []uint32{32, 64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			numSlots := w
			bd := newStandardBander(numSlots, w, true)
			h := newStandardHasher(w, 1, 7, true) // numStarts=1
			h.setOrdinalSeed(0)

			kh := h.keyHash([]byte("tiny"))
			hr := h.derive(kh)

			// start must be 0 (only one possible start).
			if hr.start != 0 {
				t.Fatalf("expected start=0 with numStarts=1, got %d", hr.start)
			}

			if !bd.Add(hr) {
				t.Error("Add failed for single-start bander")
			}
		})
	}
}

func TestAdd_HandCraftedDirectInsertion(t *testing.T) {
	// Directly verify slot contents for a known hand-crafted equation.
	bd := newStandardBander(100, 64, true)

	hr := hashResult{
		start:    7,
		coeffRow: uint128{lo: 0b10110101}, // bits 0,2,4,5,7 set
		result:   42,
	}

	if !bd.Add(hr) {
		t.Fatal("Add failed")
	}

	// With fcao=true, pivot is at start=7. Coefficient is stored as-is
	// (no shift needed since bit 0 is the pivot).
	slot := bd.getSlot(7)
	if slot.coeffRow != hr.coeffRow {
		t.Errorf("stored coeffRow = %+v, want %+v", slot.coeffRow, hr.coeffRow)
	}
	if slot.result != hr.result {
		t.Errorf("stored result = %d, want %d", slot.result, hr.result)
	}
}

func TestAdd_HandCraftedWithShift(t *testing.T) {
	// When firstCoeffAlwaysOne=false, the coefficient might not have
	// bit 0 set, requiring a right-shift before storage.
	bd := newStandardBander(100, 64, false)

	hr := hashResult{
		start:    10,
		coeffRow: uint128{lo: 0b11000}, // bits 3 and 4 set; pivot offset = 3
		result:   7,
	}

	if !bd.Add(hr) {
		t.Fatal("Add failed")
	}

	// Pivot offset = 3 → absolute column = 10 + 3 = 13.
	// Stored coeff = 0b11000 >> 3 = 0b11.
	slot := bd.getSlot(13)
	if slot.coeffRow.lo != 0b11 {
		t.Errorf("stored coeffRow.lo = 0b%b, want 0b11", slot.coeffRow.lo)
	}
	if slot.result != 7 {
		t.Errorf("stored result = %d, want 7", slot.result)
	}
}

// =============================================================================
// LARGE-SCALE STATISTICAL TESTS
// =============================================================================

func TestAdd_SuccessRate(t *testing.T) {
	// With proper sizing (numStarts ≈ 1.02*N for w=128), the banding
	// should succeed with high probability. Test across seeds to measure
	// the success rate.
	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			const numKeys = 5000
			// Paper §4: overhead ratio m/n ≈ 1 + 2.3/w.
			// Use 1.05 for safety margin.
			numStarts := uint32(float64(numKeys) * 1.05)
			numSlots := numStarts + w - 1

			h := newStandardHasher(w, numStarts, 7, true)
			successes := 0
			const numTrials = 50

			for seed := uint32(0); seed < numTrials; seed++ {
				h.setOrdinalSeed(seed)
				bd := newStandardBander(numSlots, w, true)

				allOk := true
				for i := 0; i < numKeys; i++ {
					kh := h.keyHash([]byte(fmt.Sprintf("stat_key_%d", i)))
					hr := h.derive(kh)
					if !bd.Add(hr) {
						allOk = false
						break
					}
				}
				if allOk {
					successes++
				}
			}

			rate := float64(successes) / float64(numTrials)
			t.Logf("w=%d: %d/%d seeds succeeded (%.1f%%)", w, successes, numTrials, rate*100)

			// With 5% overhead and w≥64, at least ~50% of seeds should work.
			if rate < 0.3 {
				t.Errorf("success rate %.1f%% is unexpectedly low", rate*100)
			}
		})
	}
}
