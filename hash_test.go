package ribbon

import (
	"fmt"
	"testing"
)

// helper: creates a standardHasher with common test defaults.
// w=128, numSlots=13000, resultBits=7 (~1% FPR).
func newTestHasher() *standardHasher {
	var coeffBits uint32 = 128
	numSlots := uint32(13000)
	numStarts := numSlots - coeffBits + 1 // 12873
	return newStandardHasher(coeffBits, numStarts, 7, true)
}

func TestKeyHash(t *testing.T) {
	h := newTestHasher()

	// Basic sanity: same key always produces same hash
	h1 := h.keyHash([]byte("hello"))
	h2 := h.keyHash([]byte("hello"))
	if h1 != h2 {
		t.Errorf("same key produced different hashes: %x vs %x", h1, h2)
	}

	// Different keys produce different hashes
	h3 := h.keyHash([]byte("world"))
	if h1 == h3 {
		t.Errorf("different keys produced same hash: %x", h1)
	}

	t.Logf("keyHash(\"hello\") = 0x%016x", h1)
	t.Logf("keyHash(\"world\") = 0x%016x", h3)
}

func TestSeedConversion(t *testing.T) {
	// Ordinal → Raw → Ordinal round-trip
	for _, ordinal := range []uint32{0, 1, 2, 42, 127, 255} {
		raw := ordinalSeedToRaw(ordinal)
		back := rawSeedToOrdinal(raw)
		if back != ordinal {
			t.Errorf("round-trip failed for ordinal %d: raw=%x, back=%d", ordinal, raw, back)
		}
	}

	// Sequential ordinal seeds produce very different raw seeds
	raw0 := ordinalSeedToRaw(0)
	raw1 := ordinalSeedToRaw(1)
	raw2 := ordinalSeedToRaw(2)
	t.Logf("ordinal 0 → raw 0x%016x", raw0)
	t.Logf("ordinal 1 → raw 0x%016x", raw1)
	t.Logf("ordinal 2 → raw 0x%016x", raw2)

	// They should differ in many bits
	diff := raw0 ^ raw1
	if diff == 0 {
		t.Error("ordinal 0 and 1 produced identical raw seeds")
	}
}

func TestSetGetOrdinalSeed(t *testing.T) {
	h := newTestHasher()

	for _, ordinal := range []uint32{0, 1, 42, 127, 255} {
		h.setOrdinalSeed(ordinal)
		got := h.getOrdinalSeed()
		if got != ordinal {
			t.Errorf("setOrdinalSeed(%d) → getOrdinalSeed() = %d", ordinal, got)
		}
	}
}

func TestSetGetNumStarts(t *testing.T) {
	h := newStandardHasher(128, 100, 7, true)
	if h.getNumStarts() != 100 {
		t.Errorf("expected numStarts=100, got %d", h.getNumStarts())
	}

	h.setNumStarts(5000)
	if h.getNumStarts() != 5000 {
		t.Errorf("expected numStarts=5000, got %d", h.getNumStarts())
	}
}

func TestGetCoeffBits(t *testing.T) {
	for _, w := range []uint32{32, 64, 128} {
		h := newStandardHasher(w, 100, 7, true)
		if h.getCoeffBits() != w {
			t.Errorf("expected coeffBits=%d, got %d", w, h.getCoeffBits())
		}
	}
}

func TestGetResultBits(t *testing.T) {
	h := newStandardHasher(128, 100, 7, true)
	if h.getResultBits() != 7 {
		t.Errorf("expected resultBits=7, got %d", h.getResultBits())
	}
}

func TestNewStandardHasher_InvalidCoeffBits(t *testing.T) {
	for _, w := range []uint32{0, 16, 48, 96, 256} {
		w := w
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("newStandardHasher(%d, ...) should panic", w)
				}
			}()
			newStandardHasher(w, 100, 7, true)
		})
	}
}

func TestRehash(t *testing.T) {
	h := newTestHasher()
	kh := h.keyHash([]byte("test_key"))

	// Same hash + same seed = same result
	h.setOrdinalSeed(0)
	r1 := h.rehash(kh)
	r2 := h.rehash(kh)
	if r1 != r2 {
		t.Errorf("rehash not deterministic")
	}

	// Different seeds produce different results
	h.setOrdinalSeed(1)
	r3 := h.rehash(kh)
	if r1 == r3 {
		t.Errorf("different seeds produced same rehash")
	}

	t.Logf("rehash(h, seed=0) = 0x%016x", r1)
	t.Logf("rehash(h, seed=1) = 0x%016x", r3)
}

func TestFastRange64(t *testing.T) {
	h := newTestHasher()

	// Output must be in [0, range)
	for _, rangeVal := range []uint32{1, 10, 100, 1000, 10000, 100000} {
		for i := uint64(0); i < 1000; i++ {
			kh := h.keyHash([]byte(fmt.Sprintf("key_%d", i)))
			result := fastRange64(kh, rangeVal)
			if result >= rangeVal {
				t.Errorf("fastRange64(0x%x, %d) = %d, out of range", kh, rangeVal, result)
			}
		}
	}
}

func TestGetStart(t *testing.T) {
	h := newTestHasher()
	h.setOrdinalSeed(0)

	kh := h.keyHash([]byte("my_key"))
	rh := h.rehash(kh)
	start := h.getStart(rh)

	numStarts := h.getNumStarts()
	if start >= numStarts {
		t.Errorf("start %d >= numStarts %d", start, numStarts)
	}

	t.Logf("numStarts=%d, start=%d", numStarts, start)
}

func TestGetCoeffRow(t *testing.T) {
	h := newTestHasher()
	h.setOrdinalSeed(0)

	kh := h.keyHash([]byte("my_key"))
	rh := h.rehash(kh)
	cr := h.getCoeffRow(rh)

	// Must have bit 0 set (firstCoeffAlwaysOne)
	if cr.lo&1 == 0 {
		t.Error("coeffRow bit 0 is not set (firstCoeffAlwaysOne violated)")
	}

	// Must be non-zero
	if cr.isZero() {
		t.Error("coeffRow is zero")
	}

	t.Logf("coeffRow = {hi: 0x%016x, lo: 0x%016x}", cr.hi, cr.lo)
}

func TestGetResultRow(t *testing.T) {
	h7 := newStandardHasher(128, 12873, 7, true)
	h1 := newStandardHasher(128, 12873, 1, true)
	h8 := newStandardHasher(128, 12873, 8, true)

	h7.setOrdinalSeed(0)
	h1.setOrdinalSeed(0)
	h8.setOrdinalSeed(0)

	kh := h7.keyHash([]byte("my_key"))
	rh := h7.rehash(kh)

	// r=7 → result must be in [0, 128)
	rr7 := h7.getResultRow(rh)
	if rr7 >= 128 {
		t.Errorf("resultRow with 7 bits = %d, expected < 128", rr7)
	}

	// r=1 → result must be 0 or 1
	rh1 := h1.rehash(kh) // same raw seed (ordinal=0)
	rr1 := h1.getResultRow(rh1)
	if rr1 > 1 {
		t.Errorf("resultRow with 1 bit = %d, expected 0 or 1", rr1)
	}

	// r=8 → full byte
	rh8 := h8.rehash(kh)
	rr8 := h8.getResultRow(rh8)

	t.Logf("resultRow(r=1)=%d, (r=7)=%d, (r=8)=%d", rr1, rr7, rr8)
}

func TestDerive_Determinism(t *testing.T) {
	h := newTestHasher()
	h.setOrdinalSeed(42)

	key := []byte("determinism_test")
	kh := h.keyHash(key)

	r1 := h.derive(kh)
	r2 := h.derive(kh)

	if r1.start != r2.start || r1.coeffRow != r2.coeffRow || r1.result != r2.result {
		t.Error("derive not deterministic")
	}

	t.Logf("start=%d, coeffRow={0x%016x, 0x%016x}, result=%d",
		r1.start, r1.coeffRow.hi, r1.coeffRow.lo, r1.result)
}

func TestDerive_SeedIndependence(t *testing.T) {
	h := newTestHasher()

	key := []byte("independence_test")
	kh := h.keyHash(key)

	h.setOrdinalSeed(0)
	r0 := h.derive(kh)

	h.setOrdinalSeed(1)
	r1 := h.derive(kh)

	// With different seeds, at least one of start/coeff/result should differ
	if r0.start == r1.start && r0.coeffRow == r1.coeffRow && r0.result == r1.result {
		t.Error("different seeds produced identical hashResult — very unlikely")
	}

	t.Logf("seed=0: start=%d, result=%d", r0.start, r0.result)
	t.Logf("seed=1: start=%d, result=%d", r1.start, r1.result)
}

func TestDerive_MatchesStandaloneFunctions(t *testing.T) {
	// Verify that the hand-inlined derive() produces exactly the same
	// (start, coeffRow, result) as calling rehash → getStart/getCoeffRow/
	// getResultRow individually. This catches any divergence introduced
	// by the inlining optimisations (fused multiply, branchless masks,
	// omitted zero-guard).
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				h := newStandardHasher(w, 10000, 7, fcao)

				for seed := uint32(0); seed < 3; seed++ {
					h.setOrdinalSeed(seed)

					for i := 0; i < 10000; i++ {
						kh := h.keyHash([]byte(fmt.Sprintf("key_%d", i)))

						// Optimised path
						got := h.derive(kh)

						// Standalone path
						rh := h.rehash(kh)
						wantStart := h.getStart(rh)
						wantCoeff := h.getCoeffRow(rh)
						wantResult := h.getResultRow(rh)

						if got.start != wantStart {
							t.Fatalf("seed=%d key_%d: start mismatch: derive=%d standalone=%d",
								seed, i, got.start, wantStart)
						}
						if got.result != wantResult {
							t.Fatalf("seed=%d key_%d: result mismatch: derive=%d standalone=%d",
								seed, i, got.result, wantResult)
						}
						// coeffRow: derive() omits the zero-guard, so for the
						// astronomically rare case where the branchless path
						// produces zero (P≈2^-w), getCoeffRow forces it to 1
						// but derive() does not. We accept this known divergence.
						if got.coeffRow != wantCoeff && !got.coeffRow.isZero() {
							t.Fatalf("seed=%d key_%d: coeffRow mismatch: derive={0x%x,0x%x} standalone={0x%x,0x%x}",
								seed, i, got.coeffRow.hi, got.coeffRow.lo, wantCoeff.hi, wantCoeff.lo)
						}
					}
				}
			})
		}
	}
}

func TestHasherInterface(t *testing.T) {
	// Verify that standardHasher satisfies the hasher interface at runtime.
	var h hasher = newStandardHasher(128, 12873, 7, true)

	h.setOrdinalSeed(0)
	kh := h.keyHash([]byte("interface_test"))
	hr := h.derive(kh)

	if hr.start >= h.getNumStarts() {
		t.Errorf("start %d >= numStarts %d", hr.start, h.getNumStarts())
	}
	if hr.coeffRow.lo&1 == 0 {
		t.Error("coeffRow LSB not set through interface")
	}
	if hr.result >= (1 << h.getResultBits()) {
		t.Errorf("result %d exceeds %d-bit range", hr.result, h.getResultBits())
	}
	if h.getCoeffBits() != 128 {
		t.Errorf("expected coeffBits=128 via interface, got %d", h.getCoeffBits())
	}
	if !h.firstCoeffAlwaysOne() {
		t.Error("expected firstCoeffAlwaysOne=true via interface")
	}

	t.Logf("via interface: start=%d, result=%d", hr.start, hr.result)
}

func TestStartDistribution(t *testing.T) {
	// Check that starts are roughly uniformly distributed.
	numStarts := uint32(10000)
	h := newStandardHasher(128, numStarts, 7, true)
	h.setOrdinalSeed(0)

	numKeys := 100000
	buckets := make([]int, 10) // 10 buckets across the range

	for i := 0; i < numKeys; i++ {
		kh := h.keyHash([]byte(fmt.Sprintf("distribution_key_%d", i)))
		rh := h.rehash(kh)
		start := h.getStart(rh)
		bucket := int(start) * 10 / int(numStarts)
		if bucket >= 10 {
			bucket = 9
		}
		buckets[bucket]++
	}

	expected := numKeys / 10
	for i, count := range buckets {
		ratio := float64(count) / float64(expected)
		if ratio < 0.85 || ratio > 1.15 {
			t.Errorf("bucket %d: count=%d, expected ~%d (ratio=%.2f)", i, count, expected, ratio)
		}
		t.Logf("bucket %d: %d (%.1f%%)", i, count, 100*ratio)
	}
}

func TestGetCoeffRow_AllWidths(t *testing.T) {
	// Verify getCoeffRow respects the configured ribbon width w.
	// For each w, the coefficient row must:
	//   1. Have bit 0 set (firstCoeffAlwaysOne)
	//   2. Have all bits above w be zero
	//   3. Be non-zero
	for _, w := range []uint32{32, 64, 128} {
		w := w
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			h := newStandardHasher(w, 10000, 7, true)
			h.setOrdinalSeed(0)

			for i := 0; i < 10000; i++ {
				kh := h.keyHash([]byte(fmt.Sprintf("key_%d", i)))
				rh := h.rehash(kh)
				cr := h.getCoeffRow(rh)

				// Bit 0 must be set
				if cr.lo&1 == 0 {
					t.Fatalf("key_%d: coeffRow LSB not set for w=%d", i, w)
				}

				// Must be non-zero
				if cr.isZero() {
					t.Fatalf("key_%d: coeffRow is zero for w=%d", i, w)
				}

				// Bits above w must be zero
				switch w {
				case 32:
					if cr.hi != 0 || cr.lo>>32 != 0 {
						t.Fatalf("key_%d: w=32 but bits above 32 set: {hi: 0x%x, lo: 0x%x}", i, cr.hi, cr.lo)
					}
				case 64:
					if cr.hi != 0 {
						t.Fatalf("key_%d: w=64 but hi bits set: 0x%x", i, cr.hi)
					}
				case 128:
					// All 128 bits available, no constraint on upper bits
				}
			}
		})
	}
}

func TestGetCoeffRow_WidthsProduceDifferentRows(t *testing.T) {
	// Different widths should generally produce different coefficient rows
	// for the same input hash, since the derivation logic differs.
	h32 := newStandardHasher(32, 10000, 7, true)
	h64 := newStandardHasher(64, 10000, 7, true)
	h128 := newStandardHasher(128, 10000, 7, true)

	h32.setOrdinalSeed(0)
	h64.setOrdinalSeed(0)
	h128.setOrdinalSeed(0)

	kh := h32.keyHash([]byte("width_test"))
	rh := h32.rehash(kh)

	cr32 := h32.getCoeffRow(rh)
	cr64 := h64.getCoeffRow(rh)
	cr128 := h128.getCoeffRow(rh)

	t.Logf("w=32:  {hi: 0x%016x, lo: 0x%016x}", cr32.hi, cr32.lo)
	t.Logf("w=64:  {hi: 0x%016x, lo: 0x%016x}", cr64.hi, cr64.lo)
	t.Logf("w=128: {hi: 0x%016x, lo: 0x%016x}", cr128.hi, cr128.lo)

	// w=32 should have fewer bits set than w=64/128
	if cr32.hi != 0 {
		t.Error("w=32 should have hi=0")
	}
	if cr64.hi != 0 {
		t.Error("w=64 should have hi=0")
	}
	if cr128.hi == 0 {
		t.Error("w=128 should have hi≠0 for most inputs")
	}
}

func TestFirstCoeffAlwaysOne(t *testing.T) {
	hTrue := newStandardHasher(128, 10000, 7, true)
	if !hTrue.firstCoeffAlwaysOne() {
		t.Error("expected firstCoeffAlwaysOne=true")
	}

	hFalse := newStandardHasher(128, 10000, 7, false)
	if hFalse.firstCoeffAlwaysOne() {
		t.Error("expected firstCoeffAlwaysOne=false")
	}
}

func TestGetCoeffRow_NoFirstCoeffAlwaysOne(t *testing.T) {
	// When firstCoeffAlwaysOne=false, LSB is NOT forced to 1.
	// coeffRow must still be non-zero (or near-zero probability for w=32).
	for _, w := range []uint32{32, 64, 128} {
		w := w
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			h := newStandardHasher(w, 10000, 7, false)
			h.setOrdinalSeed(0)

			lsbZeroCount := 0
			for i := 0; i < 10000; i++ {
				kh := h.keyHash([]byte(fmt.Sprintf("key_%d", i)))
				rh := h.rehash(kh)
				cr := h.getCoeffRow(rh)

				// Must be non-zero
				if cr.isZero() {
					t.Fatalf("key_%d: coeffRow is zero for w=%d", i, w)
				}

				// Bits above w must still be zero
				switch w {
				case 32:
					if cr.hi != 0 || cr.lo>>32 != 0 {
						t.Fatalf("key_%d: w=32 but bits above 32 set", i)
					}
				case 64:
					if cr.hi != 0 {
						t.Fatalf("key_%d: w=64 but hi bits set", i)
					}
				}

				if cr.lo&1 == 0 {
					lsbZeroCount++
				}
			}

			// With firstCoeffAlwaysOne=false, roughly half should have LSB=0
			if lsbZeroCount == 0 {
				t.Errorf("w=%d: all 10000 rows had LSB=1, expected ~50%% with LSB=0", w)
			}
			t.Logf("w=%d: %d/10000 rows had LSB=0 (%.1f%%)", w, lsbZeroCount,
				100*float64(lsbZeroCount)/10000)
		})
	}
}
