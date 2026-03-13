package ribbonGo

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// TEST HELPERS
// =============================================================================

// generateStringKeys creates numKeys deterministic string keys with the
// given prefix. Each key is of the form "<prefix>_<index>".
func generateStringKeys(prefix string, numKeys int) []string {
	keys := make([]string, numKeys)
	for i := range keys {
		keys[i] = fmt.Sprintf("%s_%d", prefix, i)
	}
	return keys
}

// =============================================================================
// NEW — constructor tests
// =============================================================================

func TestNew(t *testing.T) {
	r := New()
	if r == nil {
		t.Fatal("New() returned nil")
	}

	// Verify default config values are applied.
	if r.cfg.CoeffBits != 128 {
		t.Errorf("cfg.CoeffBits = %d, want 128", r.cfg.CoeffBits)
	}
	if r.cfg.ResultBits != 7 {
		t.Errorf("cfg.ResultBits = %d, want 7", r.cfg.ResultBits)
	}
	if !r.cfg.FirstCoeffAlwaysOne {
		t.Error("cfg.FirstCoeffAlwaysOne = false, want true")
	}

	// Filter should be nil before Build is called.
	if r.f != nil {
		t.Error("filter should be nil before Build()")
	}
}

func TestNewWithConfig_AllValidWidths(t *testing.T) {
	for _, w := range []uint32{32, 64, 128} {
		for _, r := range []uint{1, 4, 7, 8} {
			name := fmt.Sprintf("w=%d/r=%d", w, r)
			t.Run(name, func(t *testing.T) {
				cfg := Config{
					CoeffBits:           w,
					ResultBits:          r,
					FirstCoeffAlwaysOne: true,
				}
				rb := NewWithConfig(cfg)
				if rb == nil {
					t.Fatal("NewWithConfig returned nil")
				}
				if rb.cfg.CoeffBits != w {
					t.Errorf("cfg.CoeffBits = %d, want %d", rb.cfg.CoeffBits, w)
				}
				if rb.cfg.ResultBits != r {
					t.Errorf("cfg.ResultBits = %d, want %d", rb.cfg.ResultBits, r)
				}
			})
		}
	}
}

func TestNewWithConfig_NormalizesMaxSeeds(t *testing.T) {
	// MaxSeeds=0 should be normalised to the default (256).
	rb := NewWithConfig(Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
		MaxSeeds:            0,
	})
	if rb.cfg.MaxSeeds != 256 {
		t.Errorf("MaxSeeds = %d, want 256 (normalised from 0)", rb.cfg.MaxSeeds)
	}

	// Explicit MaxSeeds should be preserved.
	rb = NewWithConfig(Config{
		CoeffBits:           64,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
		MaxSeeds:            50,
	})
	if rb.cfg.MaxSeeds != 50 {
		t.Errorf("MaxSeeds = %d, want 50", rb.cfg.MaxSeeds)
	}
}

func TestNewWithConfig_InvalidCoeffBits(t *testing.T) {
	for _, w := range []uint32{0, 16, 48, 96, 256} {
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			defer func() {
				if rec := recover(); rec == nil {
					t.Errorf("NewWithConfig with CoeffBits=%d should panic", w)
				}
			}()
			NewWithConfig(Config{
				CoeffBits:  w,
				ResultBits: 7,
			})
		})
	}
}

func TestNewWithConfig_InvalidResultBits(t *testing.T) {
	for _, r := range []uint{0, 9, 16, 64} {
		t.Run(fmt.Sprintf("r=%d", r), func(t *testing.T) {
			defer func() {
				if rec := recover(); rec == nil {
					t.Errorf("NewWithConfig with ResultBits=%d should panic", r)
				}
			}()
			NewWithConfig(Config{
				CoeffBits:  128,
				ResultBits: r,
			})
		})
	}
}

// =============================================================================
// NEW FROM KEYS — one-step constructor tests
// =============================================================================

func TestNewFromKeys(t *testing.T) {
	const numKeys = 1000
	keys := generateStringKeys("from_keys", numKeys)

	r, err := NewFromKeys(keys)
	if err != nil {
		t.Fatalf("NewFromKeys failed: %v", err)
	}
	if r == nil {
		t.Fatal("NewFromKeys returned nil Ribbon")
	}

	// Verify default config.
	if r.cfg.CoeffBits != 128 {
		t.Errorf("cfg.CoeffBits = %d, want 128", r.cfg.CoeffBits)
	}
	if r.cfg.ResultBits != 7 {
		t.Errorf("cfg.ResultBits = %d, want 7", r.cfg.ResultBits)
	}

	// All inserted keys must be found (zero false negatives).
	for i, key := range keys {
		if !r.Contains(key) {
			t.Fatalf("false negative for key %d: %q", i, key)
		}
	}
}

func TestNewFromKeys_Empty(t *testing.T) {
	r, err := NewFromKeys(nil)
	if err != nil {
		t.Fatalf("NewFromKeys(nil) failed: %v", err)
	}
	if r.Contains("anything") {
		t.Error("empty filter should return false")
	}
}

func TestNewFromKeys_EmptySlice(t *testing.T) {
	r, err := NewFromKeys([]string{})
	if err != nil {
		t.Fatalf("NewFromKeys([]string{}) failed: %v", err)
	}
	if r.Contains("probe") {
		t.Error("empty filter should return false")
	}
}

func TestNewFromKeysWithConfig(t *testing.T) {
	const numKeys = 1000

	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				cfg := Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: fcao,
				}
				keys := generateStringKeys("from_keys_cfg", numKeys)

				r, err := NewFromKeysWithConfig(cfg, keys)
				if err != nil {
					t.Fatalf("NewFromKeysWithConfig failed: %v", err)
				}
				if r == nil {
					t.Fatal("NewFromKeysWithConfig returned nil")
				}
				if r.cfg.CoeffBits != w {
					t.Errorf("cfg.CoeffBits = %d, want %d", r.cfg.CoeffBits, w)
				}

				for i, key := range keys {
					if !r.Contains(key) {
						t.Fatalf("false negative for key %d: %q", i, key)
					}
				}
			})
		}
	}
}

func TestNewFromKeysWithConfig_InvalidConfig(t *testing.T) {
	defer func() {
		if rec := recover(); rec == nil {
			t.Error("NewFromKeysWithConfig with invalid config should panic")
		}
	}()
	NewFromKeysWithConfig(Config{CoeffBits: 99, ResultBits: 7}, []string{"key"})
}

func TestNewFromKeysWithConfig_Empty(t *testing.T) {
	cfg := Config{
		CoeffBits:           64,
		ResultBits:          8,
		FirstCoeffAlwaysOne: true,
	}
	r, err := NewFromKeysWithConfig(cfg, nil)
	if err != nil {
		t.Fatalf("NewFromKeysWithConfig(nil) failed: %v", err)
	}
	if r.Contains("anything") {
		t.Error("empty filter should return false")
	}
}

func TestNewFromKeys_EquivalentToNewPlusBuild(t *testing.T) {
	// NewFromKeys should produce the same results as New() + Build().
	const numKeys = 500
	keys := generateStringKeys("equiv", numKeys)

	r1, err := NewFromKeys(keys)
	if err != nil {
		t.Fatalf("NewFromKeys failed: %v", err)
	}

	r2 := New()
	if err := r2.Build(keys); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Both filters must agree on all inserted keys.
	for _, key := range keys {
		got1 := r1.Contains(key)
		got2 := r2.Contains(key)
		if got1 != got2 {
			t.Fatalf("disagreement on key %q: NewFromKeys=%v, New+Build=%v",
				key, got1, got2)
		}
	}
}

// =============================================================================
// BUILD — construction via the public API
// =============================================================================

func TestRibbon_Build_AllConfigs(t *testing.T) {
	// Build a filter with 1000 keys for each of the 6 valid
	// (coeffBits, firstCoeffAlwaysOne) configurations.
	// All inserted keys must be found (zero false negatives).
	const numKeys = 1000

	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			name := fmt.Sprintf("w=%d/fcao=%v", w, fcao)
			t.Run(name, func(t *testing.T) {
				rb := NewWithConfig(Config{
					CoeffBits:           w,
					ResultBits:          7,
					FirstCoeffAlwaysOne: fcao,
				})

				keys := generateStringKeys("ribbon_build", numKeys)
				if err := rb.Build(keys); err != nil {
					t.Fatalf("Build failed: %v", err)
				}

				for i, key := range keys {
					if !rb.Contains(key) {
						t.Fatalf("false negative for key %d: %q", i, key)
					}
				}
			})
		}
	}
}

func TestRibbon_Build_Empty(t *testing.T) {
	// Building with zero keys should succeed and Contains should
	// always return false.
	rb := New()
	if err := rb.Build(nil); err != nil {
		t.Fatalf("Build(nil) failed: %v", err)
	}

	if rb.Contains("anything") {
		t.Error("empty filter should return false for any key")
	}
	if rb.Contains("") {
		t.Error("empty filter should return false for empty string")
	}
}

func TestRibbon_Build_EmptySlice(t *testing.T) {
	rb := New()
	if err := rb.Build([]string{}); err != nil {
		t.Fatalf("Build([]string{}) failed: %v", err)
	}
	if rb.Contains("probe") {
		t.Error("empty filter should return false")
	}
}

func TestRibbon_Build_SingleKey(t *testing.T) {
	rb := New()
	if err := rb.Build([]string{"the_one_key"}); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	if !rb.Contains("the_one_key") {
		t.Fatal("false negative for the single inserted key")
	}

	// Most non-members should return false.
	fps := 0
	const numProbes = 10000
	for i := 0; i < numProbes; i++ {
		if rb.Contains(fmt.Sprintf("other_key_%d", i)) {
			fps++
		}
	}
	fpRate := float64(fps) / float64(numProbes)
	t.Logf("FP rate for single-key filter: %.2f%% (%d/%d)",
		fpRate*100, fps, numProbes)

	// With r=7, expected FPR ≈ 0.78%. Allow generous margin.
	if fpRate > 0.05 {
		t.Errorf("FP rate %.2f%% is suspiciously high for r=7", fpRate*100)
	}
}

func TestRibbon_Build_Rebuild(t *testing.T) {
	// Build may be called multiple times. Each call replaces the
	// previous filter entirely.
	rb := New()

	keys1 := generateStringKeys("set_one", 500)
	if err := rb.Build(keys1); err != nil {
		t.Fatalf("first Build failed: %v", err)
	}
	for _, k := range keys1 {
		if !rb.Contains(k) {
			t.Fatalf("false negative after first Build: %q", k)
		}
	}

	// Rebuild with a completely different key set.
	keys2 := generateStringKeys("set_two", 500)
	if err := rb.Build(keys2); err != nil {
		t.Fatalf("second Build failed: %v", err)
	}
	for _, k := range keys2 {
		if !rb.Contains(k) {
			t.Fatalf("false negative after second Build: %q", k)
		}
	}

	// Old keys from set_one should mostly NOT be found (they weren't
	// in the second build set). We tolerate some false positives.
	fps := 0
	for _, k := range keys1 {
		if rb.Contains(k) {
			fps++
		}
	}
	fpRate := float64(fps) / float64(len(keys1))
	t.Logf("after rebuild: %.2f%% of old keys still match (FP)", fpRate*100)
	if fpRate > 0.05 {
		t.Errorf("%.2f%% of old keys still match — rebuild may not have replaced the filter", fpRate*100)
	}
}

func TestRibbon_Build_LargeScale(t *testing.T) {
	// Large-scale test: 50K keys via the public API.
	if testing.Short() {
		t.Skip("skipping large-scale test in short mode")
	}

	const numKeys = 50000
	keys := generateStringKeys("large_ribbon", numKeys)

	rb := New()
	if err := rb.Build(keys); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	for i, key := range keys {
		if !rb.Contains(key) {
			t.Fatalf("false negative for key %d", i)
		}
	}
}

// =============================================================================
// CONTAINS — membership query tests
// =============================================================================

func TestRibbon_Contains_BeforeBuild(t *testing.T) {
	// Contains must return false if Build has never been called.
	rb := New()
	if rb.Contains("anything") {
		t.Error("Contains should return false before Build()")
	}
	if rb.Contains("") {
		t.Error("Contains should return false for empty string before Build()")
	}
}

func TestRibbon_Contains_EmptyStringKey(t *testing.T) {
	// An empty string is a valid key.
	rb := New()
	if err := rb.Build([]string{""}); err != nil {
		t.Fatalf("Build with empty string key failed: %v", err)
	}
	if !rb.Contains("") {
		t.Fatal("false negative for empty string key")
	}
}

func TestRibbon_Contains_UnicodeKeys(t *testing.T) {
	// Unicode strings should work correctly — they're just bytes.
	keys := []string{
		"日本語",
		"中文",
		"한국어",
		"العربية",
		"🎉🎊🎈",
		"café",
		"naïve",
	}

	rb := New()
	if err := rb.Build(keys); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	for _, key := range keys {
		if !rb.Contains(key) {
			t.Fatalf("false negative for unicode key: %q", key)
		}
	}
}

func TestRibbon_Contains_LongKeys(t *testing.T) {
	// Very long keys should hash correctly.
	rb := New()

	long1 := string(make([]byte, 10000)) // 10 KB of zeroes
	long2 := string(make([]byte, 10001)) // 10 KB + 1 byte of zeroes

	keys := []string{long1, long2, "short"}
	if err := rb.Build(keys); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	for _, key := range keys {
		if !rb.Contains(key) {
			t.Fatalf("false negative for key of length %d", len(key))
		}
	}
}

// =============================================================================
// FALSE POSITIVE RATE — statistical validation via the public API
// =============================================================================

func TestRibbon_FalsePositiveRate(t *testing.T) {
	// The definitive FPR validation through the public API.
	// Build with 10,000 keys, query 1,000,000 non-members, and verify
	// the measured FPR matches the theoretical rate 2^(-r).
	if testing.Short() {
		t.Skip("skipping FPR test in short mode (1M queries)")
	}

	const numKeys = 10000
	const numNonMembers = 1000000

	for _, r := range []uint{7, 8} {
		name := fmt.Sprintf("r=%d", r)
		t.Run(name, func(t *testing.T) {
			rb := NewWithConfig(Config{
				CoeffBits:           128,
				ResultBits:          r,
				FirstCoeffAlwaysOne: true,
			})

			keys := generateStringKeys("fpr_ribbon_member", numKeys)
			if err := rb.Build(keys); err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			// Verify zero false negatives first.
			for i, key := range keys {
				if !rb.Contains(key) {
					t.Fatalf("false negative for key %d", i)
				}
			}

			// Query 1,000,000 non-member keys.
			fps := 0
			for i := 0; i < numNonMembers; i++ {
				if rb.Contains(fmt.Sprintf("fpr_ribbon_non_member_%d", i)) {
					fps++
				}
			}

			fpRate := float64(fps) / float64(numNonMembers)
			expectedRate := 1.0 / float64(uint64(1)<<r)

			t.Logf("FP rate: %.4f%% (%d / %d), expected ≈ %.4f%%",
				fpRate*100, fps, numNonMembers, expectedRate*100)

			if fpRate > expectedRate*3.0 {
				t.Errorf("FP rate %.4f%% is much higher than expected %.4f%%",
					fpRate*100, expectedRate*100)
			}
			if fpRate < expectedRate*0.3 {
				t.Errorf("FP rate %.4f%% is suspiciously low (expected ≈ %.4f%%)",
					fpRate*100, expectedRate*100)
			}
		})
	}
}

func TestRibbon_FalsePositiveRate_AllResultBits(t *testing.T) {
	// Verify FPR scales correctly with r: each additional result bit
	// should roughly halve the false-positive rate.
	if testing.Short() {
		t.Skip("skipping multi-r FPR test in short mode")
	}

	const numKeys = 5000
	const numNonMembers = 500000

	prevFPRate := 1.0
	for _, r := range []uint{1, 2, 4, 7, 8} {
		name := fmt.Sprintf("r=%d", r)
		t.Run(name, func(t *testing.T) {
			rb := NewWithConfig(Config{
				CoeffBits:           128,
				ResultBits:          r,
				FirstCoeffAlwaysOne: true,
			})

			keys := generateStringKeys("fpr_ribbon_rbits", numKeys)
			if err := rb.Build(keys); err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			fps := 0
			for i := 0; i < numNonMembers; i++ {
				if rb.Contains(fmt.Sprintf("fpr_ribbon_rbits_non_%d", i)) {
					fps++
				}
			}

			fpRate := float64(fps) / float64(numNonMembers)
			expectedRate := 1.0 / float64(uint64(1)<<r)

			t.Logf("r=%d: FP rate %.4f%% (expected %.4f%%), ratio to prev: %.2f",
				r, fpRate*100, expectedRate*100, prevFPRate/math.Max(fpRate, 1e-10))

			if fpRate > expectedRate*3.0 {
				t.Errorf("FP rate %.4f%% is much higher than expected %.4f%%",
					fpRate*100, expectedRate*100)
			}
			if fpRate < expectedRate*0.3 && r < 8 {
				t.Errorf("FP rate %.4f%% is suspiciously low (expected ≈ %.4f%%)",
					fpRate*100, expectedRate*100)
			}

			prevFPRate = fpRate
		})
	}
}

// =============================================================================
// ERROR PATH — construction failure via the public API
// =============================================================================

func TestRibbon_Build_ErrConstructionFailed(t *testing.T) {
	// Force construction failure by using MaxSeeds=1.
	// With only 1 seed attempt and many keys, failure is likely.
	rb := NewWithConfig(Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
		MaxSeeds:            1,
	})

	// Use enough keys to make single-seed failure very likely.
	keys := generateStringKeys("fail_ribbon", 10000)
	err := rb.Build(keys)

	if err == nil {
		// It's theoretically possible for 1 seed to succeed, but very unlikely
		// with such tight internal overhead. If it does, just log and skip.
		t.Log("construction unexpectedly succeeded with MaxSeeds=1 — skipping error check")
		return
	}

	if !errors.Is(err, ErrConstructionFailed) {
		t.Errorf("expected ErrConstructionFailed, got: %v", err)
	}
}

func TestRibbon_Build_ErrorDoesNotCorruptState(t *testing.T) {
	// After a failed Build, the Ribbon should still be usable —
	// either by calling Build again or by querying (returns false).
	rb := NewWithConfig(Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
		MaxSeeds:            1,
	})

	keys := generateStringKeys("fail_then_retry", 10000)
	err := rb.Build(keys)

	if err != nil {
		// Build failed (expected). Contains should return false.
		if rb.Contains("fail_then_retry_0") {
			t.Error("Contains should return false after failed Build")
		}

		// Retry with a generous seed budget should succeed.
		rb2 := NewWithConfig(Config{
			CoeffBits:           128,
			ResultBits:          7,
			FirstCoeffAlwaysOne: true,
			MaxSeeds:            256,
		})
		if err2 := rb2.Build(keys); err2 != nil {
			t.Fatalf("retry Build failed: %v", err2)
		}
		if !rb2.Contains("fail_then_retry_0") {
			t.Error("false negative after successful retry Build")
		}
	}
}

// =============================================================================
// CONFIG EDGE CASES
// =============================================================================

func TestRibbon_AllWidths(t *testing.T) {
	// Verify that all three ribbon widths produce correct filters
	// through the public API.
	const numKeys = 2000

	for _, w := range []uint32{32, 64, 128} {
		name := fmt.Sprintf("w=%d", w)
		t.Run(name, func(t *testing.T) {
			rb := NewWithConfig(Config{
				CoeffBits:           w,
				ResultBits:          7,
				FirstCoeffAlwaysOne: true,
			})

			keys := generateStringKeys("width_test", numKeys)
			if err := rb.Build(keys); err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			for i, key := range keys {
				if !rb.Contains(key) {
					t.Fatalf("false negative for key %d with w=%d", i, w)
				}
			}

			// Spot-check FPR to ensure correctness.
			fps := 0
			const numProbes = 100000
			for i := 0; i < numProbes; i++ {
				if rb.Contains(fmt.Sprintf("width_non_%d", i)) {
					fps++
				}
			}
			fpRate := float64(fps) / float64(numProbes)
			expectedRate := 1.0 / 128.0 // 2^(-7)
			t.Logf("w=%d FP rate: %.4f%% (expected ≈ %.4f%%)",
				w, fpRate*100, expectedRate*100)

			if fpRate > expectedRate*3.0 {
				t.Errorf("FP rate %.4f%% is much higher than expected %.4f%%",
					fpRate*100, expectedRate*100)
			}
		})
	}
}

func TestRibbon_AllResultBits(t *testing.T) {
	// Build and verify filters for each valid resultBits value (1..8)
	// through the public API.
	const numKeys = 500

	for r := uint(1); r <= 8; r++ {
		name := fmt.Sprintf("r=%d", r)
		t.Run(name, func(t *testing.T) {
			rb := NewWithConfig(Config{
				CoeffBits:           128,
				ResultBits:          r,
				FirstCoeffAlwaysOne: true,
			})

			keys := generateStringKeys("rbits_ribbon", numKeys)
			if err := rb.Build(keys); err != nil {
				t.Fatalf("Build failed for r=%d: %v", r, err)
			}

			for i, key := range keys {
				if !rb.Contains(key) {
					t.Fatalf("false negative for key %d with r=%d", i, r)
				}
			}
		})
	}
}

func TestRibbon_FirstCoeffAlwaysOne_Toggle(t *testing.T) {
	// Both FirstCoeffAlwaysOne=true and false should produce correct filters.
	const numKeys = 1000

	for _, fcao := range []bool{true, false} {
		name := fmt.Sprintf("fcao=%v", fcao)
		t.Run(name, func(t *testing.T) {
			rb := NewWithConfig(Config{
				CoeffBits:           128,
				ResultBits:          7,
				FirstCoeffAlwaysOne: fcao,
			})

			keys := generateStringKeys("fcao_ribbon", numKeys)
			if err := rb.Build(keys); err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			for i, key := range keys {
				if !rb.Contains(key) {
					t.Fatalf("false negative for key %d with fcao=%v", i, fcao)
				}
			}
		})
	}
}

// =============================================================================
// CONCURRENCY — thread safety of Contains after Build
// =============================================================================

func TestRibbon_Contains_ConcurrentReads(t *testing.T) {
	// After Build completes, Contains must be safe to call from
	// multiple goroutines concurrently.
	const numKeys = 5000
	const numGoroutines = 8
	const queriesPerGoroutine = 10000

	rb := New()
	keys := generateStringKeys("concurrent_ribbon", numKeys)
	if err := rb.Build(keys); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	errc := make(chan error, numGoroutines)
	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			for i := 0; i < queriesPerGoroutine; i++ {
				key := keys[i%numKeys]
				if !rb.Contains(key) {
					errc <- fmt.Errorf("goroutine %d: false negative for key %q", gid, key)
					return
				}
			}
			errc <- nil
		}(g)
	}

	for g := 0; g < numGoroutines; g++ {
		if err := <-errc; err != nil {
			t.Fatal(err)
		}
	}
}

// =============================================================================
// ErrConstructionFailed — sentinel value checks
// =============================================================================

func TestErrConstructionFailed_IsError(t *testing.T) {
	// Verify that ErrConstructionFailed satisfies the error interface
	// and has a descriptive message.
	var err error = ErrConstructionFailed

	if err.Error() == "" {
		t.Error("ErrConstructionFailed.Error() should not be empty")
	}

	// errors.Is should work for direct comparison.
	if !errors.Is(err, ErrConstructionFailed) {
		t.Error("errors.Is(ErrConstructionFailed, ErrConstructionFailed) should be true")
	}
}
