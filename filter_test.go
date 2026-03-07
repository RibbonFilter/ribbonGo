package ribbonGo

import (
	"fmt"
	"math"
	"testing"

	"github.com/zeebo/xxh3"
)

// =============================================================================
// TEST HELPERS
// =============================================================================

// generateKeys creates numKeys deterministic byte-slice keys with the
// given prefix. Each key is of the form "<prefix>_<index>".
func generateKeys(prefix string, numKeys int) [][]byte {
	keys := make([][]byte, numKeys)
	for i := range keys {
		keys[i] = []byte(fmt.Sprintf("%s_%d", prefix, i))
	}
	return keys
}

// generateHashes creates numKeys deterministic uint64 hashes from keys.
func generateHashes(prefix string, numKeys int) []uint64 {
	hashes := make([]uint64, numKeys)
	for i := range hashes {
		hashes[i] = xxh3.Hash([]byte(fmt.Sprintf("%s_%d", prefix, i)))
	}
	return hashes
}

// allConfigs returns all 6 valid (coeffBits, firstCoeffAlwaysOne) configurations.
func allConfigs() []Config {
	var cfgs []Config
	for _, w := range []uint32{32, 64, 128} {
		for _, fcao := range []bool{true, false} {
			cfgs = append(cfgs, Config{
				CoeffBits:           w,
				ResultBits:          7,
				FirstCoeffAlwaysOne: fcao,
			})
		}
	}
	return cfgs
}

// configName returns a short test name for a Config.
func configName(cfg Config) string {
	return fmt.Sprintf("w=%d/fcao=%v", cfg.CoeffBits, cfg.FirstCoeffAlwaysOne)
}

// =============================================================================
// BUILD — constructor tests
// =============================================================================

func TestBuild_AllConfigs(t *testing.T) {
	// Build a filter with 1000 keys for each of the 6 configurations.
	// All inserted keys must be found (zero false negatives).
	const numKeys = 1000

	for _, cfg := range allConfigs() {
		t.Run(configName(cfg), func(t *testing.T) {
			keys := generateKeys("build_test", numKeys)

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			// Verify all inserted keys are found.
			for i, key := range keys {
				if !f.contains(key) {
					t.Fatalf("false negative for key %d: %q", i, key)
				}
			}

			t.Logf("seed=%d, numSlots=%d, numStarts=%d",
				f.seed, f.numSlots, f.hasher.numStarts)
		})
	}
}

func TestBuild_Empty(t *testing.T) {
	// An empty filter (no keys) should always return false for queries.
	for _, cfg := range allConfigs() {
		t.Run(configName(cfg), func(t *testing.T) {
			f, err := buildFilter(nil, cfg)
			if err != nil {
				t.Fatalf("Build failed for empty input: %v", err)
			}

			if f.hasher.numStarts != 0 {
				t.Errorf("NumStarts = %d, want 0", f.hasher.numStarts)
			}
			if f.numSlots != 0 {
				t.Errorf("NumSlots = %d, want 0", f.numSlots)
			}

			// Must return false for any query.
			if f.contains([]byte("anything")) {
				t.Error("empty filter should always return false")
			}
			if f.containsHash(12345) {
				t.Error("empty filter should always return false for ContainsHash")
			}

			// FPRate should be 0 for empty filter.
			if f.fpRate() != 0.0 {
				t.Errorf("FPRate = %f, want 0.0", f.fpRate())
			}
		})
	}
}

func TestBuild_SingleKey(t *testing.T) {
	// A filter with a single key must find that key and reject most others.
	for _, cfg := range allConfigs() {
		t.Run(configName(cfg), func(t *testing.T) {
			keys := [][]byte{[]byte("the_one_key")}

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			if !f.contains(keys[0]) {
				t.Fatal("false negative for the single inserted key")
			}

			// Check that most non-members return false.
			fps := 0
			const numProbes = 10000
			for i := 0; i < numProbes; i++ {
				if f.contains([]byte(fmt.Sprintf("other_key_%d", i))) {
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
		})
	}
}

func TestBuild_InvalidConfig_CoeffBits(t *testing.T) {
	for _, w := range []uint32{0, 16, 48, 96, 256} {
		t.Run(fmt.Sprintf("w=%d", w), func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Build with CoeffBits=%d should panic", w)
				}
			}()
			buildFilter([][]byte{[]byte("key")}, Config{
				CoeffBits:  w,
				ResultBits: 7,
			})
		})
	}
}

func TestBuild_InvalidConfig_ResultBits(t *testing.T) {
	for _, r := range []uint{0, 9, 16, 64} {
		t.Run(fmt.Sprintf("r=%d", r), func(t *testing.T) {
			defer func() {
				if rec := recover(); rec == nil {
					t.Errorf("Build with ResultBits=%d should panic", r)
				}
			}()
			buildFilter([][]byte{[]byte("key")}, Config{
				CoeffBits:  128,
				ResultBits: r,
			})
		})
	}
}

// =============================================================================
// BUILD FROM HASHES — pre-hashed key construction
// =============================================================================

func TestBuildFromHashes(t *testing.T) {
	// BuildFromHashes should produce a filter that matches ContainsHash.
	for _, cfg := range allConfigs() {
		t.Run(configName(cfg), func(t *testing.T) {
			const numKeys = 500
			hashes := generateHashes("from_hashes", numKeys)

			f, err := buildFromHashes(hashes, cfg)
			if err != nil {
				t.Fatalf("BuildFromHashes failed: %v", err)
			}

			// All inserted hashes must be found.
			for i, h := range hashes {
				if !f.containsHash(h) {
					t.Fatalf("false negative for hash %d: 0x%x", i, h)
				}
			}
		})
	}
}

func TestBuildFromHashes_Empty(t *testing.T) {
	f, err := buildFromHashes(nil, defaultConfig())
	if err != nil {
		t.Fatalf("BuildFromHashes failed for empty input: %v", err)
	}
	if f.hasher.numStarts != 0 {
		t.Errorf("NumStarts = %d, want 0", f.hasher.numStarts)
	}
	if f.containsHash(0) {
		t.Error("empty filter should always return false")
	}
}

// =============================================================================
// CONTAINS — true positive tests (zero false negatives)
// =============================================================================

func TestContains_TruePositives(t *testing.T) {
	// Full pipeline: build and verify all keys across all configurations.
	// Uses a larger key set than TestBuild_AllConfigs for more thorough
	// coverage.
	for _, cfg := range allConfigs() {
		t.Run(configName(cfg), func(t *testing.T) {
			const numKeys = 5000
			keys := generateKeys("tp_test", numKeys)

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			for i, key := range keys {
				if !f.contains(key) {
					t.Fatalf("false negative for key %d (seed=%d, numStarts=%d)",
						i, f.seed, f.hasher.numStarts)
				}
			}
		})
	}
}

func TestContains_EmptyFilter(t *testing.T) {
	// An empty filter (no keys) should always return false.
	f, err := buildFilter(nil, defaultConfig())
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 1000; i++ {
		key := []byte(fmt.Sprintf("probe_%d", i))
		if f.contains(key) {
			t.Fatalf("empty filter returned true for %q", key)
		}
	}
}

// =============================================================================
// CONTAINS HASH — pre-hashed query tests
// =============================================================================

func TestContainsHash_MatchesContains(t *testing.T) {
	// ContainsHash(xxh3.Hash(key)) must return the same result as
	// Contains(key) for all keys.
	for _, cfg := range allConfigs() {
		t.Run(configName(cfg), func(t *testing.T) {
			const numKeys = 1000
			keys := generateKeys("hash_match", numKeys)

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			// Check inserted keys.
			for _, key := range keys {
				got := f.containsHash(xxh3.Hash(key))
				want := f.contains(key)
				if got != want {
					t.Fatalf("ContainsHash mismatch for key %q: got=%v, want=%v",
						key, got, want)
				}
			}

			// Check non-member keys.
			for i := 0; i < 1000; i++ {
				key := []byte(fmt.Sprintf("non_member_%d", i))
				got := f.containsHash(xxh3.Hash(key))
				want := f.contains(key)
				if got != want {
					t.Fatalf("ContainsHash mismatch for non-member %q: got=%v, want=%v",
						key, got, want)
				}
			}
		})
	}
}

// =============================================================================
// FALSE POSITIVE RATE — statistical validation
// =============================================================================

// TestContains_FalsePositiveRate is the definitive FPR validation test.
//
// It builds a filter with 10,000 random keys, then queries 1,000,000
// completely different random keys and measures the false-positive rate.
// The measured FPR must closely match the theoretical rate 2^(-r).
//
// If the math in the Bander, Solver, or Query is wrong, this test will
// fail dramatically (e.g., FPR near 50% instead of 0.78%).
//
// Paper §3: "the false-positive probability is approximately 2^(-r)."
func TestContains_FalsePositiveRate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping FPR test in short mode (1M queries)")
	}

	const numKeys = 10000
	const numNonMembers = 1000000

	for _, r := range []uint{7, 8} {
		for _, w := range []uint32{64, 128} {
			name := fmt.Sprintf("w=%d/r=%d", w, r)
			t.Run(name, func(t *testing.T) {
				keys := generateKeys("fpr_member", numKeys)

				cfg := Config{
					CoeffBits:           w,
					ResultBits:          r,
					FirstCoeffAlwaysOne: true,
				}

				f, err := buildFilter(keys, cfg)
				if err != nil {
					t.Fatalf("Build failed: %v", err)
				}

				// Verify zero false negatives first.
				for i, key := range keys {
					if !f.contains(key) {
						t.Fatalf("false negative for key %d", i)
					}
				}

				// Query 1,000,000 non-member keys.
				fps := 0
				for i := 0; i < numNonMembers; i++ {
					key := []byte(fmt.Sprintf("fpr_non_member_%d", i))
					if f.contains(key) {
						fps++
					}
				}

				fpRate := float64(fps) / float64(numNonMembers)
				expectedRate := 1.0 / float64(uint64(1)<<r)

				t.Logf("FP rate: %.4f%% (%d / %d), expected ≈ %.4f%%",
					fpRate*100, fps, numNonMembers, expectedRate*100)

				// Assert the measured FPR is within a reasonable range of
				// the theoretical value. We use a 3× margin to account for
				// statistical variance.
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
}

func TestContains_FalsePositiveRate_AllResultBits(t *testing.T) {
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
			keys := generateKeys("fpr_rbits_member", numKeys)

			cfg := Config{
				CoeffBits:           128,
				ResultBits:          r,
				FirstCoeffAlwaysOne: true,
			}

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			fps := 0
			for i := 0; i < numNonMembers; i++ {
				if f.contains([]byte(fmt.Sprintf("fpr_rbits_non_%d", i))) {
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
// ACCESSORS — metadata verification
// =============================================================================

func TestFilter_Accessors(t *testing.T) {
	cfg := Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
	}
	keys := generateKeys("accessor_test", 1000)
	f, err := buildFilter(keys, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if f.hasher.coeffBits != 128 {
		t.Errorf("CoeffBits() = %d, want 128", f.hasher.coeffBits)
	}
	if f.hasher.resultBits != 7 {
		t.Errorf("ResultBits() = %d, want 7", f.hasher.resultBits)
	}
	if !f.hasher.forceFirstCoeff {
		t.Error("FirstCoeffAlwaysOne() = false, want true")
	}
	if f.hasher.numStarts == 0 {
		t.Error("NumStarts() = 0 for non-empty filter")
	}
	if f.numSlots == 0 {
		t.Error("NumSlots() = 0 for non-empty filter")
	}
	if f.numSlots != f.hasher.numStarts+f.hasher.coeffBits-1 {
		t.Errorf("NumSlots() = %d, want %d (numStarts + w - 1)",
			f.numSlots, f.hasher.numStarts+f.hasher.coeffBits-1)
	}

	expectedFPR := 1.0 / 128.0 // 2^(-7)
	if math.Abs(f.fpRate()-expectedFPR) > 1e-10 {
		t.Errorf("FPRate() = %f, want %f", f.fpRate(), expectedFPR)
	}

	sol := f.data[:f.numSlots]
	if uint32(len(sol)) != f.numSlots {
		t.Errorf("len(SolutionData()) = %d, want %d", len(sol), f.numSlots)
	}
}

func TestFilter_Accessors_Empty(t *testing.T) {
	f, _ := buildFilter(nil, defaultConfig())

	if f.hasher.numStarts != 0 {
		t.Errorf("NumStarts() = %d, want 0", f.hasher.numStarts)
	}
	if f.numSlots != 0 {
		t.Errorf("NumSlots() = %d, want 0", f.numSlots)
	}
	if f.fpRate() != 0.0 {
		t.Errorf("FPRate() = %f, want 0.0", f.fpRate())
	}
	if f.data[:f.numSlots] != nil {
		t.Error("SolutionData() should be nil for empty filter")
	}
}

// =============================================================================
// DEFAULT CONFIG
// =============================================================================

func TestDefaultConfig(t *testing.T) {
	cfg := defaultConfig()

	if cfg.CoeffBits != 128 {
		t.Errorf("defaultConfig().CoeffBits = %d, want 128", cfg.CoeffBits)
	}
	if cfg.ResultBits != 7 {
		t.Errorf("defaultConfig().ResultBits = %d, want 7", cfg.ResultBits)
	}
	if !cfg.FirstCoeffAlwaysOne {
		t.Error("defaultConfig().FirstCoeffAlwaysOne = false, want true")
	}
}

// =============================================================================
// LARGE-SCALE INTEGRATION TEST
// =============================================================================

func TestBuild_LargeScale(t *testing.T) {
	// Large-scale integration test: 50K keys, all configs, verify all
	// queries return true.
	if testing.Short() {
		t.Skip("skipping large-scale test in short mode")
	}

	const numKeys = 50000

	for _, w := range []uint32{64, 128} {
		name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
		t.Run(name, func(t *testing.T) {
			keys := generateKeys("large_scale", numKeys)

			cfg := Config{
				CoeffBits:           w,
				ResultBits:          7,
				FirstCoeffAlwaysOne: true,
			}

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			for i, key := range keys {
				if !f.contains(key) {
					t.Fatalf("false negative for key %d (seed=%d)", i, f.seed)
				}
			}

			t.Logf("seed=%d, numSlots=%d, numStarts=%d, FPR=%.4f%%",
				f.seed, f.numSlots, f.hasher.numStarts, f.fpRate()*100)
		})
	}
}

// =============================================================================
// OVERHEAD RATIO — stress test with tight sizing
// =============================================================================

func TestBuild_TightOverhead(t *testing.T) {
	// Stress-test the retry loop with a very tight overhead ratio.
	// This forces more seed retries but should still succeed within
	// the default 256 seed budget.
	if testing.Short() {
		t.Skip("skipping tight overhead test in short mode")
	}

	const numKeys = 5000
	keys := generateKeys("tight_overhead", numKeys)

	cfg := Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
		MaxSeeds:            256,
	}
	cfg = normalizeConfig(cfg)

	// Phase 1: hash all keys.
	h := newStandardHasher(cfg.CoeffBits, 0, cfg.ResultBits, cfg.FirstCoeffAlwaysOne)
	hashes := make([]uint64, numKeys)
	for i, key := range keys {
		hashes[i] = h.keyHash(key)
	}

	// Use a very tight overhead ratio via the internal override.
	f, err := buildCoreWithOverride(hashes, cfg, 1.02)
	if err != nil {
		t.Fatalf("Build failed with tight overhead: %v", err)
	}

	// Verify correctness.
	for i, key := range keys {
		if !f.contains(key) {
			t.Fatalf("false negative for key %d", i)
		}
	}

	t.Logf("tight overhead: seed=%d, numSlots=%d", f.seed, f.numSlots)
}

// =============================================================================
// ERROR PATH — construction failure
// =============================================================================

func TestBuild_MaxSeedsExhausted(t *testing.T) {
	// Force construction failure by using an extremely tight overhead
	// ratio with a small seed budget.
	keys := generateKeys("exhaust_seeds", 1000)

	cfg := Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
		MaxSeeds:            1, // only 1 attempt
	}
	cfg = normalizeConfig(cfg)

	// Phase 1: hash all keys.
	h := newStandardHasher(cfg.CoeffBits, 0, cfg.ResultBits, cfg.FirstCoeffAlwaysOne)
	hashes := make([]uint64, len(keys))
	for i, key := range keys {
		hashes[i] = h.keyHash(key)
	}

	// Use an extremely tight overhead ratio via the internal override.
	_, err := buildCoreWithOverride(hashes, cfg, 1.001)
	if err == nil {
		// It's possible (though unlikely) that even 1 seed succeeds.
		// This is OK — the test is about verifying error handling.
		t.Log("construction succeeded with 1 seed (unlikely but possible)")
		return
	}

	if err != ErrConstructionFailed {
		t.Errorf("expected ErrConstructionFailed, got: %v", err)
	}
}

// =============================================================================
// w=32 SPECIFIC TESTS — verify the w=32 path works correctly
// =============================================================================

func TestBuild_W32_Correctness(t *testing.T) {
	// w=32 has a subtlety: BackSubstitute treats it as w=64 (since
	// coeffHi is nil for both). This test verifies that the full pipeline
	// works correctly for w=32 despite this internal representation detail.
	for _, fcao := range []bool{true, false} {
		name := fmt.Sprintf("w=32/fcao=%v", fcao)
		t.Run(name, func(t *testing.T) {
			const numKeys = 2000
			keys := generateKeys("w32_test", numKeys)

			cfg := Config{
				CoeffBits:           32,
				ResultBits:          7,
				FirstCoeffAlwaysOne: fcao,
			}

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			if f.hasher.coeffBits != 32 {
				t.Errorf("CoeffBits() = %d, want 32", f.hasher.coeffBits)
			}

			for i, key := range keys {
				if !f.contains(key) {
					t.Fatalf("false negative for key %d", i)
				}
			}

			// Check FPR for w=32.
			fps := 0
			const numProbes = 100000
			for i := 0; i < numProbes; i++ {
				if f.contains([]byte(fmt.Sprintf("w32_non_%d", i))) {
					fps++
				}
			}
			fpRate := float64(fps) / float64(numProbes)
			expectedRate := 1.0 / 128.0 // 2^(-7)
			t.Logf("w=32 FP rate: %.4f%% (expected ≈ %.4f%%)",
				fpRate*100, expectedRate*100)

			if fpRate > expectedRate*3.0 {
				t.Errorf("FP rate %.4f%% is much higher than expected %.4f%%",
					fpRate*100, expectedRate*100)
			}
		})
	}
}

// =============================================================================
// MULTI-BIT RESULT TESTS — verify different result bit widths
// =============================================================================

func TestBuild_AllResultBits(t *testing.T) {
	// Build and verify filters for each valid resultBits value (1..8).
	const numKeys = 500

	for r := uint(1); r <= 8; r++ {
		name := fmt.Sprintf("r=%d", r)
		t.Run(name, func(t *testing.T) {
			keys := generateKeys("rbits_test", numKeys)

			cfg := Config{
				CoeffBits:           128,
				ResultBits:          r,
				FirstCoeffAlwaysOne: true,
			}

			f, err := buildFilter(keys, cfg)
			if err != nil {
				t.Fatalf("Build failed for r=%d: %v", r, err)
			}

			for i, key := range keys {
				if !f.contains(key) {
					t.Fatalf("false negative for key %d with r=%d", i, r)
				}
			}
		})
	}
}
