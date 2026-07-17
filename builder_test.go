package ribbonGo

import (
	"fmt"
	"testing"
)

// =============================================================================
// ICML numSlots multiple-of-w invariant (Task 1)
//
// The Interleaved Column-Major Layout (paper §5.2) groups slots into blocks
// of exactly w rows, so numSlots must be a multiple of the ribbon width w on
// both build paths (computeNumSlots and buildCoreWithOverride).
// =============================================================================

func TestComputeNumSlots_MultipleOfW(t *testing.T) {
	for _, w := range []uint32{32, 64, 128} {
		for _, numKeys := range []int{1, 100, 1000, 100000} {
			name := fmt.Sprintf("w=%d/n=%d", w, numKeys)
			t.Run(name, func(t *testing.T) {
				numSlots := computeNumSlots(numKeys, w)
				if numSlots%w != 0 {
					t.Errorf("numSlots=%d is not a multiple of w=%d", numSlots, w)
				}
				minSlots := w * 2
				if numSlots < minSlots {
					t.Errorf("numSlots=%d < minSlots=%d", numSlots, minSlots)
				}
			})
		}
	}
}

func TestBuildCoreOverride_MultipleOfW(t *testing.T) {
	// Ratios like 1.2 previously produced non-multiple-of-w numSlots. Assert
	// both the ICML slot invariant and the numStarts consistency after rounding.
	const numKeys = 1000
	for _, cfg := range allConfigs() {
		for _, ratio := range []float64{1.2, 1.05, 1.333} {
			name := fmt.Sprintf("%s/ratio=%.3f", configName(cfg), ratio)
			t.Run(name, func(t *testing.T) {
				hashes := generateHashes("override_mow", numKeys)
				f, err := buildCoreWithOverride(hashes, normalizeConfig(cfg), ratio)
				if err != nil {
					t.Fatalf("buildCoreWithOverride failed: %v", err)
				}
				if f.numSlots%cfg.CoeffBits != 0 {
					t.Errorf("numSlots=%d not a multiple of w=%d", f.numSlots, cfg.CoeffBits)
				}
				if f.hasher.numStarts != f.numSlots-cfg.CoeffBits+1 {
					t.Errorf("numStarts=%d, want %d (numSlots - w + 1)",
						f.hasher.numStarts, f.numSlots-cfg.CoeffBits+1)
				}
			})
		}
	}
}
