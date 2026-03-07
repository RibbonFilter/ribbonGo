package ribbonGo

import (
	"errors"

	"github.com/zeebo/xxh3"
)

// =============================================================================
// PUBLIC API — the user-facing surface of the Ribbon filter library
// =============================================================================

// ErrConstructionFailed is returned by Build when the banding algorithm
// could not solve the linear system within the configured number of seed
// retries.
//
// This typically means the input contains duplicate keys (which always
// produce linearly dependent equations regardless of seed).
//
// Callers should either:
//   - Remove duplicate keys from the input.
//   - Increase Config.MaxSeeds.
var ErrConstructionFailed = errors.New("ribbon: construction failed after exhausting all seed retries")

// =============================================================================
// CONFIG — construction parameters
// =============================================================================

// Config holds the parameters for constructing a Ribbon filter.
//
// Every configurable knob described in Dillinger & Walzer (2021) is exposed
// so that users can reproduce and experiment with every trade-off:
//   - CoeffBits (w): ribbon width, controlling the space–construction trade-off.
//   - ResultBits (r): fingerprint bits, controlling the FPR.
//   - FirstCoeffAlwaysOne: whether to force the first coefficient bit to 1.
//   - MaxSeeds: the maximum number of hash seed retries before giving up.
type Config struct {
	// CoeffBits is the ribbon width w: must be 32, 64, or 128.
	//
	// Paper §4: "w ∈ {32, 64, 128}. Overhead ratio m/n ≈ 1 + 2.3/w."
	// Larger w → less space overhead per key, slower construction per key.
	//   w=32:  ~7.2% overhead, fastest construction per key
	//   w=64:  ~3.6% overhead, balanced
	//   w=128: ~1.8% overhead, most compact, slowest construction per key
	CoeffBits uint32

	// ResultBits is the number of fingerprint bits r stored per key.
	// The false-positive rate (FPR) is approximately 2^(-r).
	//
	// Paper §3: "each query computes an r-bit fingerprint and compares
	// it against the stored solution row."
	//   r=7:  FPR ≈ 0.78%
	//   r=8:  FPR ≈ 0.39%
	//   r=10: FPR ≈ 0.098%
	//
	// Must be in [1, 8]. Limited to 8 because the solution stores one
	// uint8 per slot (the paper calls these "result rows").
	ResultBits uint

	// FirstCoeffAlwaysOne controls whether bit 0 of every coefficient row
	// is forced to 1. When true, the Gaussian elimination pivot is
	// deterministic at column s(x), removing the need for a leading-zero
	// scan during banding (faster construction).
	//
	// Paper §4: "setting the first coefficient bit to 1."
	// Set to false for research/experimentation with natural coefficient
	// distributions.
	FirstCoeffAlwaysOne bool

	// MaxSeeds is the maximum number of hash seed retries before
	// construction returns ErrConstructionFailed.
	// A value of 0 uses the default (256).
	//
	// Paper §2: "if banding fails, retry with a new seed."
	// With typical overhead ratios (≥ 1.05), fewer than 10 seeds are
	// usually needed.
	MaxSeeds uint32
}

// =============================================================================
// RIBBON — the main filter type
// =============================================================================

// Ribbon is a space-efficient probabilistic filter for approximate set
// membership queries.
//
// Create one with New (default settings) or NewWithConfig (custom
// parameters), then call Build to construct the filter from a set of
// keys. After building, call Contains to test membership.
//
// A Ribbon filter guarantees:
//   - No false negatives: if a key was in the build set, Contains
//     always returns true.
//   - Configurable false positives: non-member keys return true with
//     probability ≈ 2^(-r), where r is Config.ResultBits.
//
// Thread safety: after Build completes, Contains may be called
// concurrently from multiple goroutines without synchronisation.
//
// Based on: Dillinger & Walzer, "Ribbon filter: practically smaller
// than Bloom and Xor" (2021), arXiv:2103.02515.
type Ribbon struct {
	cfg Config
	f   *filter
}

// New creates a Ribbon filter with default settings.
//
// Defaults: w=128 (most compact), r=7 (FPR ≈ 0.78%),
// firstCoeffAlwaysOne=true (fastest construction), maxSeeds=256.
//
// Paper §4: "w=128 seems to be closest to a generally good choice."
func New() *Ribbon {
	return &Ribbon{cfg: defaultConfig()}
}

// NewWithConfig creates a Ribbon filter with custom parameters.
//
// Panics if Config.CoeffBits is not 32, 64, or 128, or if
// Config.ResultBits is not in [1, 8].
func NewWithConfig(cfg Config) *Ribbon {
	validateConfig(cfg)
	return &Ribbon{cfg: normalizeConfig(cfg)}
}

// Build constructs the filter from the given keys.
//
// Keys must be unique: duplicate keys produce identical equations
// regardless of the hash seed, causing guaranteed construction failure.
// If duplicates may be present, the caller should de-duplicate first.
//
// Build may be called multiple times to rebuild the filter with
// different key sets. Each call replaces the previous filter.
//
// Returns ErrConstructionFailed if banding fails for all seed retries.
func (r *Ribbon) Build(keys []string) error {
	byteKeys := make([][]byte, len(keys))
	for i, k := range keys {
		byteKeys[i] = []byte(k)
	}

	f, err := buildFilter(byteKeys, r.cfg)
	if err != nil {
		return err
	}
	r.f = f
	return nil
}

// Contains tests whether key is probably a member of the set used to
// build this filter.
//
// Returns true if the key is probably in the set (with false-positive
// probability ≈ 2^(-r)), or false if the key is definitely not in the set.
//
// Returns false if Build has not been called yet.
//
// This is the hot path of the library — zero allocations, branchless
// coefficient derivation, and skip-zero dot product iteration.
func (r *Ribbon) Contains(key string) bool {
	if r.f == nil {
		return false
	}
	return r.f.containsHash(xxh3.Hash([]byte(key)))
}
