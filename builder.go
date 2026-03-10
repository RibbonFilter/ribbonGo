package ribbonGo

import "math"

// =============================================================================
// BUILD — internal filter construction with seed-retry loop
// =============================================================================

// defaultConfig returns the recommended configuration for general use.
//
// It selects w=128 (most compact), r=7 (FPR ≈ 0.78%), firstCoeffAlwaysOne
// (fastest construction), and lets the builder compute the optimal
// overhead ratio and seed budget automatically.
//
// Paper §4: "w=128 seems to be closest to a generally good choice."
func defaultConfig() Config {
	return Config{
		CoeffBits:           128,
		ResultBits:          7,
		FirstCoeffAlwaysOne: true,
	}
}

// buildFilter constructs a Ribbon filter from raw byte-slice keys.
//
// Phase 1 (once): each key is hashed with XXH3 to a 64-bit stored hash.
// Phase 2 (per seed attempt): stored hashes are remixed with the current
// seed and fed to the banding algorithm. On success, back-substitution
// produces the solution vector which is packaged into a filter.
//
// Paper §2: "We hash each key once (Phase 1). For each seed attempt,
// we derive (start, coefficients, result) triples (Phase 2) and
// perform on-the-fly Gaussian elimination."
//
// Keys must be unique: duplicate keys produce identical (start, coeff,
// result) triples regardless of seed, causing guaranteed linear
// dependence. If duplicates may be present, the caller should
// de-duplicate before calling buildFilter.
//
// Returns ErrConstructionFailed if banding fails for all seeds.
//
// [RocksDB: Standard128RibbonBitsBuilder::Finish in filter_policy.cc]
func buildFilter(keys []string, cfg Config) (*filter, error) {
	validateConfig(cfg)
	cfg = normalizeConfig(cfg)

	numKeys := len(keys)
	if numKeys == 0 {
		return emptyFilter(cfg), nil
	}

	// Phase 1: hash all keys once with XXH3.
	// The stored hashes are reused across all seed attempts, amortising
	// the cost of the expensive XXH3 computation.
	// Uses keyHashString to avoid string→[]byte allocation per key.
	h := newStandardHasher(cfg.CoeffBits, 0, cfg.ResultBits, cfg.FirstCoeffAlwaysOne)
	hashes := make([]uint64, numKeys)
	for i, key := range keys {
		hashes[i] = h.keyHashString(key)
	}

	return buildCore(hashes, cfg)
}

// buildFromHashes constructs a Ribbon filter from pre-computed 64-bit
// key hashes.
//
// The hashes must have been produced by the same Phase 1 hash function
// used internally (XXH3). This is useful for applications that maintain
// their own hash cache or use pre-hashed pipelines.
//
// Hashes must be unique: duplicate hashes produce identical equations
// regardless of seed, causing guaranteed linear dependence.
//
// Returns ErrConstructionFailed if banding fails for all seeds.
func buildFromHashes(hashes []uint64, cfg Config) (*filter, error) {
	validateConfig(cfg)
	cfg = normalizeConfig(cfg)

	if len(hashes) == 0 {
		return emptyFilter(cfg), nil
	}

	return buildCore(hashes, cfg)
}

// =============================================================================
// INTERNAL — construction core and helpers
// =============================================================================

// buildCore implements the seed-retry loop that orchestrates the hasher,
// bander, and solver to produce a filter.
//
// Algorithm (paper §2):
//  1. Compute numStarts and numSlots from numKeys and coeffBits.
//  2. Create a standardHasher and standardBander.
//  3. For each seed ordinal 0..MaxSeeds-1:
//     a. Set the hasher's seed.
//     b. Derive (start, coeffRow, result) for every stored hash.
//     c. Reset the bander and attempt AddRange.
//     d. If successful, call backSubstitute and return the filter.
//  4. If all seeds fail, return ErrConstructionFailed.
//
// Memory strategy:
//   - The []hashResult buffer is allocated once and reused across retries.
//   - The bander is allocated once and reset() between retries (clear()
//     compiles to optimised memset, avoiding per-element zeroing).
//   - Only the final solution allocates new memory (one []uint8).
//
// [RocksDB: BandingAdd retry loop in StandardBanding::ResetAndFindSeedToSolve]
func buildCore(hashes []uint64, cfg Config) (*filter, error) {
	numKeys := len(hashes)

	// Compute sizing.
	// The overhead ratio grows logarithmically with n (not a fixed constant).
	// RocksDB ribbon_config.cc uses empirically-derived lookup tables for
	// small n and a formula (baseFactor + log2(n) * factorPerPow2) for
	// large n. The overhead for ~5% per-seed failure probability is:
	//   w=128: ~1.04 at n=1K, ~1.05 at n=1M
	//   w=64:  ~1.05 at n=1K, ~1.12 at n=1M
	//   w=32:  ~1.11 at n=1K, ~1.30 at n=1M
	// numSlots: total columns in the banding matrix.
	// numStarts: valid start positions (= numSlots - w + 1).
	numSlots := computeNumSlots(numKeys, cfg.CoeffBits)
	numStarts := numSlots - cfg.CoeffBits + 1

	// Create the hasher (concrete *standardHasher for devirtualisation).
	h := newStandardHasher(cfg.CoeffBits, numStarts, cfg.ResultBits, cfg.FirstCoeffAlwaysOne)

	// Create the bander (allocated once, reset between retries).
	bd := newStandardBander(numSlots, cfg.CoeffBits, cfg.FirstCoeffAlwaysOne)

	// Pre-allocate the hashResult buffer (reused across all seed attempts).
	// Each hashResult is 24 bytes; for 100K keys this is ~2.4 MB — acceptable
	// for a one-time construction cost, and AddRange's software-pipelined
	// prefetching more than pays for it on large filters.
	hrs := make([]hashResult, numKeys)

	// Seed-retry loop.
	for seed := uint32(0); seed < cfg.MaxSeeds; seed++ {
		h.setOrdinalSeed(seed)

		// Phase 2: derive (start, coeffRow, result) for all keys.
		for i, kh := range hashes {
			hrs[i] = h.derive(kh)
		}

		// Reset the bander's slot arrays for this attempt.
		bd.reset()

		// Attempt banding: on-the-fly Gaussian elimination over GF(2).
		if bd.AddRange(hrs) {
			// Success! Back-substitute to compute the solution vector S.
			sol := backSubstitute(bd, cfg.ResultBits)
			return newFilterFromSolution(sol, h, seed, numSlots), nil
		}
		// Banding failed (linear dependence). Try next seed.
	}

	return nil, ErrConstructionFailed
}

// validateConfig panics on invalid configuration values.
// These are programmer errors (wrong constants), not runtime failures.
func validateConfig(cfg Config) {
	switch cfg.CoeffBits {
	case 32, 64, 128:
		// valid
	default:
		panic("ribbon: Config.CoeffBits must be 32, 64, or 128")
	}
	if cfg.ResultBits == 0 || cfg.ResultBits > 8 {
		panic("ribbon: Config.ResultBits must be in [1, 8]")
	}
}

// normalizeConfig fills in zero-valued optional fields with sensible defaults.
func normalizeConfig(cfg Config) Config {
	if cfg.MaxSeeds == 0 {
		cfg.MaxSeeds = 256
	}
	return cfg
}

// =============================================================================
// SLOT COMPUTATION — RocksDB-style dynamic overhead ratio
// =============================================================================

// bandingConfigData holds empirically-derived parameters for computing the
// number of slots from the number of keys.
//
// Unlike the simplified formula m/n ≈ 1 + 2.3/w (which is the theoretical
// minimum for ~50% per-seed success), the ACTUAL required overhead ratio
// grows logarithmically with n. For large n, this growth is significant:
//   - w=64  at n=10K needs ~1.06× but at n=100M needs ~1.14×
//   - w=128 at n=10K needs ~1.03× but at n=100M needs ~1.05×
//
// RocksDB solves this with empirically-derived lookup tables (from the
// FindOccupancy test) for 2^0..2^17 slots, and a linear extrapolation
// formula for larger slot counts:
//
//	factor = baseFactor + log2(numSlots) × factorPerPow2
//
// where factor = numSlots / numKeysAdded.
//
// The data below uses kOneIn20 construction failure chance (~5% per seed),
// without smash. This matches the RocksDB default for standard filters.
//
// [RocksDB: BandingConfigHelperData in ribbon_config.cc]
type bandingConfigData struct {
	// knownToAdd[i] = max keys that can be banded into 2^i slots with
	// ~5% per-seed failure probability. Zero means unsupported.
	knownToAdd [18]float64

	// factorPerPow2: how much the overhead factor increases per doubling
	// of the slot count. Used for extrapolation beyond the lookup table.
	factorPerPow2 float64

	// baseFactor: precomputed from the lookup table as:
	//   baseFactor = (2^17 / knownToAdd[17]) - 17 * factorPerPow2
	baseFactor float64
}

// getNumToAddForPow2 returns how many keys can be banded into 2^log2Slots
// slots with ~5% per-seed failure probability.
func (d *bandingConfigData) getNumToAddForPow2(log2Slots uint32) float64 {
	if log2Slots < 18 {
		if v := d.knownToAdd[log2Slots]; v > 0 {
			return v
		}
	}
	// Formula for large values or unsupported small values.
	factor := d.baseFactor + float64(log2Slots)*d.factorPerPow2
	if factor < 1.0 {
		factor = 1.0
	}
	return float64(uint64(1)<<log2Slots) / factor
}

// Empirical data from RocksDB's FindOccupancy test (ribbon_config.cc).
// kOneIn20, kCoeffBits=128, smash=false.
var configData128 = func() bandingConfigData {
	known := [18]float64{
		0, 0, 0, 0, 0, 0, 0, 0, // 2^0..2^7: unsupported
		248.851, // 2^8  = 256 slots
		499.532, // 2^9  = 512
		1001.26, // 2^10 = 1024
		2003.97, // 2^11 = 2048
		4005.59, // 2^12 = 4096
		8000.39, // 2^13 = 8192
		15966.6, // 2^14 = 16384
		31828.1, // 2^15 = 32768
		63447.3, // 2^16 = 65536
		126506,  // 2^17 = 131072
	}
	const fpp = 0.0038
	finalFactor := float64(uint32(1)<<17) / known[17]
	return bandingConfigData{
		knownToAdd:    known,
		factorPerPow2: fpp,
		baseFactor:    finalFactor - 17*fpp,
	}
}()

// kOneIn20, kCoeffBits=64, smash=false.
var configData64 = func() bandingConfigData {
	known := [18]float64{
		0, 0, 0, 0, 0, 0, 0, // 2^0..2^6: unsupported
		120.659, // 2^7  = 128 slots
		243.346, // 2^8  = 256
		488.168, // 2^9  = 512
		976.373, // 2^10 = 1024
		1948.86, // 2^11 = 2048
		3875.85, // 2^12 = 4096
		7704.97, // 2^13 = 8192
		15312.4, // 2^14 = 16384
		30395.1, // 2^15 = 32768
		60321.8, // 2^16 = 65536
		119813,  // 2^17 = 131072
	}
	const fpp = 0.0083
	finalFactor := float64(uint32(1)<<17) / known[17]
	return bandingConfigData{
		knownToAdd:    known,
		factorPerPow2: fpp,
		baseFactor:    finalFactor - 17*fpp,
	}
}()

// w=32: RocksDB does not support this width. We extrapolate from the
// pattern that factorPerPow2 roughly doubles as w halves:
//
//	w=128: 0.0038, w=64: 0.0083, w=32: ~0.019 (extrapolated)
//
// baseFactor similarly decreases: 0.9715 → 0.9528 → ~0.93.
// No lookup table — formula-only with conservative estimates.
var configData32 = bandingConfigData{
	knownToAdd:    [18]float64{}, // no empirical data
	factorPerPow2: 0.0200,        // conservative: ~2.4× the w=64 rate
	baseFactor:    0.9100,        // conservative: ~0.04 below w=64
}

// computeNumSlots determines the total number of banding matrix columns
// (slots) for the given number of keys and ribbon width.
//
// This implements the RocksDB-style dynamic overhead ratio, where the
// overhead grows logarithmically with n. For small n, empirically-derived
// lookup tables provide exact values. For large n, a linear extrapolation
// formula is used.
//
// The returned numSlots includes the extra (w-1) columns that form the
// "tail" of the banding matrix: numSlots = numStarts + coeffBits - 1.
//
// [RocksDB: BandingConfigHelper1MaybeSupported::GetNumSlots in ribbon_config.cc]
func computeNumSlots(numKeys int, coeffBits uint32) uint32 {
	if numKeys == 0 {
		return 0
	}

	var d *bandingConfigData
	switch coeffBits {
	case 128:
		d = &configData128
	case 64:
		d = &configData64
	case 32:
		d = &configData32
	default:
		panic("ribbon: unsupported coeffBits for slot computation")
	}

	numToAdd := float64(numKeys)
	log2NumToAdd := math.Log(numToAdd) * 1.4426950409 // 1/ln(2)
	approxLog2Slots := uint32(log2NumToAdd + 0.5)
	if approxLog2Slots > 32 {
		approxLog2Slots = 32
	}

	lowerNumToAdd := d.getNumToAddForPow2(approxLog2Slots)

	var upperNumToAdd float64
	if approxLog2Slots == 0 || lowerNumToAdd == 0 {
		// Return minimum non-zero slots (not using smash mode).
		return coeffBits * 2
	}

	if numToAdd < lowerNumToAdd {
		upperNumToAdd = lowerNumToAdd
		approxLog2Slots--
		lowerNumToAdd = d.getNumToAddForPow2(approxLog2Slots)
		if lowerNumToAdd == 0 {
			return coeffBits * 2
		}
	} else {
		upperNumToAdd = d.getNumToAddForPow2(approxLog2Slots + 1)
	}

	upperPortion := (numToAdd - lowerNumToAdd) / (upperNumToAdd - lowerNumToAdd)
	lowerNumSlots := float64(uint64(1) << approxLog2Slots)

	// Interpolation, round up (matching RocksDB).
	numSlots := uint32(upperPortion*lowerNumSlots + lowerNumSlots + 0.999999999)

	// Ensure numSlots is large enough for at least 1 start position
	// plus the (w-1) tail: numStarts = numSlots - coeffBits + 1 >= 1.
	minSlots := coeffBits * 2
	if numSlots < minSlots {
		numSlots = minSlots
	}

	return numSlots
}

// computeNumStarts derives the number of valid start positions from the
// number of keys and the overhead ratio.
//
//	numStarts = ceil(numKeys * overheadRatio)
//
// Guaranteed ≥ 1 for non-empty key sets.
func computeNumStarts(numKeys int, overheadRatio float64) uint32 {
	ns := uint32(math.Ceil(float64(numKeys) * overheadRatio))
	if ns < 1 {
		ns = 1
	}
	return ns
}

// emptyFilter returns a filter that always returns false for contains.
// Used when buildFilter is called with zero keys.
func emptyFilter(cfg Config) *filter {
	return &filter{
		hasher: *newStandardHasher(cfg.CoeffBits, 0, cfg.ResultBits, cfg.FirstCoeffAlwaysOne),
	}
}

// buildCoreWithOverride is an unexported variant of buildCore that accepts
// an explicit overhead ratio, bypassing the automatic derivation from w.
// Used only by tests that need to force pathological ratios (e.g. 1.001)
// to exercise the retry loop and error paths.
func buildCoreWithOverride(hashes []uint64, cfg Config, overheadRatio float64) (*filter, error) {
	numKeys := len(hashes)
	numStarts := computeNumStarts(numKeys, overheadRatio)
	numSlots := numStarts + cfg.CoeffBits - 1

	h := newStandardHasher(cfg.CoeffBits, numStarts, cfg.ResultBits, cfg.FirstCoeffAlwaysOne)
	bd := newStandardBander(numSlots, cfg.CoeffBits, cfg.FirstCoeffAlwaysOne)
	hrs := make([]hashResult, numKeys)

	for seed := uint32(0); seed < cfg.MaxSeeds; seed++ {
		h.setOrdinalSeed(seed)
		for i, kh := range hashes {
			hrs[i] = h.derive(kh)
		}
		bd.reset()
		if bd.AddRange(hrs) {
			sol := backSubstitute(bd, cfg.ResultBits)
			return newFilterFromSolution(sol, h, seed, numSlots), nil
		}
	}
	return nil, ErrConstructionFailed
}

// newFilterFromSolution packages a solution into a filter, padding the
// data array for bounds-check elimination in the contains hot path.
//
// The solution data from backSubstitute has length numSlots + w (where w
// is the solver's detected width — which may be 64 for w=32, see note
// in backSubstitute). We re-allocate with numSlots + 128 bytes of
// capacity so that contains can use a single `_ = data[127]` BCE proof
// for all ribbon widths:
//
//	For w=32:  max start = numSlots-32,  data[start:] has len ≥ 160  → data[127] ✓
//	For w=64:  max start = numSlots-64,  data[start:] has len ≥ 192  → data[127] ✓
//	For w=128: max start = numSlots-128, data[start:] has len ≥ 256  → data[127] ✓
//
// The extra padding bytes are always zero, matching the semantics of
// empty (unoccupied) solution slots.
func newFilterFromSolution(sol *solution, h *standardHasher, seed uint32, numSlots uint32) *filter {
	paddedLen := int(numSlots) + 128
	data := make([]uint8, paddedLen)
	copy(data, sol.data)

	return &filter{
		data:     data,
		hasher:   *h, // copy by value for cache locality
		seed:     seed,
		numSlots: numSlots,
	}
}
