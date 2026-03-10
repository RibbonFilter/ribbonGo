package ribbonGo

import (
	"math/bits"

	"github.com/zeebo/xxh3"
)

// =============================================================================
// CONSTANTS — hash pipeline multipliers and seed-mixing parameters
// =============================================================================

const (
	// kRehashFactor: large prime multiplier for seed remixing.
	// When the banding algorithm retries with a new seed, each stored hash
	// must produce a fresh (start, coeff, result) triple. Simple XOR-and-
	// multiply ensures sufficient diffusion:
	//   rehashed = (storedHash ^ rawSeed) * kRehashFactor
	//
	// [RocksDB: StandardRehasherAdapter::HashFn]
	kRehashFactor uint64 = 0x6193d459236a3a0d

	// kCoeffAndResultFactor: multiplier that derives coefficient and result
	// rows from a rehashed hash. The start position depends on the high bits
	// of the rehashed hash (via fastRange64). To ensure the coefficient row
	// and result row are statistically independent of start, we multiply by
	// a different large prime — this redistributes the bit dependencies.
	//
	// Paper §2: "each key x is assigned a start position s(x) and a
	// coefficient row c(x) ∈ {0,1}^w" — s, c, and result must be
	// pairwise independent derivations from the same hash.
	//
	// [RocksDB: kCoeffAndResultFactor in StandardHasher]
	kCoeffAndResultFactor uint64 = 0xc28f82822b650bed

	// kCoeffXor64: XOR constant for 64→128 bit expansion of coefficient row.
	// When w=128, a single 64-bit hash must fill 128 coefficient bits.
	// XOR-folding with a non-zero constant ensures:
	//   (a) upper and lower halves are not identical,
	//   (b) the result is guaranteed non-zero (if a==0 then lo = kCoeffXor64 ≠ 0).
	kCoeffXor64 uint64 = 0xc367844a6e52731d

	// Seed premixing constants (ordinal seed ↔ raw seed conversion).
	// The filter stores only a small ordinal (e.g., 0..255) to identify
	// which seed succeeded. Bijective mixing converts these sequential
	// ordinals into well-distributed 64-bit raw seeds, so that consecutive
	// seeds produce maximally different hash remixings.
	kSeedMixMask       uint64 = 0xf0f0f0f0f0f0f0f0
	kSeedMixShift      uint   = 4
	kToRawSeedFactor   uint64 = 0xc78219a23eeadd03
	kFromRawSeedFactor uint64 = 0xfe1a137d14b475ab
)

// =============================================================================
// PACKAGE-LEVEL UTILITIES
// =============================================================================

// ordinalSeedToRaw converts a small ordinal seed (e.g., 0..255) to a raw seed
// with good bit distribution.
//
// The Ribbon construction retries with a new hash seed when banding fails
// (paper §2). Only the successful seed ordinal is stored in the serialized
// filter — this bijective mixing ensures sequential ordinals produce
// maximally different raw seeds for independent rehashing.
//
//	raw = ordinal * kToRawSeedFactor
//	raw ^= (raw & kSeedMixMask) >> kSeedMixShift
func ordinalSeedToRaw(ordinal uint32) uint64 {
	raw := uint64(ordinal) * kToRawSeedFactor
	raw ^= (raw & kSeedMixMask) >> kSeedMixShift
	return raw
}

// rawSeedToOrdinal converts a raw seed back to an ordinal seed.
// Inverse of ordinalSeedToRaw — used to recover the stored ordinal
// from the internal raw seed state.
func rawSeedToOrdinal(raw uint64) uint32 {
	tmp := raw
	tmp ^= (tmp & kSeedMixMask) >> kSeedMixShift
	return uint32(tmp * kFromRawSeedFactor)
}

// fastRange64 maps a 64-bit hash uniformly into [0, rangeVal) without modulo
// bias, using 128-bit multiply: (uint128(hash) * rangeVal) >> 64.
//
// This is faster than h % rangeVal and has no bias for power-of-2 ranges.
// It depends primarily on the high bits of hash, which is important because
// getCoeffRow and getResultRow consume the low/middle bits of the same
// rehashed hash — using high bits for start avoids correlation.
//
// [RocksDB: FastRangeGeneric]
func fastRange64(hash uint64, rangeVal uint32) uint32 {
	hi, _ := bits.Mul64(uint64(rangeVal), hash)
	return uint32(hi)
}

// =============================================================================
// HASH RESULT
// =============================================================================

// hashResult holds all values derived from hashing a key for banding.
// This is the output of the full two-phase hash pipeline.
type hashResult struct {
	start    uint32  // start position in [0, numStarts)
	coeffRow uint128 // w-bit coefficient bitmask
	result   uint8   // r-bit fingerprint
}

// =============================================================================
// HASHER INTERFACE
// =============================================================================

// hasher defines the interface for Ribbon filter hash operations.
//
// Per §2 of Dillinger & Walzer (2021), each key must be mapped to a triple
// (start, coefficients, result) for the banding linear system. This mapping
// uses a two-phase pipeline to support efficient seed retries:
//
//	Phase 1 (once per key):       Key → 64-bit hash (stored for reuse)
//	Phase 2 (per seed attempt):   stored hash → rehash → (start, coeffRow, result)
//
// Custom hash strategies can be plugged in by implementing this interface.
//
// [RocksDB: StandardHasher<TypesAndSettings>]
//
// Typical usage:
//
//	h := newStandardHasher(128, numStarts, 7, true)
//
//	// Phase 1: hash keys once
//	hashes := make([]uint64, len(keys))
//	for i, key := range keys {
//	    hashes[i] = h.keyHash(key)
//	}
//
//	// Phase 2: try seeds until banding succeeds
//	for seed := uint32(0); ; seed++ {
//	    h.setOrdinalSeed(seed)
//	    for _, hash := range hashes {
//	        hr := h.derive(hash)
//	        // use hr.start, hr.coeffRow, hr.result for banding
//	    }
//	}
type hasher interface {
	// keyHash computes the initial 64-bit hash of a raw key (Phase 1).
	// Called once per key; the result is stored and reused across seed attempts.
	keyHash(key []byte) uint64

	// keyHashString is like keyHash but accepts a string directly,
	// avoiding the []byte allocation that string→[]byte conversion incurs.
	keyHashString(key string) uint64

	// setOrdinalSeed sets the hash seed from a small ordinal value (0..255).
	// Internally converts to a well-distributed raw seed.
	setOrdinalSeed(ordinal uint32)

	// getOrdinalSeed returns the current ordinal seed.
	getOrdinalSeed() uint32

	// setNumStarts updates the number of valid start positions.
	// numStarts = numSlots - coeffBits + 1
	setNumStarts(numStarts uint32)

	// getNumStarts returns the current number of valid start positions.
	getNumStarts() uint32

	// getCoeffBits returns the ribbon width w (32, 64, or 128).
	getCoeffBits() uint32

	// getResultBits returns the number of fingerprint bits used.
	getResultBits() uint

	// firstCoeffAlwaysOne returns whether bit 0 of the coefficient row is
	// always forced to 1. When true, the banding algorithm can skip the
	// leading-zero scan (faster construction). When false, the coefficient
	// row is still guaranteed non-zero through other means.
	firstCoeffAlwaysOne() bool

	// rehash mixes a pre-computed 64-bit hash with the current raw seed.
	// This is the core of Phase 2 — produces a new hash for deriving
	// start/coeff/result. Called once per key per seed attempt.
	rehash(h uint64) uint64

	// getStart computes the start slot index from a rehashed hash.
	// Returns a value in [0, numStarts).
	getStart(h uint64) uint32

	// getCoeffRow derives the w-bit coefficient row from a rehashed hash.
	// If firstCoeffAlwaysOne is true, bit 0 is forced to 1.
	// The returned row is always non-zero (required for banding).
	getCoeffRow(h uint64) uint128

	// getResultRow derives the r-bit fingerprint from a rehashed hash.
	// The result is masked to the configured number of result bits.
	getResultRow(h uint64) uint8

	// derive rehashes a stored hash with the current seed and returns
	// the complete (start, coeffRow, result) triple for banding.
	derive(h uint64) hashResult
}

// =============================================================================
// STANDARD HASHER — implementation
// =============================================================================

// standardHasher implements the hasher interface.
//
// It realises the two-phase hash pipeline described in §2 of Dillinger &
// Walzer (2021), where each key is hashed once (Phase 1) and the stored hash
// is cheaply remixed per seed attempt (Phase 2) to produce the
// (start, coefficients, result) triple for banding.
//
// Configurable parameters (paper §2 & §4):
//   - coeffBits (w): ribbon width — 32, 64, or 128.
//     Overhead ratio m/n ≈ 1 + 2.3/w; larger w = less space, slower build.
//   - resultBits (r): fingerprint bits stored per key. FPR ≈ 2^(−r).
//   - forceFirstCoeff: when true, bit 0 of every coefficient row is forced
//     to 1, so the Gaussian elimination pivot is always at column "start"
//     — no leading-zero scan needed (faster construction). When false, the
//     row is still guaranteed non-zero through other means.
//
// Hash pipeline:
//
//	Phase 1: key → XXH3_64bits(key, seed=0) → stored uint64 hash
//	Phase 2: (hash ^ rawSeed) * kRehashFactor → rehashed
//	  ├─ start:    fastRange64(rehashed, numStarts)     [high bits]
//	  ├─ coeffRow: multiply → expand/truncate to w bits  [low/middle bits]
//	  └─ result:   multiply → bswap64 → mask to r bits   [byte-swapped]
//
// [RocksDB: StandardHasher<TypesAndSettings>]
type standardHasher struct {
	rawSeed         uint64
	numStarts       uint32
	coeffBits       uint32 // ribbon width w: 32, 64, or 128
	resultBits      uint
	resultMask      uint8 // pre-computed: (1 << resultBits) - 1
	forceFirstCoeff bool  // force LSB=1 in coefficient rows

	// Pre-computed coefficient derivation masks (set once in constructor).
	// These replace the per-call switch on coeffBits in derive(), making
	// the coefficient path branchless and keeping derive() inlineable.
	//
	//   w=32:  coeffLoMask=0x00000000FFFFFFFF  coeffHiMask=0  coeffXor=0
	//   w=64:  coeffLoMask=0xFFFFFFFFFFFFFFFF  coeffHiMask=0  coeffXor=0
	//   w=128: coeffLoMask=0xFFFFFFFFFFFFFFFF  coeffHiMask=^0 coeffXor=kCoeffXor64
	coeffLoMask uint64 // truncation mask for lo half of coefficient row
	coeffHiMask uint64 // 0 for w≤64, ^0 for w=128 (enables hi = a)
	coeffXor    uint64 // 0 for w≤64, kCoeffXor64 for w=128 (XOR expansion)
	coeffOrMask uint64 // 1 if forceFirstCoeff, 0 otherwise (branchless LSB set)
}

// Compile-time check: *standardHasher implements hasher.
var _ hasher = (*standardHasher)(nil)

// newStandardHasher creates a Ribbon filter hasher with the given configuration.
//
// Parameters:
//   - coeffBits: ribbon width w — must be 32, 64, or 128.
//     Paper §4: overhead ratio m/n ≈ 1 + 2.3/w. Larger w saves space but
//     increases construction time per key.
//   - numStarts: number of valid start positions (= numSlots − w + 1).
//     Paper §2: "each key x is assigned a start position s(x) ∈ {0,…,m−w}".
//   - resultBits: fingerprint bits r. FPR ≈ 2^(−r) (paper §3).
//   - firstCoeffAlwaysOne: when true, bit 0 of every coefficient row is
//     forced to 1, so the banding pivot is deterministic at column s(x).
//     Set to false for research/experimentation with natural coefficient
//     distributions.
//
// Panics if coeffBits is not 32, 64, or 128.
func newStandardHasher(coeffBits uint32, numStarts uint32, resultBits uint, firstCoeffAlwaysOne bool) *standardHasher {
	var coeffLoMask, coeffHiMask, coeffXor uint64
	switch coeffBits {
	case 32:
		coeffLoMask = 0x00000000FFFFFFFF
	case 64:
		coeffLoMask = ^uint64(0)
	case 128:
		coeffLoMask = ^uint64(0)
		coeffHiMask = ^uint64(0)
		coeffXor = kCoeffXor64
	default:
		panic("ribbon: coeffBits must be 32, 64, or 128")
	}
	var coeffOrMask uint64
	if firstCoeffAlwaysOne {
		coeffOrMask = 1
	}
	return &standardHasher{
		numStarts:       numStarts,
		coeffBits:       coeffBits,
		resultBits:      resultBits,
		resultMask:      uint8((1 << resultBits) - 1),
		forceFirstCoeff: firstCoeffAlwaysOne,
		coeffLoMask:     coeffLoMask,
		coeffHiMask:     coeffHiMask,
		coeffXor:        coeffXor,
		coeffOrMask:     coeffOrMask,
	}
}

// --- Phase 1: Key → 64-bit hash ---

// keyHash computes the initial 64-bit hash of a key using XXH3 (Phase 1).
//
// This is the only step that touches the raw key bytes. The 64-bit output
// is stored and reused across all seed attempts, amortising the cost of
// hashing over retries (paper §2).
//
// NOTE: we use the final XXH3 spec (via zeebo/xxh3). This is NOT byte-
// compatible with RocksDB's XXPH3 (preview v0.7.2), which is fine for a
// standalone implementation — compatibility would only matter for reading
// RocksDB's on-disk filter blocks.
func (sh *standardHasher) keyHash(key []byte) uint64 {
	return xxh3.Hash(key)
}

// keyHashString hashes a string key directly using XXH3 without
// allocating a []byte copy. Uses xxh3.HashString which reads the
// string's underlying bytes via unsafe.Pointer.
func (sh *standardHasher) keyHashString(key string) uint64 {
	return xxh3.HashString(key)
}

// --- Seed management ---

// setOrdinalSeed sets the seed from a small ordinal (typically 0..255).
// Converts the ordinal to a well-distributed raw seed via bijective mixing.
// The filter serializes only this small ordinal, not the full 64-bit raw seed.
func (sh *standardHasher) setOrdinalSeed(ordinal uint32) {
	sh.rawSeed = ordinalSeedToRaw(ordinal)
}

// getOrdinalSeed returns the current ordinal seed by inverting the raw seed.
func (sh *standardHasher) getOrdinalSeed() uint32 {
	return rawSeedToOrdinal(sh.rawSeed)
}

// --- Configuration accessors ---

// setNumStarts updates the number of valid start positions.
// numStarts = numSlots - coeffBits + 1.
// Call this when the filter size changes.
func (sh *standardHasher) setNumStarts(numStarts uint32) {
	sh.numStarts = numStarts
}

// getNumStarts returns the current number of valid start positions.
func (sh *standardHasher) getNumStarts() uint32 {
	return sh.numStarts
}

// getCoeffBits returns the ribbon width w.
func (sh *standardHasher) getCoeffBits() uint32 {
	return sh.coeffBits
}

// getResultBits returns the number of fingerprint bits.
func (sh *standardHasher) getResultBits() uint {
	return sh.resultBits
}

// firstCoeffAlwaysOne returns whether bit 0 of coefficient rows is forced to 1.
func (sh *standardHasher) firstCoeffAlwaysOne() bool {
	return sh.forceFirstCoeff
}

// --- Phase 2: Rehash + derive ---

// rehash mixes a stored hash with the current raw seed (Phase 2 entry point).
//
//	rehashed = (storedHash ^ rawSeed) * kRehashFactor
//
// The Ribbon construction retries banding with a new seed on failure
// (paper §2). rehash is the cheapest possible remixing: XOR injects the
// seed, and the multiply provides good high-bit diffusion for fastRange64.
// The raw seed is already pre-mixed (via ordinalSeedToRaw), so sequential
// ordinals produce well-separated rehash outputs.
func (sh *standardHasher) rehash(h uint64) uint64 {
	return (h ^ sh.rawSeed) * kRehashFactor
}

// getStart computes the start position s(x) from a rehashed hash.
//
// Paper §2: "each key x is assigned a start position s(x) ∈ {0,…,m−w}"
// where m−w+1 = numStarts. We use fastRange64 (high-bit-dependent) so that
// start is statistically independent from the coefficient and result rows
// derived from the same hash's low/middle bits.
func (sh *standardHasher) getStart(h uint64) uint32 {
	return fastRange64(h, sh.numStarts)
}

// getCoeffRow derives the w-bit coefficient row c(x) from a rehashed hash.
//
// Paper §2: "each key x is assigned … a coefficient row c(x) ∈ {0,1}^w".
// The coefficient row must be (a) non-zero (otherwise the linear system row
// is trivial and cannot encode information) and (b) independent of start.
//
// We multiply by kCoeffAndResultFactor (a different prime than kRehashFactor)
// to redistribute bit dependencies away from the high bits that getStart
// already consumed. The derivation then depends on w:
//
//	w=32:  truncate the 64-bit product to 32 bits.
//	w=64:  use the full 64-bit product.
//	w=128: XOR-fold the 64-bit product into 128 bits:
//	       {hi: a, lo: a ^ kCoeffXor64}. This guarantees non-zero output
//	       (kCoeffXor64 ≠ 0 ⇒ if a==0 then lo≠0, if a≠0 then hi≠0) and
//	       avoids identical upper/lower halves.
//
// When forceFirstCoeff is true, bit 0 is unconditionally set to 1. This
// makes the Gaussian elimination pivot deterministic at column s(x),
// removing the need for a leading-zero scan during banding (paper §4).
//
// Non-zero guarantee: when forceFirstCoeff is false, the branchless path
// could theoretically produce a zero row (P ≈ 2^-w for w≤64; impossible
// for w=128 due to XOR expansion). A cold zero-guard catches this.
//
// Implementation: uses the same pre-computed masks as derive() — the
// per-width switch is replaced by branchless mask/XOR/OR operations,
// and the forceFirstCoeff branch is replaced by coeffOrMask.
func (sh *standardHasher) getCoeffRow(h uint64) uint128 {
	a := h * kCoeffAndResultFactor
	cr := uint128{
		hi: a & sh.coeffHiMask,
		lo: (a & sh.coeffLoMask) ^ sh.coeffXor | sh.coeffOrMask,
	}
	// Zero guard for standalone callers: if forceFirstCoeff is false and
	// the branchless path produces an all-zero row, force it to 1.
	// When forceFirstCoeff is true, coeffOrMask=1 ensures cr.lo≥1.
	// When w=128, XOR expansion ensures at least one half is non-zero.
	// Only reachable for w≤64 with !forceFirstCoeff at P≈2^(-w).
	if cr.lo == 0 && cr.hi == 0 {
		cr.lo = 1
	}
	return cr
}

// getResultRow derives the r-bit fingerprint from a rehashed hash.
//
// Paper §3: when used as a filter, each key's result row is a hash-derived
// r-bit fingerprint. The FPR for non-member keys is ≈ 2^(−r).
//
// Derivation:
//  1. a = h * kCoeffAndResultFactor  (same multiply as getCoeffRow)
//  2. Byte-swap a — moves the highest-quality bits (affected most by
//     the multiply's carry chain) into the lowest byte position, from
//     which we extract r bits.
//  3. Mask to resultBits.
func (sh *standardHasher) getResultRow(h uint64) uint8 {
	a := h * kCoeffAndResultFactor
	swapped := bits.ReverseBytes64(a)
	return uint8(swapped) & sh.resultMask
}

// derive rehashes a stored hash with the current seed and returns the
// complete (start, coeffRow, result) triple needed by the banding algorithm.
//
// Paper §2: for each key, the banding step inserts a row into the m × w
// coefficient matrix at position s(x), with coefficient bits c(x) and
// target result r(x). This function computes all three from one stored hash.
//
// Performance: this method manually inlines rehash, getStart, getCoeffRow,
// and getResultRow for two reasons:
//
//	(a) The multiply h*kCoeffAndResultFactor is shared between getCoeffRow
//	    and getResultRow — inlining lets us compute it once instead of twice.
//	(b) The per-width switch in getCoeffRow is replaced by pre-computed
//	    masks (coeffLoMask, coeffHiMask, coeffXor), and the forceFirstCoeff
//	    branch is replaced by a pre-computed OR mask (coeffOrMask), making
//	    the entire coefficient path branchless. This brings the compiler's
//	    inlining cost under budget 80, so derive() itself is inlined into
//	    the caller's hot loop (banding), removing one more call/return.
//
// Note: the zero-guard for !forceFirstCoeff is omitted here because:
//   - P(zero coeff) ≈ 2^-64 for w=64, 2^-32 for w=32, 0 for w=128.
//   - If it occurs, the banding algorithm treats the degenerate row as a
//     failure and retries with a new seed — no correctness issue.
//   - The individual getCoeffRow() method retains the guard for callers
//     who use it outside the banding loop.
func (sh *standardHasher) derive(h uint64) hashResult {
	// --- inline rehash ---
	rh := (h ^ sh.rawSeed) * kRehashFactor

	// --- shared multiply (coeff + result) ---
	a := rh * kCoeffAndResultFactor

	// --- inline getStart (fastRange64) ---
	startHi, _ := bits.Mul64(uint64(sh.numStarts), rh)

	// --- inline getCoeffRow (fully branchless) ---
	var cr uint128
	cr.hi = a & sh.coeffHiMask
	cr.lo = (a & sh.coeffLoMask) ^ sh.coeffXor | sh.coeffOrMask

	// --- inline getResultRow (reuses `a` — no second multiply) ---
	return hashResult{
		start:    uint32(startHi),
		coeffRow: cr,
		result:   uint8(bits.ReverseBytes64(a)) & sh.resultMask,
	}
}
