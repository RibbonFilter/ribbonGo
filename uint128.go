package ribbonGo

import (
	"encoding/binary"
	"math/bits"
)

// =============================================================================
// uint128 — 128-bit unsigned integer for coefficient rows
// =============================================================================

// uint128 represents a 128-bit unsigned integer.
// Used as the coefficient row type when ribbon width w=128.
// Paper §4: "r=128 seems to be closest to a generally good choice for Ribbon"
// because it scales to ~10M keys with only ~5% space overhead.
// hi holds the upper 64 bits, lo holds the lower 64 bits.
type uint128 struct {
	hi uint64
	lo uint64
}

// isZero returns true if the 128-bit value is zero.
func (u uint128) isZero() bool {
	return u.hi == 0 && u.lo == 0
}

// xor returns u XOR v.
func (u uint128) xor(v uint128) uint128 {
	return uint128{hi: u.hi ^ v.hi, lo: u.lo ^ v.lo}
}

// and returns u AND v.
func (u uint128) and(v uint128) uint128 {
	return uint128{hi: u.hi & v.hi, lo: u.lo & v.lo}
}

// or returns u OR v.
func (u uint128) or(v uint128) uint128 {
	return uint128{hi: u.hi | v.hi, lo: u.lo | v.lo}
}

// lsh returns u << n (n must be 0..127).
func (u uint128) lsh(n uint) uint128 {
	if n >= 128 {
		return uint128{}
	}
	if n >= 64 {
		return uint128{hi: u.lo << (n - 64), lo: 0}
	}
	if n == 0 {
		return u
	}
	return uint128{
		hi: (u.hi << n) | (u.lo >> (64 - n)),
		lo: u.lo << n,
	}
}

// rsh returns u >> n (n must be 0..127).
func (u uint128) rsh(n uint) uint128 {
	if n >= 128 {
		return uint128{}
	}
	if n >= 64 {
		return uint128{hi: 0, lo: u.hi >> (n - 64)}
	}
	if n == 0 {
		return u
	}
	return uint128{
		hi: u.hi >> n,
		lo: (u.lo >> n) | (u.hi << (64 - n)),
	}
}

// bitParity returns 1 if the number of set bits is odd, 0 if even.
// Used during query to XOR selected solution rows.
func (u uint128) bitParity() byte {
	x := u.hi ^ u.lo
	return byte(bits.OnesCount64(x) & 1)
}

// bit returns the value of the i-th bit (0 = LSB of lo, 127 = MSB of hi).
func (u uint128) bit(i uint) uint {
	if i >= 128 {
		return 0
	}
	if i >= 64 {
		return uint((u.hi >> (i - 64)) & 1)
	}
	return uint((u.lo >> i) & 1)
}

// trailingZeros returns the number of trailing zero bits in the 128-bit value.
// If u is zero, returns 128 (all bits are trailing zeros).
//
// Used by the banding algorithm to find the pivot offset — the lowest set bit
// in a coefficient row determines which column is the elimination pivot.
//
// Performance: compiles to a single TZCNT/BSF instruction on the lo half
// in the common case (w≤64, or w=128 when lo≠0). The hi half is only
// touched when lo is entirely zero.
func (u uint128) trailingZeros() uint {
	if u.lo != 0 {
		return uint(bits.TrailingZeros64(u.lo))
	}
	return 64 + uint(bits.TrailingZeros64(u.hi))
}

// putBytes serialises the uint128 into a 16-byte array in little-endian order.
// lo occupies bytes [0..7], hi occupies bytes [8..15].
func (u uint128) putBytes() [16]byte {
	var b [16]byte
	binary.LittleEndian.PutUint64(b[0:8], u.lo)
	binary.LittleEndian.PutUint64(b[8:16], u.hi)
	return b
}

// uint128FromBytes reads a uint128 from a 16-byte slice in little-endian order.
// Panics if len(b) < 16.
func uint128FromBytes(b []byte) uint128 {
	if len(b) < 16 {
		panic("uint128FromBytes: source slice must be at least 16 bytes")
	}
	return uint128{
		lo: binary.LittleEndian.Uint64(b[0:8]),
		hi: binary.LittleEndian.Uint64(b[8:16]),
	}
}
