package ribbon

import (
	"fmt"
	"testing"
)

func TestUint128_BasicOps(t *testing.T) {
	a := uint128{hi: 0xFF, lo: 0x01}
	b := uint128{hi: 0x0F, lo: 0x10}

	// XOR
	x := a.xor(b)
	if x.hi != 0xF0 || x.lo != 0x11 {
		t.Errorf("XOR wrong: %+v", x)
	}

	// AND
	n := a.and(b)
	if n.hi != 0x0F || n.lo != 0x00 {
		t.Errorf("AND wrong: %+v", n)
	}

	// OR
	o := a.or(b)
	if o.hi != 0xFF || o.lo != 0x11 {
		t.Errorf("OR wrong: %+v", o)
	}
}

func TestUint128_Shifts(t *testing.T) {
	// Test left shift across 64-bit boundary
	a := uint128{hi: 0, lo: 1}
	shifted := a.lsh(64)
	if shifted.hi != 1 || shifted.lo != 0 {
		t.Errorf("lsh(64) wrong: %+v", shifted)
	}

	// Test right shift across 64-bit boundary
	b := uint128{hi: 1, lo: 0}
	shifted = b.rsh(64)
	if shifted.hi != 0 || shifted.lo != 1 {
		t.Errorf("rsh(64) wrong: %+v", shifted)
	}

	// Small shifts
	c := uint128{hi: 0, lo: 0x8000000000000000}
	shifted = c.lsh(1)
	if shifted.hi != 1 || shifted.lo != 0 {
		t.Errorf("lsh(1) carry wrong: %+v", shifted)
	}
}

func TestUint128_BitParity(t *testing.T) {
	// Even number of bits → parity 0
	a := uint128{hi: 0, lo: 0x3} // bits: 11 → 2 bits → even
	if a.bitParity() != 0 {
		t.Errorf("expected parity 0 for 0x3, got %d", a.bitParity())
	}

	// Odd number of bits → parity 1
	b := uint128{hi: 0, lo: 0x7} // bits: 111 → 3 bits → odd
	if b.bitParity() != 1 {
		t.Errorf("expected parity 1 for 0x7, got %d", b.bitParity())
	}
}

// ---------------------------------------------------------------------------
// uint128 serialization: putBytes / uint128FromBytes
// ---------------------------------------------------------------------------

func TestUint128_PutBytesFromBytes_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		val  uint128
	}{
		{"zero", uint128{hi: 0, lo: 0}},
		{"lo_only", uint128{hi: 0, lo: 0xDEADBEEFCAFEBABE}},
		{"hi_only", uint128{hi: 0x0123456789ABCDEF, lo: 0}},
		{"both", uint128{hi: 0x0123456789ABCDEF, lo: 0xDEADBEEFCAFEBABE}},
		{"max", uint128{hi: 0xFFFFFFFFFFFFFFFF, lo: 0xFFFFFFFFFFFFFFFF}},
		{"one", uint128{hi: 0, lo: 1}},
		{"hi_msb", uint128{hi: 0x8000000000000000, lo: 0}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf := tc.val.putBytes()
			got := uint128FromBytes(buf[:])
			if got != tc.val {
				t.Errorf("round-trip failed: put %+v, got back %+v", tc.val, got)
			}
		})
	}
}

func TestUint128_PutBytes_KnownPattern(t *testing.T) {
	// Verify exact byte layout: lo in bytes [0..7], hi in bytes [8..15],
	// each in little-endian.
	v := uint128{hi: 0x0807060504030201, lo: 0x100F0E0D0C0B0A09}
	buf := v.putBytes()

	// lo = 0x100F0E0D0C0B0A09 little-endian → 09 0A 0B 0C 0D 0E 0F 10
	expectedLo := []byte{0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10}
	// hi = 0x0807060504030201 little-endian → 01 02 03 04 05 06 07 08
	expectedHi := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}

	for i := 0; i < 8; i++ {
		if buf[i] != expectedLo[i] {
			t.Errorf("byte[%d] = 0x%02X, want 0x%02X (lo region)", i, buf[i], expectedLo[i])
		}
	}
	for i := 0; i < 8; i++ {
		if buf[8+i] != expectedHi[i] {
			t.Errorf("byte[%d] = 0x%02X, want 0x%02X (hi region)", 8+i, buf[8+i], expectedHi[i])
		}
	}
}

func TestUint128_FromBytes_KnownPattern(t *testing.T) {
	// The 16 bytes 0x01..0x10 should decode as:
	// lo = LittleEndian(01 02 03 04 05 06 07 08) = 0x0807060504030201
	// hi = LittleEndian(09 0A 0B 0C 0D 0E 0F 10) = 0x100F0E0D0C0B0A09
	buf := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
		0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10}
	got := uint128FromBytes(buf)
	wantLo := uint64(0x0807060504030201)
	wantHi := uint64(0x100F0E0D0C0B0A09)
	if got.lo != wantLo || got.hi != wantHi {
		t.Errorf("got {hi: 0x%016X, lo: 0x%016X}, want {hi: 0x%016X, lo: 0x%016X}",
			got.hi, got.lo, wantHi, wantLo)
	}
}

func TestUint128_PutBytes_ReturnsSizeExactly16(t *testing.T) {
	v := uint128{hi: 0xAAAAAAAAAAAAAAAA, lo: 0xBBBBBBBBBBBBBBBB}
	buf := v.putBytes()
	if len(buf) != 16 {
		t.Fatalf("expected 16 bytes, got %d", len(buf))
	}
	got := uint128FromBytes(buf[:])
	if got != v {
		t.Errorf("got %+v, want %+v", got, v)
	}
}

func TestUint128_FromBytes_PanicOnShortSlice(t *testing.T) {
	shortLens := []int{0, 1, 8, 15}

	for _, length := range shortLens {
		t.Run(fmt.Sprintf("len=%d", length), func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("uint128FromBytes did not panic for len=%d", length)
				}
				msg, ok := r.(string)
				if !ok || msg != "uint128FromBytes: source slice must be at least 16 bytes" {
					t.Fatalf("unexpected panic: %v", r)
				}
			}()
			buf := make([]byte, length)
			uint128FromBytes(buf)
		})
	}
}

// ---------------------------------------------------------------------------
// trailingZeros — find the lowest set bit position
// ---------------------------------------------------------------------------

func TestUint128_TrailingZeros(t *testing.T) {
	cases := []struct {
		name string
		val  uint128
		want uint
	}{
		// lo-only values (common case for w≤64)
		{"lo_bit0", uint128{lo: 1}, 0},
		{"lo_bit1", uint128{lo: 2}, 1},
		{"lo_bit63", uint128{lo: 1 << 63}, 63},
		{"lo_mixed", uint128{lo: 0x80}, 7},
		{"lo_all_ones", uint128{lo: ^uint64(0)}, 0},

		// hi-only values (lo=0, w=128 scenarios)
		{"hi_bit0", uint128{hi: 1}, 64},
		{"hi_bit1", uint128{hi: 2}, 65},
		{"hi_bit63", uint128{hi: 1 << 63}, 127},
		{"hi_mixed", uint128{hi: 0x100}, 72},

		// both halves set — lo takes precedence
		{"both_lo_wins", uint128{hi: 1, lo: 4}, 2},
		{"both_lo_bit0", uint128{hi: ^uint64(0), lo: 1}, 0},

		// zero value (edge case — should return 128)
		{"zero", uint128{}, 128},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.val.trailingZeros()
			if got != tc.want {
				t.Errorf("trailingZeros({hi: 0x%016x, lo: 0x%016x}) = %d, want %d",
					tc.val.hi, tc.val.lo, got, tc.want)
			}
		})
	}
}
