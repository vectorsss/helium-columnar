//! MSB-first bit reader / writer used by the Gorilla XOR coder.

pub(crate) struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32, // position within the last byte, 0..=7
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            bit_pos: 0,
        }
    }

    pub fn write_bit(&mut self, bit: bool) {
        if self.bit_pos == 0 {
            self.buf.push(0);
        }
        if bit {
            // SAFETY: we just pushed a byte above when bit_pos == 0, so
            // buf is non-empty here.
            if let Some(last) = self.buf.last_mut() {
                *last |= 1 << (7 - self.bit_pos);
            }
        }
        self.bit_pos = (self.bit_pos + 1) % 8;
    }

    /// Write the lowest `count` bits of `value`, MSB-first.
    pub fn write_bits(&mut self, value: u64, count: u32) {
        debug_assert!(count <= 64);
        for b in (0..count).rev() {
            self.write_bit((value >> b) & 1 != 0);
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    pub fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.bytes.len() {
            return None;
        }
        let bit = (self.bytes[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit != 0)
    }

    pub fn read_bits(&mut self, count: u32) -> Option<u64> {
        debug_assert!(count <= 64);
        let mut v: u64 = 0;
        for _ in 0..count {
            v = (v << 1) | (self.read_bit()? as u64);
        }
        Some(v)
    }
}
