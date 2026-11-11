
/* Keccak-256 grind: find nonce with keccak256(seed[32] || nonce_be256) <= target */
__constant ulong RC[24] = {
  0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,
  0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
  0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
  0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL, 0x8000000000008003UL,
  0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,
  0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant int ROT[25] = {
   0,  1, 62, 28, 27,
  36, 44,  6, 55, 20,
   3, 10, 43, 25, 39,
  41, 45, 15, 21,  8,
  18,  2, 61, 56, 14
};

inline ulong rotl64(ulong x, int n) {
  return n == 0 ? x : ((x << n) | (x >> (64 - n)));
}

inline ulong bswap64(ulong x) {
  return ((x & 0xFFUL) << 56) |
         (((x >> 8) & 0xFFUL) << 48) |
         (((x >> 16) & 0xFFUL) << 40) |
         (((x >> 24) & 0xFFUL) << 32) |
         (((x >> 32) & 0xFFUL) << 24) |
         (((x >> 40) & 0xFFUL) << 16) |
         (((x >> 48) & 0xFFUL) << 8) |
         ((x >> 56) & 0xFFUL);
}

inline void keccakf(ulong *s) {
  for (int r = 0; r < 24; r++) {
    ulong C[5], D[5], B[25];
    for (int x = 0; x < 5; x++) C[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
    for (int x = 0; x < 5; x++) D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
    for (int i = 0; i < 25; i++) s[i] ^= D[i % 5];
    for (int x = 0; x < 5; x++)
      for (int y = 0; y < 5; y++)
        B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(s[x + 5 * y], ROT[x + 5 * y]);
    for (int x = 0; x < 5; x++)
      for (int y = 0; y < 5; y++)
        s[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
    s[0] ^= RC[r];
  }
}

inline ulong ld64le_priv(__private const uchar *p) {
  ulong v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | (ulong)p[i];
  return v;
}

__kernel void grind(
    __global const uchar *prefix, /* 64 bytes: agentId || seed */
    ulong base,
    ulong count,
    __global const ulong *target,
    __global volatile uint *foundCount,
    __global ulong *foundNonce,
    uint maxFound
) {
  ulong gid = (ulong)get_global_id(0);
  if (gid >= count) return;
  ulong nonce = base + gid;

  uchar msg[136];
  #pragma unroll
  for (int i = 0; i < 136; i++) msg[i] = 0;
  #pragma unroll
  for (int i = 0; i < 64; i++) msg[i] = prefix[i];
  #pragma unroll
  for (int i = 0; i < 8; i++) msg[95 - i] = (uchar)((nonce >> (8 * i)) & 0xFF);
  msg[96] = 0x01;
  msg[135] = 0x80;

  ulong s[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) s[i] = 0;
  #pragma unroll
  for (int i = 0; i < 17; i++) s[i] = ld64le_priv(msg + 8 * i);

  keccakf(s);

  ulong w0 = bswap64(s[0]);
  ulong w1 = bswap64(s[1]);
  ulong w2 = bswap64(s[2]);
  ulong w3 = bswap64(s[3]);

  int cmp = 0;
  if (w0 != target[0]) cmp = (w0 < target[0]) ? -1 : 1;
  else if (w1 != target[1]) cmp = (w1 < target[1]) ? -1 : 1;
  else if (w2 != target[2]) cmp = (w2 < target[2]) ? -1 : 1;
  else if (w3 != target[3]) cmp = (w3 < target[3]) ? -1 : 1;

  if (cmp <= 0) {
    uint idx = atomic_inc(foundCount);
    if (idx < maxFound) foundNonce[idx] = nonce;
  }
}


__kernel void hashwords(__global const uchar *prefix, ulong nonce, __global ulong *out) {
  uchar msg[136];
  for (int i = 0; i < 136; i++) msg[i] = 0;
  for (int i = 0; i < 64; i++) msg[i] = prefix[i];
  for (int i = 0; i < 8; i++) msg[95 - i] = (uchar)((nonce >> (8 * i)) & 0xFF);
  msg[96] = 0x01;
  msg[135] = 0x80;
  ulong s[25];
  for (int i = 0; i < 25; i++) s[i] = 0;
  for (int i = 0; i < 17; i++) s[i] = ld64le_priv(msg + 8 * i);
  keccakf(s);
  out[0] = bswap64(s[0]); out[1] = bswap64(s[1]); out[2] = bswap64(s[2]); out[3] = bswap64(s[3]);
}
