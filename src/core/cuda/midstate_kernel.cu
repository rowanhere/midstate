typedef unsigned int u32;
typedef unsigned long long u64;

struct Params {
    u32 midstate[8];
    u32 target[8];
    u32 pool[8];
    u32 base_lo;
    u32 base_hi;
    u32 n_nonces;
    u32 iters;
    u32 has_pool;
    u32 pad0;
    u32 pad1;
    u32 pad2;
};

struct Winners {
    u32 count;
    u32 cap;
    u32 pad0;
    u32 pad1;
    u32 nonce_lo[256];
    u32 nonce_hi[256];
    u32 kind[256];
};

__constant__ int MSG[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8 },
    { 3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1 },
    {10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6 },
    {12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4 },
    { 9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7 },
    {11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13 }
};

__device__ __forceinline__ u32 rotr32(u32 x, u32 n) {
    return (x >> n) | (x << (32u - n));
}

__device__ __forceinline__ u32 bswap32(u32 x) {
    return ((x & 0x000000ffu) << 24) |
           ((x & 0x0000ff00u) << 8)  |
           ((x & 0x00ff0000u) >> 8)  |
           ((x & 0xff000000u) >> 24);
}

__device__ __forceinline__ void g(u32 v[16], int a, int b, int c, int d, u32 x, u32 y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr32(v[d] ^ v[a], 16u);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12u);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr32(v[d] ^ v[a], 8u);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7u);
}

__device__ __forceinline__ void compress(const u32 m[16], u32 block_len, u32 out[8]) {
    u32 v[16] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0u, 0u, block_len, 11u
    };

    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        g(v, 0, 4,  8, 12, m[MSG[r][ 0]], m[MSG[r][ 1]]);
        g(v, 1, 5,  9, 13, m[MSG[r][ 2]], m[MSG[r][ 3]]);
        g(v, 2, 6, 10, 14, m[MSG[r][ 4]], m[MSG[r][ 5]]);
        g(v, 3, 7, 11, 15, m[MSG[r][ 6]], m[MSG[r][ 7]]);
        g(v, 0, 5, 10, 15, m[MSG[r][ 8]], m[MSG[r][ 9]]);
        g(v, 1, 6, 11, 12, m[MSG[r][10]], m[MSG[r][11]]);
        g(v, 2, 7,  8, 13, m[MSG[r][12]], m[MSG[r][13]]);
        g(v, 3, 4,  9, 14, m[MSG[r][14]], m[MSG[r][15]]);
    }

    out[0] = v[0] ^ v[8];
    out[1] = v[1] ^ v[9];
    out[2] = v[2] ^ v[10];
    out[3] = v[3] ^ v[11];
    out[4] = v[4] ^ v[12];
    out[5] = v[5] ^ v[13];
    out[6] = v[6] ^ v[14];
    out[7] = v[7] ^ v[15];
}

__device__ __forceinline__ u64 nonce_for(const Params* p, u32 gid) {
    u32 lo = p->base_lo + gid;
    u32 carry = lo < p->base_lo ? 1u : 0u;
    u32 hi = p->base_hi + carry;
    return ((u64)hi << 32) | (u64)lo;
}

__device__ __forceinline__ void first_compress(const Params* p, u32 gid, u32 h[8]) {
    u32 m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = p->midstate[i];
    u64 nonce = nonce_for(p, gid);
    m[8] = (u32)nonce;
    m[9] = (u32)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 16; ++i) m[i] = 0u;
    compress(m, 40u, h);
}

__device__ __forceinline__ void iterate_hash(u32 h[8]) {
    u32 m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = h[i];
    #pragma unroll
    for (int i = 8; i < 16; ++i) m[i] = 0u;
    compress(m, 32u, h);
}

__device__ __forceinline__ bool lt8(const u32 h[8], const u32 target[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        u32 k = bswap32(h[i]);
        if (k < target[i]) return true;
        if (k > target[i]) return false;
    }
    return false;
}

extern "C" __global__ void k_init(const Params* p, u32* state) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p->n_nonces) return;
    u32 h[8];
    first_compress(p, gid, h);
    u32 off = gid * 8u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[off + i] = h[i];
}

extern "C" __global__ void k_step(const Params* p, u32* state) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p->n_nonces) return;
    u32 h[8];
    u32 off = gid * 8u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = state[off + i];
    for (u32 i = 0; i < p->iters; ++i) {
        iterate_hash(h);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[off + i] = h[i];
}

extern "C" __global__ void k_test(const Params* p, u32* state, Winners* out) {
    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p->n_nonces) return;
    u32 h[8];
    u32 off = gid * 8u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = state[off + i];

    u32 kind = 0xffffffffu;
    if (lt8(h, p->target)) {
        kind = 0u;
    } else if (p->has_pool != 0u && lt8(h, p->pool)) {
        kind = 1u;
    }

    if (kind != 0xffffffffu) {
        u32 idx = atomicAdd(&out->count, 1u);
        if (idx < out->cap) {
            u64 nonce = nonce_for(p, gid);
            out->nonce_lo[idx] = (u32)nonce;
            out->nonce_hi[idx] = (u32)(nonce >> 32);
            out->kind[idx] = kind;
        }
    }
}
