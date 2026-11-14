// metal/src/qwen36/msl/q36_rope.metal - the two steps between a layer's q/k/v projections and its attention:
//
//     y = x / sqrt(mean(x^2) + eps) * w        (per HEAD: attn_q_norm / attn_k_norm are head-wide, 256 values)
//     y = rope(y)                             (NEOX pairs (i, i + 32) over the first 64 dims)
//
// qsa_prep does both for Flash-Next, with the indexer's rows and the int8 KV cache wrapped around them; qwen35moe has
// neither, and its KV is bf16, so the port needed the plain form.  The RoPE table is the same one qsa.metal reads - the
// host's float2 [pos][32] of (cos, sin) built in double (setup_host.cpp build_rope_table, NEOX, base from the file's
// rope.freq_base) - so a q36 position rotates exactly as a Flash-Next one does.
//
// One threadgroup a head row: QD = 256 threads, one component each, the row's mean of squares over the threadgroup
// (sum32 per 32-lane group, then the groups' partials).  The row is staged in threadgroup memory because RoPE pairs a
// component with the one 32 lanes away, which is not in this thread's registers.
//
//   raw   fp32 [head][256]   the projection's row for this token
//   w     fp32 [256]         the head's norm weight
//   cs    float2 [pos][32]   the RoPE table
//   out   fp32 head rows (queries) or bf16 head rows (the KV ring): two instantiations of the same kernel
kernel void q36_head_norm_rope_f32(device const float* raw [[buffer(0)]], device const float* w [[buffer(1)]],
                                   device const float2* cs [[buffer(2)]], device float* out [[buffer(3)]],
                                   constant uint& pos [[buffer(4)]], constant float& eps [[buffer(5)]],
                                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                   ushort sg_ [[simdgroup_index_in_threadgroup]], ushort lane_ [[thread_index_in_simdgroup]]) {
    constexpr uint QD = 256, HALF = 32, QROT = 2 * HALF;
    const uint h = tg.x;
    // The projection's row is per head [q(QD) | gate(QD)], as the CUDA engine lays it out
    // (q36_engine.cu: qg[(t*H+head)*2*HD] for q, +HD for the gate), so a head's q starts at h*2*QD - NOT h*QD, which read the
    // first eight heads' q and the last eight heads' GATE as if they were queries.
    device const float* r = raw + (size_t) h * 2 * QD;
    const float x = r[tid];
    const float ss = sum32(x * x);
    threadgroup float part[8];
    const ushort sg = sg32(tid, sg_);
    const ushort lane = lane32(tid, lane_);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8u; ++g) tot += part[g];
    const float rs = rsqrt(tot / (float) QD + eps);
    threadgroup float row[QD];
    row[tid] = x * rs * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = row[tid];
    if (tid < QROT) {   // NEOX: lane i pairs with i + 32, both halves of the rotation written from the staged row
        const uint i = uint(tid) & (HALF - 1u);
        const float2 c = cs[(size_t) pos * HALF + i];
        const float a = row[i], b = row[i + HALF];
        o = tid < HALF ? a * c.x - b * c.y : a * c.y + b * c.x;
    }
    out[(size_t) h * QD + tid] = o;
}

kernel void q36_head_norm_rope_bf16(device const float* raw [[buffer(0)]], device const float* w [[buffer(1)]],
                                    device const float2* cs [[buffer(2)]], device uint16_t* out [[buffer(3)]],
                                    constant uint& pos [[buffer(4)]], constant float& eps [[buffer(5)]],
                                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                    ushort sg_ [[simdgroup_index_in_threadgroup]], ushort lane_ [[thread_index_in_simdgroup]]) {
    constexpr uint QD = 256, HALF = 32, QROT = 2 * HALF;
    const uint h = tg.x;
    // The projection's row is per head [q(QD) | gate(QD)], as the CUDA engine lays it out
    // (q36_engine.cu: qg[(t*H+head)*2*HD] for q, +HD for the gate), so a head's q starts at h*2*QD - NOT h*QD, which read the
    // first eight heads' q and the last eight heads' GATE as if they were queries.
    device const float* r = raw + (size_t) h * 2 * QD;
    const float x = r[tid];
    const float ss = sum32(x * x);
    threadgroup float part[8];
    const ushort sg = sg32(tid, sg_);
    const ushort lane = lane32(tid, lane_);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8u; ++g) tot += part[g];
    const float rs = rsqrt(tot / (float) QD + eps);
    threadgroup float row[QD];
    row[tid] = x * rs * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = row[tid];
    if (tid < QROT) {
        const uint i = uint(tid) & (HALF - 1u);
        const float2 c = cs[(size_t) pos * HALF + i];
        const float a = row[i], b = row[i + HALF];
        o = tid < HALF ? a * c.x - b * c.y : a * c.y + b * c.x;
    }
    // bf16 by round-to-nearest-even on the top half, as the port's other bf16 writers do (no bfloat type on macOS 11)
    uint u = as_type<uint>(o);
    u += 0x7fffu + ((u >> 16) & 1u);
    out[(size_t) h * QD + tid] = uint16_t(u >> 16);
}
