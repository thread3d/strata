with open("metal/kernels/moe.metal", "r") as f:
    content = f.read()

old = """template <> struct XD<T_Q4_K> {
    // Load RAW activations for down projection. Group scales applied in unit_combine via sc.
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float v0 = h[base + j];
            const float v1 = h[base + 16 + j];
            sum0 += v0; sum1 += v1;
            xs[j] = v0;
            xs[16 + j] = v1;
        }
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

new = """template <> struct XD<T_Q4_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        xd16(h + base, xs, sum0, 4);
        xd16(h + base + 16, xs + 16, sum1, 4);
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

if old in content:
    content = content.replace(old, new)
    print("Replaced Q4_K")
else:
    print("Q4_K not found!")

# Q5_K
old5 = """template <> struct XD<T_Q5_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float v0 = h[base + j];
            const float v1 = h[base + 16 + j];
            sum0 += v0; sum1 += v1;
            xs[j] = v0;
            xs[16 + j] = v1;
        }
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

new5 = """template <> struct XD<T_Q5_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        xd16(h + base, xs, sum0, 4);
        xd16(h + base + 16, xs + 16, sum1, 4);
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

if old5 in content:
    content = content.replace(old5, new5)
    print("Replaced Q5_K")
else:
    print("Q5_K not found!")

# Q6_K
old6 = """template <> struct XD<T_Q6_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float v0 = h[base + j];
            const float v1 = h[base + 16 + j];
            sum0 += v0; sum1 += v1;
            xs[j] = v0;
            xs[16 + j] = v1;
        }
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

new6 = """template <> struct XD<T_Q6_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        xd16(h + base, xs, sum0, 2);
        xd16(h + base + 16, xs + 16, sum1, 2);
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

if old6 in content:
    content = content.replace(old6, new6)
    print("Replaced Q6_K")
else:
    print("Q6_K not found!")

# Q3_K
old3 = """template <> struct XD<T_Q3_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        // Q3_K: 32 contiguous values per unit, sums per 16. Block size 110 bytes.
        // Scales at offset 32*u? Actually Q3_K has scales per 32 values.
        const int base = u * 32;
        // Q3_K: each block 110 bytes = 32 values * 3.4375 bytes. Scales at beginning?
        // For simplicity, no scaling for Q3_K (similar to Q5_1)
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float v0 = h[base + j];
            const float v1 = h[base + 16 + j];
            sum0 += v0; sum1 += v1;
            xs[j] = v0; xs[16 + j] = v1;
        }
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

new3 = """template <> struct XD<T_Q3_K> {
    static inline void load(device const float* h, device const uchar* weight_row, int u, thread float* xs, thread float* xsum) {
        (void)weight_row;
        const int base = u * 32;
        float sum0 = 0.0f, sum1 = 0.0f;
        xd16(h + base, xs, sum0, 2);
        xd16(h + base + 16, xs + 16, sum1, 2);
        xsum[0] = sum0; xsum[1] = sum1;
    }
};"""

if old3 in content:
    content = content.replace(old3, new3)
    print("Replaced Q3_K")
else:
    print("Q3_K not found!")

with open("metal/kernels/moe.metal", "w") as f:
    f.write(content)

print("Done writing")