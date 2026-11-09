// metal/tools/metal_caps.m - what this Mac's Metal can do, for every GPU it reports.
//
// The Metal engine asks the runtime three questions before it picks a kernel (probe_mpp, probe_bf16 in src/mtl_rt.mm), and
// the DeepSeek engine's kernels pass raw device addresses, which need one more.  This prints the answers on a machine you are
// looking at for the first time, plus the two facts that decide whether a memory range can be handed to the GPU without a copy.
// No GPU work beyond one pipeline build per probe and one 4 KB buffer per no-copy test.
//
//   clang -fobjc-arc -O1 -framework Metal -framework Foundation metal/tools/metal_caps.m -o /tmp/metal_caps && /tmp/metal_caps
//
// Measured on an Intel Mac Pro (Radeon RX 580, macOS 11.7.11): metal3=0, gpuAddress absent, no residency set, no bf16, no MPP,
// and newBufferWithBytesNoCopy refuses EVERY read-only mapping while accepting writable memory of any size - which is why the
// model's pages are mapped PROT_READ | PROT_WRITE + MAP_PRIVATE (docs/MAC.md, "An Intel Mac").
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <stdio.h>

static void report_capabilities(id<MTLDevice> d) {
    NSArray* all = MTLCopyAllDevices();
    printf("device %-34s maxBufferLength %.1f GB, max threads/threadgroup %lu\n", [[d name] UTF8String],
           [d maxBufferLength] / 1e9, (unsigned long) [d maxThreadsPerThreadgroup].width);
    // MTLGPUFamilyMetal3 is 5001 and MTLGPUFamilyMac2 is 2002: an SDK older than macOS 13 has neither enum, so the numbers go in
    if ([d respondsToSelector:@selector(supportsFamily:)]) {
        printf("  families: metal3=%d mac2=%d common3=%d common2=%d apple7=%d\n",
               (int) [d supportsFamily:(MTLGPUFamily) 5001], (int) [d supportsFamily:(MTLGPUFamily) 2002],
               (int) [d supportsFamily:(MTLGPUFamily) 2003], (int) [d supportsFamily:(MTLGPUFamily) 1000],
               (int) [d supportsFamily:(MTLGPUFamily) 1007]);
    }
    // The simdgroup width decides whether a 32-lane kernel is right - Metal's is 32 on Apple silicon and 64 on this AMD card -
    // and neither threadsPerSimdgroup nor simdgroupsPerThreadgroup is in an older SDK, so it is measured: a kernel writes the
    // two MSL constants out.  (This is probe_simd_width in src/mtl_rt.mm, which the runtime asks before it compiles anything.)
    unsigned lanes = 0, sgs = 0;
    {
        // the two come in as kernel arguments with their own attributes (they are not globals in MSL)
        const char* src = "kernel void w(device uint* o [[buffer(0)]], uint tid [[thread_index_in_threadgroup]],\n"
                          "                  uint lanes [[threads_per_simdgroup]], uint sgs [[simdgroups_per_threadgroup]]) {\n"
                          "    if (tid == 0) { o[0] = lanes; o[1] = sgs; }\n"
                          "}";
        NSError* e = nil;
        id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&e];
        id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"w"] : nil;
        id<MTLComputePipelineState> ps = fn ? [d newComputePipelineStateWithFunction:fn error:&e] : nil;
        id<MTLBuffer> out = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (ps && out) {
            id<MTLCommandQueue> q = [d newCommandQueue];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const unsigned* v = (const unsigned*) [out contents];
            lanes = v[0];
            sgs = v[1];
        }
    }
    if (lanes) printf("  simdgroup: %u lanes, %u simdgroups in a 256-thread threadgroup\n", lanes, sgs);
    else {
        NSError* why = nil;
        id<MTLLibrary> probe = [d newLibraryWithSource:[NSString stringWithUTF8String:
            "kernel void w(device uint* o [[buffer(0)]], uint tid [[thread_index_in_threadgroup]],\n"
            "              uint lanes [[threads_per_simdgroup]]) { if (tid == 0) o[0] = lanes; }"] options:nil error:&why];
        (void) probe;
        printf("  simdgroup: could not be measured here (%s)\n",
               why ? [[why localizedDescription] UTF8String] : "no pipeline");
    }
    // the two the runtime probes, and the one the DeepSeek engine's raw-pointer kernels need
    printf("  gpuAddress API: %s\n", [d respondsToSelector:@selector(gpuAddress)] ? "present" : "absent");
    printf("  MTLResidencySet: %s\n", NSClassFromString(@"MTLResidencySet") ? "present" : "absent");
    const char* bf16 = "kernel void p(device float* o [[buffer(0)]]) { bfloat x = bfloat(1.5f); o[0] = float(x); }";
    const char* mpp = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n"
                      "kernel void q(device float* o [[buffer(0)]]) { o[0] = 1.0f; }";
    NSError* err = nil;
    printf("  bf16 (bfloat in MSL): %s\n",
           [d newLibraryWithSource:[NSString stringWithUTF8String:bf16] options:nil error:&err] ? "compiles" : "absent");
    err = nil;
    printf("  MetalPerformancePrimitives: %s\n",
           [d newLibraryWithSource:[NSString stringWithUTF8String:mpp] options:nil error:&err] ? "compiles" : "absent");
    (void) all;
}

// Can this range be handed to the GPU without a copy?  Measured over four kinds of memory at one size.
static void report_no_copy(id<MTLDevice> d) {
    const size_t n = 1 << 20;
    struct { const char* what; void* p; } cases[4];
    void* h = malloc(n + 16384);
    cases[0].what = "anonymous mmap, writable";
    cases[0].p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    cases[1].what = "anonymous mmap, read-only";
    cases[1].p = mmap(NULL, n, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    cases[2].what = "malloc, 16 KiB aligned";
    cases[2].p = (void*) (((uintptr_t) h + 16383) & ~(uintptr_t) 16383);
    cases[3].what = "read-only file mmap";
    cases[3].p = NULL;
    NSString* tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:@"metal_caps.bin"];
    const int fd = open([tmp fileSystemRepresentation], O_CREAT | O_RDWR, 0600);
    if (fd >= 0) {
        ftruncate(fd, (off_t) n);
        cases[3].p = mmap(NULL, n, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
    }
    printf("newBufferWithBytesNoCopy, 1 MB:\n");
    for (int i = 0; i < 4; ++i) {
        if (cases[i].p == MAP_FAILED || cases[i].p == NULL) { printf("  %-32s (no memory)\n", cases[i].what); continue; }
        id<MTLBuffer> b = [d newBufferWithBytesNoCopy:cases[i].p length:n options:MTLResourceStorageModeShared deallocator:nil];
        printf("  %-32s %s\n", cases[i].what, b ? "accepted" : "REFUSED");
    }
    if (cases[3].p && cases[3].p != MAP_FAILED) munmap(cases[3].p, n);
    [NSFileManager.defaultManager removeItemAtPath:tmp error:nil];
}

int main(void) { @autoreleasepool {
    NSArray* devices = MTLCopyAllDevices();
    if (devices.count == 0) { printf("no Metal device (an ssh session needs one: MTLCreateSystemDefaultDevice returns nil there)\n"); return 1; }
    for (id<MTLDevice> d in devices) { report_capabilities(d); report_no_copy(d); }
    return 0; } }
