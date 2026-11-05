#!/bin/sh
# jetson/build_engine.sh - build the Strata CUDA engine on a Jetson AGX Xavier (sm_72, arm64 Linux, CUDA 11.4) into a
# ready-made engine folder.
#
#   jetson/build_engine.sh                      # engine -> ./out-jetson/engine
#   ARCHS=72 BUILD_DSV41=0 ARCHS=72 OUT=/tmp/eng JOBS=6 jetson/build_engine.sh
#
# Run jetson/toolchain.sh first (or have a C++20 host compiler and CMake 3.24+ of your own), then source
# ~/toolchains/strata-env.sh.  The result is the folder setup.py keeps a ready-made engine in.
#
# What is different from spark/build_engine.sh, and why:
#
#   * CUDA 11.4, not 13.  JetPack 5 is the last release for Xavier and it carries 11.4.  nvcc 11.4 has no -std=c++20,
#     so CMakeLists.txt builds the CUDA translation units as C++17 (no .cu source needs C++20 syntax), and the sites
#     that use launch/graph/memcpy entry points newer than 11.4 take their pre-CUDA-12 path
#     (include/strata/platform/cuda_capabilities.hpp): no cluster launches, no programmatic dependent launch, no
#     batched memcpy, no per-node graph priorities.  Each of those is a speed path, not a different result.
#   * sm_72, not sm_121.  Compute capability 7.2 is below the 7.5 the engine supports, so the build takes
#     -DSTRATA_EXPERIMENTAL_SM60=ON (the community Pascal/Volta path, #236).  docs/JETSON.md is explicit about what
#     that does not promise.
#   * No STRATA_PORTABLE.  That flag builds ggml-cpu for an ARMv8.2+dotprod+fp16+i8mm baseline (what a Grace or a
#     Graviton has).  Xavier's Carmel core has fp16 but no dotprod and no i8mm, so those instructions would fault.
#     The build stays -mcpu=native, for this board.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
archs=${ARCHS:-72}
out=${OUT:-$repo/out-jetson/engine}
build=${BUILD:-$repo/build-jetson}
jobs=${JOBS:-$(nproc)}
py=${PYTHON:-python3}
cmake=${CMAKE:-cmake}
llama=${STRATA_GGML_DIR:-$repo/third_party/llama.cpp}

say() { printf '%s\n' "$*" >&2; }

case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) ;;
  *) say "build_engine.sh: this builds on arm64 Linux (the Jetson); use spark/build_engine.sh on a DGX Spark and" \
         "packaging/build_cuda_linux.sh on an x86 PC"; exit 2 ;;
esac
# /usr/local/cuda/bin is on PATH for an interactive shell on a JetPack board, not for every one
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  PATH=/usr/local/cuda/bin:$PATH
  export PATH
fi
command -v nvcc >/dev/null 2>&1 || { say "build_engine.sh: no nvcc; JetPack 5 installs the CUDA 11.4 toolkit at" \
                                        "/usr/local/cuda-11.4 (docs/JETSON.md)"; exit 2; }

cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$cudafull" in
  11.4) ;;
  11.*) say "build_engine.sh: note: CUDA $cudafull has not been tried on this board; 11.4 is what JetPack 5 ships" ;;
  12.*) say "build_engine.sh: CUDA $cudafull works too, but the engine then takes the newer launch and graph paths;" \
            "nothing here needs it" ;;
  *) say "build_engine.sh: CUDA ${cudafull:-missing} is not a toolkit this board can run: JetPack 5 caps it at 11.4" >&2; exit 2 ;;
esac

# a C++20 host compiler, which is not the board's GCC 9
if ! command -v "$CXX" >/dev/null 2>&1; then
  say "build_engine.sh: CXX ('$CXX') is not on PATH.  Run sh jetson/toolchain.sh and . ~/toolchains/strata-env.sh first." >&2
  exit 2
fi

if [ ! -f "$llama/ggml/CMakeLists.txt" ]; then
  # llama.cpp at the pinned commit (ggml for the engine), exactly as setup.py fetches it
  (cd "$repo" && "$py" -c "import setup; print(setup.get_llama_cpp())")
  [ -f "$llama/ggml/CMakeLists.txt" ] || { say "build_engine.sh: no llama.cpp tree at $llama after fetching it" \
                                             "(STRATA_GGML_DIR=<tree> names another)" >&2; exit 2; }
fi

# the engine prints its commit with --version: a checkout gives it to CMake itself, a tree copy without .git needs COMMIT
want_dsv41=OFF
[ "${BUILD_DSV41:-0}" = 1 ] && want_dsv41=ON
if [ "$want_dsv41" = ON ] && [ ! -e "$repo/.git" ] && [ ! -f "$repo/BUILD_ID" ]; then
  [ -n "${COMMIT:-}" ] || { say "build_engine.sh: $repo has no .git and no BUILD_ID: set COMMIT=<hex> so the DeepSeek" \
                                "engine can print its commit" >&2; exit 2; }
  printf '%s\n' "$(printf '%s' "$COMMIT" | cut -c1-9)" > "$repo/BUILD_ID"
fi

real=""; for a in $(echo "$archs" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}
cd "$repo"
# -DSTRATA_EXPERIMENTAL_SM60=ON admits compute capability below 7.5 both at configure time and in the runtime device
# check (src/core/device.cu), which is what lets sm_72 build and run at all.
"$cmake" -G Ninja -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_BUILD_DSV41="$want_dsv41" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF \
    '-DCMAKE_INSTALL_RPATH=$ORIGIN' "-DCMAKE_CUDA_ARCHITECTURES=${real}" -DSTRATA_GGML_DIR="$llama"
rm -f "$build/strata-dsv41"
if [ "$want_dsv41" = ON ]; then
  "$cmake" --build "$build" --target strata strata_kq strata-dsv41 -j "$jobs"
else
  "$cmake" --build "$build" --target strata strata_kq -j "$jobs"
fi

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
rm -rf "$out"
mkdir -p "$out"
cp "$build/strata" "$out/"
[ -f "$build/libstrata_kq.so" ] && cp "$build/libstrata_kq.so" "$out/"
[ "$want_dsv41" = ON ] && [ -f "$build/strata-dsv41" ] && cp "$build/strata-dsv41" "$out/"
# the CUDA libraries the engine loads (cuBLAS, cuBLASLt, cudart) go beside it, so the board needs only its driver
"$py" packaging/collect_libs.py --binary "$build/strata" --search /usr/local/cuda/lib64 --out "$out" --also 'libcublasLt.so*'
echo "{\"version\": \"$ver\", \"source\": \"jetson/build_engine.sh\", \"platform\": \"linux-aarch64\", \"archs\": \"$archs\", \"ptx\": false, \"cuda\": \"$cudafull\", \"experimental_sm60\": true}" > "$out/BUILD.json"
"$out/strata" --version 2>/dev/null || true
echo "built: $out"
