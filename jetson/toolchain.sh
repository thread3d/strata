#!/bin/sh
# jetson/toolchain.sh - the toolchain a Jetson AGX Xavier needs to build Strata's engine, installed without root.
#
# The board's own releases stop at GCC 9 and CMake 3.16 (Ubuntu 20.04, JetPack 5), which is neither a C++20 compiler
# nor the CMake 3.24 the tree asks for; and the CUDA 11.4 toolkit JetPack 5 carries accepts a host compiler no newer
# than GCC 10, so GCC 10 - the newest it does accept - is the version to install.  Nothing here changes the system:
#
#   cmake, ninja   pip --user, so they land in ~/.local/bin
#   GCC 10         Ubuntu's own gcc-10 packages, downloaded and unpacked into ~/toolchains/gcc10 instead of installed
#
# The unpacked GCC 10 links against the board's existing libstdc++6 (Ubuntu 20.04's is built from GCC 10), so the
# engine needs no LD_LIBRARY_PATH at run time.  Run it as the user who will build (no sudo, no password):
#
#   sh jetson/toolchain.sh
#   . ~/toolchains/strata-env.sh       # adds gcc-10, cmake and ninja to PATH for this shell
#   sh jetson/build_engine.sh
#
# docs/JETSON.md explains what this build is and what it cannot do.
set -eu

prefix=${TOOLCHAIN_PREFIX:-$HOME/toolchains}
gccdir=$prefix/gcc10
bindir=$HOME/.local/bin
min_cmake=3.24

say() { printf '%s\n' "$*" >&2; }

# 0 when the cmake on PATH is at least $min_cmake (3.24), 1 otherwise
cmake_new_enough() {
    command -v cmake >/dev/null 2>&1 || return 1
    cmake --version 2>/dev/null | head -1 | sed 's/^cmake version //' |
        awk -F. -v m="$min_cmake" '{ split(m, w, "."); if ($1+0 > w[1]+0 || ($1+0 == w[1]+0 && $2+0 >= w[2]+0)) exit 0; exit 1 }'
}

case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) ;;
  *) say "toolchain.sh: this installs the toolchain for an arm64 Linux board (the Jetson AGX Xavier), not $(uname -s)-$(uname -m)"; exit 2 ;;
esac

# ---- cmake and ninja (pip --user) ---------------------------------------------------------------
if ! cmake_new_enough; then
  say "installing cmake and ninja into $bindir (pip --user)"
  python3 -m pip install --user --upgrade "cmake>=3.24" ninja
  PATH=$bindir:$PATH
  cmake_new_enough || { say "toolchain.sh: cmake is still older than $min_cmake; add $bindir to PATH ahead of the" \
                             "system one, or name another with CMAKE=<path> when building"; exit 2; }
fi
command -v ninja >/dev/null 2>&1 || python3 -m pip install --user ninja
say "cmake: $(cmake --version | head -1)  ninja: $(ninja --version 2>/dev/null || echo missing)"

# ---- GCC 10 -------------------------------------------------------------------------------------
if [ ! -x "$gccdir/usr/bin/g++-10" ]; then
  ver=$(apt-cache policy gcc-10 2>/dev/null | sed -n 's/.*Candidate: //p' | head -1)
  [ -n "$ver" ] || { say "toolchain.sh: this system's package list has no gcc-10; install a C++20 compiler by hand"; exit 2; }
  say "unpacking gcc-10 $ver into $gccdir (no system change: apt-get download + dpkg-deb -x)"
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  ( cd "$tmp" && apt-get download gcc-10 g++-10 cpp-10 libgcc-10-dev libstdc++-10-dev >/dev/null )
  mkdir -p "$gccdir"
  for f in "$tmp"/*.deb; do dpkg-deb -x "$f" "$gccdir"; done
fi

# ---- the environment file the build reads -------------------------------------------------------
mkdir -p "$prefix"
cat > "$prefix/strata-env.sh" <<EOF
# written by jetson/toolchain.sh - source it before building Strata on this board
PATH=$bindir:$gccdir/usr/bin:\$PATH
export PATH
CC=$gccdir/usr/bin/gcc-10
CXX=$gccdir/usr/bin/g++-10
export CC CXX
EOF

# ---- check it does what the build needs: C++20 concepts, and <concepts> (GCC 9 has neither) ---------
probe=$prefix/.cxx20-probe.cpp
cat > "$probe" <<'EOF'
#include <concepts>
#include <array>
#include <string>
template <class T> concept Addable = requires(T a, T b) { a + b; };
template <Addable T> T sum(T a, T b) { return a + b; }
int main() {
    std::array<int, 3> a{1, 2, 3};
    std::string s = "blk.0.weight";
    return sum(a[0], a[1]) == 3 && s.ends_with(".weight") ? 0 : 1;
}
EOF
if ! "$gccdir/usr/bin/g++-10" -std=c++20 "$probe" -o "$prefix/.cxx20-probe" 2>"$prefix/.cxx20-probe.log"; then
  say "toolchain.sh: the unpacked g++-10 cannot compile a C++20 program:"; sed -n '1,10p' "$prefix/.cxx20-probe.log" >&2; exit 2
fi
rm -f "$probe" "$prefix/.cxx20-probe" "$prefix/.cxx20-probe.log"

say ""
say "toolchain ready:"
say "  cmake   $(cmake --version | head -1)"
say "  gcc-10  $("$gccdir/usr/bin/g++-10" --version | head -1)"
say ""
say "build with:  . $prefix/strata-env.sh && sh jetson/build_engine.sh"
