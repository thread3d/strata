// include/strata/qwen36/engine.hpp - the Qwen3.6-35B-A3B (GGUF architecture qwen35moe) engine path on Metal:
// `strata` hands a qwen35moe model to it at the top of main() (the family switch, guarded by STRATA_HAVE_QWEN36),
// and strata-q36 is a thin wrapper around the same entry point.  See metal/src/qwen36/q36_engine.cpp.
#pragma once

namespace strata::qwen36 {

/// The engine's command line (strata's own flag names: --native / --model, --max-context, --serve,
/// --tokens / --ids, --max-new / --n).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::qwen36
