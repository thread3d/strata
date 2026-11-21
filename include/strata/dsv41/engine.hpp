// include/strata/dsv41/engine.hpp - the DeepSeek V4.1-Flash-1017 (ds4) engine path as a library:
// `strata` hands a --pack directory to it at the top of main() (the family switch), and strata-dsv41 is a thin
// wrapper around the same entry point.  See src/dsv41/ds4_main.cpp.
// DeepSeek packs are directory-based (ds4.txt + engram.bin + GGUF shards), not a single GGUF, so there is no
// general.architecture key to read: the switch detects a ds4 pack by the presence of ds4.txt in the --pack directory.
#pragma once

namespace strata::dsv41 {

/// The engine's command line (strata's own flag names where they mean the same thing: --pack / --model,
/// --max-context, --serve, --tokens / --ids, --max-new / --n; the family's own: --gpus, --vram-experts,
/// --cpu-threads, --expert-profile, --ssd-cache, --dspark, --experts, --tp, --prefill-chunk, --vision).  Returns
/// the process exit code.
int run(int argc, char** argv);

}  // namespace strata::dsv41
