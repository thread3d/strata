with open("metal/src/ref_quant.h", "r") as f:
    content = f.read()

old = "inline size_t row_bytes(int t, int n) { return (size_t)n / block_elems(t) * block_bytes(t); }"
new = "inline size_t row_bytes(int t, int n) { return (size_t)((n + block_elems(t) - 1) / block_elems(t)) * block_bytes(t); }"

if old in content:
    content = content.replace(old, new)
    print("Replaced")
else:
    print("Not found")
    for i, line in enumerate(content.split('\n')):
        if 'row_bytes' in line and 'inline' in line:
            print(f'Line {i}: {repr(line)}')

with open("metal/src/ref_quant.h", "w") as f:
    f.write(content)