#!/usr/bin/env python3
"""Generate scheduling-only variants of the pinned two-component BF16 attention."""
import argparse
import hashlib
from pathlib import Path


REFERENCE_SHA256 = "dc3eaf7c379b38535269d6651de4ba93c714873e142f479d6abcd7150a6accfb"


def generate(source, *, skip_masked_qk=False, prefetch_values=False, compact_values=False):
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("expected the pinned attention_shared_k16.loom source")
    if not any((skip_masked_qk, prefetch_values, compact_values)):
        raise ValueError("select at least one scheduling change")

    if skip_masked_qk:
        # Odd QK waves compute only masked scores. Preserve their score stores
        # and all barriers, but omit their dot products and unused K staging.
        start = source.index("%ksrow1 =")
        end = source.index("kernel.barrier<workgroup>", start)
        source = source[:start] + source[end:]
        start = source.index("%qkzero =")
        end = source.index("%kr0 =", start)
        dot = source[start:end].replace("%scaled =", "%active_scaled =")
        branch = (
            "%qk_active = index.cmp eq, %part, %c0 : index\n"
            "%scaled = scf.if %qk_active -> (vector<8xf32>) {\n"
            + dot
            + "scf.yield %active_scaled : vector<8xf32>\n"
            "} else { scf.yield %zv : vector<8xf32> }\n"
        )
        source = source[:start] + branch + source[end:]

    if prefetch_values:
        start = source.index("%stagevc0 =")
        end = source.index("kernel.barrier<workgroup>", start)
        block = source[start:end].splitlines()
        loads = "\n".join(line for line in block if not line.startswith("vector.store")) + "\n"
        stores = "\n".join(line for line in block if line.startswith("vector.store")) + "\n"
        if stores.count("vector.store") != 1:
            raise ValueError("expected one 16-key value load")
        source = source[:start] + stores + source[end:]
        marker = "%qk_active =" if skip_masked_qk else "%qkzero ="
        source = source.replace(marker, loads + "scf.schedule.fence\n" + marker, 1)

    if compact_values:
        # Only columns 0..15 are consumed. Retain padding to spread LDS banks.
        source = source.replace("view<256x40xbf16>", "view<256x24xbf16>")
        source = source.replace("%stats_off = index.constant 64512", "%stats_off = index.constant 56320")
        source = source.replace("%lds_bytes = index.constant 64768", "%lds_bytes = index.constant 56576")

    # The reference's opening layout description no longer describes variants.
    header = (
        "// Experimental BF16 16-key scheduling; both probability components retained.\n"
        f"// skip_masked_qk={skip_masked_qk}, prefetch_values={prefetch_values}, "
        f"compact_values={compact_values}.\n"
        "// Arithmetic order of active QK, softmax and PV is unchanged.\n"
    )
    return header + source[source.index("amdgpu.target"):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("kernels/attention_shared_k16.loom"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-masked-qk", action="store_true")
    parser.add_argument("--prefetch-values", action="store_true")
    parser.add_argument("--compact-values", action="store_true")
    args = parser.parse_args()
    candidate = generate(
        args.source.read_text(), skip_masked_qk=args.skip_masked_qk,
        prefetch_values=args.prefetch_values, compact_values=args.compact_values,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(candidate)


if __name__ == "__main__":
    main()
