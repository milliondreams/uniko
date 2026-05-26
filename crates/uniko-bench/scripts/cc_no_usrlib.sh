#!/bin/bash
# Linker wrapper that drops `-L /usr/lib` from the cc argv.
#
# On this Fedora install /usr/lib holds the 32-bit i386 multilib stub
# of libcuda.so, which the linker picks up before /usr/lib64/libcuda.so
# and errors with `incompatible with elf64-x86-64`. Stripping the
# -L /usr/lib pair forces the linker to find the 64-bit lib in the
# usual search paths.
#
# Used via `RUSTFLAGS="-C linker=<path-to-this-script>"` during a
# gpu-cuda build. See crates/uniko-bench/run.sh.

args=()
i=1
while [ $i -le $# ]; do
    a="${!i}"
    if [ "$a" = "-L" ]; then
        j=$((i+1))
        b="${!j}"
        if [ "$b" = "/usr/lib" ]; then
            i=$((i+2))
            continue
        fi
    fi
    args+=("$a")
    i=$((i+1))
done
exec cc "${args[@]}"
