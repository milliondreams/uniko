#!/usr/bin/env python3
"""Guard: every version in the workspace must derive from the single source of
truth, `[workspace.package].version` in the root Cargo.toml.

Most version sources inherit automatically and need no check: crate versions via
`version.workspace = true`, the Python *distribution* version via
`dynamic = ["version"]` (maturin reads the crate version), and the runtime
`uniko.__version__` via `env!("CARGO_PKG_VERSION")` exported from the extension.

This guard covers the two spots Cargo/maturin CANNOT auto-inherit:

1. The internal path+version pins in `[workspace.dependencies]`
   (`uniko-* = { path = ..., version = "X" }`). `cargo publish` requires a
   version on path deps, and Cargo has no way to inherit the workspace-package
   version into them — so they are the one manual-sync spot. Bump with
   `cargo set-version --workspace <ver>` (updates these too); this guard catches
   a hand-edit that forgets one.
2. No wheel `pyproject.toml` may hardcode `version = ` — each must use
   `dynamic = ["version"]` so maturin reads the crate (= workspace) version.

Exits non-zero on any mismatch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_CARGO = REPO / "Cargo.toml"
BINDINGS = ["uniko-py", "uniko-cuda", "uniko-metal"]


def workspace_version(text: str) -> str | None:
    m = re.search(
        r"^\[workspace\.package\][^\[]*?^version = \"([^\"]+)\"", text, re.M | re.S
    )
    return m.group(1) if m else None


def main() -> int:
    text = ROOT_CARGO.read_text()
    ws = workspace_version(text)
    if not ws:
        print("ERROR: [workspace.package].version not found in Cargo.toml", file=sys.stderr)
        return 1

    ok = True

    # 1. Internal path+version pins must match the workspace version.
    pins = re.findall(r'^(uniko-[a-z]+) = \{[^}]*\bversion = "([^"]+)"', text, re.M)
    if not pins:
        print("ERROR: no internal uniko-* pins found in [workspace.dependencies]", file=sys.stderr)
        ok = False
    for name, ver in pins:
        if ver != ws:
            ok = False
            print(f'ERROR: [workspace.dependencies] {name} pins "{ver}" != workspace "{ws}"', file=sys.stderr)

    # 2. Wheel pyprojects must use dynamic version, never a hardcoded one.
    for b in BINDINGS:
        pp = REPO / "bindings" / b / "pyproject.toml"
        if not pp.exists():
            continue
        ptext = pp.read_text()
        if re.search(r"^version = ", ptext, re.M):
            ok = False
            print(f'ERROR: bindings/{b}/pyproject.toml hardcodes a version; use dynamic = ["version"]', file=sys.stderr)
        elif 'dynamic = ["version"]' not in ptext.replace("'", '"'):
            ok = False
            print(f'ERROR: bindings/{b}/pyproject.toml must declare dynamic = ["version"]', file=sys.stderr)

    if not ok:
        print(
            f'\nAll versions must derive from [workspace.package].version = "{ws}". '
            f"Bump with `cargo set-version --workspace <ver>`.",
            file=sys.stderr,
        )
        return 1

    print(f'OK: workspace version "{ws}"; {len(pins)} internal pins match and all wheel pyprojects are dynamic.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
