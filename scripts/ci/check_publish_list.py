#!/usr/bin/env python3
"""Guard: the crates.io publish list in `.github/workflows/release.yml`
(`PUBLISH_ORDER`) must be uniko's complete publishable dependency closure, in
topological order.

`cargo publish` resolves each crate's dependencies against the crates.io index,
so a workspace crate can only be published after every workspace crate it
depends on (normal + build deps; path-only dev-deps are stripped, but VERSIONED
dev-deps are kept and gate too). The release workflow publishes a hardcoded
ordered list. When a workspace crate is added to the graph but not the list, or
a versioned dev-dependency cycle is introduced, the release fails mid-publish
with "no matching package named ..." — and crates.io is append-only, so a bad
partial publish can't be undone. This check recomputes the closure from
`cargo metadata` and compares it to the list so the failure surfaces in CI,
before tagging.

Requires `cargo` on PATH. Exits non-zero with a diff on mismatch.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE_YML = REPO / ".github/workflows/release.yml"
ROOT_CRATE = "uniko-api"


def publishable_closure() -> tuple[set[str], dict[str, set[str]]]:
    """Return (closure, normal+build dep graph) for ROOT_CRATE's workspace deps."""
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=REPO
        )
    )
    pkgs = {p["name"]: p for p in meta["packages"]}
    internal = set(pkgs)
    nopublish = {n for n, p in pkgs.items() if p.get("publish") == []}

    graph: dict[str, set[str]] = {}
    for name, p in pkgs.items():
        deps: set[str] = set()
        for d in p["dependencies"]:
            if d["name"] not in internal:
                continue
            # Normal + build deps always gate publishing. Dev-deps gate too IF
            # they carry a version requirement: `cargo publish` keeps versioned
            # dev-dependencies in the published manifest, and refuses to package
            # until they exist on crates.io. Path-only dev-deps (req "*") are
            # stripped and don't gate — that's the escape hatch for a true
            # A<->B cycle (see crates/uniko-cortex/Cargo.toml dev-deps).
            if d["kind"] in (None, "build") or (
                d["kind"] == "dev" and d.get("req", "*") != "*"
            ):
                deps.add(d["name"])
        graph[name] = deps

    seen: set[str] = set()
    stack = [ROOT_CRATE]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(graph.get(c, ()))
    closure = {c for c in seen if c not in nopublish}
    return closure, graph


def publish_list_from_workflow() -> list[str]:
    text = RELEASE_YML.read_text()
    # uniko keeps the ordered list in the `PUBLISH_ORDER: "a b c"` env var.
    m = re.search(r'^\s*PUBLISH_ORDER:\s*"([^"]*)"', text, re.MULTILINE)
    if not m:
        print("ERROR: PUBLISH_ORDER not found in release.yml", file=sys.stderr)
        return []
    return m.group(1).split()


def main() -> int:
    closure, graph = publishable_closure()
    listed = publish_list_from_workflow()
    listed_set = set(listed)

    ok = bool(listed)

    missing = closure - listed_set
    if missing:
        ok = False
        print(f"ERROR: publishable crates MISSING from release.yml: {sorted(missing)}", file=sys.stderr)

    extra = listed_set - closure
    if extra:
        ok = False
        print(f"ERROR: release.yml lists crates NOT in {ROOT_CRATE}'s publishable closure: {sorted(extra)}", file=sys.stderr)

    if len(listed) != len(listed_set):
        ok = False
        dupes = sorted({c for c in listed if listed.count(c) > 1})
        print(f"ERROR: release.yml publishes crates more than once: {dupes}", file=sys.stderr)

    # Topological order: each crate must appear after its in-closure deps. A true
    # versioned dev-dependency cycle makes this unsatisfiable in ANY order, so it
    # is reported here — forcing the path-only-dev-dep fix.
    pos = {c: i for i, c in enumerate(listed)}
    for c in listed:
        for dep in graph.get(c, ()):
            if dep in closure and dep in pos and pos[dep] > pos[c]:
                ok = False
                print(f"ERROR: {c} is published before its dependency {dep}", file=sys.stderr)

    if not ok:
        print(
            f"\nUpdate the `PUBLISH_ORDER` list in {RELEASE_YML.relative_to(REPO)} "
            f"to {ROOT_CRATE}'s full publishable dependency closure in topological "
            f"order (or break a versioned dev-dep cycle by making the back-edge "
            f"path-only).",
            file=sys.stderr,
        )
        return 1

    print(f"OK: release.yml publishes all {len(closure)} crates in {ROOT_CRATE}'s closure, in topological order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
