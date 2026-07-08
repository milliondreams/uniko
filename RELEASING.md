<!-- SPDX-License-Identifier: Apache-2.0 -->
# Releasing uniko

This repo ships two artifact families from a single tag:

- **Rust crates** → [crates.io] (6 publishable crates, one shared workspace version).
- **Python wheels** → [PyPI] (the `uniko` package, built with maturin).

Both are published by `.github/workflows/release.yml`, triggered by pushing a
`v*` tag. Authentication is **OIDC Trusted Publishing** for both registries —
there are **no API tokens stored in the repo**.

---

## TL;DR — cutting a release

1. Bump the version with **`cargo set-version --workspace <ver>`** (cargo-edit) —
   this updates `[workspace.package].version` **and** the internal
   `[workspace.dependencies]` pins atomically. Everything else derives from it:
   crate versions (`version.workspace = true`), the Python wheel version
   (`dynamic = ["version"]`), and runtime `uniko.__version__`
   (`env!("CARGO_PKG_VERSION")`). Do **not** hand-edit versions — the
   `check_version_sync` CI/guard step fails if the pins drift or a pyproject
   hardcodes a version. Land it on `main` and make sure CI is green.
2. Tag and push:
   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. The workflow runs **validation + builds unconditionally**, then **pauses** the
   publish jobs on the `release` environment.
4. Review the dry-run logs and downloadable wheel/sdist artifacts on the run.
5. When satisfied, **approve** the `release` environment in the Actions UI. The
   crates publish to crates.io (in dependency order), the wheels + sdist publish
   to PyPI, and a GitHub Release is created with the artifacts attached.

> The tag must match the workspace version, or the `guard` job fails. A
> pre-release suffix (`v0.1.0-rc.1`) is accepted for rehearsal — it compares only
> the base version, so the published crate/wheel still carry the workspace
> version (`0.1.0`). Don't approve a rehearsal run unless you actually mean to
> publish that version.

---

## What the workflow does

| Job | Gated? | Purpose |
| --- | --- | --- |
| `guard` | no | Fails unless the tag matches the workspace version. |
| `validate-crates` | no | `cargo publish --dry-run` for the leaf crate; `cargo package --no-verify` for the rest. |
| `build-wheels` | no | Builds abi3 wheels for Linux (x86_64, aarch64), macOS (arm64), Windows (x64). |
| `build-sdist` | no | Builds the source distribution. |
| `publish-crates` | **yes** | Publishes the 6 crates to crates.io via OIDC, in dependency order. |
| `publish-pypi` | **yes** | Publishes all wheels + sdist to PyPI via OIDC. |
| `github-release` | **yes** | Creates the GitHub Release and attaches all artifacts. |

The publish order is fixed by the internal dependency graph:

```
uniko-store → uniko-pipes → uniko-extract → uniko-cortex → uniko-memory → uniko-api
```

`uniko-bench` and `bindings/uniko-py` are `publish = false` — the bench harness
is internal, and the Python binding ships as a wheel, not a crate.

### The dry-run caveat (read this)

`cargo publish --dry-run` packages a crate, rewrites its **path deps to
version-only deps**, then compiles that packaged form. The leaf crate
(`uniko-store`) verifies end-to-end this way. The other five **cannot** be fully
dry-run published before a real release: their packaged form requires
`uniko-store 0.1.0` (etc.) to already exist on crates.io. So `validate-crates`
runs `cargo package --no-verify` for them — which validates the manifest and the
packaged file list, but not a from-scratch compile. **Full chain verification
only happens during the real, ordered publish.** This is an inherent crates.io
limitation, not a workaround.

---

## One-time setup (required before the first real release)

These are configured **outside this repo** and only need to be done once.

### 1. GitHub `release` environment (the approval gate)

In the repo: **Settings → Environments → New environment → `release`**. Add
yourself / the release team under **Required reviewers**. This is what makes the
publish jobs pause for manual approval. Optionally restrict it to the `main`
branch and to tags matching `v*`.

### 2. crates.io Trusted Publishing

For **each** of the 6 publishable crates, on crates.io:
**Crate → Settings → Trusted Publishing → Add**, with:

- Repository owner/name: `rustic-ai/uniko`
- Workflow filename: `release.yml`
- Environment: `release`

> **First publish of a brand-new crate name:** crates.io trusted publishing can
> only be configured on a crate that already exists. If a name has never been
> published, do a **one-time manual bootstrap** of that version from a maintainer
> machine (`cargo publish -p <crate>`), then add the trusted publisher for
> subsequent releases. Confirm this per crate before relying on the gated job.

> **Bootstrap from a clean clone, not your dev tree.** `cargo publish` first runs a
> libgit2 status walk from the repo root. A working tree with large *gitignored* dirs
> (`data/` benchmark KBs, `target/`, `.uni_cache/`, `.venv/` — millions of files) makes
> that walk fail with `failed to retrieve git status … Failed to update the excludes
> stack`. `--allow-dirty` does **not** help (it suppresses the dirty *warning*, not the
> walk). Publish from a fresh checkout instead, which contains only tracked files:
>
> ```sh
> git clone --local . /tmp/uniko-publish && cd /tmp/uniko-publish
> cargo publish -p uniko-store   # then the rest, in dependency order
> ```
>
> CI is unaffected — it always checks out clean.

### 3. PyPI Trusted Publishing

The three wheel-variant projects — `uniko`, `uniko-cuda`, `uniko-metal` — are
all already registered on PyPI (each has a `0.0.0` placeholder release), so add
a **regular** trusted publisher on each existing project. (A "pending publisher"
is only for a name that does *not* exist yet — no longer our case.)

On PyPI, for **each** of `uniko`, `uniko-cuda`, `uniko-metal`:
**Project → Manage → Publishing → Add a trusted publisher (GitHub)**, with:

- Owner: `rustic-ai`
- Repository name: `uniko`
- Workflow name: `release.yml`
- Environment name: `release`

(To rehearse against [TestPyPI], register the same publisher there and point
`publish-pypi` at it temporarily.)

#### File-size limit (required — all three variants exceed 100 MB)

PyPI's default per-file limit is 100 MB, and every variant is over it: the base
`uniko` wheel statically bundles ONNX Runtime (~113 MiB), and `uniko-cuda` /
`uniko-metal` additionally embed candle+mistralrs GPU kernels. Request a
file-size-limit increase for **each** project via
<https://pypi.org/help/#file-size-limit> (cite the bundled ONNX Runtime +
candle/mistralrs kernels; rustic-ai/uni-db obtained the same increases for its
`uni-db-cuda`/`-metal` variants). The `github-release` job attaches the wheels
to the GitHub Release (2 GB/asset) as a fallback if a PyPI upload is still
rejected.

**PyPI publishing is disabled by a flag until those increases land.** The
`publish-pypi` job is gated on the repository variable `PYPI_PUBLISH_ENABLED`
and will not run while it is unset. Everything else still works — wheels build
and validate, crates.io publishes, and `github-release` attaches the wheels
(the interim distribution channel). **To enable PyPI once the size increases are
approved:** Settings → Secrets and variables → Actions → **Variables** → set
`PYPI_PUBLISH_ENABLED` = `true`. (It must be a repository *variable*, not an env
value — a job-level `if:` can't read workflow `env`.)

---

## Rehearsing without going live

Because the publishes are gated, the safest rehearsal is:

1. Push a pre-release tag, e.g. `git push origin v0.1.0-rc.1`.
2. Let `validate-crates`, `build-wheels`, and `build-sdist` run.
3. Download the wheel artifacts from the run and smoke-test locally:
   ```sh
   pip install ./uniko-0.1.0-cp310-abi3-manylinux_2_28_x86_64.whl
   python -c "import uniko; print(uniko.__version__)"
   ```
4. **Do not approve** the `release` environment. Cancel the run (or just leave the
   gated jobs unapproved) and delete the rc tag.

---

## Known risks

- **`ort` links only from pyke's prebuilt binaries.** The `ort` crate (and
  uni-db's `provider-onnx`) does not build ONNX Runtime from source — it
  downloads a prebuilt binary for the exact target triple at build time, and
  fails hard (`cargo::error`) for any triple pyke does not publish. Two
  consequences are already baked into the matrix:
  - **aarch64 Linux builds on a native ARM runner** (`ubuntu-24.04-arm`), not an
    x86_64 cross-build, so pyke fetches the native aarch64 ORT bundle inside the
    ARM manylinux container. This removes the old QEMU/cross-link fragility — if
    this cell regresses, keep it native rather than reaching for emulation.
  - **macOS is aarch64-only.** rc.12 ships no `x86_64-apple-darwin` binary and
    our `onnx` feature is always-on, so an Intel-macOS wheel cannot link. Intel
    Macs are EOL; do not re-add x86_64 macOS without building ORT from source.
- **`ort = 2.0.0-rc.12` is a pre-release.** crates.io accepts crates that depend
  on pre-release versions, so this does not block publishing.

[crates.io]: https://crates.io
[PyPI]: https://pypi.org/project/uniko/
[TestPyPI]: https://test.pypi.org/
