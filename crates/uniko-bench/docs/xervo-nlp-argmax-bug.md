# Bug: xervo `OnnxNlpModel::analyze` fails on every input (`argmax_axis` axis=2)

**Status:** confirmed, blocker for the NLP-API migration
**Component:** `uni-xervo` `0.13.0` — `local/onnx` provider, `NlpModel` impl
**Repo to file against:** `milliondreams/uni-xervo`
**Found:** 2026-06-07, while running the uniko↔xervo NLP A/B parity check
**Fix size:** one character (`2` → `1`)

---

## Summary

xervo `0.13.0` ships an `OnnxNlpModel` implementing the higher-level
`NlpModel` trait (`runtime.nlp_model(alias).analyze(..)`). It is
**non-functional**: every call to `analyze()` returns

```
ONNX invocation failure for alias 'nlp': argmax_axis only supports axis=1, got 2
```

The failure is **unconditional** — it does not depend on which
`NlpTasks` are requested. A POS-only request fails identically to a
full `NlpTasks::ALL` request. The model can never produce output for any
caller.

This is the impl the doc comment at `src/traits/nlp.rs:141` still
describes as shipping "in a follow-up release": the code landed but was
never exercised, so the bug went unnoticed.

---

## Symptom

Reproduced via the uniko `nlp-parity` bench bin (which resolves a
`task = nlp` alias backed by the canonical `kniv-deberta` cascade and
calls `analyze`):

```
$ ./target/debug/nlp-parity
INFO nlp_parity: sentences to compare count=12
Error: xervo analyze 'Barack Obama was born in Hawaii in 1961.':
       ONNX invocation failure for alias 'nlp': argmax_axis only supports axis=1, got 2
```

uniko's own pipeline (raw-tensor path, same model, same ONNX artifact)
succeeds on the identical input — so this is isolated to xervo's decode,
not the model weights or the environment.

---

## Root cause

`uni-xervo/src/provider/local_onnx/nlp.rs`

The dependency-parse head decode calls `argmax_axis` with `axis = 2`:

```rust
// nlp.rs:257  (inside the per-chunk forward loop)
let dep_heads = argmax_axis(&outputs.arc_scores, 2)?; // [seq, seq] -> head per token
```

But `argmax_axis` operates on a **2-D** `Array2<f32>` and only supports
`axis == 1`:

```rust
// nlp.rs:527-536
/// Argmax over `axis` of a 2-D matrix; works only when axis=1 (per-row argmax).
fn argmax_axis(arr: &Array2<f32>, axis: usize) -> Result<Vec<usize>> {
    if axis != 1 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: "nlp".to_string(),
            cause: format!("argmax_axis only supports axis=1, got {axis}"),
        });
    }
    argmax_last_axis(arr)
}
```

`arc_scores` is `[seq, seq]` (a 2-D matrix of head-attachment scores).
The intent — stated in the call-site comment, "head per token" — is a
**per-row argmax over candidate heads**, which for an `Array2` is
`axis = 1`. The `2` is almost certainly a leftover from a 3-D
`(batch, seq, seq)` mental model; on a 2-D array, axis 2 does not exist,
so the guard rejects it every time.

### Why it is unconditional (not DEP-gated)

The failing call at line 257 runs **before** the per-task gate. The
`NlpTasks::DEP` check only happens later, when assembling the token:

```rust
// nlp.rs:282-289 — the task gate is HERE, far downstream of line 257
let dep = if request.tasks.contains(NlpTasks::DEP) {
    Some(DepLink { head: dep_heads[i], relation: /* ... */ })
} else {
    None
};
```

Because `dep_heads` is computed eagerly at line 257 regardless of the
requested tasks, `analyze()` errors out before it ever reaches this
gate. Requesting only POS, only NER, or only CLS fails just the same.

---

## The fix

`uni-xervo/src/provider/local_onnx/nlp.rs:257`

```diff
-        let dep_heads = argmax_axis(&outputs.arc_scores, 2)?; // [seq, seq] -> head per token
+        let dep_heads = argmax_axis(&outputs.arc_scores, 1)?; // [seq, seq] -> head per token
```

`argmax_axis(arr, 1)` delegates to `argmax_last_axis`, which is exactly
"for each row (token), pick the column (candidate head) with the max
score" — the intended semantics.

> Optional hardening: since `argmax_axis` only ever supports `axis=1`,
> the `axis` parameter is dead weight that invites exactly this mistake.
> Consider dropping the parameter and calling `argmax_last_axis`
> directly, or `debug_assert!`-ing the axis at the call site.

---

## Why it shipped (test gap)

The only end-to-end coverage of this path is
`tests/onnx_models_expensive_test.rs:559`:

```rust
#[tokio::test]
#[ignore]                       // <-- never runs in CI
async fn kniv_deberta_nlp_cascade_end_to_end() {
    require_expensive_tests!(); // <-- needs the HF model download
    ...
    let req = NlpRequest { text, tasks: NlpTasks::ALL };
    let results = model.analyze(vec![req]).await.expect("analyze"); // would panic today
    ...
}
```

It is both `#[ignore]`d and gated behind `require_expensive_tests!()`
(model download), so it does not run by default. If it *were* run it
would panic at `.expect("analyze")` with the argmax error — i.e. the
existing test already encodes the right scenario; it just never
executes.

### Recommended regression guard (cheap, no model download)

`argmax_axis` is a private free function. A unit test inside
`nlp.rs`'s `#[cfg(test)] mod tests` pins the contract without any model
or ONNX runtime:

```rust
#[test]
fn argmax_axis_supports_per_row_argmax_used_by_dep_decode() {
    use ndarray::array;
    // arc_scores is [seq, seq]; DEP decode wants the best head per token
    // (per-row argmax = axis 1). Row 0 -> col 2, row 1 -> col 0.
    let arc_scores = array![[0.1_f32, 0.2, 0.9], [0.7, 0.3, 0.1]];

    // The correct axis for a 2-D matrix:
    assert_eq!(argmax_axis(&arc_scores, 1).unwrap(), vec![2, 0]);

    // The value passed at nlp.rs:257 must NOT be used — it errors:
    assert!(argmax_axis(&arc_scores, 2).is_err());
}
```

This test fails to compile/pass only if someone reintroduces an
unsupported axis at the call site (paired with un-`#[ignore]`-ing the
e2e test under the expensive-test job for full coverage).

---

## How to reproduce

### A) In this repo, end-to-end (needs model + ort)

```bash
cargo build -p uniko-bench --bin nlp-parity --no-default-features
./target/debug/nlp-parity            # errors on the first sentence
```

### B) Isolated regression test (this repo)

`crates/uniko-bench/tests/xervo_nlp_argmax_repro.rs` — a `#[tokio::test]`
(`#[ignore]`, model-gated) that resolves a `task = nlp` alias and calls
`analyze`. It asserts the *correct, post-fix* behavior, so today it fails
with the argmax error and after the upstream fix it passes:

```bash
cargo test -p uniko-bench --no-default-features \
  --test xervo_nlp_argmax_repro -- --ignored --nocapture
```

---

## Impact on uniko

The uniko↔xervo NLP A/B parity check (deciding whether uniko can drop its
in-crate tokenize+decode in favor of `NlpModel::analyze`) is **blocked**
until xervo ships the fix: xervo produces no output to diff against. No
uniko production code was changed — the migration is gated on a fixed
`uni-xervo` release.

Once a fixed version is available:

1. bump `uni-xervo` in the workspace,
2. run `nlp-parity` to get the six-axis parity report
   (tokenization / POS / NER / DEP / CLS / SRL),
3. proceed (or not) with the catalog + call-site swap based on the
   measured divergence.
