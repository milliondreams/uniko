# Summary

<!-- Describe what this PR changes and why. -->

## Type of change

<!-- Check all that apply. -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would change existing behavior)
- [ ] Refactor (no functional change)
- [ ] Documentation
- [ ] CI / build / tooling

## Checklist

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo nextest run` passes
- [ ] `cargo deny check` passes (licenses / advisories)
- [ ] uni-db seal respected (product crates do not `use uni_db` or call `.db()` directly — access goes through `uniko-store`)
- [ ] Documentation updated where relevant

## Linked issues

<!-- e.g. Closes #123, Fixes #456 -->
