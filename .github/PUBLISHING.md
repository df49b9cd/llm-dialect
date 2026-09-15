# Publishing to crates.io

Publishing is fully automated via [publish.yml](workflows/publish.yml), which runs on
version tags and publishes to crates.io using
[Trusted Publishing](https://crates.io/docs/trusted-publishing) (OIDC, no stored API
tokens).

## One-time setup on crates.io

The crate `llm-dialect` does not exist yet on crates.io, so the first publish must be
manual; trusted publishing is then attached to the released crate.

1.  Authenticate locally:

        cargo login

2.  Publish `0.1.0` from the commit tagged `v0.1.0` (verify everything first):

        cargo publish --all-features --dry-run
        cargo publish --all-features

3.  Open https://crates.io/crates/llm-dialect/settings → **Trusted Publishing** →
    **Add trusted publisher** and enter:

    | Field              | Value          |
    | ------------------ | -------------- |
    | Repository owner   | `df49b9cd`     |
    | Repository name    | `llm-dialect`  |
    | Workflow filename  | `publish.yml`  |
    | Environment        | *(leave empty)*|

## Releasing a new version

1.  Bump `version` in [../Cargo.toml](../Cargo.toml) (keep the README install snippet in
    sync) and merge to `master`.
2.  Tag the merge commit and push:

        git tag v0.2.0 && git push origin v0.2.0

3.  The **publish** workflow runs: full CI checks (`cargo check` across versions and
    feature sets, tests, doc tests), verifies the tag matches the manifest version, then
    `cargo publish --all-features` via a short-lived OIDC token. The action revokes the
    token when the job ends.
