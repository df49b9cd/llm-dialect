# Publishing to crates.io

Publishing is automated via [publish.yml](workflows/publish.yml), which runs on
version tags and publishes to crates.io via
[Trusted Publishing](https://crates.io/docs/trusted-publishing) (OIDC, no stored API
tokens).

## Trusted-publishing registration (one-time, done)

The crate page → **Settings → Trusted Publishing → Add trusted publisher**:

| Field              | Value          |
| ------------------ | -------------- |
| Repository owner   | `df49b9cd`     |
| Repository name    | `llm-dialect`  |
| Workflow filename  | `publish.yml`  |
| Environment        | *(leave empty)*|

`v0.1.0` predates this registration (it was published manually with `cargo login` +
`cargo publish`), so its tag-push run fails the OIDC exchange at
`rust-lang/crates-io-auth-action`. From `v0.1.1` on, tagging is sufficient.

## Releasing a new version

1.  Bump `version` in [../Cargo.toml](../Cargo.toml) (keep the README install snippet in
    sync) and merge to `master`.
2.  Tag the merge commit and push:

        git tag v0.1.1 && git push origin v0.1.1

3.  The **publish** workflow runs: full CI checks (`cargo check` across versions and
    feature sets, tests, doc tests), verifies the tag matches the manifest version,
    refuses to republish an existing version, then `cargo publish --all-features` via a
    short-lived OIDC token. The action revokes the token when the job ends.
