//! Purity lint for the translation layer.
//!
//! The dialect translation code is the crate's sans-I/O core: request parsers
//! (`*/req.rs`), the items model, the canonical types, and the per-chunk
//! framer state machines must stay pure data transformation — no async
//! runtime, no HTTP client, no I/O types, no wall-clock, no randomness (ids
//! and time are supplied by the driver shells). This test fails `cargo test`
//! the moment a runtime token lands in a pure file, so the boundary is
//! enforced rather than aspirational.
//!
//! The three axum stream shells (`anthropic_stream_response`,
//! `openai_stream_response`, `responses_stream_response`) are the sanctioned
//! driver boundary: they are thin pumps and deliberately runtime-coupled.
//! Their regions (and the test-only `#[cfg(test)]` / `#[cfg(all(test,
//! feature = "axum"))]` modules) are excluded below.

use std::path::Path;

/// Files that must stay pure (paths relative to `src/`).
const PURE_FILES: &[&str] = &[
    "dialect/anthropic/req.rs",
    "dialect/anthropic/out.rs",
    "dialect/anthropic/stream.rs",
    "dialect/deflate.rs",
    "dialect/openai_chat/req.rs",
    "dialect/openai_chat/stream.rs",
    "dialect/openai_responses/req.rs",
    "dialect/openai_responses/out.rs",
    "dialect/openai_responses/stream.rs",
    "items.rs",
    "canonical.rs",
];

/// Forbidden outside the sanctioned driver regions: async runtime, HTTP
/// client/server types, I/O traits, wall clock, randomness, database.
const FORBIDDEN: &[&str] = &[
    "tokio",
    "reqwest",
    "axum",
    "sqlx",
    "uuid",
    "SystemTime",
    "Instant",
    "Utc::now",
    "rand::",
];

/// Lines inside these markers are sanctioned driver/factory regions — the
/// pump, id/clock minting, and `sequence_number` stamping. Their bodies are
/// excluded below; everything outside them must stay pure.
const SANCTIONED_STARTS: &[&str] = &[
    "pub fn anthropic_stream_response",
    "pub fn openai_stream_response",
    "pub fn responses_stream_response",
    "pub fn full(",
    // id minting is sanctioned exactly like ChatResponse::full
    "pub fn responses_body_from_chat(",
    "fn open_text_item(",
    "fn open_reasoning_item(",
    "pub struct ResponsesFramer {",
    "pub struct OpenAiFramer {",
];

#[test]
fn dialect_translation_layer_stays_pure() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations: Vec<String> = Vec::new();

    for rel in PURE_FILES {
        let path = src_dir.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let lines: Vec<&str> = src.lines().collect();

        let mut sanctioned: std::collections::HashSet<usize> = Default::default();
        for (i, line) in lines.iter().enumerate() {
            if SANCTIONED_STARTS.iter().any(|s| line.contains(s)) {
                let mut d = 0i64;
                let mut started = false;
                for (j, l) in lines.iter().enumerate().skip(i) {
                    d += l.matches('{').count() as i64 - l.matches('}').count() as i64;
                    if d > 0 {
                        started = true;
                    }
                    if started && d <= 0 {
                        for k in i..=j {
                            sanctioned.insert(k);
                        }
                        break;
                    }
                }
            }
        }
        let mut tests: std::collections::HashSet<usize> = Default::default();
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim();
            let is_test = t.starts_with("#[cfg(test)]")
                || t.starts_with("#[cfg(all(test")
                || t.starts_with("#[test]")
                || t.starts_with("#[tokio::test]");
            if is_test {
                let mut d = 0i64;
                let mut started = false;
                for (j, l) in lines.iter().enumerate().skip(i + 1) {
                    d += l.matches('{').count() as i64 - l.matches('}').count() as i64;
                    if d > 0 {
                        started = true;
                    }
                    if started && d <= 0 {
                        for k in i..=j {
                            tests.insert(k);
                        }
                        break;
                    }
                }
            }
        }

        for (i, line) in lines.iter().enumerate() {
            if sanctioned.contains(&i) || tests.contains(&i) {
                continue;
            }
            if line.trim().starts_with("//") {
                continue;
            }
            if line.trim().starts_with("#[cfg(") && line.contains("axum") {
                continue;
            }
            for tok in FORBIDDEN {
                if line.contains(tok) {
                    violations.push(format!(
                        "{rel}:{}: forbidden token `{tok}`: {}",
                        i + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "purity violations:\n{}",
        violations.join("\n")
    );
}
