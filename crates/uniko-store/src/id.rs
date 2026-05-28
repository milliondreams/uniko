/// ID generation for uniko nodes.
///
/// **ADR-1:** All `*_id` fields use UUID v7 (time-sortable, monotonically increasing)
/// when not caller-provided. Exception: `chunk_id` uses deterministic `{parent_id}:{index}`
/// to enable idempotent re-chunking. Deterministic IDs derived from content
/// (procedure / topic / entity) use [`stable_hex64`].
use sha2::Digest;
use uuid::Uuid;

/// Generate a new UUID v7 (time-sortable, monotonically increasing).
///
/// Used for all `*_id` fields when not caller-provided. Returns lowercase
/// hyphenated format: `"xxxxxxxx-xxxx-7xxx-yxxx-xxxxxxxxxxxx"`.
pub fn new_id() -> String {
    Uuid::now_v7().to_string()
}

/// Generate a deterministic chunk ID: `{parent_id}:{index}`.
///
/// Enables idempotent re-chunking — the same parent and index always produce
/// the same chunk ID, so re-processing a document doesn't create duplicates.
pub fn chunk_id(parent_id: &str, index: usize) -> String {
    format!("{parent_id}:{index}")
}

/// Validate that a string is a valid UUID v7.
///
/// Checks both that the string parses as a UUID and that its version field is 7.
pub fn is_valid_id(id: &str) -> bool {
    Uuid::parse_str(id)
        .map(|u| u.get_version_num() == 7)
        .unwrap_or(false)
}

/// Streaming SHA-256 hasher wrapper used by [`stable_hex64`].
///
/// Exists so callers can feed bytes without importing the `sha2` crate
/// or the [`Digest`] trait. Internal hashing primitive is fixed at
/// SHA-256; the stable-id contract is that the first 64 bits of the
/// digest form the hex-encoded suffix.
pub struct StableHasher(sha2::Sha256);

impl StableHasher {
    /// Feed `data` into the hasher.
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
}

/// Build a deterministic, persistence-safe ID of the form `"{prefix}_{u64:016x}"`.
///
/// The closure controls exactly which bytes are fed (including any
/// separators) so distinct call sites can preserve their own framing
/// rules without colliding. The first 8 bytes of the SHA-256 digest
/// are interpreted as a big-endian `u64` and rendered as 16 hex digits.
///
/// SHA-256 was chosen over [`std::collections::hash_map::DefaultHasher`]
/// because the latter is not guaranteed stable across rustc versions,
/// and these IDs end up persisted on graph nodes.
///
/// # Examples
///
/// ```
/// use uniko_store::id::stable_hex64;
/// let id = stable_hex64("ent", |h| {
///     h.update(b"alice");
///     h.update(b"\x00");
///     h.update(b"person");
/// });
/// assert!(id.starts_with("ent_"));
/// assert_eq!(id.len(), "ent_".len() + 16);
/// ```
pub fn stable_hex64(prefix: &str, feed: impl FnOnce(&mut StableHasher)) -> String {
    let mut h = StableHasher(sha2::Sha256::new());
    feed(&mut h);
    let digest = h.0.finalize();
    let lead = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 digest is always >= 8 bytes"),
    );
    format!("{prefix}_{lead:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_new_id_uniqueness() {
        let ids: HashSet<String> = (0..1000).map(|_| new_id()).collect();
        assert_eq!(ids.len(), 1000, "generated IDs must be unique");
    }

    #[test]
    fn test_new_id_is_uuid_v7() {
        let id = new_id();
        let parsed = Uuid::parse_str(&id).expect("should parse as UUID");
        assert_eq!(parsed.get_version_num(), 7, "must be UUID v7");
    }

    #[test]
    fn test_new_id_monotonic() {
        let ids: Vec<String> = (0..100).map(|_| new_id()).collect();
        for window in ids.windows(2) {
            assert!(
                window[0] <= window[1],
                "IDs must be lexicographically ordered: {} > {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_chunk_id_format() {
        assert_eq!(chunk_id("abc", 3), "abc:3");
        assert_eq!(chunk_id("parent-uuid-here", 0), "parent-uuid-here:0");
        assert_eq!(chunk_id("x", 999), "x:999");
    }

    #[test]
    fn test_chunk_id_deterministic() {
        let a = chunk_id("parent", 5);
        let b = chunk_id("parent", 5);
        assert_eq!(a, b, "same inputs must produce same output");
    }

    #[test]
    fn test_stable_hex64_format() {
        let id = stable_hex64("proc", |h| {
            h.update(b"agent\x00a\x00b");
        });
        assert!(id.starts_with("proc_"), "expected proc_ prefix, got {id}");
        assert_eq!(id.len(), "proc_".len() + 16, "expected 16 hex chars");
        assert!(
            id[5..].chars().all(|c| c.is_ascii_hexdigit()),
            "expected lowercase hex suffix"
        );
    }

    #[test]
    fn test_stable_hex64_deterministic() {
        let feed = |h: &mut StableHasher| {
            h.update(b"alice");
            h.update(b"\x00");
            h.update(b"person");
        };
        let a = stable_hex64("ent", feed);
        let b = stable_hex64("ent", feed);
        assert_eq!(a, b, "same input must yield same id");
    }

    #[test]
    fn test_stable_hex64_distinct_inputs() {
        let a = stable_hex64("ent", |h| h.update(b"alice"));
        let b = stable_hex64("ent", |h| h.update(b"bob"));
        assert_ne!(a, b, "distinct inputs must (almost certainly) differ");
    }

    #[test]
    fn test_stable_hex64_separator_matters() {
        // Two-part hash with vs. without separator must differ —
        // otherwise call sites that rely on separators (procedures,
        // entities) could alias on inputs like ("ab", "c") vs ("a", "bc").
        let with_sep = stable_hex64("x", |h| {
            h.update(b"ab");
            h.update(b"\x00");
            h.update(b"c");
        });
        let without_sep = stable_hex64("x", |h| {
            h.update(b"ab");
            h.update(b"c");
        });
        assert_ne!(with_sep, without_sep);
    }

    #[test]
    fn test_is_valid_id() {
        let id = new_id();
        assert!(is_valid_id(&id), "generated ID must be valid");

        assert!(!is_valid_id("not-a-uuid"), "random string must be invalid");
        assert!(!is_valid_id(""), "empty string must be invalid");

        // UUID v4 should be rejected (wrong version)
        let v4 = uuid::Uuid::new_v4().to_string();
        assert!(!is_valid_id(&v4), "UUID v4 must be rejected");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_id_always_valid(_seed in 0u64..10000) {
            let id = new_id();
            prop_assert!(is_valid_id(&id), "generated ID must always be valid: {}", id);
        }

        #[test]
        fn proptest_chunk_id_no_collision(
            parent1 in "[a-z]{1,10}",
            parent2 in "[a-z]{1,10}",
            idx1 in 0usize..1000,
            idx2 in 0usize..1000,
        ) {
            if parent1 != parent2 || idx1 != idx2 {
                let a = chunk_id(&parent1, idx1);
                let b = chunk_id(&parent2, idx2);
                prop_assert_ne!(a, b, "different inputs must produce different chunk IDs");
            }
        }
    }
}
