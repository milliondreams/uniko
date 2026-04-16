/// ID generation for uniko nodes.
///
/// **ADR-1:** All `*_id` fields use UUID v7 (time-sortable, monotonically increasing)
/// when not caller-provided. Exception: `chunk_id` uses deterministic `{parent_id}:{index}`
/// to enable idempotent re-chunking.
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
