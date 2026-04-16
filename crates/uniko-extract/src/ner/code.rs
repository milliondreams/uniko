//! Code entity extraction using tree-sitter AST parsing.
//!
//! Extracts function, class, struct, and import names from source code
//! with high confidence (0.9) since AST parsing is deterministic.

// Rust guideline compliant

#![cfg(feature = "code-parse")]

use tree_sitter::Parser;

use super::types::{EntityType, ExtractionSource, RawEntity};

/// Extract code entities from source content.
///
/// Walks the full AST (all nesting levels) to find function/method
/// definitions, class/struct/enum definitions, and import statements.
pub fn extract_code_entities(content: &str, language: &str) -> Vec<RawEntity> {
    let mut parser = match parser_for_language(language) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let mut entities = Vec::new();
    walk_node(tree.root_node(), content, language, &mut entities);
    entities
}

/// Initialize a tree-sitter parser for the given language.
fn parser_for_language(lang: &str) -> Option<Parser> {
    let language_fn = match lang {
        "python" | "py" => tree_sitter_python::LANGUAGE,
        "rust" | "rs" => tree_sitter_rust::LANGUAGE,
        "javascript" | "js" => tree_sitter_javascript::LANGUAGE,
        "typescript" | "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX,
        _ => return None,
    };
    let mut parser = Parser::new();
    parser.set_language(&language_fn.into()).ok()?;
    Some(parser)
}

/// Recursively walk the AST collecting named declarations and imports.
fn walk_node(
    node: tree_sitter::Node<'_>,
    source: &str,
    language: &str,
    entities: &mut Vec<RawEntity>,
) {
    if let Some(raw) = classify_node(node, source, language) {
        entities.push(raw);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_node(child, source, language, entities);
    }
}

/// Classify a single AST node, returning a [`RawEntity`] if it is a
/// named declaration or import.
fn classify_node(
    node: tree_sitter::Node<'_>,
    source: &str,
    language: &str,
) -> Option<RawEntity> {
    let kind = node.kind();
    let (entity_type, name) = match (language, kind) {
        // Python
        (_, "function_definition" | "decorated_definition") => {
            (EntityType::CodeSymbol, extract_name(node, source)?)
        }
        (_, "class_definition") => (EntityType::CodeSymbol, extract_name(node, source)?),
        ("python" | "py", "import_statement" | "import_from_statement") => {
            (EntityType::CodeImport, extract_import_name(node, source)?)
        }
        // Rust
        (_, "function_item") => (EntityType::CodeSymbol, extract_name(node, source)?),
        (_, "struct_item") => (EntityType::CodeSymbol, extract_name(node, source)?),
        (_, "enum_item") => (EntityType::CodeSymbol, extract_name(node, source)?),
        (_, "trait_item") => (EntityType::CodeSymbol, extract_name(node, source)?),
        ("rust" | "rs", "use_declaration") => {
            (EntityType::CodeImport, extract_use_path(node, source)?)
        }
        // JS/TS
        (_, "function_declaration") => (EntityType::CodeSymbol, extract_name(node, source)?),
        (_, "class_declaration") => (EntityType::CodeSymbol, extract_name(node, source)?),
        (_, "method_definition") => (EntityType::CodeSymbol, extract_name(node, source)?),
        ("javascript" | "js" | "typescript" | "ts" | "tsx", "import_statement") => {
            (EntityType::CodeImport, extract_import_source(node, source)?)
        }
        _ => return None,
    };

    Some(RawEntity {
        surface_form: name.clone(),
        canonical_name: name, // code symbols are case-sensitive
        entity_type,
        confidence: 0.9,
        source: ExtractionSource::CodeAst,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
    })
}

/// Extract the `identifier`, `type_identifier`, or `name` child from
/// a declaration node.
fn extract_name(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        // Rust uses `type_identifier` for struct/enum/trait names.
        if matches!(child.kind(), "identifier" | "name" | "type_identifier") {
            return Some(source[child.start_byte()..child.end_byte()].to_string());
        }
        // For decorated definitions (Python), look one level deeper.
        if child.kind() == "function_definition" || child.kind() == "class_definition" {
            return extract_name(child, source);
        }
    }
    None
}

/// Extract the imported module name from a Python import statement.
fn extract_import_name(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "dotted_name" | "aliased_import" | "module_name" => {
                let text = source[child.start_byte()..child.end_byte()].to_string();
                // For "from os.path import join", get "os.path".
                return Some(text.split(" as ").next().unwrap_or(&text).trim().to_string());
            }
            _ => {}
        }
    }
    // Fallback: grab the text after "import " keyword.
    let text = &source[node.start_byte()..node.end_byte()];
    text.strip_prefix("import ")
        .or_else(|| text.split("import ").nth(1))
        .map(|s| s.split(" as ").next().unwrap_or(s).trim().to_string())
}

/// Extract the crate/module path from a Rust `use` declaration.
fn extract_use_path(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let text = &source[node.start_byte()..node.end_byte()];
    // "use tokio::sync::mpsc;" → "tokio"
    let path = text
        .strip_prefix("use ")?
        .trim_end_matches(';')
        .trim();
    let root = path.split("::").next()?;
    Some(root.to_string())
}

/// Extract the source string from a JS/TS import statement.
fn extract_import_source(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "string" || child.kind() == "string_literal" {
            let text = &source[child.start_byte()..child.end_byte()];
            return Some(text.trim_matches(|c| c == '\'' || c == '"').to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_functions() {
        let code = "def hello():\n    pass\n\ndef world():\n    pass\n";
        let ents = extract_code_entities(code, "python");
        let names: Vec<&str> = ents
            .iter()
            .filter(|e| e.entity_type == EntityType::CodeSymbol)
            .map(|e| e.canonical_name.as_str())
            .collect();
        assert!(names.contains(&"hello"));
        assert!(names.contains(&"world"));
    }

    #[test]
    fn test_python_class() {
        let code = "class AuthService:\n    def login(self):\n        pass\n";
        let ents = extract_code_entities(code, "python");
        assert!(ents
            .iter()
            .any(|e| e.canonical_name == "AuthService"
                && e.entity_type == EntityType::CodeSymbol));
    }

    #[test]
    fn test_python_import() {
        let code = "import numpy\nfrom os import path\n";
        let ents = extract_code_entities(code, "python");
        let imports: Vec<&str> = ents
            .iter()
            .filter(|e| e.entity_type == EntityType::CodeImport)
            .map(|e| e.canonical_name.as_str())
            .collect();
        assert!(imports.contains(&"numpy"));
    }

    #[test]
    fn test_rust_struct_and_fn() {
        let code = "struct Config { x: i32 }\nfn process() -> i32 { 42 }\n";
        let ents = extract_code_entities(code, "rust");
        assert!(ents.iter().any(|e| e.canonical_name == "Config"));
        assert!(ents.iter().any(|e| e.canonical_name == "process"));
    }

    #[test]
    fn test_rust_use() {
        let code = "use tokio::sync::mpsc;\n";
        let ents = extract_code_entities(code, "rust");
        assert!(ents
            .iter()
            .any(|e| e.canonical_name == "tokio"
                && e.entity_type == EntityType::CodeImport));
    }

    #[test]
    fn test_js_function_and_class() {
        let code = "function handler() {}\nclass Component {}\n";
        let ents = extract_code_entities(code, "javascript");
        assert!(ents.iter().any(|e| e.canonical_name == "handler"));
        assert!(ents.iter().any(|e| e.canonical_name == "Component"));
    }

    #[test]
    fn test_unsupported_language() {
        let ents = extract_code_entities("PERFORM MOVE.", "cobol");
        assert!(ents.is_empty());
    }
}
