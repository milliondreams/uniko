//! Filter predicates for graph queries.
//!
//! [`Filter`] translates to parameterized Cypher `WHERE` clauses, preventing
//! injection by using numbered `$p{N}` parameters for all values.

// Rust guideline compliant

use uni_db::Value;

use super::validate_property_name;
use crate::error::Result;

/// Predicate tree for filtering nodes or edges in queries.
///
/// Converted to Cypher via [`Filter::to_cypher`].
#[derive(Debug, Clone)]
pub enum Filter {
    /// `{property} = $val`
    Eq(String, Value),
    /// `{property} <> $val`
    Ne(String, Value),
    /// `{property} > $val`
    Gt(String, Value),
    /// `{property} >= $val`
    Gte(String, Value),
    /// `{property} < $val`
    Lt(String, Value),
    /// `{property} <= $val`
    Lte(String, Value),
    /// `{property} IN [$vals]`
    In(String, Vec<Value>),
    /// `{property} IS NULL`
    IsNull(String),
    /// `{property} IS NOT NULL`
    IsNotNull(String),
    /// All children must match.
    And(Vec<Filter>),
    /// At least one child must match.
    Or(Vec<Filter>),
    /// Negation.
    Not(Box<Filter>),
}

/// Output from [`Filter::to_cypher`].
pub(crate) struct CypherFilter {
    /// Cypher boolean expression fragment (e.g. `n.status = $p0`).
    pub fragment: String,
    /// Parameter bindings consumed by the fragment.
    pub params: Vec<(String, Value)>,
    /// Next unused parameter index.
    pub next_offset: usize,
}

impl Filter {
    /// Compile the filter into a Cypher boolean expression.
    ///
    /// `var` is the Cypher variable to match against (e.g. `"n"`).
    /// `offset` is the starting index for generated parameter names.
    ///
    /// # Errors
    ///
    /// Returns an error if any property name is invalid.
    pub(crate) fn to_cypher(&self, var: &str, offset: usize) -> Result<CypherFilter> {
        match self {
            Self::Eq(prop, val) => scalar_op(var, prop, "=", val, offset),
            Self::Ne(prop, val) => scalar_op(var, prop, "<>", val, offset),
            Self::Gt(prop, val) => scalar_op(var, prop, ">", val, offset),
            Self::Gte(prop, val) => scalar_op(var, prop, ">=", val, offset),
            Self::Lt(prop, val) => scalar_op(var, prop, "<", val, offset),
            Self::Lte(prop, val) => scalar_op(var, prop, "<=", val, offset),
            Self::In(prop, vals) => {
                validate_property_name(prop)?;
                let param = format!("p{offset}");
                let fragment = format!("{var}.{prop} IN ${param}");
                let list = Value::List(vals.clone());
                Ok(CypherFilter {
                    fragment,
                    params: vec![(param, list)],
                    next_offset: offset + 1,
                })
            }
            Self::IsNull(prop) => {
                validate_property_name(prop)?;
                Ok(CypherFilter {
                    fragment: format!("{var}.{prop} IS NULL"),
                    params: vec![],
                    next_offset: offset,
                })
            }
            Self::IsNotNull(prop) => {
                validate_property_name(prop)?;
                Ok(CypherFilter {
                    fragment: format!("{var}.{prop} IS NOT NULL"),
                    params: vec![],
                    next_offset: offset,
                })
            }
            Self::And(children) => compound(var, children, " AND ", offset),
            Self::Or(children) => compound(var, children, " OR ", offset),
            Self::Not(inner) => {
                let child = inner.to_cypher(var, offset)?;
                Ok(CypherFilter {
                    fragment: format!("NOT ({})", child.fragment),
                    params: child.params,
                    next_offset: child.next_offset,
                })
            }
        }
    }
}

fn scalar_op(
    var: &str,
    prop: &str,
    op: &str,
    val: &Value,
    offset: usize,
) -> Result<CypherFilter> {
    validate_property_name(prop)?;
    let param = format!("p{offset}");
    let fragment = format!("{var}.{prop} {op} ${param}");
    Ok(CypherFilter {
        fragment,
        params: vec![(param, val.clone())],
        next_offset: offset + 1,
    })
}

fn compound(
    var: &str,
    children: &[Filter],
    joiner: &str,
    mut offset: usize,
) -> Result<CypherFilter> {
    if children.is_empty() {
        return Ok(CypherFilter {
            fragment: "true".into(),
            params: vec![],
            next_offset: offset,
        });
    }
    let mut fragments = Vec::with_capacity(children.len());
    let mut params = Vec::new();
    for child in children {
        let c = child.to_cypher(var, offset)?;
        fragments.push(c.fragment);
        params.extend(c.params);
        offset = c.next_offset;
    }
    let fragment = if fragments.len() == 1 {
        fragments.into_iter().next().unwrap()
    } else {
        format!("({})", fragments.join(joiner))
    };
    Ok(CypherFilter {
        fragment,
        params,
        next_offset: offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_filter() {
        let f = Filter::Eq("status".into(), Value::String("active".into()));
        let c = f.to_cypher("n", 0).unwrap();
        assert_eq!(c.fragment, "n.status = $p0");
        assert_eq!(c.params.len(), 1);
        assert_eq!(c.next_offset, 1);
    }

    #[test]
    fn test_and_filter() {
        let f = Filter::And(vec![
            Filter::Eq("kind".into(), Value::String("agent".into())),
            Filter::Gt("frequency".into(), Value::Int(5)),
        ]);
        let c = f.to_cypher("n", 0).unwrap();
        assert_eq!(c.fragment, "(n.kind = $p0 AND n.frequency > $p1)");
        assert_eq!(c.params.len(), 2);
        assert_eq!(c.next_offset, 2);
    }

    #[test]
    fn test_not_filter() {
        let f = Filter::Not(Box::new(Filter::IsNull("embedding".into())));
        let c = f.to_cypher("n", 0).unwrap();
        assert_eq!(c.fragment, "NOT (n.embedding IS NULL)");
    }

    #[test]
    fn test_invalid_property_rejected() {
        let f = Filter::Eq("bad prop!".into(), Value::Int(1));
        assert!(f.to_cypher("n", 0).is_err());
    }
}
