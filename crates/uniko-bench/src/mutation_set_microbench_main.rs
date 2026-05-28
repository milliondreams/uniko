//! Persistent-Lance microbench discriminating which Entity index drives
//! the 17 ms/row cost in MutationSetExec observed in production.
//!
//! Uses SchemaBuilder (Rust API) since Cypher DDL doesn't expose vector
//! or Hash-scalar index types. Variants progressively add the indexes
//! present on production Entity until per-row cost matches.
//!
//! Production Entity indexes (uniko-store/schema/entities.rs):
//!   entity_id: Hash, name: Hash + FullText, entity_type: Hash,
//!   unstable: Hash, embedding: Vector(HnswSq m=16, ef=100, cosine).
//!
//! Run: cargo run -p uniko-bench --release --bin mutation-set-microbench

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{
    DataType, IndexType, ScalarType, Uni, Value, VectorAlgo, VectorIndexCfg, VectorMetric,
};

use uniko_bench::now_value;

#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    hash_indexes: bool,
    fulltext_name: bool,
    vector_embedding: bool,
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "noindex",
        hash_indexes: false,
        fulltext_name: false,
        vector_embedding: false,
    },
    Variant {
        name: "hash_only",
        hash_indexes: true,
        fulltext_name: false,
        vector_embedding: false,
    },
    Variant {
        name: "fulltext_name",
        hash_indexes: true,
        fulltext_name: true,
        vector_embedding: false,
    },
    Variant {
        name: "vector_only",
        hash_indexes: false,
        fulltext_name: false,
        vector_embedding: true,
    },
    Variant {
        name: "production",
        hash_indexes: true,
        fulltext_name: true,
        vector_embedding: true,
    },
];

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    for v in VARIANTS {
        let tmp = tempfile::tempdir()?;
        let db = Uni::open(tmp.path().to_string_lossy().to_string())
            .build()
            .await?;
        run_variant(*v, &db).await?;
        println!();
    }
    Ok(())
}

async fn run_variant(v: Variant, db: &Uni) -> anyhow::Result<()> {
    println!("══════ variant: {} ══════", v.name);
    let mut builder = db
        .schema()
        .label("Entity")
        .property("entity_id", DataType::String)
        .property("name", DataType::String)
        .property_nullable("entity_type", DataType::String)
        .property_nullable("first_seen", DataType::DateTime)
        .property_nullable("last_seen", DataType::DateTime)
        .property_nullable("frequency", DataType::Int64)
        .property_nullable("confidence", DataType::Float64)
        .property_nullable("invalidation_count", DataType::Int64)
        .property_nullable("last_invalidation_at", DataType::DateTime)
        .property_nullable("unstable", DataType::Bool)
        .property_nullable("embedding", DataType::Vector { dimensions: 768 });
    if v.hash_indexes {
        builder = builder
            .index("entity_id", IndexType::Scalar(ScalarType::Hash))
            .index("name", IndexType::Scalar(ScalarType::Hash))
            .index("entity_type", IndexType::Scalar(ScalarType::Hash))
            .index("unstable", IndexType::Scalar(ScalarType::Hash));
    }
    if v.fulltext_name {
        builder = builder.index("name", IndexType::FullText);
    }
    if v.vector_embedding {
        builder = builder.index(
            "embedding",
            IndexType::Vector(VectorIndexCfg {
                algorithm: VectorAlgo::HnswSq {
                    m: 16,
                    ef_construction: 100,
                    partitions: None,
                },
                metric: VectorMetric::Cosine,
                embedding: None,
            }),
        );
    }
    builder.done().apply().await?;

    let session = db.session();

    // Insert 4000 entities with embeddings
    const N: usize = 4000;
    let vids: Vec<i64> = {
        let tx = session.tx().await?;
        let mut rows = Vec::with_capacity(N);
        for i in 0..N {
            let mut h = HashMap::new();
            h.insert("entity_id".into(), Value::String(format!("e:{i}")));
            h.insert("name".into(), Value::String(format!("entity_{i}")));
            h.insert("entity_type".into(), Value::String("person".into()));
            h.insert("first_seen".into(), now_value());
            h.insert("last_seen".into(), now_value());
            h.insert("frequency".into(), Value::Int(1));
            h.insert("confidence".into(), Value::Float(0.5));
            h.insert("invalidation_count".into(), Value::Int(0));
            h.insert("unstable".into(), Value::Bool(false));
            if v.vector_embedding {
                h.insert(
                    "embedding".into(),
                    Value::Vector((0..768).map(|k| (k % 13) as f32 / 13.0).collect()),
                );
            }
            rows.push(h);
        }
        let r = tx.bulk_insert_vertices("Entity", rows).await?;
        tx.commit().await?;
        r.iter().map(|v| v.as_u64() as i64).collect()
    };

    const UPDATE: &str = "\
        UNWIND $updates AS u \
        MATCH (n:Entity) WHERE id(n) = u.nid \
        SET n.frequency = u.new_frequency, \
            n.last_seen = $now, \
            n.confidence = u.new_confidence";

    println!(
        "{:>6} {:>10} {:>10} {:>16} {:>16}",
        "batch", "wall_ms", "ms/row", "MutSet_cum_ms", "MutSet_ms/row"
    );
    for &batch in &[1usize, 3, 10, 100] {
        let updates: Vec<Value> = vids[..batch]
            .iter()
            .enumerate()
            .map(|(i, &nid)| {
                let mut m = HashMap::new();
                m.insert("nid".into(), Value::Int(nid));
                m.insert("new_frequency".into(), Value::Int((i as i64) + 2));
                m.insert("new_confidence".into(), Value::Float(0.7));
                Value::Map(m)
            })
            .collect();
        let mut walls = Vec::new();
        let mut muts = Vec::new();
        for _ in 0..5 {
            let tx = session.tx().await?;
            let t = Instant::now();
            let (_r, prof) = tx
                .execute_with(UPDATE)
                .param("updates", Value::List(updates.clone()))
                .param("now", now_value())
                .profile()
                .await?;
            let wall = t.elapsed().as_secs_f64() * 1000.0;
            tx.commit().await?;
            walls.push(wall);
            let m = prof
                .runtime_stats
                .iter()
                .find(|o| o.operator == "MutationSetExec")
                .map(|o| o.time_ms)
                .unwrap_or(0.0);
            muts.push(m);
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let w = med(walls);
        let m = med(muts);
        println!(
            "{:>6} {:>10.2} {:>10.4} {:>16.2} {:>16.4}",
            batch,
            w,
            w / batch as f64,
            m,
            m / batch as f64
        );
    }
    Ok(())
}
