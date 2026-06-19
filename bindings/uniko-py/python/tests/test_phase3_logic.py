"""Phase 3 tests: scopes, rules, assume, abduce."""

from __future__ import annotations

import uniko


async def test_recall_in_and_query_in_with_scope() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    await agent.session("s1").observe(uniko.Turn("alice", "alpha notes about rockets"))
    await agent.session("s2").observe(uniko.Turn("alice", "beta notes about gardening"))

    bundle = await agent.recall_in("rockets", uniko.Scope().sessions(["s1"]))
    assert isinstance(bundle.items, list)

    rows = await agent.query_in(
        "MATCH (n) RETURN n LIMIT 3", uniko.Scope().sessions(["s1"])
    )
    assert isinstance(rows, list)


async def test_define_and_run_rule() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    nid = await agent.define_rule(
        "py_rule", "CREATE RULE py_rule AS MATCH (n:Episode) YIELD KEY n"
    )
    assert isinstance(nid, int)
    assert nid > 0

    rows = await agent.run_rule("py_rule", ["n"])
    assert isinstance(rows, list)


async def test_assume_is_hypothetical_and_rolls_back() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")

    rows = await (
        agent.assume(
            "ASSUME { CREATE (:Fact {fact_id: 'srv-port', subject: 'srv', "
            "predicate: 'port', object: '9090'}) }"
        )
        .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .run()
    )
    # The hypothetical fact is visible inside the assume body.
    assert isinstance(rows, list)
    assert len(rows) == 1

    # ...but the mutation must not survive into the real graph.
    after = await agent.query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
    assert after == []


async def test_abduce_returns_modifications() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    await agent.define_rule(
        "reachable", "CREATE RULE reachable AS MATCH (a:Episode) YIELD KEY a"
    )
    result = await agent.abduce("ABDUCE reachable WHERE a.kind = 'query'")
    # ABDUCE yields ranked, validated graph modifications.
    assert isinstance(result.modifications, list)
    for mod in result.modifications:
        assert "modification" in mod
        assert isinstance(mod["validated"], bool)
        assert isinstance(mod["cost"], float)
