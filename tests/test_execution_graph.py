"""Tests for execution graph rendering and loading."""

from pathlib import Path

from chronicle.execution_graph import ExecutionGraph

TRACE_DIR = Path(__file__).parent.parent / "fixtures" / "traces" / "deletion-incident-001"


def test_execution_graph_loads_trace():
    graph = ExecutionGraph.load(TRACE_DIR)
    assert graph.trace_id == "trace-deletion-incident-001"
    assert len(graph.timeline()) == 3


def test_execution_graph_parent_chain():
    graph = ExecutionGraph.load(TRACE_DIR)
    timeline = graph.timeline()
    assert timeline[0].node_id == "agent"
    assert timeline[1].node_id == "delete_file"
    assert timeline[1].parent_envelope_id == timeline[0].envelope_id
    assert timeline[2].parent_envelope_id == timeline[1].envelope_id


def test_execution_graph_mermaid():
    graph = ExecutionGraph.load(TRACE_DIR)
    mermaid = graph.to_mermaid()
    assert "agent@1" in mermaid
    assert "delete_file@1" in mermaid
    assert "-->" in mermaid


def test_parent_calls_same_subagent_twice_waterfall():
    """Same sub-agent twice under one parent; each has nested llm + tool spans."""
    graph = ExecutionGraph.load(
        Path(__file__).parent.parent
        / "fixtures"
        / "traces"
        / "parent-calls-subagent-twice"
    )
    timeline = graph.timeline()
    orch = next(e for e in timeline if e.node_id == "orchestrator")
    researchers = [e for e in timeline if e.parent_envelope_id == orch.envelope_id]
    assert [(e.node_id, e.invocation_index) for e in researchers] == [
        ("researcher", 1),
        ("researcher", 2),
    ]
    for r in researchers:
        kids = sorted(
            (e for e in timeline if e.parent_envelope_id == r.envelope_id),
            key=lambda e: e.sequence,
        )
        assert [e.boundary_kind for e in kids] == ["llm", "tool"]
        assert [e.node_id for e in kids] == ["llm", "web_search"]

    tree = graph.to_otel_tree()
    assert "orchestrator#1" in tree
    assert "researcher#1" in tree
    assert "researcher#2" in tree
    assert "llm#1" in tree and "llm#2" in tree
    assert "web_search#1" in tree and "web_search#2" in tree
    waterfall = graph.to_otel_waterfall()
    assert "█" in waterfall
    assert "llm#1" in waterfall and "web_search#2" in waterfall
