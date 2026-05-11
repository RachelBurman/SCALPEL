"""
Assembles the SCALPEL agent as a LangGraph StateGraph.

The graph has three nodes and one decision point:

    [planner] ──tool call──► [tools] ──► back to planner
         │
         └──no tool call──► [synthesise] ──► END

MemorySaver provides in-memory checkpointing so the graph can resume a
conversation across multiple .invoke() calls within the same process
(used later for multi-turn agent sessions in Phase 3+).
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from scalpel.agent.state import AgentState
from scalpel.agent.nodes import planner, synthesise, route_after_planner, tool_node


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("tools", tool_node)
    builder.add_node("synthesise", synthesise)

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "synthesise": "synthesise"},
    )
    builder.add_edge("tools", "planner")
    builder.add_edge("synthesise", END)

    return builder.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    """Return the compiled graph, building it once on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
