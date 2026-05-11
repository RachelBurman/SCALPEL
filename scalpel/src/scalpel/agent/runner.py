"""
Entry point for running the SCALPEL agent.

stream_run() is a generator that yields events as the agent works.
Each event is a dict like {"planner": {...}} or {"tools": {...}}.
The CLI iterates over these to display live progress, then reads the
final report from the last synthesise event.
"""

from langchain_core.messages import HumanMessage

from scalpel.agent.graph import get_graph
from scalpel.agent.state import AgentState


def stream_run(goal: str, thread_id: str = "default"):
    """
    Run the agent and yield graph events as they happen.

    Each yielded value is a dict mapping a node name to its state update.
    Callers can inspect the events to display progress, then read
    event["synthesise"]["report"] from the final event for the result.

    Args:
        goal: The research question or instruction for the agent.
        thread_id: Identifies the conversation for checkpointing.
                   Use a unique value per session to keep histories separate.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "goal": goal,
        "messages": [HumanMessage(content=goal)],
        "iteration": 0,
        "report": "",
    }

    yield from graph.stream(initial_state, config=config)
