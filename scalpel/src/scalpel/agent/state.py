"""
AgentState — the single shared data structure that flows through the graph.

Every node reads from this state and returns a partial update. LangGraph
merges the updates automatically. The `messages` field uses the add_messages
reducer, which means new messages are *appended* rather than replacing the
list — so every tool call and response accumulates across the full run.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    goal: str                                          # Original user goal — never changes
    messages: Annotated[list[BaseMessage], add_messages]  # Accumulating conversation
    iteration: int                                     # Loop counter for the safety guard
    report: str                                        # Final output written by synthesise
