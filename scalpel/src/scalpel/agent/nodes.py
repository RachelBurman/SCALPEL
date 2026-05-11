"""
Graph nodes and routing logic for the SCALPEL agent.

Three nodes:
  planner   — LLM decides what to do next (may emit tool calls or a final answer)
  tools     — LangGraph's ToolNode executes whatever tool the planner called
  synthesise — extracts the final answer and writes the report

One routing function sits between planner and the rest:
  - tool call in last message → go to tools → back to planner
  - no tool call (or iteration limit hit) → go to synthesise → END
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from scalpel.agent.state import AgentState
from scalpel.agent.tools import SCALPEL_TOOLS
from scalpel.config import settings

MAX_ITERATIONS = 15

_SYSTEM_PROMPT = """\
You are SCALPEL's autonomous research agent. Your purpose is to investigate \
scientific literature and investment claims systematically using evidence-based reasoning.

You have access to a library of indexed academic papers and the following tools:
- list_papers: See what is available before you start
- search_papers: Semantic search across all indexed papers
- summarize_paper / extract_claims: Understand what a paper argues
- critique_methodology / critique_statistics: Evaluate research quality in depth
- bullshit_score_paper: Score overall scientific rigour (0 = excellent, 10 = poor)
- cross_reference_company: Check if research supports or contradicts company claims
- get_retrieval_params: Get optimised retrieval settings for a specific query

Work through the goal step by step:
1. Call list_papers first to orient yourself
2. Search for relevant evidence with targeted queries
3. Drill into specific papers that look relevant
4. Cross-reference findings across the library where useful
5. When you have enough evidence to answer the goal thoroughly, write your \
complete final analysis as a structured markdown report — do not call any more \
tools after that

Your final message must be the full report. Do not end with "I am done" — end \
with the actual analysis.\
"""


_llm_with_tools = None


def _get_llm_with_tools():
    """Build and cache the LLM with tools bound. Created once per process."""
    global _llm_with_tools
    if _llm_with_tools is not None:
        return _llm_with_tools

    if settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_model,
            temperature=settings.model_temperature,
        )
    else:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.model_temperature,
        )

    _llm_with_tools = llm.bind_tools(SCALPEL_TOOLS)
    return _llm_with_tools


def planner(state: AgentState) -> dict:
    """
    The brain of the agent. Calls the LLM with the full message history and
    the tool list. The LLM either:
      - returns an AIMessage with tool_calls → the router sends us to tools
      - returns a plain AIMessage → the router sends us to synthesise
    """
    messages = state["messages"]

    # Prepend system prompt if this is the first call
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(messages)

    response = _get_llm_with_tools().invoke(messages)

    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def synthesise(state: AgentState) -> dict:
    """
    Terminal node. Finds the agent's final text response (the last AIMessage
    that is not a tool call) and sets it as the report.

    If the final message is too short (the LLM said "done" instead of writing
    a report), makes one more LLM call to produce a proper summary.
    """
    report = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            report = msg.content
            break

    # Safety net: if the model didn't write a real report, prompt it to
    if len(report.strip()) < 200:
        if settings.llm_provider == "openrouter":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                model=settings.openrouter_model,
                temperature=settings.model_temperature,
            )
        else:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                base_url=settings.ollama_host,
                model=settings.ollama_model,
                temperature=settings.model_temperature,
            )

        prompt = HumanMessage(
            content=(
                "Based on all the research you have conducted above, write a comprehensive "
                "final report addressing the original goal. Include key findings, evidence "
                "quality assessment, and conclusions. Format as markdown."
            )
        )
        response = llm.invoke(list(state["messages"]) + [prompt])
        report = response.content

    return {"report": report}


def route_after_planner(state: AgentState) -> str:
    """
    Decides where to go after the planner runs.

    Returns "tools" if the LLM emitted a tool call, "synthesise" if it
    wrote a final answer, or forces "synthesise" if the iteration limit
    has been reached (safety guard against infinite loops).
    """
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "synthesise"

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "synthesise"


# LangGraph's built-in node that executes whatever tool the planner called.
# It reads the tool name and args from the AIMessage's tool_calls field,
# runs the matching function from SCALPEL_TOOLS, and wraps the result in
# a ToolMessage so the planner can read it on the next iteration.
tool_node = ToolNode(SCALPEL_TOOLS)
