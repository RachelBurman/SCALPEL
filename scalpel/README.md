# SCALPEL 🔪

**Scientific Critique & Analysis Pipeline for Evidence Literature**

A personal AI research assistant that ingests, critiques, and cross-references academic papers — and investment research too. Built to cut through the noise and find the signal, and to score both papers and company claims on how much bullshit they contain.

Now extended with an **autonomous agentic layer** — give SCALPEL a research goal and it decides which papers to retrieve, which analyses to run, and how to sequence its own reasoning without explicit per-step instruction.

---

## What it does

- 📄 **Paper Ingestion** — PDF extraction and direct arXiv fetching
- 🔍 **Semantic Search** — Vector search across your entire paper library
- 🧠 **AI Critique** — Methodology, statistics, claims, limitations
- 💩 **Bullshit Score** — Scientific rigour scored 0–10 with red flags
- 🔬 **RAG Evaluation** — LLM-as-judge scoring of groundedness, relevance, and confidence calibration
- 📈 **BEAR Module** — Bias Evaluation and Analysis of Research: investment analysis with bull/bear cases, per-claim BS scores, and cross-referencing of company claims against your paper library
- 🤖 **Autonomous Agent** — LangGraph agent that plans and sequences multi-step analyses without instruction
- ⚡ **C++ Chunker** — pybind11 extension for fast semantic boundary detection
- 🖥️ **Web UI** — Streamlit interface for RAG querying, evaluation scores, and analytics

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (local) | Ollama (Qwen 2.5 and others) |
| LLM (cloud) | OpenRouter — any free or paid model |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector DB | LanceDB (`papers` + `markets` collections) |
| Agent Framework | LangGraph (StateGraph + ToolNode + MemorySaver) |
| PDF Processing | PyMuPDF |
| Market Data | yfinance (Yahoo Finance) |
| CLI | Typer + Rich |
| Web UI | Streamlit + Plotly |

---

## Setup

```bash
# Install core dependencies
poetry install

# Install the agentic layer
pip install scalpel[agent]

# For BEAR (investment analysis)
pip install scalpel[bear]

# For the Streamlit web UI
pip install scalpel[ui]

# Configure provider and model interactively
scalpel setup

# Pull the embedding model (required for search)
ollama pull nomic-embed-text
```

### Environment variables (`.env`)

```
LLM_PROVIDER=openrouter          # or "ollama"
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemma-2-9b-it:free
```

### Model requirement for the agent

The autonomous agent requires a model that supports **tool/function calling**. Small models (Qwen 2.5 0.5b, 1.5b) are not reliable for this. Recommended:

- **Ollama:** `qwen2.5:7b` or larger (`ollama pull qwen2.5:7b`)
- **OpenRouter:** any model — the cloud models all support tool calling

---

## Usage

### Autonomous Agent

Give SCALPEL a research goal. It decides what to do.

```bash
scalpel agent "critique the statistical methods across all indexed papers"
scalpel agent "find papers on attention mechanisms and score their methodology"
scalpel agent "which papers in the library have the highest bullshit scores and why?"
scalpel agent "summarise and cross-reference all papers on transformer architectures"

# Save the report
scalpel agent "evaluate all papers on sample size adequacy" --output report.md
```

The agent prints its reasoning steps live:

```
🔪 SCALPEL Agent

Goal: critique the statistical methods across all indexed papers

Step 1 → list_papers()
  ↳ Found 3 paper(s) in the library...
Step 2 → search_papers('statistical methods significance testing')
  ↳ Found 5 chunk(s) for 'statistical methods'...
Step 3 → critique_statistics('/path/to/paper.pdf')
  ↳ STATISTICS CRITIQUE — Attention Is All You Need...
Step 4  Writing report...

╭──────── 🔪 Agent Report ────────╮
│ ## Statistical Critique         │
│ ...                             │
╰─────────────────────────────────╯
```

The agent has access to 9 tools drawn from the existing SCALPEL pipeline:

| Tool | What the agent uses it for |
|------|--------------------------|
| `list_papers` | Orient itself — what's in the library? |
| `search_papers` | Retrieve relevant evidence for any query |
| `summarize_paper` | Get a quick overview before going deeper |
| `extract_claims` | Pull out falsifiable claims for cross-checking |
| `critique_methodology` | Evaluate study design and controls |
| `critique_statistics` | Evaluate statistical validity and effect sizes |
| `bullshit_score_paper` | Score overall rigour — 0 = excellent, 10 = poor |
| `cross_reference_company` | Check if research supports company claims (BEAR) |
| `get_retrieval_params` | Use k-NN history to optimise retrieval settings |

### Papers

```bash
# Add papers
scalpel add data/papers/my_paper.pdf
scalpel add-arxiv 1706.03762

# Analyse
scalpel bs data/papers/my_paper.pdf         # Bullshit score
scalpel analyze data/papers/my_paper.pdf    # Summary + bullshit score
scalpel analyze --full data/papers/...      # Full critique suite
scalpel critique data/papers/... --focus methods

# Search and evaluate
scalpel search "attention mechanism"
scalpel eval "How does self-attention work?"

# Library management
scalpel list
scalpel remove "paper title"
scalpel stats
scalpel config
scalpel model    # Switch LLM without re-running setup
```

### BEAR — Investment Analysis

```bash
# Ingest a company (fetches from Yahoo Finance)
scalpel bear add AAPL
scalpel bear add TSLA

# Full investment report: bull case, bear case, key assumptions, per-claim BS scores
scalpel bear analyse NVDA

# Bullshit score only — how credible are the company's claims?
scalpel bear bs TSLA

# Cross-reference company claims against your scientific paper library
scalpel bear cross MRNA

# Compare two companies head-to-head
scalpel bear compare AAPL MSFT

# Library management
scalpel bear list
scalpel bear remove TSLA
```

### Web UI

```bash
scalpel ui    # Opens Streamlit interface in browser
```

---

## Project Structure

```
scalpel/
├── src/scalpel/
│   ├── agent/           # Autonomous agentic layer (LangGraph)
│   │   ├── state.py        #   AgentState TypedDict (goal, messages, iteration, report)
│   │   ├── tools.py        #   9 SCALPEL capabilities wrapped as @tool functions
│   │   ├── nodes.py        #   planner, synthesise, route_after_planner, tool_node
│   │   ├── graph.py        #   StateGraph assembly + MemorySaver checkpointing
│   │   └── runner.py       #   stream_run() generator for the CLI
│   ├── ingestion/       # PDF extraction, chunking, arXiv fetcher
│   ├── embeddings/      # LanceDB 'papers' vector store + Ollama embeddings
│   ├── analysis/        # LLM client, critique engine, prompts
│   ├── evaluation/      # LLM-as-judge: groundedness, relevance, confidence
│   ├── retrieval/       # k-NN retrieval parameter optimisation + data collection
│   ├── rl/              # RAGEnvironment Gym-like wrapper (legacy)
│   ├── bear/            # BEAR: market data, 'markets' vector store, investment reports
│   │   ├── fetcher.py      #   yfinance ingestion (financials, earnings, news)
│   │   ├── ingestion.py    #   LanceDB 'markets' collection
│   │   ├── analyst.py      #   LLM investment reports + per-claim BS scoring
│   │   └── cross_reference.py  # Cross-references papers ↔ market claims
│   ├── cpp/             # pybind11 C++ extension (fast chunker, optional)
│   └── interface/       # CLI (Typer + Rich) + Streamlit web UI
├── data/papers/         # Your paper library (gitignored)
├── data/lancedb/        # Vector database (gitignored)
└── pyproject.toml
```

### How the agent graph works

```
User goal (str)
      │
      ▼
  [planner]  ◄─────────────────────────────────┐
      │                                         │
      ├── tool call in response? ──Yes──► [tools] (ToolNode executes it)
      │
      └── no tool call (or iteration limit hit)
            │
            ▼
       [synthesise]
            │
            ▼
        final report ──► saved to data/reports/
```

State flows through the graph as an `AgentState` TypedDict. The `messages` field accumulates the full conversation (system prompt, tool calls, tool results, AI reasoning) using LangGraph's `add_messages` reducer. The planner sees the entire history on every iteration so it always knows what has already been done.

---

## Bullshit Scoring

Both the paper analyser and BEAR use a 0–10 bullshit scale:

| Score | Rating | Meaning |
|-------|--------|---------|
| 0–2 | Highly Credible | Strong evidence, conservative claims |
| 3–4 | Mostly Credible | Minor gaps, well-supported overall |
| 5–6 | Mixed Signals | Some claims outrun the evidence |
| 7–8 | Heavy Spin | Corporate/academic narrative dominates |
| 9–10 | Pure Narrative | Little to no evidential grounding |

---

## Roadmap

### ✅ Phase 1 — Tool wrapping (complete)
All SCALPEL capabilities exposed as typed LangGraph tools. The LLM can call any of them by name with structured arguments. Wrappers translate between LLM-friendly string inputs and SCALPEL's internal dataclasses (`ExtractedPaper`, `BullshitScore`, etc.).

### ✅ Phase 2 — Core agent graph (complete)
`StateGraph` with three nodes (`planner → tools → planner` loop, then `synthesise`). Live step-by-step CLI output via `graph.stream()`. `scalpel agent "<goal>"` command. MemorySaver checkpointing for within-session continuity.

### 🔜 Phase 3 — Planning and multi-step reasoning
- Explicit plan generation before acting: agent produces a `plan: list[str]` and ticks steps off as it goes
- Plan-tracking in `AgentState` so the planner knows where it is in the sequence
- Re-retrieval loop: if evaluation score is below threshold, the agent tries different retrieval params rather than giving up
- Loop guard already in place (`MAX_ITERATIONS = 15`)

### 🔜 Phase 4 — Long-term memory
- New `agent_sessions` LanceDB table storing `(goal, report)` pairs as embeddings
- `recall(goal)` searches prior sessions and prepends relevant past analyses to context
- Agent builds up institutional memory across runs — a paper critiqued last week informs this week's analysis
- `scalpel agent history` command to browse past sessions

### 🔜 Phase 5 — Portfolio polish
- `--stream` flag: token-by-token streaming of the agent's reasoning using LangGraph's streaming mode
- `--trace` flag: writes full graph execution trace to JSON (node sequence, state at each step, tool call arguments)
- Streamlit UI panel: goal text box, live reasoning step display, rendered final report
- `AGENT.md` architecture document with graph diagram for interview/portfolio use

---

*Built with spite and caffeine.*
