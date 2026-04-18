# SCALPEL 🔪

**Scientific Critique & Analysis Pipeline for Evidence Literature**

A personal AI research assistant that ingests, critiques, and cross-references academic papers. Built to cut through the noise and find the signal — and to score papers on how much bullshit they contain.

---

## What it does

- 📄 **Paper Ingestion** — PDF extraction and direct arXiv fetching
- 🔍 **Semantic Search** — Vector search across your entire paper library
- 🧠 **AI Critique** — Methodology, statistics, claims, limitations
- 💩 **Bullshit Score** — Scientific rigour scored 0–10 with red flags
- 🔬 **RAG Evaluation** — LLM-as-judge scoring of groundedness, relevance, and confidence calibration
- 🤖 **RL Retrieval Agent** — PPO agent that learns optimal retrieval parameters from evaluation feedback

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (cloud) | OpenRouter — any free or paid model |
| LLM (local) | Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector DB | LanceDB |
| PDF Processing | PyMuPDF |
| RL Training | PyTorch (PPO from scratch) |
| CLI | Typer + Rich |

---

## Setup

```bash
# Install dependencies
poetry install

# For RL training
poetry install --extras rl

# Configure provider and model interactively
scalpel setup

# Pull the embedding model (required for search)
ollama pull nomic-embed-text
```

### Environment variables (`.env`)

```
LLM_PROVIDER=openrouter          # or "ollama"
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

---

## Usage

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
scalpel stats
scalpel config

# Switch model without re-running setup
scalpel model

# Train the RL retrieval agent
scalpel train-rl
scalpel train-rl --resume              # Continue from last checkpoint
scalpel train-rl --iterations 20 --rollouts 30
```

---

## Project Structure

```
scalpel/
├── src/scalpel/
│   ├── ingestion/       # PDF extraction, chunking, arXiv fetcher
│   ├── embeddings/      # LanceDB vector store + Ollama embeddings
│   ├── analysis/        # LLM client, critique engine, prompts
│   ├── evaluation/      # LLM-as-judge: groundedness, relevance, confidence
│   ├── rl/              # PPO retrieval agent (PyTorch)
│   │   ├── environment.py  # RAG pipeline as RL environment + result cache
│   │   ├── policy.py       # Actor-Critic network
│   │   ├── agent.py        # PPO algorithm from scratch
│   │   └── train.py        # Training loop with checkpoint resume
│   └── interface/       # CLI (Typer) + Textual TUI
├── data/papers/         # Your paper library (gitignored)
├── models/rl/           # RL checkpoints (gitignored)
└── pyproject.toml
```

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | LLM Evaluation Framework (groundedness, relevance, confidence) | ✅ Complete |
| 2 | PPO Retrieval Optimisation (PyTorch, from scratch) | ✅ Complete |
| 3 | C++ acceleration (pybind11 chunker) | 🔜 Next |
| 4 | Streamlit interface | 🔜 Planned |

---

*Built with spite and caffeine.*
