# SCALPEL 🔪

**Scientific Critique & Analysis Pipeline for Evidence Literature**

A personal AI research assistant that ingests, critiques, and cross-references academic papers — and now investment research too. Built to cut through the noise and find the signal, and to score both papers and company claims on how much bullshit they contain.

---

## What it does

- 📄 **Paper Ingestion** — PDF extraction and direct arXiv fetching
- 🔍 **Semantic Search** — Vector search across your entire paper library
- 🧠 **AI Critique** — Methodology, statistics, claims, limitations
- 💩 **Bullshit Score** — Scientific rigour scored 0–10 with red flags
- 🔬 **RAG Evaluation** — LLM-as-judge scoring of groundedness, relevance, and confidence calibration
- 📈 **BEAR Module** — Bias Evaluation and Analysis of Research: investment analysis with bull/bear cases, per-claim BS scores, and cross-referencing of company claims against your paper library
- ⚡ **C++ Chunker** — pybind11 extension for fast semantic boundary detection
- 🖥️ **Web UI** — Streamlit interface for RAG querying, evaluation scores, and analytics

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (cloud) | OpenRouter — any free or paid model |
| LLM (local) | Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector DB | LanceDB (`papers` + `markets` collections) |
| PDF Processing | PyMuPDF |
| Market Data | yfinance (Yahoo Finance) |
| CLI | Typer + Rich |
| Web UI | Streamlit + Plotly |

---

## Setup

```bash
# Install core dependencies
poetry install

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

---

## Usage

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
scalpel bear analyse AAPL

# Bullshit score only — how credible are the company's claims?
scalpel bear bs TSLA

# Cross-reference company claims against your scientific paper library
scalpel bear cross AAPL

# Compare two companies head-to-head
scalpel bear compare AAPL MSFT

# Library management
scalpel bear list
scalpel bear remove TSLA
```

### Reports

Every analysis command auto-saves a markdown report to `data/reports/` with a timestamped filename:

```
data/reports/
├── bear_analyse_AAPL_2026-04-25_21-25.md
├── bear_cross_AAPL_2026-04-25_21-51.md
├── bear_compare_AAPL_vs_MSFT_2026-04-25_22-13.md
└── ...
```

You can also specify a custom path with `--output`:

```bash
scalpel bear analyse AAPL --output reports/my_report.md
scalpel bear cross AAPL --output reports/aapl_cross.md
scalpel critique paper.pdf --output reports/critique.md
```

### Example Reports

The following reports were generated as part of initial testing against the paper library:

| Report | Command |
|--------|---------|
| [AAPL Investment Report](scalpel/data/reports/bear_analyse_AAPL_2026-04-25_21-25.md) | `scalpel bear analyse AAPL` |
| [MSFT Investment Report](scalpel/data/reports/bear_analyse_MSFT_2026-04-25_21-36.md) | `scalpel bear analyse MSFT` |
| [AAPL Cross-Reference](scalpel/data/reports/bear_cross_AAPL_2026-04-25_21-51.md) | `scalpel bear cross AAPL` |
| [MSFT Cross-Reference](scalpel/data/reports/bear_cross_MSFT_2026-04-25_22-04.md) | `scalpel bear cross MSFT` |
| [AAPL vs MSFT Comparison](scalpel/data/reports/bear_compare_AAPL_vs_MSFT_2026-04-25_22-13.md) | `scalpel bear compare AAPL MSFT` |

Cross-references were run against a library of 10 papers covering stock price prediction, LLM-based sentiment analysis, earnings announcement modelling, tech sector financial metrics, product diversification and firm value, and AI capability valuation.

### Web UI

```bash
scalpel ui    # Opens Streamlit interface in browser
```

---

## Project Structure

```
scalpel/
├── src/scalpel/
│   ├── ingestion/       # PDF extraction, chunking, arXiv fetcher
│   ├── embeddings/      # LanceDB 'papers' vector store + Ollama embeddings
│   ├── analysis/        # LLM client, critique engine, prompts
│   ├── evaluation/      # LLM-as-judge: groundedness, relevance, confidence
│   ├── retrieval/       # Retrieval parameter optimisation + data collection
│   ├── bear/            # BEAR: market data fetcher, 'markets' vector store,
│   │   ├── fetcher.py      #   yfinance ingestion (financials, earnings, news)
│   │   ├── ingestion.py    #   LanceDB 'markets' collection
│   │   ├── analyst.py      #   LLM investment reports + per-claim BS scoring
│   │   └── cross_reference.py  # Cross-references papers ↔ market claims
│   ├── cpp/             # pybind11 C++ extension (fast chunker)
│   └── interface/       # CLI (Typer + Rich) + Streamlit web UI
├── data/papers/         # Your paper library (gitignored)
├── data/lancedb/        # Vector database (gitignored)
├── data/reports/        # Generated markdown reports (gitignored)
└── pyproject.toml
```

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

*Built with spite and caffeine.*
