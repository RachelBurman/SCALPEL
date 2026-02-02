# SCALPEL 🔪

**Scientific Critique & Analysis Pipeline for Evidence Literature**

A personal AI research assistant that summarizes, critiques, and cross-references academic papers. Built to cut through the noise and find the signal.

## Features (Planned)

- 📄 **Paper Ingestion** — PDF extraction and arXiv link handling
- 🔍 **Methodology Critique** — Flag statistical red flags, small sample sizes, questionable methods
- 📊 **Claim Extraction** — Pull out key claims and findings
- 🔗 **Cross-Reference** — Compare against your paper library for contradictions/support
- 💩 **Bullshit Score** — Because some papers need one

## Tech Stack

- **LLM**: Qwen 2.5 via Ollama (local, cost-effective)
- **PDF Processing**: PyMuPDF
- **Vector DB**: ChromaDB (coming soon)
- **Interface**: CLI first, then Streamlit

## Setup

```bash
# Install dependencies
poetry install

# Copy environment config
cp .env.example .env

# Make sure Ollama is running with Qwen 2.5
ollama pull qwen2.5

# Run (coming soon)
poetry run scalpel
```

## Project Structure

```
scalpel/
├── src/scalpel/
│   ├── config.py        # Settings & configuration
│   ├── ingestion/       # PDF & arXiv handlers
│   ├── embeddings/      # Vector DB logic
│   ├── analysis/        # The brain - critique engine
│   └── interface/       # CLI/UI
├── tests/
├── data/papers/         # Your paper library
└── pyproject.toml
```

## Status

🚧 **Phase 1: Foundation** — In Progress

---

*Built with spite and caffeine.*