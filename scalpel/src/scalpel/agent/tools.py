"""
LangGraph tool wrappers for SCALPEL.

Each function is decorated with @tool, which gives the LangGraph planner
a typed, described interface to each SCALPEL capability. The wrappers
translate between LLM-friendly inputs (strings, ints) and SCALPEL's
internal types (ExtractedPaper, BullshitScore, etc.).
"""

from pathlib import Path

import numpy as np
import ollama
from langchain_core.tools import tool

from scalpel.analysis.engine import get_analyzer
from scalpel.bear.cross_reference import get_cross_referencer
from scalpel.config import settings
from scalpel.embeddings.vector_store import get_store
from scalpel.ingestion import extract_pdf
from scalpel.retrieval.cache import RetrievalCache
from scalpel.retrieval.optimizer import RetrievalOptimizer


@tool
def list_papers() -> str:
    """
    List all scientific papers currently in the SCALPEL library.

    Use this first when given a research goal, to understand what papers
    are available before deciding which ones to retrieve or analyse.
    Returns paper titles, chunk counts, and file paths.
    """
    papers = get_store().list_papers()
    if not papers:
        return "The paper library is empty. No papers have been indexed yet."

    lines = [f"Found {len(papers)} paper(s) in the library:\n"]
    for p in papers:
        lines.append(f"- {p['title']} ({p['chunk_count']} chunks) — path: {p['path']}")
    return "\n".join(lines)


@tool
def search_papers(query: str, n_results: int = 5, section_filter: str = "") -> str:
    """
    Search the paper library for chunks most relevant to a query.

    Use this to retrieve evidence about a specific topic, claim, or
    methodology. Results are ranked by semantic similarity.

    Args:
        query: Natural language question or topic to search for.
        n_results: Number of chunks to return (default 5, max 10).
        section_filter: Restrict to a specific paper section, e.g. "Methods",
            "Results", "Discussion". Leave empty to search all sections.

    Returns chunks with paper title, section, similarity score, and text.
    """
    n_results = min(n_results, 10)
    filter_val = section_filter if section_filter else None
    results = get_store().search(query, n_results=n_results, section_filter=filter_val)

    if not results:
        return f"No results found for query: '{query}'"

    lines = [f"Found {len(results)} chunk(s) for '{query}':\n"]
    for i, r in enumerate(results, 1):
        section = f" [{r.section}]" if r.section else ""
        lines.append(
            f"--- Result {i} | {r.paper_title}{section} (score: {r.score:.3f}) ---"
        )
        lines.append(r.text[:600] + ("..." if len(r.text) > 600 else ""))
        lines.append("")

    return "\n".join(lines)


@tool
def summarize_paper(paper_path: str, mode: str = "brief") -> str:
    """
    Summarise a paper from the library.

    Use this to get an overview of a paper before deciding whether to
    critique it in depth. Prefer 'brief' for a fast scan; use 'full'
    when you need a thorough understanding of the paper's contributions.

    Args:
        paper_path: Full file path to the PDF, as returned by list_papers.
        mode: 'brief' (default) uses the abstract and key sections;
              'full' reads the entire paper.

    Returns a structured summary of the paper's contributions and methods.
    """
    try:
        paper = extract_pdf(paper_path)
        result = get_analyzer().summarize(paper, mode=mode, verbose=False)  # type: ignore[arg-type]
        return f"SUMMARY — {result.paper_title}\n\n{result.content}"
    except Exception as e:
        return f"Error summarising '{paper_path}': {e}"


@tool
def extract_claims(paper_path: str) -> str:
    """
    Extract the major claims made in a paper.

    Use this to identify what a paper is asserting before cross-checking
    those claims against other papers or critiquing their basis. Focuses
    on the Abstract, Results, Discussion, and Conclusion sections.

    Args:
        paper_path: Full file path to the PDF, as returned by list_papers.

    Returns a structured list of claims with supporting evidence and
    credibility notes.
    """
    try:
        paper = extract_pdf(paper_path)
        result = get_analyzer().extract_claims(paper, verbose=False)
        return f"CLAIMS — {result.paper_title}\n\n{result.content}"
    except Exception as e:
        return f"Error extracting claims from '{paper_path}': {e}"


@tool
def critique_methodology(paper_path: str) -> str:
    """
    Critique the methodology of a paper.

    Use this when you need to evaluate study design, experimental controls,
    sample sizes, confounders, or the validity of the methods used. Focuses
    on Methods, Experimental, and Results sections.

    Args:
        paper_path: Full file path to the PDF, as returned by list_papers.

    Returns a detailed critique of the paper's methodological choices and
    weaknesses.
    """
    try:
        paper = extract_pdf(paper_path)
        result = get_analyzer().critique_methodology(paper, verbose=False)
        return f"METHODOLOGY CRITIQUE — {result.paper_title}\n\n{result.content}"
    except Exception as e:
        return f"Error critiquing methodology of '{paper_path}': {e}"


@tool
def critique_statistics(paper_path: str) -> str:
    """
    Critique the statistical analysis of a paper.

    Use this when you need to evaluate statistical methods, significance
    thresholds, effect sizes, p-hacking risk, or whether the conclusions
    actually follow from the reported numbers.

    Args:
        paper_path: Full file path to the PDF, as returned by list_papers.

    Returns a detailed critique of the paper's statistical approach and
    validity.
    """
    try:
        paper = extract_pdf(paper_path)
        result = get_analyzer().critique_statistics(paper, verbose=False)
        return f"STATISTICS CRITIQUE — {result.paper_title}\n\n{result.content}"
    except Exception as e:
        return f"Error critiquing statistics of '{paper_path}': {e}"


@tool
def bullshit_score_paper(paper_path: str) -> str:
    """
    Calculate the overall scientific rigour score for a paper.

    Scores the paper 0–10 across methodology, statistics, claims, and
    transparency. Lower is better. Use this to get a headline quality
    signal and decide how much weight to give a paper's conclusions.

    Args:
        paper_path: Full file path to the PDF, as returned by list_papers.

    Returns a structured score with per-dimension breakdown and specific
    red flags.
    """
    try:
        paper = extract_pdf(paper_path)
        score = get_analyzer().bullshit_score(paper, verbose=False)

        lines = [
            f"BULLSHIT SCORE — {paper.title}",
            f"Overall: {score.overall_score}/10 ({score.rating})",
        ]
        if score.methodology_score is not None:
            lines += [
                f"  Methodology:  {score.methodology_score}/10",
                f"  Statistics:   {score.statistics_score}/10",
                f"  Claims:       {score.claims_score}/10",
                f"  Transparency: {score.transparency_score}/10",
            ]
        if score.red_flags:
            lines.append("\nRed flags:")
            lines.extend(f"  - {flag}" for flag in score.red_flags)
        if score.summary:
            lines += ["", score.summary]

        return "\n".join(lines)
    except Exception as e:
        return f"Error scoring '{paper_path}': {e}"


@tool
def cross_reference_company(ticker: str) -> str:
    """
    Cross-reference a company's claims against the scientific paper library.

    Use this when analysing investment or technology claims to check whether
    the scientific literature supports, contradicts, or is silent on what a
    company asserts. Requires the company to have been added with
    'scalpel bear add <ticker>' first.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA', 'MSFT', 'GOOG'.

    Returns a structured analysis of supported, contradicted, and
    unsupported claims with citations to specific papers.
    """
    try:
        result = get_cross_referencer().cross_reference(ticker.upper(), verbose=False)
        return (
            f"CROSS-REFERENCE — {result.company_name} ({result.ticker})\n\n"
            f"{result.analysis}"
        )
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error cross-referencing '{ticker}': {e}"


@tool
def get_retrieval_params(query: str) -> str:
    """
    Get the optimal retrieval parameters for a query using the k-NN optimizer.

    Uses past retrieval experience to recommend n_chunks, chunk_trunc,
    similarity_threshold, and whether to rerank by section. Falls back to
    sensible defaults when no relevant history exists yet.

    Use this before search_papers when retrieval quality matters, such as
    when searching for evidence to support or refute a specific claim.

    Args:
        query: The research query you are about to run.

    Returns recommended parameter values to pass to search_papers.
    """
    try:
        cache = RetrievalCache()
        optimizer = RetrievalOptimizer(cache)
        n = optimizer.build_index()

        embedding_response = ollama.embed(
            model=settings.embedding_model,
            input=query,
            truncate=True,
        )
        query_embedding = np.array(embedding_response["embeddings"][0], dtype=np.float32)

        params = optimizer.get_params(query_embedding)
        source = f"k-NN index ({n} entries)" if n > 0 else "defaults (no history yet)"

        return (
            f"Recommended retrieval parameters for: '{query}'\n"
            f"Source: {source}\n"
            f"  n_results:            {params.n_chunks}\n"
            f"  chunk_truncation:     {params.chunk_trunc} chars\n"
            f"  similarity_threshold: {params.similarity_threshold}\n"
            f"  rerank_by_section:    {params.rerank}"
        )
    except Exception as e:
        return (
            f"Error getting retrieval params: {e}. "
            f"Using defaults: n=5, trunc=512, threshold=0.0, rerank=False"
        )


# All tools collected for easy import by the agent graph.
SCALPEL_TOOLS = [
    list_papers,
    search_papers,
    summarize_paper,
    extract_claims,
    critique_methodology,
    critique_statistics,
    bullshit_score_paper,
    cross_reference_company,
    get_retrieval_params,
]
