"""
Analysis module for SCALPEL.

The brain of the operation. LLM-powered paper analysis and critique.

# Quick start
from scalpel.analysis import summarize, bullshit_score, full_analysis
from scalpel.ingestion import extract_pdf

paper = extract_pdf("path/to/paper.pdf")

# Individual analyses
summary = summarize(paper)
summary.display()

bs = bullshit_score(paper)
bs.display()  # Shows score with pretty formatting

# Full analysis (everything at once)
results = full_analysis(paper)
results["summary"].display()
results["methodology"].display()
results["bullshit_score"].display()

# Available analysis functions:
# - summarize(paper)           Summarize key points
# - critique_methodology(paper) Evaluate study design
# - critique_statistics(paper)  Evaluate statistical analysis
# - critique_full(paper)        Comprehensive critique
# - extract_claims(paper)       Extract and rate claims
# - identify_limitations(paper) Find stated/unstated limitations
# - bullshit_score(paper)       The crown jewel: 0-10 rigor score
# - full_analysis(paper)        Run all of the above
"""

from scalpel.analysis.llm_client import (
    LLMClient,
    LLMResponse,
    get_client,
    generate,
)
from scalpel.analysis.engine import (
    Analyzer,
    AnalysisResult,
    BullshitScore,
    get_analyzer,
    summarize,
    critique_methodology,
    critique_statistics,
    critique_full,
    extract_claims,
    identify_limitations,
    bullshit_score,
    full_analysis,
)
from scalpel.analysis.prompts import (
    PromptTemplate,
    get_template,
    list_templates,
    TEMPLATES,
    AnalysisType,
)

__all__ = [
    # LLM Client
    "LLMClient",
    "LLMResponse",
    "get_client",
    "generate",
    # Analyzer
    "Analyzer",
    "AnalysisResult",
    "BullshitScore",
    "get_analyzer",
    "summarize",
    "critique_methodology",
    "critique_statistics",
    "critique_full",
    "extract_claims",
    "identify_limitations",
    "bullshit_score",
    "full_analysis",
    # Prompts
    "PromptTemplate",
    "get_template",
    "list_templates",
    "TEMPLATES",
    "AnalysisType",
]