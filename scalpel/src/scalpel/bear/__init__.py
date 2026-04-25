"""
BEAR — Bias Evaluation and Analysis of Research.

Investment analysis module that fetches market data, embeds it into a
'markets' LanceDB collection, and cross-references company claims against
the scientific paper library.
"""

from scalpel.bear.fetcher import CompanyData, CompanySection, fetch_ticker
from scalpel.bear.ingestion import MarketSearchResult, MarketStore, get_market_store
from scalpel.bear.analyst import (
    BEARAnalyst,
    BSScoreResult,
    Claim,
    InvestmentReport,
    bullshit_score,
    compare,
    full_report,
    get_analyst,
)
from scalpel.bear.cross_reference import (
    CrossReferenceResult,
    CrossReferencer,
    cross_reference,
    get_cross_referencer,
)

__all__ = [
    # Fetcher
    "CompanyData",
    "CompanySection",
    "fetch_ticker",
    # Ingestion
    "MarketStore",
    "MarketSearchResult",
    "get_market_store",
    # Analyst
    "BEARAnalyst",
    "InvestmentReport",
    "BSScoreResult",
    "Claim",
    "get_analyst",
    "full_report",
    "bullshit_score",
    "compare",
    # Cross-reference
    "CrossReferencer",
    "CrossReferenceResult",
    "get_cross_referencer",
    "cross_reference",
]
