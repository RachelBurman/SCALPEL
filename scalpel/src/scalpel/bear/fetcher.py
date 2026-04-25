"""
Market data fetcher for BEAR module.

Pulls ticker data from Yahoo Finance: financials, earnings, news,
key metrics, and analyst recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from scalpel.console import console


@dataclass
class CompanySection:
    """A named section of company data, analogous to PaperSection."""

    name: str
    content: str


@dataclass
class CompanyData:
    """Market and fundamental data for a company."""

    ticker: str
    company_name: str
    sector: str
    industry: str
    sections: list[CompanySection]
    metadata: dict = field(default_factory=dict)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"== {s.name.upper()} ==\n{s.content}" for s in self.sections)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, decimals: int = 2, pct: bool = False, large: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if v != v:  # NaN
            return "N/A"
        if pct:
            return f"{v * 100:.{decimals}f}%"
        if large:
            return f"{v:,.0f}"
        return f"{v:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value) if value else "N/A"


def _df_to_text(df: Any, label: str = "") -> str:
    """Convert a yfinance DataFrame to readable text (4 most recent periods)."""
    try:
        if df is None or df.empty:
            return "Data not available."
        lines = []
        if label:
            lines.append(label)
        cols = list(df.columns[:4])
        for idx in df.index:
            vals = []
            for col in cols:
                raw = df.loc[idx, col]
                try:
                    v = float(raw)
                    vals.append(f"{v:,.0f}" if not (v != v) else "N/A")
                except (TypeError, ValueError):
                    vals.append(str(raw))
            lines.append(f"  {idx}: {' | '.join(vals)}")
        return "\n".join(lines)
    except Exception:
        return "Data not available."


def fetch_ticker(ticker: str, verbose: bool = True) -> CompanyData:
    """
    Fetch comprehensive market data for a ticker symbol from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA')
        verbose: Print progress to console

    Returns:
        CompanyData with all available market data sections

    Raises:
        ImportError: If yfinance is not installed
        ValueError: If ticker is not found or returns no data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for BEAR. Install with: pip install scalpel[bear]"
        )

    if verbose:
        console.print(f"[cyan]Fetching[/cyan] market data for [bold]{ticker.upper()}[/bold]...")

    t = yf.Ticker(ticker)
    info = t.info or {}

    company_name = info.get("longName") or info.get("shortName") or ticker.upper()
    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"

    if not info.get("symbol") and not info.get("longName") and not info.get("shortName"):
        raise ValueError(f"Ticker '{ticker}' not found or returned no data.")

    sections: list[CompanySection] = []

    # --- Overview ---
    description = info.get("longBusinessSummary") or "No description available."
    overview_lines = [
        f"Company: {company_name} ({ticker.upper()})",
        f"Sector: {sector}",
        f"Industry: {industry}",
        f"Exchange: {info.get('exchange', 'N/A')}",
        f"Country: {info.get('country', 'N/A')}",
        f"Website: {info.get('website', 'N/A')}",
        f"Employees: {info.get('fullTimeEmployees', 'N/A')}",
        "",
        description,
    ]
    sections.append(CompanySection("Overview", "\n".join(overview_lines)))

    # --- Key Metrics ---
    metrics_lines = [
        f"Market Cap: {_fmt(_safe_float(info.get('marketCap')), large=True)} USD",
        f"Enterprise Value: {_fmt(_safe_float(info.get('enterpriseValue')), large=True)} USD",
        f"Trailing P/E: {_fmt(info.get('trailingPE'))}",
        f"Forward P/E: {_fmt(info.get('forwardPE'))}",
        f"Price/Sales (TTM): {_fmt(info.get('priceToSalesTrailing12Months'))}",
        f"Price/Book: {_fmt(info.get('priceToBook'))}",
        f"EV/EBITDA: {_fmt(info.get('enterpriseToEbitda'))}",
        f"EV/Revenue: {_fmt(info.get('enterpriseToRevenue'))}",
        f"Revenue Growth (YoY): {_fmt(info.get('revenueGrowth'), pct=True)}",
        f"Earnings Growth (YoY): {_fmt(info.get('earningsGrowth'), pct=True)}",
        f"Profit Margin: {_fmt(info.get('profitMargins'), pct=True)}",
        f"Operating Margin: {_fmt(info.get('operatingMargins'), pct=True)}",
        f"ROE: {_fmt(info.get('returnOnEquity'), pct=True)}",
        f"ROA: {_fmt(info.get('returnOnAssets'), pct=True)}",
        f"Debt/Equity: {_fmt(info.get('debtToEquity'))}",
        f"Current Ratio: {_fmt(info.get('currentRatio'))}",
        f"Quick Ratio: {_fmt(info.get('quickRatio'))}",
        f"Free Cash Flow: {_fmt(_safe_float(info.get('freeCashflow')), large=True)} USD",
        f"Revenue (TTM): {_fmt(_safe_float(info.get('totalRevenue')), large=True)} USD",
        f"Gross Profit: {_fmt(_safe_float(info.get('grossProfits')), large=True)} USD",
        f"EBITDA: {_fmt(_safe_float(info.get('ebitda')), large=True)} USD",
        f"Beta: {_fmt(info.get('beta'))}",
        f"52-Week High: {_fmt(info.get('fiftyTwoWeekHigh'))}",
        f"52-Week Low: {_fmt(info.get('fiftyTwoWeekLow'))}",
        f"Analyst Target (mean): {_fmt(info.get('targetMeanPrice'))}",
        f"Analyst Recommendation: {info.get('recommendationKey', 'N/A')}",
        f"Analysts Covering: {info.get('numberOfAnalystOpinions', 'N/A')}",
    ]
    sections.append(CompanySection("Key Metrics", "\n".join(metrics_lines)))

    # --- Financials ---
    try:
        fin = getattr(t, "income_stmt", None) or getattr(t, "financials", None)
        bs = getattr(t, "balance_sheet", None)
        cf = getattr(t, "cashflow", None)
        financials_text = (
            _df_to_text(fin, "Income Statement (annual, USD):")
            + "\n\n"
            + _df_to_text(bs, "Balance Sheet (annual, USD):")
            + "\n\n"
            + _df_to_text(cf, "Cash Flow (annual, USD):")
        )
    except Exception:
        financials_text = "Financial statements not available."
    sections.append(CompanySection("Financials", financials_text))

    # --- Earnings History ---
    try:
        earnings_lines = []
        qe = getattr(t, "quarterly_earnings", None)
        if qe is None or (hasattr(qe, "empty") and qe.empty):
            qe = getattr(t, "earnings", None)
        if qe is not None and hasattr(qe, "empty") and not qe.empty:
            earnings_lines.append("Quarterly Earnings (Estimate vs Actual EPS):")
            for idx, row in qe.iterrows():
                est_key = next((k for k in row.index if "estimate" in str(k).lower()), None)
                act_key = next((k for k in row.index if "actual" in str(k).lower()), None)
                est = _fmt(row.get(est_key) if est_key else None)
                act = _fmt(row.get(act_key) if act_key else None)
                earnings_lines.append(f"  {str(idx)[:10]}: Estimate={est} | Actual={act}")

        next_date = info.get("nextEarningsDate") or info.get("earningsTimestamp")
        if next_date:
            try:
                date_str = datetime.utcfromtimestamp(int(next_date)).strftime("%Y-%m-%d")
                earnings_lines.append(f"\nNext Earnings Date: {date_str}")
            except (TypeError, ValueError, OSError):
                pass

        earnings_text = "\n".join(earnings_lines) if earnings_lines else "Earnings data not available."
    except Exception:
        earnings_text = "Earnings data not available."
    sections.append(CompanySection("Earnings", earnings_text))

    # --- Analyst Recommendations ---
    try:
        recs = t.recommendations
        rec_lines = []
        if recs is not None and not recs.empty:
            rec_lines.append("Recent Analyst Recommendations:")
            recent = recs.tail(12)
            for idx, row in recent.iterrows():
                firm = (
                    row.get("Firm") or row.get("firm") or
                    row.get("Company") or "Unknown"
                )
                grade = row.get("To Grade") or row.get("toGrade") or row.get("Action") or ""
                from_grade = row.get("From Grade") or row.get("fromGrade") or ""
                date_str = str(idx)[:10]
                arrow = f"{from_grade} → {grade}" if from_grade and from_grade != grade else grade
                rec_lines.append(f"  {date_str}: {firm} — {arrow}")

            grade_col = next(
                (c for c in recs.columns if "grade" in c.lower() and "to" in c.lower()),
                None,
            )
            if grade_col:
                counts = recs[grade_col].value_counts()
                rec_lines.append("\nRating Distribution (all-time):")
                for g, n in counts.items():
                    rec_lines.append(f"  {g}: {n}")

        n_analysts = info.get("numberOfAnalystOpinions")
        if n_analysts:
            rec_lines.append(
                f"\nCurrent Coverage: {n_analysts} analysts  "
                f"Target: {_fmt(info.get('targetMeanPrice'))} "
                f"(Low {_fmt(info.get('targetLowPrice'))}, High {_fmt(info.get('targetHighPrice'))})"
            )

        recommendations_text = "\n".join(rec_lines) if rec_lines else "Analyst recommendations not available."
    except Exception:
        recommendations_text = "Analyst recommendations not available."
    sections.append(CompanySection("Analyst Recommendations", recommendations_text))

    # --- Recent News ---
    try:
        news_items = t.news or []
        news_lines = []
        for item in news_items[:15]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            ts = item.get("providerPublishTime") or item.get("publishedAt") or 0
            try:
                date_str = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d") if ts else "N/A"
            except (TypeError, ValueError, OSError):
                date_str = "N/A"
            summary = (item.get("summary") or item.get("description") or "")[:200]
            news_lines.append(f"[{date_str}] {title} ({publisher})")
            if summary:
                news_lines.append(f"  {summary}")
        news_text = "\n".join(news_lines) if news_lines else "No recent news available."
    except Exception:
        news_text = "News data not available."
    sections.append(CompanySection("Recent News", news_text))

    metadata = {
        "symbol": info.get("symbol", ticker.upper()),
        "exchange": info.get("exchange", ""),
        "currency": info.get("currency", "USD"),
        "market_cap": _safe_float(info.get("marketCap")),
        "sector": sector,
        "industry": industry,
    }

    if verbose:
        console.print(
            f"[green]✓[/green] Fetched [bold]{company_name}[/bold] "
            f"([cyan]{sector}[/cyan]) — {len(sections)} data sections"
        )

    return CompanyData(
        ticker=ticker.upper(),
        company_name=company_name,
        sector=sector,
        industry=industry,
        sections=sections,
        metadata=metadata,
    )
