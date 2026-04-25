"""
Investment analyst for BEAR module.

Generates structured investment reports using the existing LLM client,
with bull case, bear case, key assumptions flagged, and a bullshit score
for each major claim.
"""

import re
from dataclasses import dataclass, field

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from scalpel.analysis.llm_client import LLMClient, get_client
from scalpel.bear.ingestion import MarketStore, get_market_store
from scalpel.console import console


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = (
    "You are a rigorous, skeptical financial analyst. Provide honest, evidence-based "
    "investment analysis. Be direct, flag unsupported claims, and never let corporate "
    "spin go unchallenged. Lower bullshit scores mean more credible claims."
)

_REPORT_PROMPT = """\
Analyze {company_name} ({ticker}) as an investment opportunity.

COMPANY DATA:
{context}

Provide analysis in EXACTLY this format — do not add extra sections:

BULL CASE:
• [reason backed by specific data]
• [reason backed by specific data]
• [reason backed by specific data]

BEAR CASE:
• [reason backed by specific data]
• [reason backed by specific data]
• [reason backed by specific data]

KEY ASSUMPTIONS:
• [assumption the bull case requires to hold]
• [assumption the bull case requires to hold]

CLAIMS ANALYSIS:
CLAIM: [a major company claim or narrative]
BS SCORE: [0-10]
REASONING: [why this score — be specific]

CLAIM: [another major claim]
BS SCORE: [0-10]
REASONING: [why this score]

OVERALL BS SCORE: [0-10]
SUMMARY: [2-3 sentence synthesis of the investment thesis]\
"""

_BS_SYSTEM = (
    "You are a financial analyst specialising in identifying corporate bullshit, spin, "
    "and unsupported claims. Be precise about what evidence does and does not support."
)

_BS_PROMPT = """\
Evaluate the credibility of claims made by {company_name} ({ticker}).

COMPANY DATA:
{context}

For each major claim or narrative, provide:
CLAIM: [the specific claim]
BS SCORE: [0-10, where 0=iron-clad evidence, 10=pure spin/fantasy]
REASONING: [brief explanation citing data]

CLAIM: [next claim]
BS SCORE: [0-10]
REASONING: [explanation]

OVERALL BS SCORE: [0-10]
SUMMARY: [What makes this company credible or not]\
"""

_COMPARE_SYSTEM = (
    "You are a financial analyst specialising in competitive investment analysis. "
    "Be objective, data-driven, and decisive about which company is the better opportunity."
)

_COMPARE_PROMPT = """\
Compare {ticker1} ({name1}) vs {ticker2} ({name2}) as investment opportunities.

{ticker1} DATA:
{context1}

{ticker2} DATA:
{context2}

RELATIVE STRENGTHS {ticker1}:
• [specific advantage]

RELATIVE STRENGTHS {ticker2}:
• [specific advantage]

VALUATION COMPARISON:
[Side-by-side key metrics and what the market is pricing in for each]

RISK COMPARISON:
[Key risks for each company]

VERDICT:
[Which is the better investment and why — be decisive]

BS SCORE {ticker1}: [0-10]
BS SCORE {ticker2}: [0-10]\
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A single company claim with its bullshit evaluation."""

    text: str
    bs_score: float
    reasoning: str

    @property
    def color(self) -> str:
        if self.bs_score <= 3:
            return "green"
        if self.bs_score <= 6:
            return "yellow"
        return "red"


@dataclass
class InvestmentReport:
    """Full investment analysis report for a company."""

    ticker: str
    company_name: str
    bull_case: list[str]
    bear_case: list[str]
    key_assumptions: list[str]
    claims: list[Claim]
    overall_bs_score: float
    summary: str
    raw_response: str
    model_used: str = ""

    @property
    def bs_rating(self) -> str:
        s = self.overall_bs_score
        if s <= 2:
            return "Highly Credible"
        if s <= 4:
            return "Mostly Credible"
        if s <= 6:
            return "Mixed Signals"
        if s <= 8:
            return "Heavy Spin"
        return "Pure Narrative"

    @property
    def bs_color(self) -> str:
        if self.overall_bs_score <= 4:
            return "green"
        if self.overall_bs_score <= 6:
            return "yellow"
        return "red"

    def display(self) -> None:
        console.print(Panel(
            f"[bold]{self.company_name}[/bold] ({self.ticker})\n"
            f"Overall BS Score: [{self.bs_color}][bold]{self.overall_bs_score:.1f}/10[/bold] — "
            f"{self.bs_rating}[/{self.bs_color}]",
            title="[bold cyan]BEAR Investment Report[/bold cyan]",
            border_style="cyan",
        ))

        bull_text = "\n".join(f"• {p}" for p in self.bull_case) or "No bull case identified."
        console.print(Panel(bull_text, title="[bold green]Bull Case[/bold green]", border_style="green"))

        bear_text = "\n".join(f"• {p}" for p in self.bear_case) or "No bear case identified."
        console.print(Panel(bear_text, title="[bold red]Bear Case[/bold red]", border_style="red"))

        if self.key_assumptions:
            assumptions_text = "\n".join(f"• {a}" for a in self.key_assumptions)
            console.print(Panel(
                assumptions_text,
                title="[bold yellow]Key Assumptions[/bold yellow]",
                border_style="yellow",
            ))

        if self.claims:
            table = Table(title="Claim-by-Claim BS Analysis", show_lines=True)
            table.add_column("Claim", style="white", max_width=48)
            table.add_column("BS", justify="center", width=6)
            table.add_column("Reasoning", style="dim", max_width=42)
            for claim in self.claims:
                score_str = f"[{claim.color}]{claim.bs_score:.0f}/10[/{claim.color}]"
                table.add_row(claim.text[:100], score_str, claim.reasoning[:120])
            console.print(table)

        console.print(Panel(
            self.summary or "No summary generated.",
            title="[bold]Summary[/bold]",
            border_style="dim",
        ))

    def to_markdown(self) -> str:
        """Return the investment report as a markdown string."""
        bull_items = [f"- {p}" for p in self.bull_case] or ["- No bull case identified."]
        bear_items = [f"- {p}" for p in self.bear_case] or ["- No bear case identified."]
        lines = [
            f"# BEAR Investment Report — {self.company_name} ({self.ticker})",
            "",
            f"**Overall BS Score: {self.overall_bs_score:.1f}/10 — {self.bs_rating}**",
            "",
            "## Bull Case",
            *bull_items,
            "",
            "## Bear Case",
            *bear_items,
        ]
        if self.key_assumptions:
            lines += ["", "## Key Assumptions", *[f"- {a}" for a in self.key_assumptions]]
        if self.claims:
            lines += ["", "## Claim-by-Claim Analysis", "| Claim | BS Score | Reasoning |", "| --- | --- | --- |"]
            for c in self.claims:
                lines.append(f"| {c.text} | {c.bs_score:.0f}/10 | {c.reasoning} |")
        lines += ["", "## Summary", self.summary or "No summary generated."]
        if self.model_used:
            lines += ["", "---", f"*Model: {self.model_used}*"]
        return "\n".join(lines)


@dataclass
class BSScoreResult:
    """Bullshit score result for a company."""

    ticker: str
    company_name: str
    overall_score: float
    claims: list[Claim]
    summary: str
    raw_response: str

    @property
    def rating(self) -> str:
        s = self.overall_score
        if s <= 2:
            return "Highly Credible"
        if s <= 4:
            return "Mostly Credible"
        if s <= 6:
            return "Mixed Signals"
        if s <= 8:
            return "Heavy Spin"
        return "Pure Narrative"

    @property
    def color(self) -> str:
        if self.overall_score <= 4:
            return "green"
        if self.overall_score <= 6:
            return "yellow"
        return "red"

    def display(self) -> None:
        color = self.color
        lines = [
            f"[bold {color}]{self.overall_score:.1f}/10[/bold {color}] — {self.rating}",
            "",
            self.summary or "",
        ]
        if self.claims:
            lines.append("\n[bold]Claims:[/bold]")
            for claim in self.claims:
                lines.append(
                    f"  [{claim.color}]{claim.bs_score:.0f}[/{claim.color}]  {claim.text[:90]}"
                )

        console.print(Panel(
            "\n".join(lines).strip(),
            title=f"[bold]BS Score — {self.company_name} ({self.ticker})[/bold]",
            border_style=color,
        ))

    def to_markdown(self) -> str:
        """Return the BS score result as a markdown string."""
        lines = [
            f"# BS Score — {self.company_name} ({self.ticker})",
            "",
            f"**{self.overall_score:.1f}/10 — {self.rating}**",
            "",
            self.summary or "",
        ]
        if self.claims:
            lines += ["", "## Claims", "| Claim | BS Score | Reasoning |", "| --- | --- | --- |"]
            for c in self.claims:
                lines.append(f"| {c.text} | {c.bs_score:.0f}/10 | {c.reasoning} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_bullet_section(text: str, header: str) -> list[str]:
    """Extract bullet points from a named section in LLM output."""
    pattern = rf"(?i){re.escape(header)}[:\s]*\n((?:[ \t]*[•\-\*\d]+[\.\)]?\s*.+\n?)+)"
    match = re.search(pattern, text)
    if not match:
        return []
    items = []
    for line in match.group(1).splitlines():
        line = re.sub(r"^[ \t]*[•\-\*\d]+[\.\)]?\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def _parse_claims(text: str) -> list[Claim]:
    """Parse CLAIM / BS SCORE / REASONING triplets from LLM output."""
    claims = []
    # Split on "CLAIM:" boundaries; index 0 is text before the first claim
    parts = re.split(r"(?i)\nCLAIM:\s*", "\n" + text)
    for part in parts[1:]:
        claim_match = re.match(r"(.+?)(?=\nBS SCORE:)", part, re.DOTALL | re.IGNORECASE)
        if not claim_match:
            continue
        claim_text = claim_match.group(1).strip()

        score_match = re.search(r"(?i)BS SCORE:\s*(\d+(?:\.\d+)?)", part)
        score = float(score_match.group(1)) if score_match else 5.0
        score = min(10.0, max(0.0, score))

        reasoning_match = re.search(
            r"(?i)REASONING:\s*(.+?)(?=\nCLAIM:|\nOVERALL|$)", part, re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        claims.append(Claim(text=claim_text, bs_score=score, reasoning=reasoning))
    return claims


def _parse_overall_bs(text: str) -> float:
    match = re.search(r"(?i)OVERALL BS SCORE:\s*\[?(\d+(?:\.\d+)?)\]?", text)
    return min(10.0, max(0.0, float(match.group(1)))) if match else 5.0


def _parse_summary(text: str) -> str:
    match = re.search(r"(?i)SUMMARY:\s*\n?(.+?)$", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_company_name(context: str, fallback: str) -> str:
    for line in context.splitlines():
        if line.startswith("Company:"):
            return line.split(":", 1)[1].split("(")[0].strip()
    return fallback


# ---------------------------------------------------------------------------
# Analyst class
# ---------------------------------------------------------------------------

class BEARAnalyst:
    """
    Investment analyst using the existing LLM client.

    Generates structured reports with bull/bear cases, assumption flagging,
    and per-claim bullshit scores.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        store: MarketStore | None = None,
    ):
        self.client = client or get_client()
        self.store = store or get_market_store()

    def _get_context(self, ticker: str) -> str:
        context = self.store.get_company_context(ticker)
        if not context:
            raise ValueError(
                f"No data for '{ticker}'. "
                f"Run [cyan]scalpel bear add {ticker}[/cyan] first."
            )
        return context

    def full_report(self, ticker: str, verbose: bool = True) -> InvestmentReport:
        """Generate a full investment report: bull/bear cases, assumptions, BS scores."""
        ticker = ticker.upper()
        if verbose:
            console.print(f"\n[cyan]Generating investment report for[/cyan] [bold]{ticker}[/bold]...")

        context = self._get_context(ticker)
        company_name = _extract_company_name(context, ticker)

        prompt = _REPORT_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            context=context[:12000],
        )
        response = self.client.generate(prompt=prompt, system=_REPORT_SYSTEM)
        text = response.content

        report = InvestmentReport(
            ticker=ticker,
            company_name=company_name,
            bull_case=_parse_bullet_section(text, "BULL CASE"),
            bear_case=_parse_bullet_section(text, "BEAR CASE"),
            key_assumptions=_parse_bullet_section(text, "KEY ASSUMPTIONS"),
            claims=_parse_claims(text),
            overall_bs_score=_parse_overall_bs(text),
            summary=_parse_summary(text),
            raw_response=text,
            model_used=self.client.model,
        )

        if verbose:
            console.print(f"[green]✓[/green] Report generated")

        return report

    def bullshit_score(self, ticker: str, verbose: bool = True) -> BSScoreResult:
        """Calculate bullshit score for a company's claims and narratives."""
        ticker = ticker.upper()
        if verbose:
            console.print(f"\n[cyan]Evaluating claims for[/cyan] [bold]{ticker}[/bold]...")

        context = self._get_context(ticker)
        company_name = _extract_company_name(context, ticker)

        prompt = _BS_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            context=context[:12000],
        )
        response = self.client.generate(prompt=prompt, system=_BS_SYSTEM)
        text = response.content

        result = BSScoreResult(
            ticker=ticker,
            company_name=company_name,
            overall_score=_parse_overall_bs(text),
            claims=_parse_claims(text),
            summary=_parse_summary(text),
            raw_response=text,
        )

        if verbose:
            console.print(f"[green]✓[/green] BS score calculated")

        return result

    def compare(self, ticker1: str, ticker2: str, verbose: bool = True) -> str:
        """Generate a comparative analysis of two companies."""
        ticker1, ticker2 = ticker1.upper(), ticker2.upper()
        if verbose:
            console.print(
                f"\n[cyan]Comparing[/cyan] [bold]{ticker1}[/bold] vs [bold]{ticker2}[/bold]..."
            )

        context1 = self._get_context(ticker1)
        context2 = self._get_context(ticker2)

        name1 = _extract_company_name(context1, ticker1)
        name2 = _extract_company_name(context2, ticker2)

        prompt = _COMPARE_PROMPT.format(
            ticker1=ticker1, name1=name1,
            ticker2=ticker2, name2=name2,
            context1=context1[:7000],
            context2=context2[:7000],
        )
        response = self.client.generate(prompt=prompt, system=_COMPARE_SYSTEM)

        if verbose:
            console.print(f"[green]✓[/green] Comparison complete")

        return response.content


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

_default_analyst: BEARAnalyst | None = None


def get_analyst() -> BEARAnalyst:
    """Get or create the default BEAR analyst."""
    global _default_analyst
    if _default_analyst is None:
        _default_analyst = BEARAnalyst()
    return _default_analyst


def full_report(ticker: str, **kwargs) -> InvestmentReport:
    return get_analyst().full_report(ticker, **kwargs)


def bullshit_score(ticker: str, **kwargs) -> BSScoreResult:
    return get_analyst().bullshit_score(ticker, **kwargs)


def compare(ticker1: str, ticker2: str, **kwargs) -> str:
    return get_analyst().compare(ticker1, ticker2, **kwargs)
