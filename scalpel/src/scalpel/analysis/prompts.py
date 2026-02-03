"""
Prompt Templates for SCALPEL Scientific Critique.

Structured prompts for different analysis types.
Combines clean architecture with rigorous analysis prompts.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptTemplate:
    """A prompt template with system and user components."""
    
    name: str
    description: str
    system_prompt: str
    user_template: str
    
    def format(self, **kwargs) -> tuple[str, str]:
        """
        Format the template with provided variables.
        
        Returns:
            Tuple of (system_prompt, formatted_user_prompt)
        """
        return self.system_prompt, self.user_template.format(**kwargs)


# === System Prompts ===

SCIENTIFIC_CRITIC_SYSTEM = """You are SCALPEL, an expert scientific critique system. Your role is to provide rigorous, evidence-based analysis of academic papers and research claims.

Guidelines:
- Be precise and cite specific evidence from the provided context
- Identify methodological strengths AND limitations objectively
- Flag unsupported claims, statistical issues, and logical gaps
- Consider alternative interpretations of data
- Maintain scientific rigor without being dismissive
- Acknowledge uncertainty where appropriate
- Use clear, structured responses with headers when helpful

When analyzing, consider:
1. Study design and methodology
2. Statistical analysis and interpretation
3. Logical coherence of arguments
4. Evidence quality and sufficiency
5. Potential confounders and biases
6. Generalizability of findings
7. Consistency with broader literature"""

RAG_CONTEXT_SYSTEM = """You are SCALPEL, a scientific analysis assistant. You have access to relevant excerpts from academic papers to inform your responses.

When using the provided context:
- Cite specific passages when making claims
- Acknowledge when context is insufficient
- Distinguish between what the papers claim and what you conclude
- Be precise about which paper/section information comes from

Format citations as: [Paper Title, Section] or [Paper Title] when section is unknown."""

SUMMARIZATION_SYSTEM = """You are SCALPEL, a scientific summarization expert. Create clear, accurate summaries that:
- Preserve key findings and methodology
- Highlight main contributions
- Note important limitations
- Use precise scientific language
- Maintain appropriate technical depth"""

BULLSHIT_DETECTOR_SYSTEM = """You are SCALPEL, a scientific rigor evaluator. Your role is to assess the quality and trustworthiness of research with brutal honesty.

You evaluate:
- Methodology soundness
- Statistical validity
- Claim calibration (do conclusions match evidence?)
- Transparency about limitations
- Red flags suggesting unreliable findings

Be direct. Be honest. Don't be unnecessarily cruel, but don't pull punches on legitimate concerns. Science depends on rigorous critique."""


# === Prompt Templates ===

CRITIQUE_METHODS = PromptTemplate(
    name="critique_methods",
    description="Critique the methodology of a study",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Analyze the methodology of this research:

**Research Context:**
{context}

**Focus your critique on:**
1. Study design appropriateness
2. Sample size and selection
3. Control conditions
4. Measurement validity
5. Potential confounders
6. Reproducibility

Provide a structured methodological critique."""
)

CRITIQUE_STATISTICS = PromptTemplate(
    name="critique_statistics",
    description="Evaluate statistical analysis",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Evaluate the statistical analysis in this research:

**Research Context:**
{context}

**Examine:**
1. Appropriateness of statistical tests
2. Sample size adequacy (power)
3. Effect sizes and practical significance
4. Multiple testing corrections
5. Confidence intervals and uncertainty
6. Data presentation clarity

Identify any statistical concerns or exemplary practices."""
)

CRITIQUE_CLAIMS = PromptTemplate(
    name="critique_claims",
    description="Evaluate claims and evidence alignment",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Analyze the alignment between claims and evidence:

**Research Context:**
{context}

**Evaluate:**
1. Are claims supported by the presented evidence?
2. Are there overstatements or unjustified generalizations?
3. Is causation inferred from correlation inappropriately?
4. Are limitations acknowledged appropriately?
5. Are alternative explanations considered?

Identify specific claims and assess their evidential support."""
)

CRITIQUE_FULL = PromptTemplate(
    name="critique_full",
    description="Comprehensive paper critique",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Provide a comprehensive scientific critique:

**Research Context:**
{context}

**Structure your analysis as:**

## Summary
Brief overview of the study

## Strengths
What the paper does well

## Methodological Concerns
Issues with study design or execution

## Statistical Issues
Problems with data analysis or interpretation

## Logical/Interpretive Issues  
Gaps in reasoning or unsupported conclusions

## Overall Assessment
Summary evaluation with key takeaways"""
)

RAG_QUERY = PromptTemplate(
    name="rag_query",
    description="Answer a question using retrieved context",
    system_prompt=RAG_CONTEXT_SYSTEM,
    user_template="""Answer the following question using the provided research context:

**Question:**
{question}

**Relevant Research Excerpts:**
{context}

Provide a thorough, evidence-based answer citing the sources where relevant."""
)

SUMMARIZE_PAPER = PromptTemplate(
    name="summarize_paper",
    description="Summarize a paper's key points",
    system_prompt=SUMMARIZATION_SYSTEM,
    user_template="""Summarize this research paper:

**Paper Content:**
{context}

**Provide:**
1. **Objective:** What the study aimed to do
2. **Methods:** How they did it (briefly)
3. **Key Findings:** Main results
4. **Conclusions:** Authors' interpretations
5. **Limitations:** Acknowledged or apparent
6. **Significance:** Why it matters"""
)

SUMMARIZE_BRIEF = PromptTemplate(
    name="summarize_brief",
    description="Brief abstract-level summary",
    system_prompt=SUMMARIZATION_SYSTEM,
    user_template="""Provide a one-paragraph summary (3-5 sentences) of this paper.

**Abstract:**
{abstract}

**Key Content:**
{context}

**One-paragraph summary:**"""
)

COMPARE_PAPERS = PromptTemplate(
    name="compare_papers",
    description="Compare findings across papers",
    system_prompt=RAG_CONTEXT_SYSTEM,
    user_template="""Compare these research excerpts on the topic: {topic}

**Research Excerpts:**
{context}

**Analyze:**
1. Points of agreement
2. Points of disagreement or tension
3. Methodological differences that might explain discrepancies
4. Gaps in the collective evidence
5. Synthesis: What can we conclude overall?"""
)

EXTRACT_CLAIMS = PromptTemplate(
    name="extract_claims",
    description="Extract and categorize claims from text",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Extract and categorize the claims made in this research:

**Research Context:**
{context}

**For each major claim, identify:**
1. The claim itself (quoted or paraphrased)
2. Type: [Finding / Interpretation / Speculation / Background]
3. Evidence strength: [Strong / Moderate / Weak / None cited]
4. Location/section it appears in

Format as a structured list of claims with assessments."""
)

IDENTIFY_LIMITATIONS = PromptTemplate(
    name="identify_limitations",
    description="Identify stated and unstated limitations",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Identify the limitations of this research, both stated and unstated.

**Research Context:**
{context}

**Analyze:**
1. **Stated Limitations:** What do the authors acknowledge?
2. **Unstated Limitations:** What issues aren't addressed?
3. **Generalizability:** How broadly do findings apply?
4. **Potential Biases:** What biases might affect results?
5. **Missing Elements:** What's conspicuously absent?

Provide a thorough limitations analysis."""
)

BULLSHIT_SCORE = PromptTemplate(
    name="bullshit_score",
    description="The crown jewel — scientific rigor score from 0-10",
    system_prompt=BULLSHIT_DETECTOR_SYSTEM,
    user_template="""Evaluate the scientific rigor of this paper content and assign a "Bullshit Score" from 0-10.

**Scoring Guide:**
- 0-2: Excellent. Rigorous methodology, appropriate claims, transparent limitations
- 3-4: Good. Minor issues but fundamentally sound
- 5-6: Concerning. Notable methodological issues or overclaims
- 7-8: Poor. Significant problems that undermine conclusions
- 9-10: Severe issues. Major red flags suggesting unreliable findings

**Research Context:**
{context}

**Evaluate these dimensions (0-10 each, where 0 is best):**
1. **Methodology Rigor**: Is the approach sound?
2. **Statistical Validity**: Are analyses appropriate?
3. **Claim Calibration**: Do conclusions match evidence?
4. **Transparency**: Are limitations acknowledged?

**Respond in this exact format:**

BULLSHIT SCORE: [X/10]

BREAKDOWN:
- Methodology: [X/10] - [brief reason]
- Statistics: [X/10] - [brief reason]  
- Claims: [X/10] - [brief reason]
- Transparency: [X/10] - [brief reason]

RED FLAGS:
[List any serious concerns, or "None identified" if clean]

SUMMARY:
[2-3 sentence overall assessment]"""
)

QUICK_ASSESS = PromptTemplate(
    name="quick_assess",
    description="Quick red flag check",
    system_prompt=SCIENTIFIC_CRITIC_SYSTEM,
    user_template="""Quickly assess this paper section for any obvious red flags or notable strengths.

**Content:**
{context}

**Quick Assessment (2-3 sentences):**"""
)

CROSS_REFERENCE = PromptTemplate(
    name="cross_reference",
    description="Cross-reference claims across papers",
    system_prompt=RAG_CONTEXT_SYSTEM,
    user_template="""Analyze how the following claim relates to other research:

**Claim to evaluate:**
{claim}

**Related research excerpts:**
{context}

**Analyze:**
1. Does the related research support, contradict, or have no bearing on this claim?
2. What specific evidence is relevant?
3. How should this affect our confidence in the original claim?

**Cross-Reference Analysis:**"""
)


# Registry of all templates
TEMPLATES: dict[str, PromptTemplate] = {
    # Critique templates
    "critique_methods": CRITIQUE_METHODS,
    "critique_statistics": CRITIQUE_STATISTICS,
    "critique_claims": CRITIQUE_CLAIMS,
    "critique_full": CRITIQUE_FULL,
    # Summarization templates
    "summarize_paper": SUMMARIZE_PAPER,
    "summarize_brief": SUMMARIZE_BRIEF,
    # Extraction templates
    "extract_claims": EXTRACT_CLAIMS,
    "identify_limitations": IDENTIFY_LIMITATIONS,
    # RAG templates
    "rag_query": RAG_QUERY,
    "compare_papers": COMPARE_PAPERS,
    "cross_reference": CROSS_REFERENCE,
    # The crown jewel
    "bullshit_score": BULLSHIT_SCORE,
    # Quick tools
    "quick_assess": QUICK_ASSESS,
}


def get_template(name: str) -> PromptTemplate:
    """Get a template by name."""
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return TEMPLATES[name]


def list_templates() -> list[dict]:
    """List all available templates with descriptions."""
    return [
        {"name": t.name, "description": t.description}
        for t in TEMPLATES.values()
    ]


# Type for all available analysis types
AnalysisType = Literal[
    "critique_methods",
    "critique_statistics", 
    "critique_claims",
    "critique_full",
    "summarize_paper",
    "summarize_brief",
    "extract_claims",
    "identify_limitations",
    "rag_query",
    "compare_papers",
    "cross_reference",
    "bullshit_score",
    "quick_assess",
]


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("[bold]SCALPEL Prompt Templates[/bold]\n")
    
    table = Table(title=f"{len(TEMPLATES)} Available Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    
    for template in TEMPLATES.values():
        table.add_row(template.name, template.description)
    
    console.print(table)