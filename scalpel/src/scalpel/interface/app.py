"""
SCALPEL Streamlit UI

Provides a clean web interface for:
  - RAG query with retrieved chunks and similarity scores
  - LLM response display
  - Evaluation scores (groundedness, relevance, confidence calibration)
  - Collection stats and similarity index explorer
"""

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SCALPEL",
    page_icon="🔪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.chunk-card {
    background: #1e1e2e;
    border-left: 4px solid #7c3aed;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    line-height: 1.6;
}
.score-high  { color: #22c55e; font-weight: 700; }
.score-mid   { color: #f59e0b; font-weight: 700; }
.score-low   { color: #ef4444; font-weight: 700; }
.section-tag {
    background: #7c3aed22;
    color: #a78bfa;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78rem;
    margin-left: 8px;
}
.eval-bar-label { font-size: 0.82rem; color: #9ca3af; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to vector store...")
def get_store():
    from scalpel.embeddings import get_store
    return get_store()


@st.cache_resource(show_spinner="Loading LLM client...")
def get_llm_client():
    from scalpel.analysis.llm_client import get_client
    return get_client()


@st.cache_resource(show_spinner="Loading evaluator...")
def get_evaluator():
    from scalpel.evaluation import get_evaluator
    return get_evaluator()


def load_retrieval_cache(
    cache_path: Path = Path("models/retrieval/step_cache.json"),
) -> list[dict]:
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return list(raw.values())
    except Exception:
        return []


@st.cache_resource(show_spinner="Building similarity index...")
def get_optimizer(cache_path: str = "models/retrieval/step_cache.json"):
    from scalpel.retrieval.cache import RetrievalCache
    from scalpel.retrieval.optimizer import RetrievalOptimizer
    cache = RetrievalCache(path=Path(cache_path))
    optimizer = RetrievalOptimizer(cache)
    optimizer.build_index()
    return optimizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score: float, out_of: float = 10.0) -> str:
    ratio = score / out_of
    if ratio >= 0.7:
        return "score-high"
    elif ratio >= 0.4:
        return "score-mid"
    return "score-low"


def render_score_bar(label: str, value: float, max_val: float = 10.0):
    ratio = value / max_val
    color = "#22c55e" if ratio >= 0.7 else "#f59e0b" if ratio >= 0.4 else "#ef4444"
    fig = go.Figure(go.Bar(
        x=[value],
        y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{value:.1f}/{max_val:.0f}"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.update_layout(
        height=45,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, max_val], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🔪 SCALPEL")
    st.caption("Scientific Critique & Analysis Pipeline")
    st.divider()

    n_chunks = st.slider("Chunks to retrieve", 3, 10, 5)
    use_optimizer = st.toggle(
        "Use similarity optimizer",
        value=False,
        help="Picks retrieval params based on similar past queries. "
             "Run `scalpel collect` first to build the index.",
    )
    st.divider()

    # Library stats
    try:
        store = get_store()
        stats = store.get_stats()
        st.metric("Papers indexed", stats["total_papers"])
        st.metric("Total chunks", stats["total_chunks"])
        st.caption(f"Embedding: `{stats['embedding_model']}`")
    except Exception as e:
        st.warning(f"Vector store unavailable: {e}")

    st.divider()

    try:
        from scalpel.config import settings
        st.caption(f"Provider: **{settings.llm_provider}**")
        st.caption(f"Model: `{settings.active_model}`")
    except Exception:
        pass


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_query, tab_rl = st.tabs(["🔍 Query & Evaluate", "📊 Collection Stats"])


# ── Query tab ─────────────────────────────────────────────────────────────────

with tab_query:
    st.markdown("## Ask a question")

    query = st.text_input(
        "Query",
        placeholder="e.g. How does the attention mechanism work?",
        label_visibility="collapsed",
    )

    run = st.button("Run", type="primary", disabled=not query)

    if run and query:
        # ── Retrieval ─────────────────────────────────────────────────────────
        with st.spinner("Retrieving chunks..."):
            try:
                store = get_store()

                if use_optimizer:
                    optimizer = get_optimizer()
                    if optimizer.index_size == 0:
                        st.warning(
                            "Similarity index is empty — using default retrieval. "
                            "Run `scalpel collect` to build the index."
                        )
                        results = store.search(query, n_results=n_chunks)
                        params_used = None
                    else:
                        import numpy as np
                        import ollama as _ollama
                        from scalpel.config import settings as _s
                        emb = _ollama.embed(model=_s.embedding_model, input=query)
                        state = np.array(emb["embeddings"][0], dtype=np.float32)
                        params = optimizer.get_params(state)
                        results = store.search(query, n_results=params.n_chunks)
                        if params.similarity_threshold > 0:
                            results = [r for r in results if r.score >= params.similarity_threshold]
                        if params.rerank:
                            priority = {"Methods", "Results", "Methodology", "Experimental"}
                            results.sort(key=lambda r: (0 if r.section in priority else 1, -r.score))
                        params_used = params
                else:
                    results = store.search(query, n_results=n_chunks)
                    params_used = None
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.stop()

        if not results:
            st.warning("No results found. Make sure papers are indexed with `scalpel add`.")
            st.stop()

        # ── Chunks ────────────────────────────────────────────────────────────
        col_chunks, col_response = st.columns([1, 1], gap="large")

        with col_chunks:
            st.markdown(f"### Retrieved chunks  `{len(results)}`")
            if params_used:
                st.caption(f"RL params: {params_used}")

            for i, r in enumerate(results):
                pct = int(r.score * 100)
                css_class = score_color(r.score, 1.0)
                section_html = (
                    f'<span class="section-tag">{r.section}</span>'
                    if r.section else ""
                )
                preview = r.text[:400].replace("\n", " ")
                if len(r.text) > 400:
                    preview += "…"
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
                    f'<b style="color:#a78bfa">{r.paper_title[:50]}</b>'
                    f'<span class="{css_class}">{pct}%</span>'
                    f'</div>'
                    f'{section_html}'
                    f'<div style="margin-top:8px;color:#d1d5db">{preview}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Generation ────────────────────────────────────────────────────────
        chunk_texts = [r.text for r in results]
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{t}" for i, t in enumerate(chunk_texts)
        )

        with col_response:
            st.markdown("### Response")
            with st.spinner("Generating response..."):
                try:
                    client = get_llm_client()
                    from scalpel.analysis.prompts import get_template
                    template = get_template("rag_query")
                    sys_p, usr_p = template.format(question=query, context=context)
                    llm_resp = client.generate(prompt=usr_p, system=sys_p)
                    response_text = llm_resp.content
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.stop()

            st.markdown(response_text)

        # ── Evaluation ────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Evaluation scores")

        with st.spinner("Evaluating response..."):
            try:
                evaluator = get_evaluator()
                score = evaluator.evaluate(query, chunk_texts, response_text, verbose=False)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.stop()

        col_g, col_r, col_c, col_o = st.columns(4)
        with col_g:
            st.markdown('<div class="eval-bar-label">Groundedness</div>', unsafe_allow_html=True)
            render_score_bar("", score.groundedness)
        with col_r:
            st.markdown('<div class="eval-bar-label">Relevance</div>', unsafe_allow_html=True)
            render_score_bar("", score.relevance)
        with col_c:
            st.markdown('<div class="eval-bar-label">Confidence Calibration</div>', unsafe_allow_html=True)
            render_score_bar("", score.confidence_calibration)
        with col_o:
            st.markdown('<div class="eval-bar-label">Overall</div>', unsafe_allow_html=True)
            render_score_bar("", score.overall)

        if score.unsupported_claims:
            with st.expander(f"⚠️ {len(score.unsupported_claims)} unsupported claim(s)"):
                for claim in score.unsupported_claims:
                    st.markdown(f"- {claim}")

        if score.reasoning:
            with st.expander("Judge reasoning"):
                st.markdown(score.reasoning)

        # ── Session history ───────────────────────────────────────────────────
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "query": query,
            "overall": score.overall,
            "groundedness": score.groundedness,
            "relevance": score.relevance,
            "confidence": score.confidence_calibration,
            "n_chunks": len(results),
        })

        if len(st.session_state.history) > 1:
            st.divider()
            st.markdown("### Session history")
            fig = go.Figure()
            queries_short = [h["query"][:30] + "…" for h in st.session_state.history]
            for metric, color, name in [
                ("overall",       "#a78bfa", "Overall"),
                ("groundedness",  "#22c55e", "Groundedness"),
                ("relevance",     "#38bdf8", "Relevance"),
                ("confidence",    "#f59e0b", "Confidence"),
            ]:
                fig.add_trace(go.Scatter(
                    x=queries_short,
                    y=[h[metric] for h in st.session_state.history],
                    name=name,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(range=[0, 10], gridcolor="#374151"),
                xaxis=dict(gridcolor="#374151"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Collection Stats tab ──────────────────────────────────────────────────────

with tab_rl:
    st.markdown("## Collection Stats")
    st.caption(
        "Run `scalpel collect` to gather (query, params, score) data. "
        "The more you collect, the smarter the similarity optimizer becomes."
    )

    entries = load_retrieval_cache()

    if not entries:
        st.info(
            "No collection data yet. Run `scalpel collect` to start building "
            "the retrieval parameter index."
        )
    else:
        import pandas as pd

        df = pd.DataFrame(entries)

        # ── Summary metrics ───────────────────────────────────────────────────
        unique_queries = df["query"].nunique() if "query" in df.columns else 0
        mean_score = df["eval_score"].mean() if "eval_score" in df.columns else 0
        best_score = df["eval_score"].max() if "eval_score" in df.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total entries", len(df))
        col2.metric("Unique queries", unique_queries)
        col3.metric("Mean eval score", f"{mean_score:.2f}/10")
        col4.metric("Best eval score", f"{best_score:.2f}/10")

        st.divider()

        # ── Score distribution ────────────────────────────────────────────────
        if "eval_score" in df.columns:
            st.markdown("### Score distribution")
            fig_hist = go.Figure(go.Histogram(
                x=df["eval_score"],
                nbinsx=20,
                marker_color="#7c3aed",
                opacity=0.8,
            ))
            fig_hist.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis=dict(title="Eval score (0–10)", gridcolor="#374151"),
                yaxis=dict(title="Count", gridcolor="#374151"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Best params per query ─────────────────────────────────────────────
        if "query" in df.columns and "eval_score" in df.columns:
            st.markdown("### Best parameters per query")
            best_rows = (
                df.sort_values("eval_score", ascending=False)
                .drop_duplicates(subset=["query"])
                .reset_index(drop=True)
            )

            display_cols = [c for c in ["query", "eval_score", "action"] if c in best_rows.columns]
            if display_cols:
                best_rows["query_short"] = best_rows["query"].str[:60]
                st.dataframe(
                    best_rows[["query_short", "eval_score", "action"]].rename(
                        columns={"query_short": "query", "eval_score": "score"}
                    ),
                    use_container_width=True,
                )

        # ── Raw data ──────────────────────────────────────────────────────────
        with st.expander("Raw cache entries"):
            show_cols = [c for c in ["query", "action", "eval_score", "reward"] if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True)
