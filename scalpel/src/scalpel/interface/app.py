"""
SCALPEL Streamlit UI — Phase 4

Provides a clean web interface for:
  - RAG query with retrieved chunks and similarity scores
  - LLM response display
  - Evaluation scores (groundedness, relevance, confidence calibration)
  - RL agent training history charts
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


def load_rl_history(save_dir: Path = Path("models/rl")) -> list[dict]:
    history_path = save_dir / "training_history.json"
    if not history_path.exists():
        return []
    try:
        return json.loads(history_path.read_text())
    except Exception:
        return []


def load_rl_agent(save_dir: Path = Path("models/rl")):
    """Load the latest trained RL agent if available."""
    final = save_dir / "agent_final.pt"
    checkpoints = sorted(save_dir.glob("agent_iter_*.pt"))
    target = final if final.exists() else (checkpoints[-1] if checkpoints else None)
    if target is None:
        return None
    try:
        import torch
        from scalpel.rl.agent import PPOAgent
        agent = PPOAgent()
        agent.load(target)
        agent.policy.eval()
        return agent
    except Exception:
        return None


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
    use_rl   = st.toggle("Use RL agent for retrieval", value=False,
                          help="Requires a trained agent in models/rl/")
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

tab_query, tab_rl = st.tabs(["🔍 Query & Evaluate", "🤖 RL Training History"])


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

                if use_rl:
                    agent = load_rl_agent()
                    if agent is None:
                        st.warning("No trained RL agent found — using default retrieval.")
                        results = store.search(query, n_results=n_chunks)
                        params_used = None
                    else:
                        import numpy as np
                        import ollama as _ollama
                        from scalpel.config import settings as _s
                        emb = _ollama.embed(model=_s.embedding_model, input=query)
                        state = np.array(emb["embeddings"][0], dtype=np.float32)
                        params = agent.get_params(state)
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


# ── RL History tab ────────────────────────────────────────────────────────────

with tab_rl:
    st.markdown("## RL Agent Training History")

    history = load_rl_history()

    if not history:
        st.info(
            "No training history found. Run `scalpel train-rl` to start training "
            "the retrieval optimisation agent."
        )
    else:
        iterations = [h["iteration"] for h in history]

        # ── Summary metrics ───────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Iterations completed", len(history))
        col2.metric("Best mean reward",
                    f"{max(h['mean_reward'] for h in history):+.3f}")
        col3.metric("Latest mean eval score",
                    f"{history[-1]['mean_eval_score']:.2f}/10")
        if "policy_loss" in history[-1]:
            col4.metric("Latest policy loss",
                        f"{history[-1]['policy_loss']:.4f}")

        st.divider()

        # ── Reward chart ──────────────────────────────────────────────────────
        st.markdown("### Mean reward per iteration")
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=iterations,
            y=[h["mean_reward"] for h in history],
            name="Mean reward",
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.1)",
        ))
        fig_reward.add_hline(y=0, line_dash="dash", line_color="#6b7280",
                             annotation_text="baseline")
        fig_reward.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            yaxis=dict(gridcolor="#374151", title="Reward"),
            xaxis=dict(gridcolor="#374151", title="Iteration", dtick=1),
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        # ── Eval score chart ──────────────────────────────────────────────────
        st.markdown("### Mean evaluation score per iteration")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=iterations,
            y=[h["mean_eval_score"] for h in history],
            name="Mean eval score",
            mode="lines+markers",
            line=dict(color="#22c55e", width=2),
            marker=dict(size=8),
        ))
        fig_eval.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            yaxis=dict(range=[0, 10], gridcolor="#374151", title="Score / 10"),
            xaxis=dict(gridcolor="#374151", title="Iteration", dtick=1),
        )
        st.plotly_chart(fig_eval, use_container_width=True)

        # ── Loss chart ────────────────────────────────────────────────────────
        if "policy_loss" in history[0]:
            st.markdown("### PPO losses")
            fig_loss = go.Figure()
            for key, color, name in [
                ("policy_loss", "#f59e0b", "Policy loss"),
                ("value_loss",  "#38bdf8", "Value loss"),
                ("entropy",     "#a78bfa", "Entropy"),
            ]:
                if key in history[0]:
                    fig_loss.add_trace(go.Scatter(
                        x=iterations,
                        y=[h.get(key, 0) for h in history],
                        name=name,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                    ))
            fig_loss.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(gridcolor="#374151"),
                xaxis=dict(gridcolor="#374151", title="Iteration", dtick=1),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # ── Raw table ─────────────────────────────────────────────────────────
        with st.expander("Raw training data"):
            import pandas as pd
            df = pd.DataFrame(history)
            st.dataframe(df, use_container_width=True)
