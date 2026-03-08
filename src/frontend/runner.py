"""
Streamlit Pipeline Runner – runs the full pipeline directly (no FastAPI)
with live progress and results display.

Launch: streamlit run src/frontend/runner.py
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import nest_asyncio
nest_asyncio.apply()

import asyncio
import sqlite3
import threading
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from src.env import load_env

load_env()

from src.state.schema import PipelineState
from src.graph.builder import build_pipeline_graph
from src.debug.run_db import ensure_run, set_run_status, finalize_run_from_state

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pipeline Runner", page_icon="🚀", layout="wide")

DB_PATH = Path(__file__).resolve().parents[2] / "logs" / "runs.sqlite"

AGENT_ORDER = ["coordinator", "research", "analysis", "writing", "quality"]

def _is_step_ok(status: str) -> bool:
    # Backward-compat with older runs that used "completed".
    return (status or "").strip().lower() in ("ok", "completed", "success", "succeeded")


def _is_step_running(status: str) -> bool:
    return (status or "").strip().lower() in ("running", "in_progress")


def _is_step_error(status: str) -> bool:
    return (status or "").strip().lower() in ("error", "failed", "failure")


def _agent_latest_status(steps_df: pd.DataFrame) -> dict:
    """
    Return a mapping agent_name -> {status, duration_ms} based on the latest step_index per agent.

    This handles pipelines where the same agent can be called multiple times (e.g., quality revision loops).
    """
    latest: dict = {}
    if steps_df is None or steps_df.empty:
        return latest

    df = steps_df.copy()
    if "step_index" in df.columns:
        df = df.sort_values("step_index")

    if "agent_name" not in df.columns:
        return latest

    for agent_name, grp in df.groupby("agent_name", dropna=False):
        if grp.empty:
            continue
        row = grp.iloc[-1].to_dict()
        latest[str(agent_name)] = {
            "status": row.get("status"),
            "duration_ms": row.get("duration_ms"),
        }
    return latest


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
for _key, _default in [
    ("pipeline_running", False),
    ("pipeline_result", None),
    ("run_id", None),
    ("run_completed", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Helper – read agent_steps from SQLite
# ---------------------------------------------------------------------------

def _read_agent_steps(run_id: str) -> pd.DataFrame:
    """Return agent_steps rows for the given run_id, or empty DataFrame."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=5)
        df = pd.read_sql_query(
            "SELECT * FROM agent_steps WHERE run_id = ? ORDER BY step_index",
            con,
            params=(run_id,),
        )
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def _read_llm_calls(run_id: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=5)
        df = pd.read_sql_query(
            "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id",
            con,
            params=(run_id,),
        )
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _run_pipeline_thread(state: dict, result_holder: dict):
    """Execute the pipeline graph in a dedicated event loop / thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        graph = build_pipeline_graph()
        result = loop.run_until_complete(graph.ainvoke(state))
        result_holder["result"] = result
        result_holder["status"] = "completed"
        try:
            finalize_run_from_state(result, status="completed")
        except Exception:
            pass
    except Exception as e:
        result_holder["error"] = str(e)
        result_holder["status"] = "failed"
        # Best-effort finalization so Monitor gets ended_at/status/artifacts even on failure.
        try:
            if hasattr(state, "model_dump"):
                state_dict = state.model_dump()
            elif isinstance(state, dict):
                state_dict = dict(state)
            else:
                state_dict = {}
            state_dict["pipeline_status"] = "failed"
            state_dict["error_message"] = str(e)
            finalize_run_from_state(state_dict, status="failed")
        except Exception:
            pass
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("🚀 Pipeline Runner")

advanced_mode = st.sidebar.checkbox(
    "⚙️ Advanced Mode",
    value=False,
    help="Show all configuration options"
)

topic = st.sidebar.text_area(
    "Topic",
    height=100,
    help="Enter a healthcare topic to research",
    placeholder="e.g., Effectiveness of telemedicine interventions for Type 2 diabetes management in adults",
)

if advanced_mode:
    scope_instructions = st.sidebar.text_area(
        "Scope Instructions (optional)",
        height=80,
        placeholder="e.g., Focus on adults 18+, exclude pediatric cases",
    )
    target_audience = st.sidebar.selectbox(
        "Target Audience",
        options=["clinical practitioners", "general public", "researchers", "policy makers"],
    )
    report_format = st.sidebar.selectbox(
        "Report Format",
        options=["clinical_brief", "full_report"],
    )
    max_iterations = st.sidebar.slider("Max Quality Iterations", min_value=1, max_value=5, value=3)
else:
    scope_instructions = ""
    target_audience = "clinical practitioners"
    report_format = "clinical_brief"
    max_iterations = 3

st.sidebar.markdown("---")

run_clicked = st.sidebar.button(
    "Run Pipeline",
  
    disabled=not topic.strip(),
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🚀 Pipeline Runner")
st.caption("Run the multi-agent healthcare intelligence pipeline directly — no API server needed.")

# ---- Handle RUN click ----
if run_clicked and topic.strip():
    # Clear previous results
    st.session_state.pipeline_result = None
    st.session_state.run_completed = False

    run_id = "streamlit-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    st.session_state.run_id = run_id

    # Register the run so Monitor can see it and so finalize_run_from_state works in Streamlit mode.
    try:
        ensure_run(
            run_id=run_id,
            source="streamlit",
            topic=topic.strip(),
            scope_instructions=scope_instructions.strip(),
            target_audience=target_audience,
            report_format=report_format,
            status="pending",
            log_path=str(Path("logs") / "runs" / f"{run_id}.jsonl"),
        )
        set_run_status(run_id, "running")
    except Exception:
        pass

    state = PipelineState(
        run_id=run_id,
        topic=topic.strip(),
        scope_instructions=scope_instructions.strip(),
        target_audience=target_audience,
        report_format=report_format,
        requested_at=datetime.now(),
        research_queries=[],
        scope_boundaries={"in_scope": [], "out_of_scope": []},
        priority_subtopics=[],
        raw_sources=[],
        deduplicated_sources=[],
        research_summary="",
        clinical_claims=[],
        evidence_gaps=[],
        contradictions=[],
        statistical_findings=[],
        analysis_narrative="",
        report_sections={},
        report_markdown="",
        citations=[],
        report_word_count=0,
        quality_issues=[],
        quality_verdict="pending",
        quality_score=0.0,
        revision_instructions="",
        quality_iteration=0,
        max_quality_iterations=max_iterations,
        should_revise=False,
        current_agent="",
        agent_history=[],
        pipeline_status="pending",
        error_message=None,
    )

    result_holder: dict = {"status": "running"}

    thread = threading.Thread(target=_run_pipeline_thread, args=(state, result_holder), daemon=True)
    thread.start()

    # Live-progress polling
    status_placeholder = st.empty()
    status_placeholder.info("Running pipeline...")
    progress_placeholder = st.empty()
    start_time = time.time()

    while result_holder["status"] == "running":
        elapsed = time.time() - start_time
        steps_df = _read_agent_steps(run_id)

        lines: list[str] = []
        latest = _agent_latest_status(steps_df)

        for agent in AGENT_ORDER:
            st_info = latest.get(agent, {})
            st_status = (st_info.get("status") or "").strip()
            if _is_step_ok(st_status):
                dur = ""
                dur_ms = st_info.get("duration_ms")
                if dur_ms is not None and pd.notna(dur_ms):
                    dur = f" ({float(dur_ms) / 1000:.1f}s)"
                lines.append(f"✅ {agent.title()}{dur}")
            elif _is_step_running(st_status):
                lines.append(f"⏳ {agent.title()}…")
            elif _is_step_error(st_status):
                lines.append(f"❌ {agent.title()}")
            else:
                lines.append(f"⬜ {agent.title()}")

        lines.append(f"\n_Elapsed: {elapsed:.0f}s_")
        progress_placeholder.markdown("\n\n".join(lines))
        time.sleep(2)

    total_time = time.time() - start_time

    if result_holder["status"] == "completed":
        status_placeholder.success("Pipeline completed!")
        st.session_state.pipeline_result = result_holder["result"]
        st.session_state.run_completed = True
    else:
        status_placeholder.error("Pipeline failed.")
        st.error(f"Pipeline error: {result_holder.get('error', 'Unknown error')}")

# ---- Display results ----
if st.session_state.run_completed and st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    run_id = st.session_state.run_id

    st.success(f"✅ Pipeline completed successfully")

    tab_report, tab_sources, tab_quality, tab_metrics = st.tabs(
        ["📄 Report", "📚 Sources", "✅ Quality Review", "📊 Metrics"]
    )

    # --- Report tab ---
    with tab_report:
        md = result.get("report_markdown", "")
        if md:
            st.markdown(md)
        else:
            st.info("No report content was generated.")

    # --- Sources tab ---
    with tab_sources:
        sources = result.get("deduplicated_sources", [])
        if sources:
            st.subheader(f"Sources ({len(sources)})")
            for i, src in enumerate(sources, 1):
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                score = src.get("relevance_score")
                score_str = f" — relevance {score:.2f}" if score is not None else ""
                if url:
                    st.markdown(f"{i}. [{title}]({url}){score_str}")
                else:
                    st.markdown(f"{i}. {title}{score_str}")
        else:
            st.info("No sources available.")

    # --- Quality tab ---
    with tab_quality:
        qscore = result.get("quality_score", 0.0)
        verdict = result.get("quality_verdict", "N/A")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{qscore:.0f} / 100")
            st.progress(min(qscore / 100, 1.0))
        with col2:
            verdict_emoji = {"pass": "✅", "revise": "🔄", "reject": "❌"}
            st.metric("Verdict", f"{verdict_emoji.get(verdict, '❓')} {verdict.upper()}")

        issues = result.get("quality_issues", [])
        if issues:
            st.subheader("Issues")
            for issue in issues:
                sev = issue.get("severity", "")
                sec = issue.get("section", "")
                desc = issue.get("description", "")
                st.markdown(f"- **[{sev.upper()}]** _{sec}_ — {desc}")
        else:
            st.info("No quality issues found.")

    # --- Metrics tab ---
    with tab_metrics:
        steps_df = _read_agent_steps(run_id)
        llm_df = _read_llm_calls(run_id)

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_dur = steps_df["duration_ms"].sum() / 1000 if not steps_df.empty and "duration_ms" in steps_df.columns else 0
            st.metric("Total Duration", f"{total_dur:.1f}s")
        with c2:
            total_tok = int(llm_df["total_tokens"].sum()) if not llm_df.empty and "total_tokens" in llm_df.columns else 0
            st.metric("Total Tokens", f"{total_tok:,}")
        with c3:
            st.metric("Sources", len(result.get("deduplicated_sources", [])))
        with c4:
            st.metric("Claims", len(result.get("clinical_claims", [])))

        # Agent duration chart
        if not steps_df.empty and "duration_ms" in steps_df.columns:
            try:
                import plotly.express as px

                dur_df = steps_df[["agent_name", "duration_ms"]].copy()
                dur_df["duration_s"] = dur_df["duration_ms"] / 1000
                fig = px.bar(
                    dur_df,
                    y="agent_name",
                    x="duration_s",
                    orientation="h",
                    title="Agent Duration (seconds)",
                    labels={"agent_name": "Agent", "duration_s": "Duration (s)"},
                )
                fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": list(reversed(AGENT_ORDER))})
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(steps_df[["agent_name", "duration_ms"]])

        # Token usage chart
        if not llm_df.empty and "input_tokens" in llm_df.columns:
            try:
                import plotly.express as px

                tok_agg = llm_df.groupby("agent_name")[["input_tokens", "output_tokens"]].sum().reset_index()
                tok_melted = tok_agg.melt(id_vars="agent_name", value_vars=["input_tokens", "output_tokens"],
                                          var_name="type", value_name="tokens")
                fig2 = px.bar(
                    tok_melted,
                    x="agent_name",
                    y="tokens",
                    color="type",
                    barmode="group",
                    title="Token Usage per Agent",
                    labels={"agent_name": "Agent", "tokens": "Tokens"},
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.dataframe(llm_df[["agent_name", "input_tokens", "output_tokens"]])

        if steps_df.empty and llm_df.empty:
            st.info("No metrics data found in the database for this run.")

elif not st.session_state.run_completed and not run_clicked:
    # Landing page
    st.markdown("""
    ### How it works

    This app runs the **Healthcare Content Intelligence Pipeline** end-to-end,
    streaming progress as each agent completes.

    | # | Agent | Role |
    |---|-------|------|
    | 1 | **Coordinator** | Plans research queries and defines scope |
    | 2 | **Research** | Searches medical literature via Tavily |
    | 3 | **Analysis** | Extracts clinical claims, evidence gaps & contradictions |
    | 4 | **Writing** | Composes a structured clinical report |
    | 5 | **Quality** | Reviews and may request revisions (up to N iterations) |

    ---
    💡 **Sample topic:** _Effectiveness of telemedicine interventions for Type 2 diabetes management in adults_
    """)
