"""
Streamlit monitoring dashboard for the Healthcare Content Intelligence Pipeline.
Reads from logs/runs.sqlite to display run metrics, agent performance, and quality data.
"""

import json
import os
import sys
import sqlite3
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.env import load_env

load_env()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pipeline Monitor",
    page_icon="📊",
    layout="wide",
)

DB_PATH = Path(__file__).resolve().parents[2] / "logs" / "runs.sqlite"
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs" / "runs"
ARTIFACTS_DIR = Path(os.getenv("RUN_ARTIFACTS_DIR", "").strip()) if os.getenv("RUN_ARTIFACTS_DIR", "").strip() else (Path(__file__).resolve().parents[2] / "logs" / "artifacts")

def _st_dataframe(df: pd.DataFrame, **kwargs):
    """
    Streamlit 1.12 compatibility wrapper.

    Newer Streamlit versions support `use_container_width=...`; 1.12 does not.
    """
    try:
        return st.dataframe(df, **kwargs)
    except TypeError:
        kwargs.pop("use_container_width", None)
        return st.dataframe(df, **kwargs)


def _artifact_path(run_id: str, name: str) -> Path:
    return ARTIFACTS_DIR / str(run_id) / name


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh (every 30s)", value=False)
    refresh_now = st.button("🔄 Refresh now")
    st.markdown("---")
    max_display_runs = st.slider("Recent runs to display", min_value=5, max_value=100, value=20, step=5,
                                  help="Limit charts to N most recent runs to avoid clutter")
    st.markdown("---")
    st.caption(f"**DB path:** `{DB_PATH}`")
    st.caption(f"**Exists:** {'✅' if DB_PATH.exists() else '❌'}")

if auto_refresh:
    st.caption("Auto-refresh is enabled. This page will refresh every ~30 seconds.")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache(ttl=30)
def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a read-only query and return a DataFrame."""
    try:
        with _get_connection() as conn:
            conn.execute("PRAGMA query_only = ON")
            return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


@st.cache(ttl=30)
def load_run_logs(run_id: str) -> list[dict]:
    """Load per-run JSONL log events from logs/runs/<run_id>.jsonl."""
    log_file = LOGS_DIR / f"{run_id}.jsonl"
    if not log_file.exists():
        return []
    try:
        events = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
    except Exception:
        return []


@st.cache(ttl=30)
def load_artifact_text(run_id: str, name: str) -> str:
    path = _artifact_path(run_id, name)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


@st.cache(ttl=30)
def load_artifact_json(run_id: str, name: str):
    path = _artifact_path(run_id, name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _report_preview_from_logs(log_events: list[dict]) -> str:
    if not log_events:
        return ""
    # Prefer the last agent_state snapshot that includes a report preview.
    preview = ""
    for ev in log_events:
        if ev.get("type") != "agent_state":
            continue
        st_obj = ev.get("state") or {}
        rp = st_obj.get("report_preview") or ""
        if isinstance(rp, str) and rp.strip():
            preview = rp.strip()
    return preview


if refresh_now:
    # Streamlit 1.12 uses st.cache (not st.cache_data).
    try:
        query_df.clear()
    except Exception:
        pass
    try:
        load_run_logs.clear()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------
if not DB_PATH.exists():
    st.title("📊 Pipeline Monitor")
    st.info(
        f"Database not found at `{DB_PATH}`.\n\n"
        "Run the pipeline at least once to generate the monitoring database."
    )
    st.stop()

st.title("📊 Pipeline Monitor")

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
try:
    kpi = query_df(
        """
        SELECT
            COUNT(*)                                          AS total_runs,
            ROUND(100.0 * SUM(CASE WHEN status IN ('ok', 'completed') THEN 1 ELSE 0 END) / MAX(COUNT(*), 1), 1) AS success_rate,
            ROUND(AVG(total_ms) / 1000.0, 1)                AS avg_duration_s,
            ROUND(AVG(quality_score_final), 2)               AS avg_quality
        FROM runs
        """
    ).copy()

    if kpi.empty or kpi.iloc[0]["total_runs"] == 0:
        st.info("No runs recorded yet.")
        st.stop()

    row = kpi.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏃 Total Runs", int(row["total_runs"]))
    c2.metric("✅ Success Rate", f"{row['success_rate']}%")
    c3.metric("⏱️ Avg Duration", f"{row['avg_duration_s']}s")
    c4.metric("⭐ Avg Quality Score", row["avg_quality"] if pd.notna(row["avg_quality"]) else "N/A")
except Exception as e:
    st.error(f"Error loading KPIs: {e}")

# ---------------------------------------------------------------------------
# Tabs layout
# ---------------------------------------------------------------------------
st.markdown("---")

# Initialize drill-down selection before tabs
selected_run = "(none)"

tab_history, tab_agents, tab_tokens, tab_quality, tab_errors = st.tabs([
    "📋 Run History",
    "⏳ Agent Performance",
    "🪙 Token Usage",
    "⭐ Quality",
    "🚨 Errors",
])

# ---------------------------------------------------------------------------
# Tab: Run History
# ---------------------------------------------------------------------------
with tab_history:
    try:
        runs_df = query_df(
            """
            SELECT
                run_id,
                SUBSTR(topic, 1, 50) AS topic,
                status,
                ROUND(total_ms / 1000.0, 1) AS duration_s,
                quality_score_final AS quality_score,
                quality_verdict_final AS quality_verdict,
                created_at
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max_display_runs,),
        ).copy()

        if runs_df.empty:
            st.info("No runs found.")
        else:
            _st_dataframe(runs_df, use_container_width=True)

            run_ids = ["(none)"] + runs_df["run_id"].tolist()
            selected_run = st.selectbox("Select a run for drill-down", run_ids, key="run_select")

            # --- Inline drill-down ---
            if selected_run != "(none)":
                st.markdown("---")
                st.subheader(f"🔍 Drill-Down: `{selected_run}`")

                # Run metadata and artifact-backed content
                run_meta_df = query_df("SELECT * FROM runs WHERE run_id = ? LIMIT 1", (selected_run,)).copy()
                run_meta = run_meta_df.iloc[0].to_dict() if not run_meta_df.empty else {}

                total_ms = float(run_meta.get("total_ms") or 0.0)
                duration_s = total_ms / 1000.0
                tok_total = int(run_meta.get("llm_total_tokens") or 0)
                src_cnt = int(run_meta.get("deduplicated_sources_count") or 0)
                claim_cnt = int(run_meta.get("clinical_claims_count") or 0)
                status = run_meta.get("status") or "N/A"
                verdict = run_meta.get("quality_verdict_final") or "N/A"
                qscore = run_meta.get("quality_score_final")
                err = (run_meta.get("error_message") or "").strip()

                h1, h2, h3, h4, h5 = st.columns(5)
                h1.metric("Status", str(status))
                h2.metric("Duration", f"{duration_s:.1f}s")
                h3.metric("Total Tokens", f"{tok_total:,}")
                h4.metric("Sources", src_cnt)
                h5.metric("Claims", claim_cnt)

                report_md = load_artifact_text(selected_run, "report.md")
                log_events = load_run_logs(selected_run)
                if err:
                    st.warning(f"Error: {err}")

                dd_overview, dd_report, dd_sources, dd_quality, dd_logs = st.tabs(
                    ["📊 Overview", "📄 Report", "📚 Sources", "✅ Quality", "🪵 Logs"]
                )

                with dd_overview:
                    # Agent steps timeline (Gantt-style)
                    steps_df = query_df(
                        """
                        SELECT step_index, agent_name, duration_ms, status,
                               raw_sources_count, deduplicated_sources_count,
                               clinical_claims_count, citations_count, report_word_count
                        FROM agent_steps
                        WHERE run_id = ?
                        ORDER BY step_index
                        """,
                        (selected_run,),
                    ).copy()

                    if steps_df.empty:
                        st.info("No agent steps recorded for this run.")
                    else:
                        st.markdown("**Agent Steps Timeline**")
                        steps_df["duration_ms"] = steps_df["duration_ms"].fillna(0)
                        steps_df["duration_s"] = steps_df["duration_ms"] / 1000.0
                        fig_gantt = px.bar(
                            steps_df,
                            x="duration_s",
                            y="agent_name",
                            color="status",
                            orientation="h",
                            title=f"Agent Steps — {selected_run}",
                            labels={"duration_s": "Duration (s)", "agent_name": "Agent"},
                        )
                        fig_gantt.update_layout(
                            yaxis={
                                "categoryorder": "array",
                                "categoryarray": steps_df["agent_name"].tolist()[::-1],
                            }
                        )
                        st.plotly_chart(fig_gantt, use_container_width=True)

                        # Step-by-step content growth
                        growth_cols = [
                            "raw_sources_count",
                            "deduplicated_sources_count",
                            "clinical_claims_count",
                            "citations_count",
                            "report_word_count",
                        ]
                        available = [c for c in growth_cols if c in steps_df.columns]
                        growth_data = steps_df[["step_index"] + available].dropna(how="all", subset=available)
                        if not growth_data.empty:
                            st.markdown("**Step-by-Step Content Growth**")
                            melted_growth = growth_data.melt(
                                id_vars=["step_index"],
                                value_vars=available,
                                var_name="metric",
                                value_name="count",
                            )
                            melted_growth["count"] = melted_growth["count"].fillna(0)
                            fig_growth = px.line(
                                melted_growth,
                                x="step_index",
                                y="count",
                                color="metric",
                                title=f"Content Growth — {selected_run}",
                                labels={"step_index": "Step", "count": "Count"},
                                markers=True,
                            )
                            st.plotly_chart(fig_growth, use_container_width=True)

                    # LLM calls table
                    llm_df = query_df(
                        """
                        SELECT agent_name, model, call_kind, duration_ms,
                               input_tokens, output_tokens, status
                        FROM llm_calls
                        WHERE run_id = ?
                        ORDER BY id
                        """,
                        (selected_run,),
                    ).copy()

                    if llm_df.empty:
                        st.info("No LLM calls recorded for this run.")
                    else:
                        st.markdown("**LLM Calls**")
                        display_llm = llm_df.copy()
                        display_llm["duration_s"] = (display_llm["duration_ms"].fillna(0) / 1000.0).round(2)
                        display_llm = display_llm.drop(columns=["duration_ms"])
                        _st_dataframe(display_llm, use_container_width=True)

                        # Per-agent token breakdown
                        step_tokens = query_df(
                            """
                            SELECT
                                COALESCE(s.agent_name, l.agent_name) AS agent_name,
                                SUM(l.input_tokens) AS input_tokens,
                                SUM(l.output_tokens) AS output_tokens
                            FROM llm_calls l
                            LEFT JOIN agent_steps s ON l.step_id = s.id
                            WHERE l.run_id = ?
                            GROUP BY COALESCE(s.agent_name, l.agent_name)
                            """,
                            (selected_run,),
                        ).copy()

                        if not step_tokens.empty:
                            st.markdown("**Per-Agent Token Breakdown**")
                            melted_tok = step_tokens.melt(
                                id_vars=["agent_name"],
                                value_vars=["input_tokens", "output_tokens"],
                                var_name="type",
                                value_name="tokens",
                            )
                            fig_step_tok = px.bar(
                                melted_tok,
                                x="agent_name",
                                y="tokens",
                                color="type",
                                barmode="group",
                                title=f"Token Breakdown — {selected_run}",
                                labels={"agent_name": "Agent", "tokens": "Tokens"},
                            )
                            st.plotly_chart(fig_step_tok, use_container_width=True)

                with dd_report:
                    if report_md.strip():
                        st.markdown(report_md)
                    else:
                        preview = _report_preview_from_logs(log_events)
                        st.info("No `report.md` artifact found for this run.")
                        if preview:
                            st.markdown("**Report Preview**")
                            st.markdown(preview)
                        st.caption(f"Expected: `{_artifact_path(selected_run, 'report.md')}`")

                with dd_sources:
                    sources = load_artifact_json(selected_run, "sources.json")
                    if isinstance(sources, list) and sources:
                        st.subheader(f"Sources ({len(sources)})")
                        for i, src in enumerate(sources, 1):
                            if not isinstance(src, dict):
                                continue
                            title = src.get("title", "Untitled")
                            url = src.get("url", "")
                            score = src.get("relevance_score")
                            score_str = f" — relevance {score:.2f}" if score is not None else ""
                            if url:
                                st.markdown(f"{i}. [{title}]({url}){score_str}")
                            else:
                                st.markdown(f"{i}. {title}{score_str}")
                    else:
                        st.info("No `sources.json` artifact found for this run.")
                        st.caption(f"Expected: `{_artifact_path(selected_run, 'sources.json')}`")

                with dd_quality:
                    col1, col2 = st.columns(2)
                    with col1:
                        if qscore is None or (isinstance(qscore, float) and pd.isna(qscore)):
                            st.metric("Quality Score", "N/A")
                        else:
                            qf = float(qscore)
                            st.metric("Quality Score", f"{qf:.0f} / 100")
                            st.progress(min(max(qf / 100.0, 0.0), 1.0))
                    with col2:
                        verdict_emoji = {"pass": "✅", "revise": "🔄", "reject": "❌"}
                        v = str(verdict).lower()
                        st.metric("Verdict", f"{verdict_emoji.get(v, '❓')} {str(verdict).upper()}")

                    qobj = load_artifact_json(selected_run, "quality.json") or {}
                    issues = qobj.get("quality_issues") if isinstance(qobj, dict) else None
                    if isinstance(issues, list) and issues:
                        st.subheader("Issues")
                        for issue in issues:
                            if not isinstance(issue, dict):
                                continue
                            sev = issue.get("severity", "")
                            sec = issue.get("section", "")
                            desc = issue.get("description", "")
                            st.markdown(f"- **[{str(sev).upper()}]** _{sec}_ — {desc}")
                    else:
                        st.info("No quality issues recorded (or artifact missing).")

                    rev = qobj.get("revision_instructions") if isinstance(qobj, dict) else ""
                    if isinstance(rev, str) and rev.strip():
                        st.subheader("Revision Instructions")
                        st.code(rev[:4000], language="text")

                with dd_logs:
                    if not log_events:
                        st.info(f"No log file found at `logs/runs/{selected_run}.jsonl`.")
                    else:
                        llm_io_events = [e for e in log_events if e.get("type") == "llm_io" and e.get("response")]
                        if not llm_io_events:
                            st.info("No LLM response events found in the log.")
                        else:
                            agents_seen = []
                            for ev in llm_io_events:
                                agent = ev.get("agent", "unknown")
                                if agent not in agents_seen:
                                    agents_seen.append(agent)

                            for agent in agents_seen:
                                agent_events = [e for e in llm_io_events if e.get("agent") == agent]
                                with st.expander(
                                    f"🔹 {agent.title()} ({len(agent_events)} call{'s' if len(agent_events) != 1 else ''})",
                                    expanded=False,
                                ):
                                    for idx, ev in enumerate(agent_events):
                                        if len(agent_events) > 1:
                                            st.markdown(f"**Call {idx + 1}**")

                                        meta = ev.get("meta", {})
                                        usage = meta.get("usage", {})
                                        model = ev.get("model", "N/A")
                                        provider = ev.get("provider", "N/A")

                                        col_m1, col_m2, col_m3 = st.columns(3)
                                        col_m1.caption(f"**Provider:** {provider} / **Model:** {model}")
                                        col_m2.caption(
                                            f"**Tokens:** {usage.get('prompt_tokens', usage.get('input_tokens', '?'))} in / {usage.get('completion_tokens', usage.get('output_tokens', '?'))} out"
                                        )
                                        col_m3.caption(f"**Finish:** {meta.get('finish_reason', 'N/A')}")

                                        sys_prompt = ev.get("system_prompt", "")
                                        if sys_prompt:
                                            show_sys = st.checkbox(
                                                "Show system prompt",
                                                value=False,
                                                key=f"show_sys_{selected_run}_{agent}_{idx}",
                                            )
                                            if show_sys:
                                                st.code(sys_prompt[:3000], language="text")

                                        messages = ev.get("messages", [])
                                        if messages:
                                            last_user = [m for m in messages if m.get("role") == "user"]
                                            if last_user:
                                                show_user = st.checkbox(
                                                    "Show user prompt",
                                                    value=False,
                                                    key=f"show_user_{selected_run}_{agent}_{idx}",
                                                )
                                                if show_user:
                                                    st.code(last_user[-1].get("content", "")[:3000], language="text")

                                        response = ev.get("response", "")
                                        if response:
                                            preview = response[:2000]
                                            if len(response) > 2000:
                                                preview += f"\n\n... (truncated, {len(response)} chars total)"
                                            st.code(preview, language="json")

                                        if idx < len(agent_events) - 1:
                                            st.markdown("---")
    except Exception as e:
        st.error(f"Error loading run history: {e}")
        runs_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# Tab: Agent Performance
# ---------------------------------------------------------------------------
with tab_agents:
    try:
        duration_df = query_df(
            """
            SELECT
                run_id,
                total_ms,
                coordinator_total_ms,
                research_total_ms,
                analysis_total_ms,
                writing_total_ms,
                quality_total_ms,
                created_at
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max_display_runs,),
        ).copy()

        if duration_df.empty:
            st.info("No duration data available.")
        else:
            agent_cols = [
                "coordinator_total_ms",
                "research_total_ms",
                "analysis_total_ms",
                "writing_total_ms",
                "quality_total_ms",
            ]
            labels = ["Coordinator", "Research", "Analysis", "Writing", "Quality"]

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Per-Run Stacked Bar**")
                melted = duration_df.melt(
                    id_vars=["run_id"], value_vars=agent_cols,
                    var_name="agent", value_name="ms",
                )
                melted["agent"] = melted["agent"].str.replace("_total_ms", "").str.title()
                melted["ms"] = melted["ms"].fillna(0)
                melted["seconds"] = melted["ms"] / 1000.0
                fig_bar = px.bar(
                    melted, y="run_id", x="seconds", color="agent",
                    orientation="h", labels={"seconds": "Duration (s)", "run_id": "Run"},
                    title="Agent Duration per Run",
                )
                fig_bar.update_layout(barmode="stack", yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_b:
                st.markdown("**Average Duration by Agent**")
                avgs = {lbl: duration_df[col].fillna(0).mean() / 1000.0 for lbl, col in zip(labels, agent_cols)}
                fig_pie = px.pie(
                    names=list(avgs.keys()), values=list(avgs.values()),
                    title="Average Time Share",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Duration trend line
            st.markdown("**Duration Trend**")
            trend_df = duration_df.sort_values("created_at").copy()
            trend_df["total_ms"] = trend_df["total_ms"].fillna(0)
            trend_df["total_s"] = trend_df["total_ms"] / 1000.0
            trend_df["run_index"] = range(len(trend_df))
            fig_trend = px.line(
                trend_df, x="run_index", y="total_s",
                title="Total Duration over Recent Runs",
                labels={"run_index": "Run #", "total_s": "Duration (s)"},
                markers=True,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading agent durations: {e}")

# ---------------------------------------------------------------------------
# Tab: Token Usage
# ---------------------------------------------------------------------------
with tab_tokens:
    try:
        token_df = query_df(
            """
            SELECT run_id, llm_input_tokens, llm_output_tokens, llm_total_tokens,
                   report_word_count
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max_display_runs,),
        ).copy()

        if token_df.empty:
            st.info("No token data available.")
        else:
            token_df = token_df.fillna(0)
            # Reverse to chronological order for cumulative chart
            token_df = token_df.iloc[::-1].reset_index(drop=True)
            token_df["run_index"] = range(len(token_df))
            token_df["cumulative_tokens"] = token_df["llm_total_tokens"].cumsum()

            fig_tok = go.Figure()
            fig_tok.add_trace(go.Bar(
                x=token_df["run_id"], y=token_df["llm_input_tokens"],
                name="Input Tokens",
            ))
            fig_tok.add_trace(go.Bar(
                x=token_df["run_id"], y=token_df["llm_output_tokens"],
                name="Output Tokens",
            ))
            fig_tok.add_trace(go.Scatter(
                x=token_df["run_id"], y=token_df["cumulative_tokens"],
                name="Cumulative Total", mode="lines+markers", yaxis="y2",
            ))
            fig_tok.update_layout(
                barmode="group",
                title="Token Usage per Run",
                yaxis=dict(title="Tokens"),
                yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
                xaxis=dict(title="Run"),
            )
            st.plotly_chart(fig_tok, use_container_width=True)

            # Avg tokens per agent
            st.markdown("**Average Tokens per Agent**")
            avg_agent_tokens = query_df(
                """
                SELECT agent_name,
                       ROUND(AVG(input_tokens), 0)  AS avg_input_tokens,
                       ROUND(AVG(output_tokens), 0) AS avg_output_tokens
                FROM llm_calls
                GROUP BY agent_name
                """
            ).copy()
            if not avg_agent_tokens.empty:
                melted_avg = avg_agent_tokens.melt(
                    id_vars=["agent_name"],
                    value_vars=["avg_input_tokens", "avg_output_tokens"],
                    var_name="type", value_name="tokens",
                )
                fig_avg = px.bar(
                    melted_avg, x="agent_name", y="tokens", color="type",
                    barmode="group",
                    title="Avg Input/Output Tokens per Agent",
                    labels={"agent_name": "Agent", "tokens": "Tokens"},
                )
                st.plotly_chart(fig_avg, use_container_width=True)

            # Token efficiency scatter
            eff_df = token_df[(token_df["llm_total_tokens"] > 0) & (token_df["report_word_count"] > 0)]
            if not eff_df.empty:
                st.markdown("**Token Efficiency**")
                fig_eff = px.scatter(
                    eff_df, x="llm_total_tokens", y="report_word_count",
                    title="Total Tokens vs Report Word Count",
                    labels={"llm_total_tokens": "Total Tokens", "report_word_count": "Report Words"},
                )
                st.plotly_chart(fig_eff, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading token usage: {e}")

# ---------------------------------------------------------------------------
# Tab: Quality
# ---------------------------------------------------------------------------
with tab_quality:
    try:
        quality_df = query_df(
            """
            SELECT quality_score_final, quality_verdict_final, quality_iteration_final,
                   total_ms
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max_display_runs,),
        ).copy()

        if quality_df.empty:
            st.info("No quality data available.")
        else:
            q1, q2, q3 = st.columns(3)

            with q1:
                qf = quality_df.dropna(subset=["quality_score_final"]).copy()
                qf["run_index"] = range(len(qf))
                if not qf.empty:
                    fig_scatter = px.scatter(
                        qf, x="run_index", y="quality_score_final",
                        title="Quality Score Trend",
                        labels={"run_index": "Run #", "quality_score_final": "Score"},
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No quality scores recorded.")

            with q2:
                verdict_counts = quality_df["quality_verdict_final"].dropna().value_counts()
                if not verdict_counts.empty:
                    fig_verdict = px.pie(
                        names=verdict_counts.index, values=verdict_counts.values,
                        title="Verdict Distribution",
                    )
                    st.plotly_chart(fig_verdict, use_container_width=True)
                else:
                    st.info("No verdicts recorded.")

            with q3:
                qi = quality_df["quality_iteration_final"].dropna()
                if not qi.empty:
                    fig_hist = px.histogram(
                        qi, title="Quality Iterations Distribution",
                        labels={"value": "Iterations", "count": "Frequency"},
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No iteration data.")

            # Quality vs Duration scatter
            qd = quality_df.dropna(subset=["quality_score_final", "total_ms"])
            if not qd.empty:
                st.markdown("**Quality vs Duration**")
                qd = qd.copy()
                qd["total_s"] = qd["total_ms"] / 1000.0
                fig_qd = px.scatter(
                    qd, x="total_s", y="quality_score_final",
                    title="Quality Score vs Total Duration",
                    labels={"total_s": "Duration (s)", "quality_score_final": "Quality Score"},
                )
                st.plotly_chart(fig_qd, use_container_width=True)

            # Iterations vs Score scatter
            qi_s = quality_df.dropna(subset=["quality_iteration_final", "quality_score_final"])
            if not qi_s.empty:
                st.markdown("**Iterations vs Score**")
                fig_qi = px.scatter(
                    qi_s, x="quality_iteration_final", y="quality_score_final",
                    title="Quality Iterations vs Final Score",
                    labels={"quality_iteration_final": "Iterations", "quality_score_final": "Quality Score"},
                )
                st.plotly_chart(fig_qi, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading quality scores: {e}")

# ---------------------------------------------------------------------------
# Tab: Errors
# ---------------------------------------------------------------------------
with tab_errors:
    try:
        error_df = query_df(
            """
            SELECT run_id, SUBSTR(topic, 1, 50) AS topic, error_message, created_at
            FROM runs WHERE status IN ('failed', 'error')
            ORDER BY created_at DESC
            """
        ).copy()

        if error_df.empty:
            st.success("No failed runs. 🎉")
        else:
            _st_dataframe(error_df, use_container_width=True)

            # Error rate over time
            rate_df = query_df(
                """
                SELECT
                    DATE(created_at) AS day,
                    ROUND(100.0 * SUM(CASE WHEN status IN ('failed', 'error') THEN 1 ELSE 0 END) / COUNT(*), 1) AS error_rate
                FROM runs
                GROUP BY DATE(created_at)
                HAVING COUNT(*) >= 2
                ORDER BY day
                """
            ).copy()
            if not rate_df.empty:
                fig_err = px.line(
                    rate_df, x="day", y="error_rate",
                    title="Daily Error Rate (%)",
                    labels={"day": "Date", "error_rate": "Error Rate (%)"},
                    markers=True,
                )
                st.plotly_chart(fig_err, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading error tracking: {e}")

# ---------------------------------------------------------------------------
# Auto-refresh via rerun
# ---------------------------------------------------------------------------
if auto_refresh:
    import time
    time.sleep(30)
    # Streamlit 1.12
    st.experimental_rerun()
