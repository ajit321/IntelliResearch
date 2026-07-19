"""
frontend/streamlit_app.py
IntelliResearch — Streamlit Frontend Application
Multi-Agent AI Research Platform
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
from websockets.asyncio.client import connect as websocket_connect

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliResearch",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "IntelliResearch — Multi-Agent AI Research Platform",
        "Get Help": "https://github.com/your-team/intelliresearch",
    },
)

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_BASE: str    = f"{BACKEND_URL}/api/v1"


def _api_headers() -> dict[str, str]:
    """Return an auth header when the user has a Clerk token."""
    token = st.session_state.get("auth_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _raise_backend_error(response: httpx.Response) -> None:
    """Raise a readable error for a failed backend response."""
    if not response.is_error:
        return
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text
    raise RuntimeError(f"Backend returned {response.status_code}: {detail}")


def _merge_sources(incoming: list[dict[str, Any]]) -> None:
    """Append newly streamed sources without duplicating them."""
    existing = {
        (source.get("url", ""), source.get("title", ""))
        for source in st.session_state.sources
    }
    for source in incoming:
        identity = (source.get("url", ""), source.get("title", ""))
        if identity not in existing:
            st.session_state.sources.append(source)
            existing.add(identity)


def _apply_stream_message(message: dict[str, Any], status_box: Any) -> bool:
    """Apply one WebSocket message and return True when streaming is done."""
    message_type = message.get("type")

    if message_type == "ping":
        return False

    for event in message.get("events", []):
        st.session_state.events.append(event)
        agent = event.get("agent", "system")
        action = event.get("action", "")
        if action == "started":
            st.session_state.agent_states[agent] = "running"
        elif action in {"completed", "skipped"}:
            st.session_state.agent_states[agent] = "done"
        status_box.write(event.get("message", f"{agent}: {action}"))

    _merge_sources(message.get("sources", []))

    progress = message.get("progress")
    if isinstance(progress, (int, float)):
        st.session_state.progress = max(st.session_state.progress, int(progress))

    if message.get("report") is not None:
        st.session_state.report = message["report"]
    if message.get("quality_score") is not None:
        st.session_state.quality_score = message["quality_score"]

    if message_type == "hitl_required":
        st.session_state.is_researching = False
        st.session_state.awaiting_hitl = True
        status_box.update(label="Draft report ready for review", state="complete")
        return True

    if message_type == "report_ready":
        st.session_state.progress = 100
        st.session_state.is_researching = False
        st.session_state.awaiting_hitl = False
        status_box.update(label="Research report complete", state="complete")
        return True

    if message_type == "fatal_error":
        error = message.get("error") or message.get("message") or "Unknown backend error"
        st.session_state.error_message = str(error)
        st.session_state.is_researching = False
        status_box.update(label="Research failed", state="error")
        return True

    return False


async def _create_and_stream_research(
    query: str,
    depth: str,
    uploaded_files: list[Any],
    status_box: Any,
) -> None:
    """Create a backend session, start it, and consume streamed graph updates."""
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, headers=_api_headers()) as client:
        create_response = await client.post(
            f"{API_BASE}/research",
            json={"query": query, "depth": depth},
        )
        _raise_backend_error(create_response)
        session = create_response.json()
        session_id = session["session_id"]
        st.session_state.session_id = session_id
        status_box.write("Backend session created. Connecting to the live feed...")

        async with websocket_connect(
            session["ws_url"],
            open_timeout=10,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            files = [
                (
                    "documents",
                    (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    ),
                )
                for uploaded in uploaded_files
            ]
            run_response = await client.post(
                f"{API_BASE}/research/{session_id}/run",
                data={"query": query, "depth": depth},
                files=files,
            )
            _raise_backend_error(run_response)
            status_box.write("Research agents started.")

            while True:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=150)
                message = json.loads(raw_message)
                if _apply_stream_message(message, status_box):
                    break


def _submit_hitl_decision(approved: bool, feedback: str = "") -> dict[str, Any]:
    """Submit a human review decision and return the updated report state."""
    session_id = st.session_state.session_id
    response = httpx.post(
        f"{API_BASE}/research/{session_id}/hitl",
        json={"approved": approved, "feedback": feedback},
        headers=_api_headers(),
        timeout=180.0,
    )
    _raise_backend_error(response)
    return response.json()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #050810; }

.stApp {
    background: linear-gradient(135deg, #050810 0%, #0D1117 50%, #050810 100%);
}

.agent-card {
    background: rgba(13,17,23,0.9);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
    transition: all 0.3s;
}

.agent-running {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 0 20px rgba(0,212,255,0.15);
}

.agent-done {
    border-color: rgba(16,185,129,0.5);
}

.agent-error {
    border-color: rgba(244,63,94,0.5);
}

.metric-card {
    background: rgba(22,27,39,0.9);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.score-excellent { color: #10B981; font-weight: 700; }
.score-good      { color: #F59E0B; font-weight: 700; }
.score-poor      { color: #F43F5E; font-weight: 700; }

.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    margin-bottom: 2px;
}

div[data-testid="stVerticalBlock"] > div:has(> iframe) {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    """Initialise Streamlit session state with defaults."""
    defaults: dict[str, Any] = {
        "session_id":       None,
        "is_researching":   False,
        "events":           [],
        "sources":          [],
        "report":           None,
        "quality_score":    None,
        "agent_states":     {},
        "progress":         0,
        "awaiting_hitl":    False,
        "error_message":    None,
        "auth_token":       None,
        "user_name":        "Demo User",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

AGENT_META: dict[str, dict[str, str]] = {
    "planner":          {"icon": "🎯", "label": "Planner",       "color": "#00D4FF"},
    "paper_agent":      {"icon": "📄", "label": "Paper Search",   "color": "#38BDF8"},
    "news_agent":       {"icon": "📰", "label": "News Search",    "color": "#F97316"},
    "market_agent":     {"icon": "📊", "label": "Market Intel",   "color": "#F59E0B"},
    "user_docs_agent":  {"icon": "📁", "label": "Your Documents", "color": "#7C3AED"},
    "analysis_agent":   {"icon": "🔬", "label": "Analysis",       "color": "#EC4899"},
    "insight_agent":    {"icon": "💡", "label": "Insights",       "color": "#A78BFA"},
    "citation_agent":   {"icon": "✅", "label": "Fact-Check",     "color": "#10B981"},
    "report_builder":   {"icon": "📝", "label": "Report Build",   "color": "#F43F5E"},
    "judge_agent":      {"icon": "⚖️", "label": "Judge",         "color": "#FBBF24"},
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 IntelliResearch")
    st.markdown("*Multi-Agent AI Research Platform*")
    st.divider()

    # Clerk auth status
    if st.session_state.auth_token:
        st.success(f"✅ Signed in as **{st.session_state.user_name}**")
        if st.button("Sign out"):
            st.session_state.auth_token = None
            st.rerun()
    else:
        st.info("🔐 Running in **Demo Mode** (no auth required)")
        st.markdown("*Add Clerk keys to `.env` for Google/Email login*")

    st.divider()

    # Agent status panel
    st.markdown("### 🤖 Agent Status")
    for agent_id, meta in AGENT_META.items():
        agent_state = st.session_state.agent_states.get(agent_id, "idle")
        if agent_state == "running":
            icon, colour = "⚙️", meta["color"]
        elif agent_state == "done":
            icon, colour = "✅", "#10B981"
        elif agent_state == "error":
            icon, colour = "❌", "#F43F5E"
        else:
            icon, colour = "○", "#475569"

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
            f'<span style="color:{colour}">{icon}</span>'
            f'<span style="color:{"#94A3B8" if agent_state=="idle" else "#E2E8F0"};font-size:13px">'
            f'{meta["icon"]} {meta["label"]}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Progress
    if st.session_state.is_researching:
        st.progress(st.session_state.progress / 100)
        st.caption(f"Progress: {st.session_state.progress}%")


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 24px">
    <div style="font-size:3rem;margin-bottom:8px">🧠</div>
    <h1 style="font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,#00D4FF,#7C3AED,#F43F5E);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
        IntelliResearch
    </h1>
    <p style="color:#94A3B8;font-size:1rem;margin-top:8px">
        10 AI Agents · Real-time Research · Evidence-based Reports
    </p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ─────────────────────────────────────────────────────────────
with st.container():
    col_input, col_config = st.columns([3, 1])

    with col_input:
        query = st.text_area(
            "Research Query",
            placeholder="e.g. What are the latest breakthroughs in quantum error correction and their implications?",
            height=120,
            label_visibility="collapsed",
        )

        # Voice input
        st.markdown("**🎤 Or use voice input:**")
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#00D4FF",
                neutral_color="#94A3B8",
                icon_size="1.5x",
            )
            if audio_bytes:
                st.info("🎤 Voice captured — transcription requires Whisper API key")
        except ImportError:
            st.caption("Install `audio-recorder-streamlit` for voice input")

    with col_config:
        depth = st.selectbox(
            "Depth",
            options=["quick", "deep", "expert"],
            index=1,
            format_func=lambda x: {"quick": "⚡ Quick", "deep": "🔬 Deep", "expert": "🎓 Expert"}[x],
        )

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload research papers, reports, or documents for RAG analysis",
        )

        launch = st.button(
            "🚀 Launch Research",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.is_researching,
        )


# ── Launch Logic ──────────────────────────────────────────────────────────────
if launch:
    if not query.strip():
        st.warning("Enter a research query before launching.")
    else:
        st.session_state.events          = []
        st.session_state.sources         = []
        st.session_state.report          = None
        st.session_state.quality_score   = None
        st.session_state.agent_states    = {}
        st.session_state.progress        = 0
        st.session_state.awaiting_hitl   = False
        st.session_state.error_message   = None
        st.session_state.is_researching  = True
        st.session_state.session_id      = None

        with st.status("Starting research...", expanded=True) as status_box:
            try:
                asyncio.run(
                    _create_and_stream_research(
                        query=query.strip(),
                        depth=depth,
                        uploaded_files=uploaded_files or [],
                        status_box=status_box,
                    )
                )
            except Exception as exc:
                st.session_state.error_message = str(exc)
                st.session_state.is_researching = False
                status_box.update(label="Research failed", state="error")
                status_box.write(str(exc))
        st.rerun()

if st.session_state.error_message:
    st.error(st.session_state.error_message)


# ── Live Results Area ─────────────────────────────────────────────────────────
if st.session_state.is_researching or st.session_state.report:

    tab_live, tab_sources, tab_report, tab_judge, tab_hitl = st.tabs([
        "📡 Live Feed", "📚 Sources", "📄 Report", "⚖️ Quality Score", "👤 HITL Review"
    ])

    # ── Live Feed Tab ─────────────────────────────────────────────────────────
    with tab_live:
        st.markdown("### 📡 Agent Activity Stream")
        log_container = st.container(height=400)

        with log_container:
            if not st.session_state.events:
                st.info("Agents will stream updates here in real time...")
            for event in reversed(st.session_state.events[-50:]):
                agent   = event.get("agent", "system")
                message = event.get("message", "")
                action  = event.get("action", "")
                color   = AGENT_META.get(agent, {}).get("color", "#94A3B8")
                icon    = AGENT_META.get(agent, {}).get("icon", "⚙")
                st.markdown(
                    f'<div class="log-entry" style="background:rgba(255,255,255,0.03)">'
                    f'<span style="color:{color}">[{icon} {agent.upper()}]</span> '
                    f'<span style="color:#94A3B8">{message}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Sources Tab ───────────────────────────────────────────────────────────
    with tab_sources:
        st.markdown(f"### 📚 Sources Found ({len(st.session_state.sources)})")
        if not st.session_state.sources:
            st.info("Sources will appear here as agents retrieve them...")
        else:
            for src in st.session_state.sources:
                source_type = src.get("source_type", "web")
                color_map   = {"arxiv": "#38BDF8", "wikipedia": "#10B981", "news": "#F97316", "user_doc": "#7C3AED"}
                color       = color_map.get(source_type, "#94A3B8")
                with st.expander(f"{src.get('title', 'Untitled')} [{source_type}]"):
                    st.markdown(f"**URL:** {src.get('url', 'N/A')}")
                    st.markdown(f"**Confidence:** {src.get('confidence', 0):.0%}")
                    st.markdown(src.get("content", "")[:500])

    # ── Report Tab ────────────────────────────────────────────────────────────
    with tab_report:
        report = st.session_state.report
        if not report:
            st.info("The final report will appear here when agents complete...")
        else:
            # Stats row
            stats = report.get("stats", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📚 Sources", stats.get("total_sources", 0))
            c2.metric("⚠️ Contradictions", stats.get("contradictions_found", 0))
            c3.metric("💡 Hypotheses", stats.get("hypotheses_generated", 0))
            c4.metric("✅ Verified Claims", stats.get("claims_verified", 0))

            st.divider()

            # Full report text
            st.markdown(report.get("full_text", ""))

            st.divider()

            # Export buttons
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                st.download_button(
                    "📄 Download Markdown",
                    data=report.get("full_text", ""),
                    file_name=f"intelliresearch-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
                    mime="text/markdown",
                )
            with ecol2:
                st.download_button(
                    "📊 Download JSON",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"intelliresearch-{datetime.now().strftime('%Y%m%d-%H%M')}.json",
                    mime="application/json",
                )

    # ── Quality Score Tab ─────────────────────────────────────────────────────
    with tab_judge:
        score = st.session_state.quality_score
        if not score:
            st.info("Quality evaluation will appear here after the Judge Agent runs...")
        else:
            overall = score.get("overall_score", 0)
            passed  = score.get("passed", False)

            sc_class = "score-excellent" if overall >= 8 else "score-good" if overall >= 6 else "score-poor"
            st.markdown(
                f'<div style="text-align:center;padding:24px">'
                f'<div class="{sc_class}" style="font-size:3rem">{overall:.1f}/10</div>'
                f'<div style="color:{"#10B981" if passed else "#F43F5E"};font-size:1.2rem">'
                f'{"✅ PASSED" if passed else "❌ NEEDS REVISION"}</div></div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clarity",      f'{score.get("clarity", 0):.1f}')
            c2.metric("Depth",        f'{score.get("depth", 0):.1f}')
            c3.metric("Accuracy",     f'{score.get("accuracy", 0):.1f}')
            c4.metric("Completeness", f'{score.get("completeness", 0):.1f}')

            st.markdown("**Judge Feedback:**")
            st.info(score.get("feedback", "No feedback available."))

    # ── HITL Tab ──────────────────────────────────────────────────────────────
    with tab_hitl:
        st.markdown("### 👤 Human-in-the-Loop Review")
        if not st.session_state.awaiting_hitl:
            st.info("This tab activates when the report is ready for your review.")
        else:
            st.success("🔔 The research report is awaiting your approval.")

            preview = st.session_state.report or {}
            st.markdown(preview.get("executive_summary", "")[:600])

            hitl_feedback = st.text_area(
                "Reviewer Feedback (optional)",
                placeholder="e.g. Please add more detail on the economic implications...",
                height=100,
            )

            hcol1, hcol2 = st.columns(2)
            with hcol1:
                if st.button("✅ Approve Report", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Finalising report..."):
                            result = _submit_hitl_decision(approved=True)
                        st.session_state.report = result.get("report") or st.session_state.report
                        st.session_state.quality_score = result.get("quality_score") or st.session_state.quality_score
                        st.session_state.awaiting_hitl = False
                        st.success("Report approved and finalised.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with hcol2:
                if st.button("🔄 Request Revision", use_container_width=True):
                    if hitl_feedback.strip():
                        try:
                            with st.spinner("Agents are revising the report..."):
                                result = _submit_hitl_decision(
                                    approved=False,
                                    feedback=hitl_feedback.strip(),
                                )
                            st.session_state.report = result.get("report") or st.session_state.report
                            st.session_state.quality_score = result.get("quality_score") or st.session_state.quality_score
                            st.session_state.awaiting_hitl = True
                            st.success("Revised draft ready for review.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                    else:
                        st.error("Please provide feedback before requesting revision.")
