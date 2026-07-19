"""
frontend/streamlit_app.py
IntelliResearch — Streamlit Frontend Application
Multi-Agent AI Research Platform
"""

import asyncio
import html
import json
import os
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
from websockets.asyncio.client import connect as websocket_connect

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliResearch · Research OS",
    page_icon="◈",
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
:root {
    --ir-bg: #070A12;
    --ir-surface: rgba(15, 20, 34, 0.82);
    --ir-surface-strong: #111726;
    --ir-border: rgba(148, 163, 184, 0.14);
    --ir-border-bright: rgba(84, 236, 210, 0.30);
    --ir-text: #F2F5FA;
    --ir-muted: #94A3B8;
    --ir-cyan: #54ECD2;
    --ir-blue: #60A5FA;
    --ir-violet: #A78BFA;
    --ir-success: #34D399;
    --ir-warning: #FBBF24;
    --ir-danger: #FB7185;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    color: var(--ir-text);
    background:
        radial-gradient(circle at 82% -10%, rgba(96, 165, 250, 0.15), transparent 28rem),
        radial-gradient(circle at 30% 8%, rgba(84, 236, 210, 0.09), transparent 24rem),
        linear-gradient(180deg, #090D17 0%, var(--ir-bg) 55%, #05070D 100%);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
.block-container { max-width: 1280px; padding: 2.2rem 2.4rem 5rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 20, 34, 0.98), rgba(8, 11, 20, 0.98));
    border-right: 1px solid var(--ir-border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }
[data-testid="stSidebar"] hr { border-color: var(--ir-border); }

.ir-brand { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.35rem; }
.ir-brand-mark {
    display: grid; place-items: center; width: 2.4rem; height: 2.4rem;
    border-radius: 0.8rem; color: #05100E; font-weight: 900;
    background: linear-gradient(135deg, var(--ir-cyan), var(--ir-blue));
    box-shadow: 0 0 24px rgba(84, 236, 210, 0.22);
}
.ir-brand-name { font-size: 1.03rem; font-weight: 750; letter-spacing: -0.02em; }
.ir-brand-sub { color: var(--ir-muted); font-size: 0.73rem; letter-spacing: 0.08em; text-transform: uppercase; }

.ir-mode-card {
    margin: 1.25rem 0 1.4rem; padding: 0.9rem 1rem; border-radius: 0.85rem;
    border: 1px solid rgba(84, 236, 210, 0.18); background: rgba(84, 236, 210, 0.06);
}
.ir-mode-row { display: flex; align-items: center; gap: 0.55rem; font-size: 0.82rem; font-weight: 650; }
.ir-live-dot { width: 0.48rem; height: 0.48rem; border-radius: 50%; background: var(--ir-success); box-shadow: 0 0 12px var(--ir-success); }
.ir-mode-copy { color: var(--ir-muted); font-size: 0.72rem; margin-top: 0.3rem; line-height: 1.45; }

.ir-sidebar-heading { color: #CBD5E1; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin: 0.2rem 0 0.7rem; }
.ir-agent-row {
    display: grid; grid-template-columns: 1.65rem 1fr auto; align-items: center; gap: 0.55rem;
    min-height: 2.15rem; padding: 0.35rem 0.45rem; margin-bottom: 0.15rem;
    border: 1px solid transparent; border-radius: 0.65rem;
}
.ir-agent-row.running { background: rgba(96, 165, 250, 0.08); border-color: rgba(96, 165, 250, 0.22); }
.ir-agent-row.done { background: rgba(52, 211, 153, 0.05); }
.ir-agent-index { display: grid; place-items: center; width: 1.55rem; height: 1.55rem; border-radius: 0.5rem; background: rgba(148, 163, 184, 0.08); font-size: 0.7rem; }
.ir-agent-label { color: #CBD5E1; font-size: 0.77rem; font-weight: 560; }
.ir-agent-state { color: #536078; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.06em; }
.ir-agent-row.running .ir-agent-state { color: var(--ir-blue); }
.ir-agent-row.done .ir-agent-state { color: var(--ir-success); }

.ir-hero { padding: 2.4rem 0 1.75rem; max-width: 920px; }
.ir-kicker {
    display: inline-flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;
    padding: 0.42rem 0.72rem; border: 1px solid rgba(84, 236, 210, 0.2);
    border-radius: 99px; background: rgba(84, 236, 210, 0.06);
    color: var(--ir-cyan); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
}
.ir-hero h1 { margin: 0; max-width: 850px; color: var(--ir-text); font-size: clamp(2.45rem, 5vw, 4.5rem); line-height: 1.02; letter-spacing: -0.055em; font-weight: 790; }
.ir-gradient-text { background: linear-gradient(105deg, var(--ir-cyan), #8CC8FF 52%, var(--ir-violet)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.ir-hero p { max-width: 690px; margin: 1.15rem 0 0; color: var(--ir-muted); font-size: 1.02rem; line-height: 1.7; }

.ir-stat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin: 0.4rem 0 2rem; }
.ir-stat-card { padding: 0.9rem 1rem; border-radius: 0.8rem; border: 1px solid var(--ir-border); background: rgba(15, 20, 34, 0.5); }
.ir-stat-value { color: var(--ir-text); font-size: 0.92rem; font-weight: 700; }
.ir-stat-label { color: var(--ir-muted); margin-top: 0.18rem; font-size: 0.72rem; }

.ir-section-kicker { color: var(--ir-cyan); font-size: 0.68rem; font-weight: 750; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem; }
.ir-section-title { margin: 0 0 0.35rem; color: var(--ir-text); font-size: 1.25rem; font-weight: 720; letter-spacing: -0.02em; }
.ir-section-copy { color: var(--ir-muted); font-size: 0.82rem; line-height: 1.55; margin-bottom: 1rem; }
.ir-divider { height: 1px; margin: 2rem 0 1.4rem; background: linear-gradient(90deg, transparent, var(--ir-border), transparent); }

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--ir-border) !important;
    border-radius: 1rem !important;
    background: linear-gradient(145deg, rgba(17, 23, 38, 0.85), rgba(10, 14, 24, 0.78));
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
}

.stTextArea textarea {
    min-height: 9rem; padding: 1rem 1.05rem; color: var(--ir-text) !important;
    background: rgba(5, 8, 15, 0.62) !important; border: 1px solid var(--ir-border) !important;
    border-radius: 0.85rem !important; line-height: 1.55;
}
.stTextArea textarea:focus { border-color: var(--ir-border-bright) !important; box-shadow: 0 0 0 3px rgba(84, 236, 210, 0.07) !important; }
.stTextArea label, .stFileUploader label { color: #DCE4F0 !important; font-weight: 650 !important; }

.stButton > button, .stDownloadButton > button {
    min-height: 2.75rem; border-radius: 0.72rem; border: 1px solid var(--ir-border);
    font-weight: 680; transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); border-color: rgba(84, 236, 210, 0.36); }
.stButton > button[kind="primary"] {
    color: #06110F; border: 0; background: linear-gradient(110deg, var(--ir-cyan), #7DD3FC);
    box-shadow: 0 10px 28px rgba(84, 236, 210, 0.16);
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 14px 34px rgba(84, 236, 210, 0.25); }

[data-testid="stFileUploaderDropzone"] { min-height: 7.5rem; border: 1px dashed rgba(148, 163, 184, 0.24); border-radius: 0.8rem; background: rgba(5, 8, 15, 0.42); }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--ir-cyan), var(--ir-blue)); }

.stTabs [data-baseweb="tab-list"] { gap: 0.35rem; padding: 0.35rem; border: 1px solid var(--ir-border); border-radius: 0.85rem; background: rgba(8, 11, 20, 0.62); }
.stTabs [data-baseweb="tab"] { height: 2.65rem; padding: 0 0.9rem; border-radius: 0.62rem; color: var(--ir-muted); font-size: 0.79rem; }
.stTabs [aria-selected="true"] { color: var(--ir-text) !important; background: rgba(84, 236, 210, 0.08) !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

[data-testid="stMetric"] { padding: 0.95rem 1rem; border: 1px solid var(--ir-border); border-radius: 0.8rem; background: rgba(15, 20, 34, 0.52); }
[data-testid="stMetricLabel"] { color: var(--ir-muted); }
[data-testid="stMetricValue"] { color: var(--ir-text); font-size: 1.55rem; }

.ir-log-entry { display: grid; grid-template-columns: auto 1fr; gap: 0.7rem; margin-bottom: 0.5rem; padding: 0.75rem 0.85rem; border: 1px solid rgba(148, 163, 184, 0.09); border-radius: 0.72rem; background: rgba(15, 20, 34, 0.56); }
.ir-log-icon { display: grid; place-items: center; width: 1.8rem; height: 1.8rem; border-radius: 0.55rem; background: rgba(148, 163, 184, 0.08); }
.ir-log-agent { font-size: 0.68rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
.ir-log-message { color: #B6C2D3; margin-top: 0.18rem; font-size: 0.78rem; line-height: 1.45; }

.ir-empty { padding: 2.2rem 1rem; text-align: center; color: var(--ir-muted); border: 1px dashed var(--ir-border); border-radius: 0.85rem; background: rgba(15, 20, 34, 0.28); }
.ir-score-hero { display: flex; justify-content: center; align-items: center; gap: 1.2rem; padding: 1.8rem; border: 1px solid var(--ir-border); border-radius: 1rem; background: rgba(15, 20, 34, 0.54); }
.ir-score-value { font-size: 3rem; line-height: 1; font-weight: 780; letter-spacing: -0.05em; }
.score-excellent { color: var(--ir-success); }
.score-good { color: var(--ir-warning); }
.score-poor { color: var(--ir-danger); }
.ir-score-label { color: var(--ir-muted); font-size: 0.75rem; margin-top: 0.3rem; }
.ir-review-banner { padding: 1rem 1.1rem; border: 1px solid rgba(167, 139, 250, 0.25); border-radius: 0.85rem; background: rgba(167, 139, 250, 0.08); color: #DDD6FE; }

.stAlert { border-radius: 0.8rem; border-color: var(--ir-border); }
.streamlit-expanderHeader { border-radius: 0.7rem; }

@media (max-width: 900px) {
    .block-container { padding: 1.4rem 1rem 3rem; }
    .ir-hero { padding-top: 1.3rem; }
    .ir-hero h1 { font-size: 2.6rem; }
    .ir-stat-grid { grid-template-columns: 1fr; }
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] { padding: 0 0.55rem; font-size: 0.7rem; }
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
    st.markdown(
        """
        <div class="ir-brand">
            <div class="ir-brand-mark">IR</div>
            <div>
                <div class="ir-brand-name">IntelliResearch</div>
                <div class="ir-brand-sub">Research operating system</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.auth_token:
        st.markdown(
            f"""
            <div class="ir-mode-card">
                <div class="ir-mode-row"><span class="ir-live-dot"></span>{st.session_state.user_name}</div>
                <div class="ir-mode-copy">Authenticated workspace · private research session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Sign out", use_container_width=True):
            st.session_state.auth_token = None
            st.rerun()
    else:
        st.markdown(
            """
            <div class="ir-mode-card">
                <div class="ir-mode-row"><span class="ir-live-dot"></span>Local workspace</div>
                <div class="ir-mode-copy">Authentication bypass is active. Research runs against your local API.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ir-sidebar-heading">Agent pipeline</div>', unsafe_allow_html=True)
    for index, (agent_id, meta) in enumerate(AGENT_META.items(), start=1):
        agent_state = st.session_state.agent_states.get(agent_id, "idle")
        state_label = {"running": "working", "done": "complete", "error": "error"}.get(agent_state, "queued")
        st.markdown(
            f"""
            <div class="ir-agent-row {agent_state}">
                <div class="ir-agent-index">{meta['icon'] if agent_state != 'idle' else f'{index:02d}'}</div>
                <div class="ir-agent-label">{meta['label']}</div>
                <div class="ir-agent-state">{state_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
    if st.session_state.is_researching:
        st.progress(st.session_state.progress / 100)
        st.caption(f"Research in progress · {st.session_state.progress}%")
    elif st.session_state.awaiting_hitl:
        st.caption("Draft ready · awaiting your review")
    else:
        st.caption("Ready for a new research question")


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <section class="ir-hero">
        <div class="ir-kicker">● Multi-agent research workspace</div>
        <h1>Go from open question to <span class="ir-gradient-text">decision-ready evidence.</span></h1>
        <p>Coordinate specialist agents across papers, news, market signals, and your own documents—then turn the evidence into a cited report you can review and refine.</p>
    </section>
    <div class="ir-stat-grid">
        <div class="ir-stat-card"><div class="ir-stat-value">10 specialist agents</div><div class="ir-stat-label">Parallel retrieval, analysis, synthesis, and review</div></div>
        <div class="ir-stat-card"><div class="ir-stat-value">Evidence first</div><div class="ir-stat-label">Source-backed claims with credibility checks</div></div>
        <div class="ir-stat-card"><div class="ir-stat-value">Human controlled</div><div class="ir-stat-label">Approve the draft or send it back for revision</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Input Section ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="ir-section-kicker">New research run</div>', unsafe_allow_html=True)
    st.markdown('<div class="ir-section-title">What do you want to understand?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ir-section-copy">Frame a specific question, choose the depth, and optionally add documents for the agents to analyse.</div>',
        unsafe_allow_html=True,
    )

    col_input, col_config = st.columns([1.75, 1], gap="large")

    with col_input:
        query = st.text_area(
            "Research question",
            placeholder="Example: What are the economic, environmental, and engine-performance implications of E20 petrol adoption in India?",
            height=150,
            help="Specific questions produce stronger plans, cleaner source selection, and more useful reports.",
        )
        st.caption("Tip: include the geography, time horizon, and decision you are trying to make.")

        with st.expander("Add a voice note", expanded=False):
            try:
                from audio_recorder_streamlit import audio_recorder

                audio_bytes = audio_recorder(
                    text="Record your question",
                    recording_color="#54ECD2",
                    neutral_color="#64748B",
                    icon_size="1.3x",
                )
                if audio_bytes:
                    st.info("Voice captured. Transcription requires a configured Whisper API key.")
            except ImportError:
                st.caption("Voice recording is unavailable in this environment.")

    with col_config:
        depth_choice = st.segmented_control(
            "Research depth",
            options=["Quick", "Deep", "Expert"],
            default="Deep",
            selection_mode="single",
            help="Quick is faster; Expert uses broader retrieval and deeper synthesis.",
        )
        depth = {"Quick": "quick", "Deep": "deep", "Expert": "expert"}.get(depth_choice or "Deep", "deep")

        uploaded_files = st.file_uploader(
            "Add source documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload reports, papers, notes, or internal documents for retrieval-augmented analysis.",
        )
        st.caption("PDF, DOCX, TXT, or Markdown · up to 200 MB each")

        launch = st.button(
            "Start research →",
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
    st.markdown('<div class="ir-divider"></div>', unsafe_allow_html=True)
    result_state = "Research in progress" if st.session_state.is_researching else (
        "Awaiting your review" if st.session_state.awaiting_hitl else "Research complete"
    )
    st.markdown('<div class="ir-section-kicker">Research workspace</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ir-section-title">{result_state}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ir-section-copy">Follow the agent pipeline, inspect the evidence, and review the generated brief.</div>',
        unsafe_allow_html=True,
    )

    summary_progress, summary_sources, summary_quality = st.columns(3)
    summary_progress.metric("Pipeline progress", f"{st.session_state.progress}%")
    summary_sources.metric("Sources collected", len(st.session_state.sources))
    quality_value = st.session_state.quality_score or {}
    summary_quality.metric("Quality score", f"{quality_value.get('overall_score', 0):.1f} / 10")

    tab_live, tab_sources, tab_report, tab_judge, tab_hitl = st.tabs([
        "Activity", "Evidence", "Research brief", "Quality", "Review"
    ])

    # ── Live Feed Tab ─────────────────────────────────────────────────────────
    with tab_live:
        st.markdown("### Agent activity")
        st.caption("Updates appear as each specialist finishes a stage of the research plan.")
        log_container = st.container(height=460)

        with log_container:
            if not st.session_state.events:
                st.markdown(
                    '<div class="ir-empty">The activity timeline will appear here once the agents begin.</div>',
                    unsafe_allow_html=True,
                )
            for event in reversed(st.session_state.events[-50:]):
                agent   = event.get("agent", "system")
                message = event.get("message", "")
                color   = AGENT_META.get(agent, {}).get("color", "#94A3B8")
                icon    = AGENT_META.get(agent, {}).get("icon", "⚙")
                label   = AGENT_META.get(agent, {}).get("label", agent.replace("_", " ").title())
                st.markdown(
                    f'<div class="ir-log-entry">'
                    f'<div class="ir-log-icon">{icon}</div>'
                    f'<div><div class="ir-log-agent" style="color:{color}">{html.escape(label)}</div>'
                    f'<div class="ir-log-message">{html.escape(str(message))}</div></div></div>',
                    unsafe_allow_html=True,
                )

    # ── Sources Tab ───────────────────────────────────────────────────────────
    with tab_sources:
        st.markdown("### Evidence library")
        st.caption("Every source collected by the retrieval agents, with provenance and confidence context.")
        if not st.session_state.sources:
            st.markdown(
                '<div class="ir-empty">Sources will appear here as the retrieval agents complete their searches.</div>',
                unsafe_allow_html=True,
            )
        else:
            source_types = {source.get("source_type", "web") for source in st.session_state.sources}
            confidences = [float(source.get("confidence", 0) or 0) for source in st.session_state.sources]
            source_total, source_variety, source_confidence = st.columns(3)
            source_total.metric("Total sources", len(st.session_state.sources))
            source_variety.metric("Source types", len(source_types))
            source_confidence.metric("Average confidence", f"{sum(confidences) / len(confidences):.0%}")

            for src in st.session_state.sources:
                source_type = src.get("source_type", "web")
                type_label = {
                    "arxiv": "Academic paper",
                    "wikipedia": "Reference",
                    "news": "News",
                    "market": "Market signal",
                    "user_doc": "Uploaded document",
                }.get(source_type, source_type.replace("_", " ").title())
                confidence = float(src.get("confidence", 0) or 0)
                with st.expander(f"{src.get('title', 'Untitled')}  ·  {type_label}"):
                    meta_type, meta_date, meta_confidence = st.columns(3)
                    meta_type.metric("Type", type_label)
                    meta_date.metric("Published", src.get("published_date") or "Unknown")
                    meta_confidence.metric("Confidence", f"{confidence:.0%}")
                    st.markdown(src.get("content", "")[:700])
                    source_url = str(src.get("url", ""))
                    if source_url.startswith(("http://", "https://")):
                        st.link_button("Open original source ↗", source_url)

    # ── Report Tab ────────────────────────────────────────────────────────────
    with tab_report:
        report = st.session_state.report
        st.markdown("### Research brief")
        st.caption("The synthesized narrative produced from the collected evidence and agent analysis.")
        if not report:
            st.markdown(
                '<div class="ir-empty">The research brief will appear here after synthesis and quality review.</div>',
                unsafe_allow_html=True,
            )
        else:
            report_status = "Draft · review required" if st.session_state.awaiting_hitl else "Approved research brief"
            st.markdown(
                f'<div class="ir-review-banner"><strong>{report_status}</strong><br><span style="color:#AFA7CF;font-size:0.78rem">Generated for: {html.escape(str(report.get("query", "Research question")))}</span></div>',
                unsafe_allow_html=True,
            )

            stats = report.get("stats", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sources", stats.get("total_sources", 0))
            c2.metric("Contradictions", stats.get("contradictions_found", 0))
            c3.metric("Hypotheses", stats.get("hypotheses_generated", 0))
            c4.metric("Verified claims", stats.get("claims_verified", 0))

            ecol1, ecol2 = st.columns(2)
            with ecol1:
                st.download_button(
                    "Download Markdown",
                    data=report.get("full_text", ""),
                    file_name=f"intelliresearch-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with ecol2:
                st.download_button(
                    "Download structured JSON",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"intelliresearch-{datetime.now().strftime('%Y%m%d-%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with st.container(border=True):
                st.markdown(report.get("full_text", ""))

    # ── Quality Score Tab ─────────────────────────────────────────────────────
    with tab_judge:
        score = st.session_state.quality_score
        st.markdown("### Quality assessment")
        st.caption("An independent judge agent scores the brief for clarity, depth, accuracy, and completeness.")
        if not score:
            st.markdown(
                '<div class="ir-empty">Quality signals will appear after the judge agent evaluates the draft.</div>',
                unsafe_allow_html=True,
            )
        else:
            overall = score.get("overall_score", 0)
            passed  = score.get("passed", False)

            sc_class = "score-excellent" if overall >= 8 else "score-good" if overall >= 6 else "score-poor"
            st.markdown(
                f'<div class="ir-score-hero">'
                f'<div><div class="ir-score-value {sc_class}">{overall:.1f}</div><div class="ir-score-label">overall score / 10</div></div>'
                f'<div><strong>{"Quality threshold passed" if passed else "Revision recommended"}</strong>'
                f'<div class="ir-score-label">{"Ready for human review" if passed else "Review the judge feedback before approval"}</div></div></div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clarity",      f'{score.get("clarity", 0):.1f}')
            c2.metric("Depth",        f'{score.get("depth", 0):.1f}')
            c3.metric("Accuracy",     f'{score.get("accuracy", 0):.1f}')
            c4.metric("Completeness", f'{score.get("completeness", 0):.1f}')

            with st.container(border=True):
                st.markdown("#### Judge feedback")
                st.write(score.get("feedback", "No feedback available."))

    # ── HITL Tab ──────────────────────────────────────────────────────────────
    with tab_hitl:
        st.markdown("### Human review")
        st.caption("You stay in control: approve the evidence-backed draft or send precise guidance back to the agents.")
        if not st.session_state.awaiting_hitl:
            st.markdown(
                '<div class="ir-empty">Review controls activate when the draft report and quality assessment are ready.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ir-review-banner"><strong>Decision required</strong><br><span style="font-size:0.8rem">Read the brief and quality feedback, then approve it or request a targeted revision.</span></div>',
                unsafe_allow_html=True,
            )

            preview = st.session_state.report or {}
            with st.container(border=True):
                st.markdown("#### Executive summary")
                st.write(preview.get("executive_summary", "")[:800] or "No executive summary was generated.")

            hitl_feedback = st.text_area(
                "Revision guidance",
                placeholder="Be specific—for example: compare the economic implications across urban and rural markets, and add recent policy evidence.",
                height=120,
            )

            hcol1, hcol2 = st.columns(2)
            with hcol1:
                if st.button("Approve and finalise", type="primary", use_container_width=True):
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
                if st.button("Request a revision", use_container_width=True):
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
