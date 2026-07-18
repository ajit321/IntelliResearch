# IntelliResearch — Codebase Architecture Analysis

> Engineering-grade reverse-engineering of the `IntelliResearch` codebase.
> Scope analyzed: full repo (`app/`, `frontend/`, `tests/`, config, `.env`, `requirements.txt`) — ~4,200 LOC Python.
> **Legend:** `[FACT]` = directly observed in code. `[INFER]` = inference/assumption requiring confirmation.

---

## 1. Codebase Overview

**Purpose `[FACT]`** — A multi-agent AI research platform. A user submits a research query (optionally with uploaded documents); a graph of 10 specialist LLM agents plans the research, retrieves from academic/news/market/user sources in parallel, analyzes for contradictions, generates hypotheses, fact-checks, compiles a structured report, self-grades it with an LLM-as-Judge, and pauses for human approval before finalizing.

### Tech stack `[FACT]`

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (uses `list[str]`, `X | None`, `TypedDict`) |
| Orchestration | LangGraph `StateGraph` + `MemorySaver` checkpointer |
| API | FastAPI (async), REST + WebSocket, `slowapi` limiter |
| LLM access | LiteLLM → OpenRouter (`acompletion`) |
| Model routing | GPT-4o-mini / GPT-4o / Claude 3.5 Sonnet by task type |
| Frontend | Streamlit (single-file, custom CSS, tabbed UI) |
| RAG | FAISS + HuggingFace `all-MiniLM-L6-v2` embeddings, recursive splitter |
| Retrieval sources | `arxiv`, Wikipedia REST, SerpAPI (news) |
| Auth | Clerk (JWT via Clerk backend API), bypass mode for dev |
| Observability | LangSmith tracing, `structlog` structured logging |
| Security | `bleach` sanitisation, regex guardrails |
| Resilience | Custom circuit breaker + retry/backoff decorators |
| Config | `pydantic-settings` from `.env` |

### Architectural style `[FACT/INFER]`
- **Layered + orchestrated multi-agent (blackboard) architecture.** Clean vertical layering: `frontend → API (main.py) → orchestration (graph.py) → agents → utils/rag/tools/security`. `[FACT]`
- **Blackboard pattern:** a single shared `ResearchState` TypedDict is the "blackboard" every node reads from and appends to. `[FACT]`
- **DAG/pipeline with map-reduce:** planner fans out to 4 retrieval agents (map), fans in to a linear analysis pipeline (reduce). `[FACT]`
- **Reflection loop:** Judge agent scores the report and can route back for self-correction. `[FACT]`
- **Event-driven streaming intent:** WebSocket + `asyncio.Queue` per session — *intended* but not wired (see §5). `[FACT]`

---

## 2. Module & Component Breakdown

### Directory map `[FACT]`
```
app/
├── main.py              FastAPI app: REST routes, WebSocket, lifespan, rate limiter
├── config.py            pydantic-settings; single source of config truth
├── agents/
│   ├── state.py         ResearchState TypedDict + all sub-schemas (the blackboard)
│   ├── graph.py         StateGraph wiring, conditional edges, compiled singleton
│   ├── planner.py       Decomposes query → sub-questions + agent assignments
│   ├── paper_agent.py   arXiv retrieval (circuit-broken, parallel)
│   ├── news_agent.py    SerpAPI news retrieval (+ mock fallback)
│   ├── market_agent.py  Wikipedia retrieval (mock/stub for real market APIs)
│   ├── user_docs_agent.py  RAG over uploaded docs via FAISS
│   ├── analysis_agent.py   Contradiction detection + synthesis (fan-in point)
│   ├── insight_agent.py    Hypotheses, trends, knowledge gaps
│   ├── citation_agent.py   Fact-checking + citation formatting
│   ├── report_builder.py   Compiles final structured report
│   └── judge_agent.py      LLM-as-Judge quality scoring → self-correction gate
├── utils/
│   ├── llm.py           OpenRouter wrapper, structured output, model router
│   ├── decorators.py    circuit_breaker, retry, timer, idempotent
│   └── logging.py       structlog config
├── rag/
│   ├── vectorstore.py   FAISSVectorStore (session-scoped, async)
│   └── retriever.py     thin retrieval wrapper
├── security/guardrails.py  prompt-injection + harmful-content regex, sanitisation
├── auth/clerk.py        Clerk JWT verification + bypass, FastAPI dependency
├── tools/document_loader.py  PDF/DOCX/TXT/MD text extraction
├── observability/langsmith.py  tracing env setup
└── cache/               EMPTY (only __init__.py)  ← unfulfilled abstraction
frontend/streamlit_app.py    UI (NOT wired to backend — see §5)
tests/                   test_agents.py, test_security.py
```

### Component responsibilities & coupling

| Component | Responsibility `[FACT]` | Public interface | Depends on | Depended on by |
|---|---|---|---|---|
| `config.Settings` | All config, cached singleton | `get_settings()` | env/`.env` | ~every module |
| `agents/state` | Shared state contract | `ResearchState` + TypedDicts | stdlib only | every agent, graph, main |
| `agents/graph` | Orchestration + edges | `run_research`, `resume_after_hitl`, `research_graph` | all agent nodes, config | `main` |
| Retrieval agents (paper/news/market/user_docs) | Fetch + normalize to `SourceDoc` | `*_node(state)` | httpx/arxiv/FAISS, decorators | graph |
| `analysis_agent` | Synthesis + contradictions (fan-in) | `analysis_agent_node` | `utils/llm` | graph (loop target) |
| `insight/citation/report_builder/judge` | Linear reasoning pipeline | `*_node(state)` | `utils/llm` | graph |
| `utils/llm` | LLM calls + structured parse + routing | `call_llm`, `call_llm_structured`, `get_model_for_task` | litellm, config, retry | every LLM agent |
| `utils/decorators` | Resilience/util decorators | `circuit_breaker`, `retry`, `timer`, `idempotent` | logging | agents, llm |
| `rag/vectorstore` | FAISS index lifecycle | `FAISSVectorStore` | langchain-community/hf | user_docs_agent |
| `security/guardrails` | Input validation | `sanitise_query`, `validate_query` | bleach, config | main |
| `auth/clerk` | AuthN | `CurrentUser`, `get_current_user` | httpx, config | main routes |

### Single-responsibility flags `[FACT]`
- **Good:** one node per file, cohesive; `state.py` cleanly isolates the data contract; `config.py` centralizes constants (as the `.cursorrules` mandates).
- **`utils/decorators.py`** is a grab-bag (circuit breaker + retry + timer + idempotency) but each is cohesive. `idempotent` is defined and never used.
- **`market_agent`** is misnamed vs. behavior: it queries Wikipedia, not market data (acknowledged in its own docstring as a stub).
- **`_now()` is duplicated in ~9 agent files** — copy-paste helper that belongs in `utils`.

---

## 3. Architecture Mapping

### Entry points `[FACT]`
1. **FastAPI** (`app/main.py`) — REST: `POST /api/v1/research` (create session) → `POST /api/v1/research/{id}/run` (start graph, accepts file uploads) → `POST /api/v1/research/{id}/hitl` (approve/reject) → `GET /api/v1/research/{id}/state`; plus `GET /health`, `GET /`, and `WS /api/v1/ws/{id}`.
2. **Streamlit** (`frontend/streamlit_app.py`) — user-facing UI (currently a standalone mockup, §5).

### Request lifecycle (intended) `[FACT]`
```
Client
  │ POST /api/v1/research  (auth: CurrentUser)
  ▼
sanitise_query → validate_query (guardrails)
  │ create session_id (uuid4), _session_queues[id] = asyncio.Queue()
  ▼ returns ws_url + session_id
Client ── connects WS /ws/{id} ──┐
  │ POST /research/{id}/run       │
  ▼                               │
extract_text_from_upload(docs)    │  (streaming channel — intended)
  │ asyncio.create_task(_run_graph)
  ▼
run_research() → research_graph.ainvoke(initial_state, thread_id=id)
  │
  ▼  (graph executes, see below)
interrupt_before "hitl_review"  ── pauses ──► ainvoke returns partial state
  │
Client POST /research/{id}/hitl (approved, feedback)
  ▼
resume_after_hitl → aupdate_state → ainvoke(None)  ── resumes ──► END or rebuild
```

### Agent graph topology `[FACT]` (from `graph.build_graph`)
```
                     ┌─────────────┐
                     │   planner   │  (entry)
                     └──────┬──────┘
        ┌───────────┬───────┴───────┬─────────────┐   ← parallel fan-out
        ▼           ▼               ▼             ▼
   paper_agent  news_agent    market_agent  user_docs_agent
        └───────────┴───────┬───────┴─────────────┘   ← fan-in (all 4 → analysis)
                            ▼
                     analysis_agent ◄──────────────┐
                            ▼                       │
                     insight_agent                  │ self-correction
                            ▼                       │ (judge score < threshold
                     citation_agent                 │  AND loops remain)
                            ▼                       │
                     report_builder ◄──────┐        │
                            ▼              │        │
                       judge_agent ────────┼────────┘
                            │              │
             (score OK) ────┼──► hitl_review (INTERRUPT before)
                            │        │
                            │        ├── approved ──► END
                            │        └── rejected ──► report_builder ┘
```

### Inter-module communication `[FACT]`
- **Agents ↔ agents:** *never* call each other directly. All communication is via the shared `ResearchState` and LangGraph edges. Concurrency-safe accumulation uses `Annotated[list, operator.add]` reducers on `sources`, `raw_texts`, `summaries`, `contradictions`, `hypotheses`, `trends`, `fact_checks`, `citations`, `events`.
- **API ↔ graph:** `main` calls `run_research` / `resume_after_hitl`; reads checkpoint via `research_graph.get_state`.
- **Agents ↔ LLM:** all through `utils/llm` (single choke point).
- **External services:** httpx (Wikipedia, SerpAPI, Clerk), `arxiv` client (run in executor), OpenRouter (LiteLLM).

### Data model & flow `[FACT]`
- Canonical unit is **`SourceDoc`** `{title, url, content, source_type, confidence, published_date}`. Every retrieval agent normalizes into this.
- Confidence is a **hardcoded per-source-type prior**, not computed: arXiv 0.92, Wikipedia 0.85, user_doc 0.88, news 0.72, mock news 0.60.
- Flow: `SourceDoc[]` → analysis (`summaries`, `Contradiction[]`) → insight (`Hypothesis[]`, `trends`) → citation (`FactCheckResult[]`, `Citation[]`) → report (`report` dict with `stats`) → judge (`QualityScore`).

---

## 4. Design Pattern & Decision Analysis

### Intentional patterns `[FACT]`
- **Blackboard / shared-state** (`ResearchState`).
- **Map-reduce fan-out/fan-in** over retrieval agents (LangGraph parallel super-step + `operator.add` reducers).
- **Reflection / LLM-as-Judge + self-correction loop.**
- **Human-in-the-loop interrupt** (`interrupt_before=["hitl_review"]`).
- **Decorator pattern** for cross-cutting concerns: `@circuit_breaker`, `@retry`, `@timer`.
- **Circuit Breaker** (CLOSED/OPEN/HALF_OPEN state machine, per-service registry).
- **Strategy/Router** — `get_model_for_task` maps task type → model (cost/quality tiering).
- **Singleton** — `get_settings` (`lru_cache`), embedding model, breaker registry.
- **Graceful degradation** — mock news when no SerpAPI key; fallback plans/outputs when structured parse returns `None`; empty-string returns on doc extraction failure.
- **Structured outputs** — Pydantic schema injected into system prompt, parsed with `model_validate_json` after stripping code fences.

### Anti-patterns, smells & AI-generation tells `[FACT]`
1. **`_now()` duplicated across ~9 files** — textbook AI copy-paste; belongs in `utils`.
2. **Config drift / unused config** — `cache_ttl_seconds`, `cache_max_size`, `rate_limit_requests`, `rate_limit_window`, `chroma_persist_dir`, `clerk_jwt_audience` are declared but never consumed by logic.
3. **Dead dependencies** — `chromadb` (never imported; only FAISS used), `python-jose[cryptography]` (never used; Clerk verified via HTTP), `openai`, `google-search-results`/`wikipedia-api` partly superseded by direct httpx, `playwright`/`streamlit-authenticator`/`audio-recorder` largely aspirational.
4. **Dead abstractions** — empty `cache/` package; `idempotent` decorator defined but never applied; `FAISSVectorStore.save_to_disk`/`load_from_disk` never called.
5. **`.cursorrules` references `emit_event()`** as a required convention — **no such function exists** anywhere. The convention was documented but never implemented, which is exactly why streaming is broken (§5).
6. **Inconsistent typing** — most `done_event`s are annotated `AgentEvent`; `user_docs_agent`'s is a bare `dict`.
7. **Misleading param** — `call_llm(response_format=...)` implies native structured output, but it only injects a schema into the system prompt; `acompletion` is called without `response_format`.
8. **Non-monotonic progress** — parallel agents each hardcode `progress` (25/28/28/30); last-writer-wins under concurrency.
9. **Hacky dedup** — `paper_agent` uses a walrus `seen.add()` side-effect inside a comprehension (`# type: ignore`).

### Implicit assumptions baked in `[FACT/INFER]`
- **Single-process, single-worker deployment.** All coordination state (`_session_queues`, `MemorySaver`, `_breakers`, `_idempotency_cache`) is in-process module globals. Multiple uvicorn workers would break session/queue routing. `[FACT]`
- `localhost` is hardcoded in the `ws_url`, OpenRouter referer, and frontend `BACKEND_URL` — not environment-driven. `[FACT]`
- The Judge threshold (6.0) and reducer-based accumulation assume a *single* pass; multiple correction loops **append** rather than replace summaries. `[FACT]`
- Clerk token verification via `GET api.clerk.com/v1/tokens/verify?token=` assumes that endpoint/shape; standard Clerk practice is local JWKS verification. `[INFER — verify]`

---

## 5. Risk & Improvement Signals

### 🔴 High-severity (functional/security)

1. **Real-time streaming is not wired `[FACT]`.** Agents append events to `state["events"]`, but nothing ever pushes them into `_session_queues[id]`. `grep` shows the queue only ever receives `{"type": "fatal_error"}`. The WebSocket therefore streams *nothing* but keepalive pings and errors. The `.cursorrules`-mandated `emit_event()` was never built. → The advertised "real-time agent activity" cannot function as built.

2. **Frontend is a disconnected mockup `[FACT]`.** `frontend/streamlit_app.py` defines `BACKEND_URL`/`API_BASE` but makes **no HTTP/WS call** to the backend. "Launch Research" only resets `st.session_state` and reruns (comment: *"Simulate demo mode with fake events"*). HITL Approve/Revise buttons mutate local state only — they never `POST /hitl`. → End-to-end product does not actually run through the backend.

3. **Self-correction loop counter never increments — potential infinite/costly loop `[FACT]`.** In `analysis_agent.py`:
   ```python
   "correction_loop_count": loop_count + (1 if loop_count > 0 else 0)
   ```
   Starting at 0, this evaluates `0 + 0 = 0` forever. The guard `loop_count < max_loops` in `should_self_correct` thus never trips on the counter; the loop is bounded only by the Judge eventually scoring ≥ threshold. Under a persistently low score the pipeline can loop up to (or beyond) intent, burning GPT-4o/Claude calls. **Fix:** `loop_count + 1`.

4. **Secrets management / no `.gitignore` `[FACT]`.** A populated `.env` with real API keys sits at repo root and there is **no root `.gitignore`** (only `venv/.gitignore` exists). If this repo is version-controlled, `OPENROUTER_API_KEY`, `CLERK_SECRET_KEY`, `SERPAPI_KEY`, `LANGSMITH_API_KEY` are exposed. **Action: verify `.env` is untracked; rotate keys if it was ever committed.**

5. **Rate limiting configured but not enforced `[FACT]`.** `Limiter` is instantiated and the exception handler registered, but **no route carries `@limiter.limit(...)`**. `rate_limit_*` config is unused. Endpoints are effectively unthrottled.

6. **Unauthenticated WebSocket `[FACT]`.** `WS /api/v1/ws/{id}` performs no auth and will auto-create a queue for any `session_id`. Mitigated only by UUID unguessability; no ownership check ties a socket to `CurrentUser`.

### 🟠 Medium-severity

7. **Loop-back fan-in correctness `[INFER — verify]`.** `analysis_agent` has 4 static incoming edges (the retrieval agents) *and* a conditional edge from `judge`. When the Judge routes back, only `judge` fired in that super-step. LangGraph's barrier semantics for a node with multiple static predecessors may cause the re-entry to stall or behave unexpectedly. Needs a runtime test of an actual self-correction cycle.

8. **State accumulation across correction loops `[FACT]`.** `summaries` uses `operator.add`; re-running analysis **appends** a second summary. `report_builder` joins all summaries, and `executive_summary` uses `summaries[0]` (the *stale* first pass). Corrections compound rather than supersede.

9. **In-memory everything = no durability/scaling `[FACT]`.** `MemorySaver` (self-noted as hackathon-grade), session queues, breakers, idempotency cache are process-local. Restart loses all sessions; horizontal scaling breaks routing. Production needs `SqliteSaver`/`PostgresSaver` + shared queue (Redis) + sticky or externalized session state.

10. **Indirect prompt injection unmitigated `[FACT]`.** Guardrails scan only the *user query*. Retrieved web/news/arXiv/user-doc content flows unfiltered into analysis/insight/report LLM prompts — the classic RAG injection vector is open.

11. **`top_k_retrieval` setting bypassed `[FACT]`.** `user_docs_agent` calls `retrieve_relevant_chunks(vs, query)` which defaults `k=5`, so the configured `top_k_retrieval` never applies via this path.

12. **No re-validation on `/run` `[FACT]`.** `/run` re-sanitises but does not re-run `validate_query`; validation happens only at session creation.

### 🟡 Low-severity
- `market_agent` name vs. Wikipedia behavior; hardcoded confidence priors; duplicated `_now()`; non-monotonic progress; `HALF_OPEN` breaker admits unlimited concurrent probes (returns `True`).

### Testing & docs gaps `[FACT]`
- Tests cover **planner, citation, judge** nodes and **all guardrails** (good, fully mocked). **No tests** for: `graph` wiring/edges, self-correction routing, HITL resume, `paper/news/market/user_docs` agents, `analysis`/`insight`/`report_builder`, `utils/llm`, decorators, RAG, auth, document loader. The `.cursorrules` "test for every agent node" rule is not met.
- No README, no architecture doc, no API usage doc, no deployment/runbook.

---

## Prioritized "Verify Manually" list

1. **Is `.env` tracked by git?** If yes → rotate all keys immediately, add `.gitignore`. (Highest urgency.)
2. **Run one full pipeline end-to-end** against OpenRouter and watch the WebSocket — confirm streaming is dead as analyzed, and confirm whether the graph completes past HITL.
3. **Force a low Judge score** and confirm the self-correction loop behavior (counter bug + fan-in re-entry #7).
4. **Confirm the Clerk verification endpoint** (`/v1/tokens/verify`) is correct for your Clerk setup, or migrate to JWKS.
5. **Confirm intended deployment topology** (single worker assumed) before any scaling.
6. Confirm whether the Streamlit frontend is meant to call the backend or is a demo shell.

## Suggested next steps for deeper review (incremental order)

1. **Entry & contract:** `config.py` → `agents/state.py` → `main.py` → `agents/graph.py` (understand the state contract and control flow first).
2. **Orchestration correctness:** trace `should_self_correct` / `after_hitl` and the counter/fan-in bugs (#3, #7, #8).
3. **LLM choke point:** `utils/llm.py` + `utils/decorators.py` (retry/breaker interaction, structured-output fragility).
4. **Retrieval agents:** paper → news → market → user_docs (external-call resilience, mock paths).
5. **Reasoning pipeline:** analysis → insight → citation → report_builder → judge.
6. **Data layer:** `rag/vectorstore.py`, `rag/retriever.py`, `tools/document_loader.py`.
7. **Cross-cutting:** `security/guardrails.py`, `auth/clerk.py`, `observability/langsmith.py`.
8. **Frontend:** `frontend/streamlit_app.py` (wire to backend or acknowledge as mock).

## Highest-leverage fixes (quick wins)
- `correction_loop_count + 1` (one-char class of bug, real impact).
- Implement `emit_event()` that pushes to `_session_queues[id]`, and call it in each node — unlocks the core streaming feature.
- Add `.gitignore` + rotate keys.
- Apply `@limiter.limit` to research routes.
- Wire the Streamlit client to the REST/WS API (or clearly label it demo).
- Hoist `_now()` to `utils`; drop unused deps (`chromadb`, `python-jose`) and dead config.
