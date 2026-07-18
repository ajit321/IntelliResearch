# 🧠 IntelliResearch — Complete Setup Guide for Beginners

> This guide walks you through setting up IntelliResearch from **absolute scratch**. No prior experience needed!

---

## 📌 What is IntelliResearch?

IntelliResearch is a **Multi-Agent AI Research Platform** that:
- Runs **10 AI agents** in parallel to research any topic
- Searches academic papers, news, market data, and your own documents
- Generates a full research report with citations
- Streams live progress via a beautiful **Streamlit web UI**

It has **two services** you must run simultaneously:
| Service | What it does | URL |
|---------|-------------|-----|
| **FastAPI Backend** | Runs the AI agents & REST API | `http://localhost:8000` |
| **Streamlit Frontend** | The web UI you interact with | `http://localhost:8501` |

---

## 🔑 API Keys Required

You need the following keys. Here's what each one is for:

### ✅ REQUIRED (the app won't work without this)

| Key | Variable Name | Where to Get It | Cost |
|-----|--------------|----------------|------|
| **OpenRouter API Key** | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) → Sign up → API Keys | Free tier available |

### ⚡ OPTIONAL (app works without these, but with reduced features)

| Key | Variable Name | Where to Get It | What it unlocks |
|-----|--------------|----------------|----------------|
| **SerpAPI Key** | `SERPAPI_KEY` | [serpapi.com](https://serpapi.com) | Real web search results (mock data used without it) |
| **LangSmith API Key** | `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | Agent tracing & debugging dashboard |
| **Clerk Auth Keys** | `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | [clerk.com](https://clerk.com) | User login/authentication (bypassed in dev mode) |

> [!IMPORTANT]
> **You only NEED the OpenRouter key to start.** Everything else is optional for local development.

---

## 🛠️ Step-by-Step Setup

### Step 1 — Check Python is Installed

Open **Terminal** (Mac: press `Cmd + Space`, type "Terminal", press Enter) and run:

```bash
python3 --version
```

You should see something like `Python 3.11.x` or `Python 3.12.x`. If you see an error, download Python from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Navigate to the Project Folder

```bash
cd "/Users/admin/Documents/Antigravity/Create Design/IntelliResearch"
```

> [!TIP]
> Always make sure you are inside this folder before running any commands below.

---

### Step 3 — Create a Virtual Environment

A virtual environment keeps the project's libraries separate from everything else on your Mac.

```bash
python3 -m venv venv
```

Then **activate** it:

```bash
source venv/bin/activate
```

✅ You'll see `(venv)` appear at the start of your terminal prompt — that means it's active!

> [!NOTE]
> You must run `source venv/bin/activate` **every time** you open a new Terminal window before working on this project.

---

### Step 4 — Install All Dependencies

```bash
pip install -r requirements.txt
```

This installs ~40 libraries. It will take **3–5 minutes** the first time. Be patient! ☕

---

### Step 5 — Set Up Your API Keys in the `.env` File

The `.env` file already exists in the project. You just need to fill in your API keys.

Open the `.env` file in any text editor and update these lines:

```env
# ✅ REQUIRED — Get from https://openrouter.ai
OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here

# ✅ REQUIRED — Choose a model (this is already set, no change needed)
LLM_MODEL=openrouter/openai/gpt-4o-mini

# ⚡ OPTIONAL — Get from https://serpapi.com (leave blank to use mock data)
SERPAPI_KEY=

# ⚡ OPTIONAL — Get from https://smith.langchain.com (leave blank to disable tracing)
LANGSMITH_API_KEY=

# 🔓 DEV MODE — Keep as false to skip login (no Clerk keys needed locally)
# Set AUTH_BYPASS=true in the .env file if you face auth errors
```

> [!CAUTION]
> Never share your `.env` file publicly or commit it to GitHub. It contains your secret API keys!

---

### Step 6 — Enable Auth Bypass for Local Development

Since you don't have Clerk keys set up, you need to bypass authentication.
Open `.env` and confirm this line exists (add it if missing):

```env
AUTH_BYPASS=true
```

> [!NOTE]
> The Clerk keys in the current `.env` file are placeholder values (`pk_live_your_clerk...`). Without auth bypass enabled, the backend will reject all requests.

---

### Step 7 — Create Required Data Directories

```bash
mkdir -p data/faiss_index data/chroma
```

These folders store the local vector database for your uploaded documents.

---

## 🚀 Running the Project

You need **two Terminal windows** open at the same time.

### Terminal Window 1 — Start the Backend (FastAPI)

```bash
# Navigate to project
cd "/Users/admin/Documents/Antigravity/Create Design/IntelliResearch"

# Activate virtual environment
source venv/bin/activate

# Start the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

### Terminal Window 2 — Start the Frontend (Streamlit)

Open a **new** Terminal window, then:

```bash
# Navigate to project
cd "/Users/admin/Documents/Antigravity/Create Design/IntelliResearch"

# Activate virtual environment
source venv/bin/activate

# Start the Streamlit frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

✅ You should see:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

---

### Step 8 — Open the App in Your Browser

Go to: **[http://localhost:8501](http://localhost:8501)**

You should see the IntelliResearch UI! 🎉

---

## ✅ Quick Verification

After both services are running, test the backend is healthy:

Open your browser and go to: **[http://localhost:8000/health](http://localhost:8000/health)**

You should see:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development",
  "auth_enabled": false,
  "tracing_enabled": false
}
```

You can also browse the full **API documentation** at: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📂 Project Structure Overview

```
IntelliResearch/
├── .env                    ← Your API keys go here
├── requirements.txt        ← All Python dependencies
├── app/
│   ├── main.py             ← FastAPI backend entry point
│   ├── config.py           ← Reads settings from .env
│   ├── agents/             ← 10 AI research agents (LangGraph)
│   │   ├── graph.py        ← Main agent pipeline orchestration
│   │   ├── planner.py      ← Plans the research strategy
│   │   ├── paper_agent.py  ← Fetches academic papers (arXiv)
│   │   ├── news_agent.py   ← Fetches news articles
│   │   ├── market_agent.py ← Fetches market/financial data
│   │   ├── analysis_agent.py ← Analyzes gathered data
│   │   ├── report_builder.py ← Writes the final report
│   │   └── judge_agent.py  ← Quality checks the report
│   ├── auth/               ← Authentication (Clerk)
│   ├── rag/                ← Vector search (FAISS/Chroma)
│   └── tools/              ← Document loaders, web search
├── frontend/
│   └── streamlit_app.py    ← Web UI (what you see in the browser)
└── data/
    ├── faiss_index/        ← Local vector store for your docs
    └── chroma/             ← ChromaDB persistent storage
```

---

## 🔧 Troubleshooting Common Issues

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Make sure `(venv)` is active. Run `source venv/bin/activate` |
| `OPENROUTER_API_KEY not set` | Open `.env` and add your real OpenRouter key |
| `Connection refused` on Streamlit | Make sure the FastAPI backend is running in Terminal 1 |
| `RateLimitExceeded` error | Wait 60 seconds and try again (10 requests/minute limit) |
| `Permission denied` on data folder | Run `mkdir -p data/faiss_index data/chroma` |
| Port 8000 already in use | Run `lsof -ti:8000 | xargs kill -9` then restart |
| Port 8501 already in use | Run `lsof -ti:8501 | xargs kill -9` then restart |

---

## 🛑 How to Stop the App

In each Terminal window, press **`Ctrl + C`** to stop the server.

---

## 🔑 Full API Keys Summary

| Priority | Key Name | Environment Variable | Where to Get | Required? |
|----------|---------|---------------------|-------------|-----------|
| 🔴 Critical | OpenRouter API Key | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | **YES** |
| 🟡 Optional | SerpAPI Key | `SERPAPI_KEY` | [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) | No (mock used) |
| 🟡 Optional | LangSmith Key | `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | No (tracing disabled) |
| 🟢 Skip for Dev | Clerk Publishable Key | `CLERK_PUBLISHABLE_KEY` | [clerk.com](https://clerk.com) | No (use AUTH_BYPASS=true) |
| 🟢 Skip for Dev | Clerk Secret Key | `CLERK_SECRET_KEY` | [clerk.com](https://clerk.com) | No (use AUTH_BYPASS=true) |

