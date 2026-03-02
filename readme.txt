# Keeta Merchant AI Assistant (Project K-MA)

## 1. Project Overview
An intelligent vertical agent for Keeta catering merchants, benchmarking the "1688 Merchant Assistant."
**Core Mission:** Provide data queries, business diagnosis, and automated reporting via a natural language interface while strictly enforcing data isolation and security.

## 2. Tech Stack
- **Backend:** Python 3.10+, FastAPI (Async), Celery (Scheduled tasks).
- **AI/LLM:** LangChain, OpenAI API (or compatible), FAISS (Vector Store).
- **Database:** 
  - *App Data:* PostgreSQL (User auth, chat history).
  - *Big Data Source:* Hive/Presto (Read-only for analytics).
- **Frontend:** React, TailwindCSS, ECharts.

## 3. Directory Structure (Standard for Copilot Context)
```text
/backend
  /app
    /api            # API Route Controllers
    /core           # Config, Security, Middleware (Auth)
    /services       # Business Logic (Report generation)
    /agents         # LangChain Agents
      - router.py   # Intent Classification
      - sql_agent.py# Text-to-SQL Logic
      - chat.py     # General QA
    /models         # Pydantic & ORM Models
    /utils          # Helper functions (DLP, Formatting)
  /tests            # Pytest modules