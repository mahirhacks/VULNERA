# VULNERA Web App

React frontend + FastAPI backend for C/C++ vulnerability scanning.

## Architecture

```
09_WEB/
├── front_end/     React (Vite) — UI, routing, API client
├── back_end/      FastAPI — scan jobs, ML pipeline, config, scan storage
├── scans/         Persisted scan JSON
└── web_config.yaml
```

The legacy Streamlit app has been removed. Use `front_end` + `back_end` below.

## Prerequisites

- Python 3.10+ with project ML dependencies (torch, transformers, etc.)
- Node.js 18+

## Run (development)

**Windows + `#` in folder path:** Vite cannot load modules from paths like `D:\#STUDIES\...` because `#` is treated as a URL fragment. The frontend `npm run dev` script automatically maps drive `V:` with `SUBST` when needed. To remove the mapping later: `subst V: /d`.

**Terminal 1 — backend** (from repo root):

```bash
pip install -r requirements.txt
cd 09_WEB/back_end
uvicorn main:app --port 8000
```

For LLM scans use **`--port 8000` only** — avoid `--reload`. Reload restarts the server mid-scan and kills Qwen while checkpoint shards are loading.

Dev reload (UI-only work, mock LLM):

```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend**:

```bash
cd 09_WEB/front_end
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies `/api` to the backend on port 8000.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET/PUT | `/api/config` | Read/write `web_config.yaml` |
| GET | `/api/scans` | List past scans |
| GET | `/api/scans/{id}` | Load scan result |
| DELETE | `/api/scans/{id}` | Delete scan |
| POST | `/api/scans` | Upload file → `{ job_id }` |
| GET | `/api/scans/jobs/{id}` | Job snapshot |
| GET | `/api/scans/jobs/{id}/stream` | SSE progress stream |

## Production build (optional)

```bash
cd 09_WEB/front_end && npm run build
```

Serve `front_end/dist` behind a reverse proxy or mount as static files from FastAPI.
