"""Custom WebUI backend: a REST API + static HTML/CSS/JS frontend served by the
same Gradio-managed server that hosts the classic Gradio Blocks UI.

Modules:
- jobs.py: background job manager (single active job, live log, stop support)
- api.py:  FastAPI router mounted under /api/v1 by ui/app.py --ui web
- server.py: bootstrap that launches the Gradio server and mounts everything

Heavy dependencies (torch via programs.core) are imported lazily inside the
request handlers so the server can start without the full ML stack.
"""
