"""Bootstrap for UI mode ``--ui web``.

Launches the Gradio-managed server (so ``--share`` still produces a public
``*.gradio.live`` URL that proxies every route) and then attaches:

* ``/webui/``  — the custom HTML/CSS/JS frontend (static files from ui/webui)
* ``/api/v1/…`` — the REST API from programs/webui/api.py
* ``/``        — a redirect to the custom UI

The classic Gradio Blocks UI remains available through ``ui/app.py`` without
the ``--ui web`` flag; both modes share one code base and one server stack.
"""

import os

import gradio as gr
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from starlette.routing import Route

from config.logging_setup import get_logger
from programs.webui.api import WEBUI_DIR, router

logger = get_logger("webui_server")


def _redirect_to_webui(request):
    """Send users landing on "/" straight to the custom WebUI."""
    return RedirectResponse(url="/webui/", status_code=307)


def launch_webui(args) -> None:
    """Start the server and block until interrupted (same semantics as the
    Gradio UI launch in ui/app.py: --listen/--listen-host/--listen-port/--share)."""
    if not WEBUI_DIR.is_dir() or not (WEBUI_DIR / "index.html").is_file():
        raise RuntimeError(f"Custom WebUI assets are missing at {WEBUI_DIR}.")

    # Minimal Blocks page: only exists to bootstrap the Gradio server, share
    # tunnel included. Users never see it because "/" is redirected below.
    with gr.Blocks(title="AsukaRVC") as bootstrap:
        gr.Markdown("AsukaRVC custom WebUI is served at `/webui/` — you should be redirected there.")

    app, local_url, share_url = bootstrap.launch(
        prevent_thread_lock=True,
        share=args.share_enabled,
        show_error=True,
        server_name=None if not args.listen else (args.listen_host or "0.0.0.0"),
        server_port=args.listen_port,
        ssr_mode=False,
    )

    # Attach the custom UI and REST API to the same running server. Starlette
    # resolves routes per request, so mounting after startup is safe; the
    # redirect route is inserted first so it wins over the Gradio "/" page.
    app.include_router(router)
    app.mount("/webui", StaticFiles(directory=str(WEBUI_DIR), html=True), name="webui")
    app.router.routes.insert(0, Route("/", _redirect_to_webui, methods=["GET"]))

    print("-----------")
    print("AsukaRVC V3 — custom WebUI")
    print("-----------")
    if local_url:
        print(f"* Local WebUI:     {local_url.rstrip('/')}/webui/")
        print(f"* Local REST API:  {local_url.rstrip('/')}/api/v1/health")
    if share_url:
        print(f"* Public WebUI:    {share_url.rstrip('/')}/webui/")
        print(f"* Public REST API: {share_url.rstrip('/')}/api/v1/health")
    if not share_url:
        print("* Tip: pass --share to expose this WebUI through a public gradio.live URL.")

    # Keep the main thread alive (uvicorn runs in the background thread).
    bootstrap.block_thread()
