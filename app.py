import os
from argparse import ArgumentParser

import spaces
import gradio as gr

from logging_setup import configure_logging_libs, setup_logging

configure_logging_libs()
setup_logging()

from tabs.common import IS_ZERO_GPU
from tabs.tabs_main import downloader_tab, inference_tab, upload_tab

if __name__ == '__main__':
    parser = ArgumentParser(description='AsukaRVC V3.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--builtin-player", action="store_true", default=False, help="Use the builtin audio player")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    parser.add_argument('--theme', type=str, default="gradio/base", help='Set the theme (default: base)')
    parser.add_argument("--ssr", action="store_true", help="Enable SSR (Server-Side Rendering)")
    args = parser.parse_args()

    with gr.Blocks(title='AsukaRVC V 3', theme=args.theme, fill_width=True, fill_height=True) as app:

        gr.Label(f'AsukaRVC V3 {"ZeroGPU mode" if IS_ZERO_GPU else ""} created with ❤️', show_label=False)
        if IS_ZERO_GPU:
            gr.Markdown(
                """
                <details>
                    <summary style="font-size: 1.5em;">⚠️ Important (click to expand)</summary>
                    <ul>
                        <li>🚀 This demo use a Zero GPU, which is available only for a limited time. It's recommended to use audio files that are no longer than 5 minutes. If you want to use it without time restrictions, you can duplicate the 'old CPU space'. ⏳</li>
                    </ul>
                </details>
                """
            )
            gr.Markdown("Duplicate the old CPU space for use in private: [![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm-dark.svg)](https://huggingface.co/spaces/r3gm/AICoverGen_old_stable_cpu?duplicate=true)\n\n")

        inference_tab()
        downloader_tab()
        upload_tab()

    print("-----------")
    print("AsukaRVC V3")
    print("-----------")
    app.launch(
        share=args.share_enabled,
        debug=args.share_enabled,
        show_error=True,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
        ssr_mode=args.ssr
    )
