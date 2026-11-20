import os, sys
from argparse import ArgumentParser

try:
    # Hugging Face Spaces runtime helper; plain local installs can run without it.
    import spaces
except ImportError:  # pragma: no cover
    spaces = None

import gradio as gr


# Pipeline modules resolve their storage dirs (rvc_models, song_output, logs)
# from the current working directory, so always run from the repository root
# regardless of where the interpreter was launched.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

current_dir = os.getcwd()
sys.path.append(current_dir)


from config.logging_setup import configure_logging_libs, setup_logging

configure_logging_libs()
setup_logging()

from ui.tabs.common import IS_ZERO_GPU, ensure_rvc_models_dir

# Make sure the voice model workspace (rvc_models dir + public index files)
# exists before any tab lists its contents.
ensure_rvc_models_dir()

if __name__ == '__main__':
    parser = ArgumentParser(description='AsukaRVC V3.', add_help=True)
    parser.add_argument('--ui', choices=['gradio', 'web'], default='gradio',
                        help='UI mode: "gradio" = the classic Gradio Blocks UI (default); '
                             '"web" = the custom HTML/CSS/JS UI served by the same Gradio server '
                             '(REST API under /api/v1, frontend under /webui/).')
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    ##parser.add_argument("--builtin-player", action="store_true", default=False, help="Use the builtin audio player")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    parser.add_argument('--theme', type=str, default="gradio/base", help='Set the theme (default: base)')
    parser.add_argument("--ssr", action="store_true", help="Enable SSR (Server-Side Rendering)")
    args = parser.parse_args()

    if args.ui == 'web':
        # Custom HTML/CSS/JS UI hosted by the Gradio server: the frontend is
        # mounted at /webui, the REST API at /api/v1, and --share exposes both
        # through a public *.gradio.live URL.
        from programs.webui.server import launch_webui

        launch_webui(args)
    else:
        # The classic Gradio UI needs the tab builders; the custom web UI does
        # not, so they are only imported in this branch.
        from ui.tabs.tabs_main import downloader_tab, inference_tab, training_tab, upload_tab

        # Gradio 6.0 moved the Blocks theme from the constructor to launch().
        # Older versions still expect it in the constructor, so pick the right
        # spot based on the installed major version.
        gradio_major = int(gr.__version__.split('.')[0])
        blocks_kwargs = {'title': 'AsukaRVC V3', 'fill_width': True, 'fill_height': True}
        launch_kwargs = {
            'share': args.share_enabled,
            'debug': args.share_enabled,
            'show_error': True,
            'server_name': None if not args.listen else (args.listen_host or '0.0.0.0'),
            'server_port': args.listen_port,
            'ssr_mode': args.ssr,
        }
        if gradio_major >= 6:
            launch_kwargs['theme'] = args.theme
        else:
            blocks_kwargs['theme'] = args.theme

        with gr.Blocks(**blocks_kwargs) as app:

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
            with gr.Tab("Train"):
                training_tab()

        print("-----------")
        print("AsukaRVC V3")
        print("-----------")
        app.launch(**launch_kwargs)
