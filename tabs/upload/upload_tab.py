"""Builder for the "Upload model" tab of the AsukaRVC WebUI."""

import gradio as gr

from tabs.common import upload_local_model


def upload_tab():
    """Build the "Upload model" tab. Must be called inside a gr.Blocks context."""
    with gr.Tab('Upload model'):
        gr.Markdown('## Upload locally trained RVC v2 model and index file')
        gr.Markdown('- Find model file (weights folder) and optional index file (logs/[name] folder)')
        gr.Markdown('- Compress files into zip file')
        gr.Markdown('- Upload zip file and give unique name for voice')
        gr.Markdown('- Click Upload model')
        with gr.Row():
            with gr.Column():
                zip_file = gr.File(label='Zip file')
            local_model_name = gr.Text(label='Model name')
        with gr.Row():
            model_upload_button = gr.Button('Upload model', variant='primary', scale=19)
            local_upload_output_message = gr.Text(label='Output Message', interactive=False, scale=20)
            model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)
