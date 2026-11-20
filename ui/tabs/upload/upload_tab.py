"""Builder for the "Upload model" tab of the AsukaRVC WebUI.

Accepts a .zip archive with a locally trained RVC voice model (a .pth file,
optionally a .index feature file) and installs it into the ``rvc_models``
workspace so it shows up in the Inference tab.
"""

import gradio as gr

from ui.tabs.common import upload_local_model


def upload_tab():
    """Build the "Upload model" tab. Must be called inside a gr.Blocks context."""
    with gr.Tab('Upload model'):
        gr.Markdown('## How to use')
        gr.Markdown('- Zip your locally trained model (must contain a .pth file,'
                    ' optionally a .index file)')
        gr.Markdown('- Pick a unique name for the voice model')
        gr.Markdown('- Click Upload and it becomes available in the Inference tab')
        with gr.Row():
            model_zip = gr.File(label='Zip file containing your model', file_types=['.zip'])
            model_name = gr.Textbox(label='Name your model')
        with gr.Row():
            upload_btn = gr.Button('Upload ⬆', variant='primary', scale=19)
            upload_output_message = gr.Textbox(label='Output Message', interactive=False, scale=20)
        upload_btn.click(upload_local_model, inputs=[model_zip, model_name], outputs=upload_output_message)
