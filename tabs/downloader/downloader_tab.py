"""Builder for the "Download model" tab of the AsukaRVC WebUI."""

import gradio as gr

from tabs.common import (
    download_online_model,
    filter_models,
    load_public_models,
    pub_dl_autofill,
)


def downloader_tab():
    """Build the "Download model" tab. Must be called inside a gr.Blocks context."""
    with gr.Tab('Download model'):

        with gr.Tab('From HuggingFace/Pixeldrain URL'):
            with gr.Row():
                model_zip_link = gr.Textbox(label='Download link to model')
                model_name = gr.Textbox(label='Name your model')
            with gr.Row():
                download_btn = gr.Button('Download 🌐', variant='primary', scale=19)
                dl_output_message = gr.Textbox(label='Output Message', interactive=False, scale=20)
            download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)
            gr.Markdown('## Input Examples')
            gr.Examples(
                [
                    ['https://huggingface.co/MrDawg/ToothBrushing/resolve/main/ToothBrushing.zip?download=true', 'ToothBrushing'],
                    ['https://huggingface.co/sail-rvc/Aldeano_Minecraft__RVC_V2_-_500_Epochs_/resolve/main/model.pth?download=true, https://huggingface.co/sail-rvc/Aldeano_Minecraft__RVC_V2_-_500_Epochs_/resolve/main/model.index?download=true', 'Minecraft_Villager'],
                    ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                    ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                    ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']
                ],
                [model_zip_link, model_name], [], download_online_model, cache_examples=False,
            )

        with gr.Tab('From Public Index'):
            gr.Markdown('## How to use')
            gr.Markdown('- Click Initialize public models table')
            gr.Markdown('- Filter models using tags or search bar')
            gr.Markdown('- Select a row to autofill the download link and model name')
            gr.Markdown('- Click Download')
            with gr.Row():
                pub_zip_link = gr.Textbox(label='Download link to model')
                pub_model_name = gr.Textbox(label='Model name')
            with gr.Row():
                download_pub_btn = gr.Button('Download 🌐', variant='primary', scale=19)
                pub_dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)
            filter_tags = gr.CheckboxGroup(value=[], label='Show voice models with tags', choices=[])
            search_query = gr.Text(label='Search')
            load_public_models_button = gr.Button(value='Initialize public models table', variant='primary')
            public_models_table = gr.DataFrame(value=[], headers=['Model Name', 'Description', 'Credit', 'URL', 'Tags'],
                                               label='Available Public Models', interactive=False)
            public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
            load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
            search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
            filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
            download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)
