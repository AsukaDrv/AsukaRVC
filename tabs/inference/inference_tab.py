"""Builder for the "Generate" tab of the AsukaRVC WebUI."""

import gradio as gr

from core import song_cover_pipeline, yt_download
from tabs.common import (
    AUDIO_EXTENSIONS,
    IS_ZERO_GPU,
    get_current_models,
    get_separation_model_names,
    process_file_upload,
    rvc_models_dir,
    show_backing_vocals_rvc_inputs,
    show_hop_slider,
    swap_visibility,
    update_models_list,
)


def inference_tab():
    """Build the "Generate" tab. Must be called inside a gr.Blocks context."""
    voice_models = get_current_models(rvc_models_dir)
    separation_model_names = get_separation_model_names()
    vocal_model_names = separation_model_names['vocals']
    karaoke_model_names = separation_model_names['karaoke']
    denoise_model_names = separation_model_names['denoise']
    dereverb_model_names = separation_model_names['dereverb']
    deecho_model_names = separation_model_names['deecho']

    with gr.Tab("Generate"):

        with gr.Accordion('Main Options'):
            with gr.Row():
                rvc_model = gr.Dropdown(voice_models, label='Voice Models')
                ref_btn = gr.Button('Refresh Models 🔁', variant='primary')

                with gr.Column(visible=False) as yt_link_col:
                    song_input = gr.Textbox(label='Song input')
                    show_file_upload_button = gr.Button('Upload file instead')

                with gr.Column(visible=True) as file_upload_col:
                    local_file = gr.File(label='Audio file', interactive=True,file_types=AUDIO_EXTENSIONS, height=150)
                    if not IS_ZERO_GPU:
                        with gr.Row():
                            with gr.Row(scale=2):
                                url_media_gui = gr.Textbox(value="", label="Enter URL",
                                                           placeholder="www.youtube.com/watch?v=g_9rPvbENUw", lines=1)
                            with gr.Row(scale=1):
                                url_button_gui = gr.Button("Process URL", variant="secondary")
                        url_button_gui.click(yt_download, [url_media_gui], [local_file])
                    song_input_file = gr.UploadButton('Upload 📂', file_types=['audio'], variant='primary', visible=False)
                    show_yt_link_button = gr.Button('Paste YouTube link/Path to local file instead', visible=False)
                    song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                with gr.Column():
                    pitch = gr.Slider(-3, 3, value=0, step=1, label='Pitch Change (Vocals ONLY)')
                    pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Overall Pitch Change')

                show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

        with gr.Accordion('Separation Models', open=False):
            gr.Markdown('### Choose which models to use for each separation step')
            with gr.Row():
                vocal_model_name = gr.Dropdown(vocal_model_names, value=vocal_model_names[0], label='Vocal Separation Model')
                karaoke_model_name = gr.Dropdown(karaoke_model_names, value=karaoke_model_names[0], label='Karaoke Model')
            with gr.Row():
                dereverb_model_name = gr.Dropdown(dereverb_model_names, value=dereverb_model_names[0], label='Dereverb Model')
                denoise_model_name = gr.Dropdown(denoise_model_names, value=denoise_model_names[0], label='Denoise Model')
                deecho_model_name = gr.Dropdown(deecho_model_names, value=deecho_model_names[0], label='Deecho Model')
            with gr.Row():
                use_dereverb = gr.Checkbox(value=True, label='Enable Dereverb')
                use_denoise = gr.Checkbox(value=False, label='Enable Denoise')
                use_deecho = gr.Checkbox(value=False, label='Enable Deecho')
                instruments_only = gr.Checkbox(value=False, label='Instruments Only')

        with gr.Accordion('Voice conversion options', open=False):
            with gr.Row():
                index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate')
                filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter radius')
                rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS mix rate')
                protect = gr.Slider(0, 0.5, value=0.33, label='Protect rate')
                with gr.Column():
                    f0_method = gr.Dropdown(['rmvpe', 'fcpe', 'crepe'], value='rmvpe', label='Pitch detection algorithm')
                    crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe hop length')
                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
            with gr.Row():
                steps = gr.Slider(minimum=1, maximum=3, label="Steps", value=1, step=1, interactive=True)
                extra_denoise = gr.Checkbox(True, label='Denoise')
                keep_files = gr.Checkbox((False if IS_ZERO_GPU else True), label='Keep intermediate files',
                                         interactive=(False if IS_ZERO_GPU else True))

            with gr.Row():
                split_audio = gr.Checkbox(value=False, label='Split Audio')
                autotune = gr.Checkbox(value=False, label='Autotune')
                embedder_model = gr.Dropdown(['contentvec'], value='contentvec', label='Embedder Model')

        with gr.Accordion('Backing Vocals RVC (optional)', open=False):
            infer_backing_vocals = gr.Checkbox(value=False, label='Infer Backing Vocals with separate RVC model')
            with gr.Column(visible=False) as backing_rvc_col:
                with gr.Row():
                    infer_backing_vocals_model = gr.Dropdown(voice_models, label='Backing Vocals RVC Model')
                    infer_backing_vocals_index = gr.Text(label='Backing Vocals Index Path')
                with gr.Row():
                    pitch_back = gr.Slider(-3, 3, value=0, step=1, label='Backing Vocals Pitch Change')
                    filter_radius_back = gr.Slider(0, 7, value=3, step=1, label='Backing Filter radius')
                    index_rate_back = gr.Slider(0, 1, value=0.5, label='Backing Index Rate')
                    rms_mix_rate_back = gr.Slider(0, 1, value=0.25, label='Backing RMS mix rate')
                with gr.Row():
                    protect_back = gr.Slider(0, 0.5, value=0.33, label='Backing Protect rate')
                    pitch_extract_back = gr.Dropdown(['rmvpe+', 'rmvpe', 'mangio-crepe'], value='rmvpe+', label='Backing Pitch detection')
                    hop_length_back = gr.Slider(32, 320, value=128, step=1, label='Backing Crepe hop length')
                    export_format_rvc_back = gr.Dropdown(['wav', 'flac', 'mp3'], value='wav', label='Backing RVC export format')
                with gr.Row():
                    split_audio_back = gr.Checkbox(value=False, label='Backing Split Audio')
                    autotune_back = gr.Checkbox(value=False, label='Backing Autotune')
                    embedder_model_back = gr.Dropdown(['contentvec', 'chinese-hubert-base', 'japanese-hubert-base', 'korean-hubert-base'],
                                                      value='contentvec', label='Backing Embedder Model')
            infer_backing_vocals.change(show_backing_vocals_rvc_inputs, inputs=infer_backing_vocals, outputs=backing_rvc_col)

        with gr.Accordion('Audio mixing options', open=False):
            gr.Markdown('### Volume Change (decibels)')
            with gr.Row():
                main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocals')
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocals')
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music')
                backing_vocals_volume = gr.Slider(-20, 20, value=0, step=1, label='Backing Vocals RVC Volume')

            gr.Markdown('### Reverb Control on AI Vocals')
            with gr.Row():
                reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room size')
                reverb_wet = gr.Slider(0, 1, value=0.2, label='Wetness level')
                reverb_dry = gr.Slider(0, 1, value=0.8, label='Dryness level')
                reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping level')
                reverb_width = gr.Slider(0, 1, value=1.0, label='Width')

            gr.Markdown('### Instrumental')
            with gr.Row():
                inst_pitch_shift = gr.Slider(-12, 12, value=0, step=1, label='Instrumental Pitch Shift')

            gr.Markdown('### Audio Output Format')
            with gr.Row():
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='RVC output file type')
                export_format_final = gr.Dropdown(['mp3', 'wav', 'flac'], value='mp3', label='Final output file type')

        with gr.Accordion('Hugging Face Upload (optional)', open=False):
            gr.Markdown('### Push the generated cover to a Hugging Face model repo')
            hf_upload = gr.Checkbox(value=False, label='Upload result to Hugging Face')
            with gr.Row():
                hf_repo_id = gr.Text(label='HF repo (username/repo)', placeholder='your-name/ai-covers')
                hf_token = gr.Text(label='HF token (write access)', type='password', placeholder='hf_...')

        with gr.Row():
            clear_btn = gr.ClearButton(value='Clear', components=[song_input, rvc_model, keep_files, local_file])
            generate_btn = gr.Button("Generate", variant='primary')
        ai_cover = gr.Audio(label='AI Cover')

        ref_btn.click(update_models_list, None, outputs=rvc_model)
        is_webui = gr.Number(value=1, visible=False)

        generate_btn.click(
            song_cover_pipeline,
            inputs=[
                local_file, rvc_model, pitch, keep_files, is_webui,
                main_gain, backup_gain, inst_gain,
                index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect,
                pitch_all,
                reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width,
                output_format, extra_denoise, steps,
                vocal_model_name, karaoke_model_name, dereverb_model_name, denoise_model_name, deecho_model_name,
                use_deecho, use_denoise, use_dereverb,
                instruments_only, backing_vocals_volume,
                inst_pitch_shift,
                export_format_final,
                split_audio, autotune, embedder_model,
                hf_upload, hf_repo_id, hf_token,
            ],
            outputs=[ai_cover]
        )
        clear_btn.click(
            lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 1.0, 'mp3', 'mp3', None, True, 1, True, False, False, False, 0, 0],
            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                     reverb_dry, reverb_damping, reverb_width, output_format, export_format_final,
                     ai_cover, extra_denoise, steps,
                     use_dereverb, use_denoise, use_deecho, instruments_only,
                     backing_vocals_volume, inst_pitch_shift]
        )
