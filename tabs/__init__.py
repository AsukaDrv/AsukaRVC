"""Tab modules for the AsukaRVC WebUI.

Structure:
    tabs/
        common.py                    -> shared helpers and constants
        inference/inference_tab.py   -> "Generate" tab
        downloader/downloader_tab.py -> "Download model" tab
        upload/upload_tab.py         -> "Upload model" tab

tabs_main.py (project root) re-exports the tab builders so app.py only
needs a single import:
    from tabs_main import inference_tab, downloader_tab, upload_tab
"""
