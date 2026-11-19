"""Aggregates every tab builder of the AsukaRVC WebUI.

app.py imports the tab builders from this module only:

    from tabs_main import inference_tab, downloader_tab, upload_tab, training_tab

Each builder opens its own gr.Tab context and must be called inside a
gr.Blocks context.
"""

from ui.tabs.downloader.downloader_tab import downloader_tab
from ui.tabs.inference.inference_tab import inference_tab
from ui.tabs.train.train import training_tab
from ui.tabs.upload.upload_tab import upload_tab

__all__ = ['inference_tab', 'downloader_tab', 'upload_tab', 'training_tab']
