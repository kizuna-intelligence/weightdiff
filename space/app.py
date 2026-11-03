import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT / 'out' / 'gradio'))
os.environ.setdefault('WEIGHTDIFF_OUTPUT_DIR', str(ROOT / 'out'))
sys.path.insert(0, str(ROOT / 'src'))

from weightdiff.application.app import DASHBOARD_CSS, demo
import gradio as gr

if __name__ == '__main__':
    demo.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'),
                server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')), share=False,
                theme=gr.themes.Default(font=['Arial', 'sans-serif'], font_mono=['monospace']),
                css=DASHBOARD_CSS,
                footer_links=[])
