import subprocess
import sys
import unittest
from pathlib import Path


class MainLazyImportTests(unittest.TestCase):
    def test_importing_main_does_not_load_audio_or_model_stack(self):
        source_dir = Path(__file__).resolve().parents[1] / 'src'
        probe = f"""
import sys
import types

gradio = types.ModuleType('gradio')
gradio.Progress = type('Progress', (), {{'__init__': lambda self: None}})
sys.modules['gradio'] = gradio
sys.path.insert(0, {str(source_dir)!r})

import main

heavy_modules = (
    'librosa', 'onnxruntime', 'torch', 'fairseq', 'mdx', 'rvc',
    'yt_dlp', 'pedalboard', 'pydub',
)
loaded = [name for name in heavy_modules if name in sys.modules]
if loaded:
    raise AssertionError(f'Unexpected eager imports: {{loaded}}')
"""
        result = subprocess.run(
            [sys.executable, '-c', probe],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
