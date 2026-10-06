import os
import tempfile
from pathlib import Path

import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'


def dl_model(link, model_name, dir_name):
    dir_name.mkdir(parents=True, exist_ok=True)
    target_path = dir_name / model_name
    if target_path.is_file() and target_path.stat().st_size > 0:
        print(f'{model_name} already exists; skipping download.')
        return

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{model_name}.', suffix='.download', dir=dir_name
    )
    os.close(descriptor)
    try:
        with requests.get(f'{link}{model_name}', stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            content_length = response.headers.get('Content-Length')
            expected_size = int(content_length) if content_length else None
            downloaded = 0
            with open(temporary_path, 'wb') as model_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        model_file.write(chunk)
                        downloaded += len(chunk)
            if downloaded == 0:
                raise RuntimeError(f'Downloaded model {model_name} is empty.')
            if expected_size is not None and downloaded != expected_size:
                raise RuntimeError(
                    f'Downloaded model {model_name} is incomplete '
                    f'({downloaded} of {expected_size} bytes).'
                )
        os.replace(temporary_path, target_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


if __name__ == '__main__':
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        print(f'Downloading {model}...')
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_model_names = ['hubert_base.pt', 'rmvpe.pt']
    for model in rvc_model_names:
        print(f'Downloading {model}...')
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)

    print('All models downloaded!')
