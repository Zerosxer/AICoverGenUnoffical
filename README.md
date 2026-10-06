# AICoverGen
An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio file. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

Showcase: https://www.youtube.com/watch?v=2qZuE4WM7CM

Setup Guide: https://www.youtube.com/watch?v=pdlhk4vVHQk

![](images/webui_generate.png?raw=true)

WebUI is under constant development and testing, but you can try it out right now on both local and colab!

## Table of Contents

- [Changelog](#changelog)
- [Setup](#setup)
    - [Install Git and Python](#install-git-and-python)
    - [Install ffmpeg and sox](#install-ffmpeg)
    - [Clone AICoverGen repository](#clone-aicovergen-repository)
    - [Download required models](#download-required-models)
- [Usage with WebUI](#usage-with-webui)
    - [Download RVC models via WebUI](#download-rvc-models-via-webui)
    - [Upload RVC models via WebUI](#upload-rvc-models-via-webui)
    - [Running the pipeline via WebUI](#running-the-pipeline-via-webui)
- [Usage with CLI](#usage-with-cli)
    - [Manual Download of RVC models](#manual-download-of-rvc-models)
    - [Running the pipeline via CLI](#running-the-pipeline-via-cli)
- [Terms of Use](#terms-of-use)


## Changelog

- WebUI for easier conversions and downloading of voice models
- Support for cover generations from a local audio file
- Option to keep intermediate files generated. e.g. Isolated vocals/instrumentals
- Download suggested public voice models from table with search/tag filters
- Support for Pixeldrain download links for voice models
- Implement new rmvpe pitch extraction technique for faster and higher quality vocal conversions
- Volume control for AI main vocals, backup vocals and instrumentals
- Index Rate for Voice conversion
- Reverb Control for AI main vocals
- Local network sharing option for webui
- Extra RVC options - filter_radius, rms_mix_rate, protect
- Local file upload via file browser option
- Upload of locally trained RVC v2 models via WebUI
- Pitch detection method control, e.g. rmvpe/mangio-crepe
- Pitch change for vocals and instrumentals together. Same effect as changing key of song in Karaoke.
- Audio output format option: wav or mp3.

## Update AICoverGen to latest version

Install and pull any new requirements and changes by opening a command line window in the `AICoverGen` directory and running the following commands.

```
git pull
python -m pip install "pip<24.1"
pip install -r requirements.txt
```

For colab users, simply click `Runtime` in the top navigation bar of the colab notebook and `Disconnect and delete runtime` in the dropdown menu. 
Then follow the instructions in the notebook to run the webui.

## Colab notebook

For those without a powerful enough NVIDIA GPU, you may try AICoverGen out using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SociallyIneptWeeb/AICoverGen/blob/main/AICoverGen_colab.ipynb)

For those who face issues with Google Colab notebook disconnecting after a few minutes, here's an alternative that doesn't use the WebUI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ardha27/AICoverGen-NoUI-Colab/blob/main/CoverGen_No_UI.ipynb)

For those who want to run this locally, follow the setup guide below.

## Setup

### Install Git and Python

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer. Install **Python 3.10** and create a virtual environment for this project.

### Install ffmpeg

Follow the instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) to install ffmpeg on your computer.

### Install sox

Follow the instructions [here](https://www.tutorialexample.com/a-step-guide-to-install-sox-sound-exchange-on-windows-10-python-tutorial/) to install sox and add it to your Windows path environment.

### Dependency compatibility

The supported local and Cog environment uses Python 3.10, CUDA 12.1, and the pinned packages in `requirements.txt`. The WebUI uses Gradio 6. RVC's HuBERT checkpoint loader still depends on Fairseq 0.12.2; its legacy OmegaConf metadata requires pip older than 24.1 and Cython 0.29 during installation. The project pins both and installs the compatible pip version first. Inference uses PyTorch 2.5.1 because newer PyTorch releases change checkpoint loading defaults used by Fairseq. ONNX Runtime 1.20 is pinned to match this CUDA/cuDNN runtime; newer ONNX Runtime builds require newer CUDA toolkits. The remaining audio/UI dependencies are upgraded and pinned together. CPU inference is supported but significantly slower; an NVIDIA GPU with a CUDA 12.1-compatible driver is recommended. The MDX separator uses ONNX Runtime's CUDA provider when available and otherwise reports that it is using CPU.

`deemix` and `lib` were removed because the application does not import them. On Windows, install the FFmpeg and SoX executables and add them to `PATH` (the Python `sox` package is only a wrapper). Python 3.10 wheels are used for FAISS and the native audio packages.

Model archives can be downloaded from any public HTTP(S) host. Private/local network destinations are rejected. ZIPs must contain exactly one `.pth` model and at most one `.index`; unrelated files are ignored. Downloads are limited to 4096 MiB and extracted archives to 8192 MiB by default. Set `AICOVERGEN_MAX_MODEL_DOWNLOAD_MB` or `AICOVERGEN_MAX_MODEL_UNPACKED_MB` to a larger positive MiB value for larger custom models. RVC voice checkpoints load with PyTorch's restricted `weights_only` mode; use standard RVC state-dictionary checkpoints. Fairseq's HuBERT loader still requires the bundled upstream checkpoint, so only replace that asset with a trusted file.

### Clone AICoverGen repository

Open a command line window and run these commands to clone this entire repository and install the additional dependencies required.

```
git clone https://github.com/SociallyIneptWeeb/AICoverGen
cd AICoverGen
python -m venv .venv
.venv\Scripts\activate
python -m pip install "pip<24.1"
pip install -r requirements.txt
```

### Download required models

Run the following command to download the required MDXNET vocal separation models and hubert base model.

```
python src/download_models.py
```

Cog builds run this downloader during image setup and validate that the required assets exist before accepting predictions. This makes the image larger, but avoids a first-request download dependency.


## Usage with WebUI

To run the AICoverGen WebUI, run the following command.

```
python src/webui.py
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--share`                                  | Create a public URL. This is useful for running the web UI on Google Colab. |
| `--listen`                                 | Make the web UI reachable from your local network. |
| `--listen-host LISTEN_HOST`                | The hostname that the server will use. |
| `--listen-port LISTEN_PORT`                | The listening port that the server will use. |

Once the following output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the WebUI.

### Download RVC models via WebUI

![](images/webui_dl_model.png?raw=true)

Navigate to the `ดาวน์โหลดโมเดล` tab, open `ดาวน์โหลดจากลิงก์`, and paste the download link to the RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download. You may refer to the examples for how the download link should look like.
The downloaded zip file should contain the .pth model file and an optional .index file.

Once the two input fields are filled in, click `ดาวน์โหลดโมเดล`. The downloaded model can be selected in the `สร้างเพลง` tab after clicking the refresh models button.

### Upload RVC models via WebUI

![](images/webui_upload_model.png?raw=true)

For people who have trained RVC v2 models locally and would like to use them for AI Cover generations.
Navigate to the `เพิ่มโมเดลจากเครื่อง` tab, select the ZIP file and enter a model name. The uploaded model can be selected in the `สร้างเพลง` tab after clicking the refresh models button!


### Running the pipeline via WebUI

![](images/webui_generate.png?raw=true)

- Under `1 · เพลงและเสียงร้อง`, select the voice model and enter a YouTube link or local audio path. You can also upload an audio file from your computer.
- Use `2 · ปรับเสียง AI` and `3 · มิกซ์เสียงและส่งออก` to tune conversion, mix, and output settings. The defaults are intended for general use.

Click `สร้าง AI Cover` to start. Processing time depends on your GPU and audio duration.

## Usage with CLI

### Manual Download of RVC models

Unzip (if needed) and transfer the `.pth` and `.index` files to a new folder in the [rvc_models](rvc_models) directory. Each folder should only contain one `.pth` and one `.index` file.

The directory structure should look something like this:
```
├── rvc_models
│   ├── John
│   │   ├── JohnV2.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── May
│   │   ├── May.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── MODELS.txt
│   └── hubert_base.pt
├── mdxnet_models
├── song_output
└── src
 ```

### Running the pipeline via CLI

To run the AI cover generation pipeline using the command line, run the following command.

```
python src/main.py [-h] -i SONG_INPUT -dir RVC_DIRNAME -p PITCH_CHANGE [-k | --keep-files | --no-keep-files] [-ir INDEX_RATE] [-fr FILTER_RADIUS] [-rms RMS_MIX_RATE] [-palgo PITCH_DETECTION_ALGO] [-hop CREPE_HOP_LENGTH] [-pro PROTECT] [-mv MAIN_VOL] [-bv BACKUP_VOL] [-iv INST_VOL] [-pall PITCH_CHANGE_ALL] [-rsize REVERB_SIZE] [-rwet REVERB_WETNESS] [-rdry REVERB_DRYNESS] [-rdamp REVERB_DAMPING] [-oformat OUTPUT_FORMAT]
```

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `-i SONG_INPUT`                            | Link to a song on YouTube or path to a local audio file. Should be enclosed in double quotes for Windows and single quotes for Unix-like systems. |
| `-dir MODEL_DIR_NAME`                      | Name of folder in [rvc_models](rvc_models) directory containing your `.pth` and `.index` files for a specific voice. |
| `-p PITCH_CHANGE`                          | Change pitch of AI vocals in octaves. Set to 0 for no change. Generally, use 1 for male to female conversions and -1 for vice-versa. |
| `-k`                                       | Optional. Can be added to keep all intermediate audio files generated. e.g. Isolated AI vocals/instrumentals. Leave out to save space. |
| `-ir INDEX_RATE`                           | Optional. Default 0.5. Control how much of the AI's accent to leave in the vocals. 0 <= INDEX_RATE <= 1. |
| `-fr FILTER_RADIUS`                        | Optional. Default 3. If >=3: apply median filtering median filtering to the harvested pitch results. 0 <= FILTER_RADIUS <= 7. |
| `-rms RMS_MIX_RATE`                        | Optional. Default 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1). 0 <= RMS_MIX_RATE <= 1. |
| `-palgo PITCH_DETECTION_ALGO`              | Optional. Default rmvpe. Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals). |
| `-hop CREPE_HOP_LENGTH`                    | Optional. Default 128. Controls how often it checks for pitch changes in milliseconds when using mangio-crepe algo specifically. Lower values leads to longer conversions and higher risk of voice cracks, but better pitch accuracy. |
| `-pro PROTECT`                             | Optional. Default 0.33. Control how much of the original vocals' breath and voiceless consonants to leave in the AI vocals. Set 0.5 to disable. 0 <= PROTECT <= 0.5. |
| `-mv MAIN_VOCALS_VOLUME_CHANGE`            | Optional. Default 0. Control volume of main AI vocals. Use -3 to decrease the volume by 3 decibels, or 3 to increase the volume by 3 decibels. |
| `-bv BACKUP_VOCALS_VOLUME_CHANGE`          | Optional. Default 0. Control volume of backup AI vocals. |
| `-iv INSTRUMENTAL_VOLUME_CHANGE`           | Optional. Default 0. Control volume of the background music/instrumentals. |
| `-pall PITCH_CHANGE_ALL`                   | Optional. Default 0. Change pitch/key of background music, backup vocals and AI vocals in semitones. Reduces sound quality slightly. |
| `-rsize REVERB_SIZE`                       | Optional. Default 0.15. The larger the room, the longer the reverb time. 0 <= REVERB_SIZE <= 1. |
| `-rwet REVERB_WETNESS`                     | Optional. Default 0.2. Level of AI vocals with reverb. 0 <= REVERB_WETNESS <= 1. |
| `-rdry REVERB_DRYNESS`                     | Optional. Default 0.8. Level of AI vocals without reverb. 0 <= REVERB_DRYNESS <= 1. |
| `-rdamp REVERB_DAMPING`                    | Optional. Default 0.7. Absorption of high frequencies in the reverb. 0 <= REVERB_DAMPING <= 1. |
| `-oformat OUTPUT_FORMAT`                   | Optional. Default mp3. wav for best quality and large file size, mp3 for decent quality and small file size. |


## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
