import json
import os
import tempfile
from argparse import ArgumentParser

import gradio as gr

from main import song_cover_pipeline
from model_utils import (
    download_model_archive,
    install_model_archive,
    validate_model_name,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

# ==============================================================================
# Modern & Clean UI CSS (แก้ไขบั๊กสีจม คอนทราสต์ต่ำ และ Layout เบี้ยวทั้งหมด)
# ==============================================================================
CUSTOM_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #090b12;
    --surface: #111522;
    --surface-soft: #151a28;
    --surface-hover: #1a2030;
    --border: #292f42;
    --border-soft: #22283a;
    --text: #edf0fb;
    --muted: #929bb2;
    --accent: #9b8cff;
    --accent-2: #6d5dfc;
}

* {
    font-family: 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg) !important;
}

body,
.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 24px clamp(12px, 2.4vw, 34px) 42px !important;
}

footer { display: none !important; }

/* -------------------------------------------------------------------------- */
/* Header                                                                    */
/* -------------------------------------------------------------------------- */
#app-header {
    padding: 30px 32px;
    margin: 0 0 22px;
    border: 1px solid #39345f;
    border-radius: 22px;
    background:
        radial-gradient(ellipse at 85% 0%, rgba(125,103,255,.28), transparent 38%),
        linear-gradient(120deg, #17172a, #111522 70%);
    overflow: hidden;
}

#app-header h1 {
    color: #fff;
    font-size: clamp(27px, 3vw, 38px);
    line-height: 1.2;
    margin: 0 0 9px;
    font-weight: 700;
    letter-spacing: -.8px;
}

#app-header p {
    color: #b9c0d8;
    margin: 0;
    font-size: 14px;
    line-height: 1.7;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #b5aaff;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 10px;
}

.header-chip {
    display: inline-block;
    margin-top: 18px;
    padding: 6px 11px;
    border-radius: 999px;
    background: #272440;
    border: 1px solid #49416e;
    color: #d7d0ff;
    font-size: 11px;
}

/* -------------------------------------------------------------------------- */
/* Tabs                                                                       */
/* -------------------------------------------------------------------------- */
.tabs {
    border-color: var(--border) !important;
}

.tab-nav {
    gap: 7px !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 18px !important;
    overflow-x: auto !important;
    scrollbar-width: none;
}

.tab-nav::-webkit-scrollbar { display: none; }

.tab-nav button {
    flex: 0 0 auto !important;
    color: var(--muted) !important;
    border: 1px solid transparent !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 500 !important;
    padding: 12px 17px !important;
    white-space: nowrap !important;
}

.tab-nav button.selected {
    color: #e5e0ff !important;
    border-color: #39345f !important;
    border-bottom: 2px solid var(--accent) !important;
    background: #19182a !important;
}

/* -------------------------------------------------------------------------- */
/* IMPORTANT: Only our own cards get card backgrounds/borders.               */
/* Do NOT style .block / .gr-box / .gr-panel globally: those are Gradio      */
/* wrappers and caused the old "box inside box" problem.                     */
/* -------------------------------------------------------------------------- */
.ui-card {
    width: 100% !important;
    min-width: 0 !important;
    padding: 18px !important;
    margin-bottom: 14px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 28px rgba(0,0,0,.12) !important;
}

.ui-card > .block,
.ui-card > .gr-box,
.ui-card > .gr-panel,
.ui-card .block,
.ui-card .gr-box,
.ui-card .gr-panel {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.ui-card .form,
.ui-card .form > .form,
.ui-card .panel {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.studio-layout,
.studio-layout > .form,
.studio-layout > .block {
    gap: 14px !important;
}

.studio-column {
    min-width: 0 !important;
}

.section-heading {
    color: #f0edff;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.4;
    margin: 0 0 4px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 14px;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 25px;
    height: 25px;
    margin-right: 8px;
    border-radius: 8px;
    background: #292447;
    border: 1px solid #49416e;
    color: #c7bcff;
    font-size: 12px;
}

/* -------------------------------------------------------------------------- */
/* Form controls                                                              */
/* -------------------------------------------------------------------------- */
label,
.label-wrap span,
.wrap .label-wrap span {
    color: #dce1f2 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

input,
textarea,
select,
.gr-input,
.gr-dropdown,
.wrap input,
.wrap textarea {
    background: #0e121d !important;
    color: #eef0fa !important;
    border: 1px solid #30374b !important;
    border-radius: 10px !important;
}

input::placeholder,
textarea::placeholder {
    color: #69738b !important;
}

input:focus,
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(155,140,255,.15) !important;
}

button {
    border-radius: 10px !important;
    transition: transform .16s ease, filter .16s ease, border-color .16s ease !important;
}

button:hover { filter: brightness(1.08); }

button.primary,
button.primary-action {
    background: linear-gradient(135deg, #8d79ff, #6655e8) !important;
    color: #fff !important;
    border: 1px solid #9c8dff !important;
    font-weight: 700 !important;
    box-shadow: 0 7px 22px rgba(109,93,252,.20) !important;
}

button.primary:hover,
button.primary-action:hover {
    transform: translateY(-1px);
}

button.secondary-btn {
    background: #1b2030 !important;
    color: #cbd2e6 !important;
    border: 1px solid #353d53 !important;
}

#generate-btn {
    width: 100% !important;
    min-height: 58px !important;
    font-size: 16px !important;
}

#clear-btn {
    width: 100% !important;
    margin-top: 9px;
}

/* -------------------------------------------------------------------------- */
/* Pitch: stable 2-column desktop layout, single-column mobile layout        */
/* -------------------------------------------------------------------------- */
.pitch-row {
    width: 100% !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 10px !important;
}

.pitch-row > .form,
.pitch-row > .block {
    min-width: 0 !important;
    flex: 1 1 0 !important;
}

.pitch-row .wrap,
.pitch-row .container {
    min-width: 0 !important;
}

.pitch-row input[type="range"] {
    min-width: 0 !important;
}

/* -------------------------------------------------------------------------- */
/* Advanced settings                                                         */
/* -------------------------------------------------------------------------- */
.gradio-container .accordion {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
    overflow: hidden !important;
}

.gradio-container .accordion > .label-wrap {
    padding: 15px 18px !important;
}

.gradio-container .accordion .tabitem {
    padding: 14px 2px 2px !important;
}

.gradio-container .info,
.gradio-container .wrap .info {
    color: #8792ab !important;
    font-size: 11px !important;
    line-height: 1.5 !important;
}

.gradio-container .prose,
.gradio-container .markdown {
    color: var(--text) !important;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: #f1efff !important;
}

.gradio-container hr {
    border-color: var(--border) !important;
}

.gradio-container input[type=range] {
    accent-color: var(--accent) !important;
}

/* -------------------------------------------------------------------------- */
/* Output                                                                     */
/* -------------------------------------------------------------------------- */
#output-card {
    border-color: #39345f !important;
    background: linear-gradient(145deg, #17182a, #121622) !important;
}

#output-card audio {
    width: 100% !important;
}

/* -------------------------------------------------------------------------- */
/* Model hub                                                                  */
/* -------------------------------------------------------------------------- */
.model-hub-title {
    margin: 4px 0 3px !important;
    color: #f1efff !important;
    font-size: clamp(24px, 3vw, 32px) !important;
}

.model-hub-description {
    color: var(--muted) !important;
    margin-bottom: 18px !important;
}

.model-form-row {
    width: 100% !important;
    gap: 10px !important;
}

.model-form-row > .form,
.model-form-row > .block {
    min-width: 0 !important;
}

.gradio-container table {
    background: #101420 !important;
    color: var(--text) !important;
}

.gradio-container th {
    background: #1b2030 !important;
    color: #e4e7f5 !important;
}

.gradio-container td {
    border-color: var(--border) !important;
}

.gradio-container .upload-container {
    border-color: #424968 !important;
    background: #101420 !important;
    border-radius: 12px !important;
}

/* -------------------------------------------------------------------------- */
/* Responsive                                                                 */
/* -------------------------------------------------------------------------- */
@media (max-width: 900px) {
    .gradio-container {
        padding: 16px 14px 30px !important;
    }

    #app-header {
        padding: 24px 22px;
        border-radius: 18px;
        margin-bottom: 16px;
    }

    .studio-layout {
        flex-direction: column !important;
    }

    .studio-layout > .form,
    .studio-layout > .block,
    .studio-column {
        width: 100% !important;
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }

    .ui-card {
        padding: 15px !important;
        border-radius: 14px !important;
    }
}

@media (max-width: 700px) {
    .gradio-container {
        padding: 10px 10px 24px !important;
    }

    #app-header h1 {
        font-size: 27px !important;
        letter-spacing: -.5px;
    }

    #app-header p {
        font-size: 13px;
        line-height: 1.65;
    }

    .header-chip {
        font-size: 10px;
        margin-top: 14px;
    }

    .tab-nav {
        gap: 3px !important;
        margin-bottom: 13px !important;
    }

    .tab-nav button {
        padding: 9px 12px !important;
        font-size: 12px !important;
    }

    .ui-card {
        padding: 13px !important;
        margin-bottom: 10px !important;
    }

    .pitch-row {
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        gap: 8px !important;
    }

    .pitch-row > .form,
    .pitch-row > .block {
        width: 100% !important;
        flex: 1 1 auto !important;
    }

    .model-form-row {
        flex-direction: column !important;
    }

    .model-form-row > .form,
    .model-form-row > .block {
        width: 100% !important;
        flex: 1 1 auto !important;
    }

    .section-heading {
        font-size: 14px;
    }

    .section-subtitle {
        font-size: 11px;
    }

    .gradio-container .accordion > .label-wrap {
        padding: 13px 14px !important;
    }
}

@media (max-width: 430px) {
    #app-header {
        padding: 20px 17px;
    }

    #app-header h1 {
        font-size: 24px !important;
    }

    .eyebrow {
        font-size: 9px;
        letter-spacing: 1.5px;
    }

    .ui-card {
        padding: 11px !important;
    }

    #generate-btn {
        min-height: 54px !important;
        font-size: 14px !important;
    }
}
"""



def get_current_models(models_dir):
    if not os.path.isdir(models_dir):
        return []
    models_list = [
        item for item in os.listdir(models_dir)
        if not item.startswith('.')
        and os.path.isdir(os.path.join(models_dir, item))
        and any(
            filename.lower().endswith('.pth')
            for filename in os.listdir(os.path.join(models_dir, item))
        )
    ]
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)


def load_public_models():
    models_table = []
    installed_models = get_current_models(rvc_models_dir)
    for model in public_models['voice_models']:
        if model['name'] not in installed_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)

    tags = list(public_models['tags'].keys())
    return gr.update(value=models_table), gr.update(choices=tags)


def get_uploaded_file_path(uploaded_file):
    if isinstance(uploaded_file, (str, os.PathLike)):
        file_path = os.fspath(uploaded_file)
    else:
        file_path = getattr(uploaded_file, 'path', None) or getattr(uploaded_file, 'name', None)
        if isinstance(file_path, os.PathLike):
            file_path = os.fspath(file_path)
    if not isinstance(file_path, str) or not file_path:
        raise gr.Error('ไม่สามารถอ่านพาธไฟล์ที่อัปโหลดได้')
    return file_path


def download_online_model(url, dir_name, progress=gr.Progress()):
    archive_path = None
    try:
        dir_name = validate_model_name(dir_name)
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        os.makedirs(rvc_models_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix='.model-download-', suffix='.zip', dir=rvc_models_dir, delete=False
        ) as archive_file:
            archive_path = archive_file.name
        download_model_archive(
            url,
            archive_path,
            progress=lambda downloaded, limit: progress(
                min(downloaded / limit, 0.95), desc='[~] Downloading model archive...'
            ),
        )
        progress(0.96, desc='[~] Validating and installing model...')
        install_model_archive(archive_path, rvc_models_dir, dir_name)
        return f'[+] {dir_name} Model successfully downloaded!'

    except Exception as e:
        raise gr.Error(f'ดาวน์โหลดหรือติดตั้งโมเดลไม่สำเร็จ: {e}') from e
    finally:
        if archive_path and os.path.exists(archive_path):
            os.remove(archive_path)


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        dir_name = validate_model_name(dir_name)
        zip_path = get_uploaded_file_path(zip_path)
        if not zip_path.lower().endswith('.zip'):
            raise gr.Error('กรุณาเลือกไฟล์โมเดล .zip')
        progress(0.5, desc='[~] Extracting zip...')
        install_model_archive(zip_path, rvc_models_dir, dir_name)
        return f'[+] {dir_name} Model successfully uploaded!'

    except Exception as e:
        raise gr.Error(f'เพิ่มโมเดลไม่สำเร็จ: {e}') from e


def filter_models(tags, query):
    models_table = []
    tags = tags or []
    query = (query or '').strip().lower()
    installed_models = get_current_models(rvc_models_dir)

    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            if model['name'] not in installed_models:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if model['name'] not in installed_models and query in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if model['name'] not in installed_models and all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if model['name'] not in installed_models and query in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    return gr.update(value=models_table)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    selected_model = pub_models.iloc[event.index[0]]
    return gr.update(value=selected_model.iloc[3]), gr.update(value=selected_model.iloc[0])


def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)


def process_file_upload(file):
    file_path = get_uploaded_file_path(file)
    return file_path, gr.update(value=file_path)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate an AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    
    public_models = {'voice_models': [], 'tags': {}}
    pub_json_path = os.path.join(rvc_models_dir, 'public_models.json')
    if os.path.exists(pub_json_path):
        with open(pub_json_path, encoding='utf8') as infile:
            public_models = json.load(infile)

    # Dark studio theme with high-contrast controls
    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=["Prompt", "sans-serif"],
    ).set(
        body_background_fill="#0b0d14",
        body_text_color="#edf0fb",
        block_background_fill="#121622",
        block_border_color="#282f43",
        input_background_fill="#0e121d",
        input_border_color="#30374b",
        button_primary_background_fill="#7564f5",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title='AICoverGen Studio', css=CUSTOM_CSS, theme=custom_theme) as app:
        gr.HTML(
            '<header id="app-header">'
            '<div class="eyebrow">AI AUDIO WORKSPACE · RVC V2</div>'
            '<h1>AICoverGen <span style="color:#a99aff">Studio</span></h1>'
            '<p>เปลี่ยนเสียงร้องให้เป็นสไตล์ที่คุณต้องการ ปรับแต่งมิกซ์ และสร้าง AI Cover ในพื้นที่ทำงานเดียว</p>'
            '<span class="header-chip">● LOCAL WORKSPACE &nbsp; / &nbsp; VOICE CONVERSION</span>'
            '</header>'
        )

        with gr.Tabs():
            with gr.Tab('♫  Studio'):
                with gr.Row(equal_height=False, elem_classes=['studio-layout']):
                    with gr.Column(scale=6, min_width=320, elem_classes=['studio-column']):
                        with gr.Group(elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading"><span class="step-number">01</span>Voice & Source</div><div class="section-subtitle">เลือกโมเดลเสียงและกำหนดแหล่งที่มาของเพลง</div>')
                            with gr.Row(equal_height=True):
                                rvc_model = gr.Dropdown(voice_models, label='VOICE MODEL', info='เลือกโมเดลเสียง RVC ที่ติดตั้งไว้', scale=5)
                                ref_btn = gr.Button('↻ Refresh', elem_classes=['secondary-btn'], scale=1, min_width=95)
                            with gr.Column( elem_id='source-url') as yt_link_col:
                                song_input = gr.Textbox(label='SONG SOURCE', placeholder='วาง YouTube URL หรือพาธไฟล์ .mp3 / .wav', lines=1)
                                show_file_upload_button = gr.Button('＋ ใช้ไฟล์เสียงจากเครื่องแทน', elem_classes=['secondary-btn'])
                            with gr.Column(visible=False) as file_upload_col:
                                local_file = gr.File(label='SELECTED AUDIO FILE', file_types=['audio'])
                                song_input_file = gr.UploadButton('↑ อัปโหลดไฟล์เสียง', file_types=['audio'], variant='primary', elem_classes=['primary-action'])
                                show_yt_link_button = gr.Button('← กลับไปใช้ URL / Path', elem_classes=['secondary-btn'])
                                song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                        with gr.Group(elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading"><span class="step-number">02</span>Pitch Control</div><div class="section-subtitle">ตั้งค่าระดับเสียงร้องและคีย์เพลงก่อนประมวลผล</div>')
                            with gr.Row(elem_classes=['pitch-row']):
                                pitch = gr.Slider(-3, 3, value=0, step=1, label='VOCAL PITCH · OCTAVES', info='+1 สูงขึ้นหนึ่ง octave · -1 ต่ำลงหนึ่ง octave')
                                pitch_all = gr.Slider(-12, 12, value=0, step=1, label='SONG KEY · SEMITONES', info='เปลี่ยนคีย์เสียงร้องและดนตรีพร้อมกัน')

                        with gr.Accordion('Advanced audio settings', open=False):
                            gr.Markdown('<div class="section-subtitle">ตั้งค่าเพิ่มเติมสำหรับผู้ใช้ที่ต้องการควบคุมผลลัพธ์ละเอียดขึ้น</div>')
                            with gr.Tabs():
                                with gr.Tab('Voice tuning'):
                                    with gr.Row():
                                        index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info='ความใกล้เคียงกับลักษณะเสียงโมเดล')
                                        filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius', info='ช่วยลดเสียงเพี้ยน')
                                    with gr.Row():
                                        rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate', info='สัดส่วนความดังตามต้นฉบับ')
                                        protect = gr.Slider(0, 0.5, value=0.33, label='Protect Breath', info='ช่วยรักษารายละเอียดเสียงเบา')
                                    with gr.Row():
                                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Pitch detection method')
                                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe Hop Length')
                                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                                with gr.Tab('Audio mixer'):
                                    with gr.Row():
                                        main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main vocal · dB')
                                        backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backing vocal · dB')
                                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Instrumental · dB')
                                with gr.Tab('Reverb'):
                                    with gr.Row():
                                        reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room size')
                                        reverb_wet = gr.Slider(0, 1, value=0.2, label='Wet level')
                                    with gr.Row():
                                        reverb_dry = gr.Slider(0, 1, value=0.8, label='Dry level')
                                        reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping')
                                with gr.Tab('Export'):
                                    output_format = gr.Radio(['mp3', 'wav'], value='mp3', label='Output format')
                                    keep_files = gr.Checkbox(label='เก็บไฟล์แยกเสียงร้องและดนตรีไว้ด้วย')

                    with gr.Column(scale=5, min_width=300, elem_classes=['studio-column']):
                        with gr.Group(elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading">Generation</div><div class="section-subtitle">ตรวจสอบการตั้งค่าแล้วเริ่มสร้างผลงานของคุณ</div>')
                            gr.Markdown('**พร้อมสร้าง AI Cover หรือยัง?**\n\nเลือกโมเดลเสียงและใส่แหล่งเพลงทางด้านซ้าย จากนั้นกดปุ่มด้านล่างเพื่อเริ่มประมวลผล')
                            generate_btn = gr.Button('▶  Generate AI Cover', variant='primary', size='lg', elem_classes=['primary-action'], elem_id='generate-btn')
                            clear_btn = gr.ClearButton(value='Reset inputs', components=[song_input, rvc_model, keep_files, local_file], elem_classes=['secondary-btn'], elem_id='clear-btn')
                        with gr.Group(elem_id='output-card', elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading">Your output</div><div class="section-subtitle">ผลงานที่ประมวลผลเสร็จจะแสดงที่นี่</div>')
                            ai_cover = gr.Audio(label='AI COVER PLAYER', type='filepath')
                            gr.Markdown('<div class="section-subtitle">เคล็ดลับ: หากเสียงยังไม่ตรงใจ ลองปรับ Pitch หรือค่าภายใน Advanced audio settings แล้วสร้างใหม่</div>')

            with gr.Tab('▦  Model Hub'):
                gr.Markdown('## Voice Model Library\nจัดการโมเดลเสียงได้ 3 วิธี เลือกดาวน์โหลดจาก URL ค้นหาคลังสาธารณะ หรืออัปโหลดไฟล์ ZIP จากเครื่อง')
                with gr.Tabs():
                    with gr.Tab('Direct URL'):
                        with gr.Group(elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading">Install from URL</div><div class="section-subtitle">ใส่ลิงก์ ZIP ที่เข้าถึงได้โดยตรงและตั้งชื่อโมเดล</div>')
                            model_zip_link = gr.Textbox(label='MODEL ZIP URL', placeholder='https://example.com/model.zip')
                            model_name = gr.Textbox(label='MODEL NAME', placeholder='เช่น Lisa, Gura')
                            download_btn = gr.Button('↓  Download & Install', variant='primary', elem_classes=['primary-action'])
                            dl_output_message = gr.Textbox(label='INSTALL STATUS', interactive=False)
                            download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)
                        gr.Markdown('### Example model links')
                        gr.Examples(
                            [['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'], ['https://pixeldrain.com/u/3tJmABXA', 'Gura'], ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']],
                            [model_zip_link, model_name], [], download_online_model,
                        )
                    with gr.Tab('Public library'):
                        with gr.Group(elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading">Browse public models</div><div class="section-subtitle">ค้นหาโมเดลที่ยังไม่ได้ติดตั้ง แล้วเลือกแถวเพื่อเตรียมติดตั้ง</div>')
                            with gr.Row(elem_classes=['model-form-row']):
                                pub_zip_link = gr.Textbox(label='SELECTED MODEL URL', interactive=False)
                                pub_model_name = gr.Textbox(label='MODEL NAME', interactive=False)
                            download_pub_btn = gr.Button('↓  Install selected model', variant='primary', elem_classes=['primary-action'])
                            pub_dl_output_message = gr.Textbox(label='INSTALL STATUS', interactive=False)
                        with gr.Row(elem_classes=['model-form-row']):
                            search_query = gr.Textbox(label='SEARCH MODELS', placeholder='ค้นหาจากชื่อหรือคำอธิบาย...')
                            filter_tags = gr.CheckboxGroup(value=[], label='FILTER TAGS', choices=[])
                        load_public_models_button = gr.Button('↻ Load / Refresh model list', elem_classes=['secondary-btn'])
                        public_models_table = gr.DataFrame(value=[], headers=['ชื่อโมเดล', 'รายละเอียด', 'เครดิต', 'URL', 'แท็ก'], label='AVAILABLE MODELS · คลิกแถวเพื่อเลือก', interactive=False)
                        public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                        load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                        search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)
                    with gr.Tab('Upload ZIP'):
                        with gr.Group(elem_classes=['ui-card']):
                            gr.Markdown('<div class="section-heading">Install a local model</div><div class="section-subtitle">เลือกไฟล์ ZIP และกำหนดชื่อโมเดลก่อนติดตั้ง</div>')
                            zip_file = gr.File(label='MODEL ARCHIVE · ZIP', file_types=['.zip'])
                            local_model_name = gr.Textbox(label='MODEL NAME', placeholder='ตั้งชื่อโมเดลที่จำง่าย')
                            model_upload_button = gr.Button('↑  Upload & Install', variant='primary', elem_classes=['primary-action'])
                            local_upload_output_message = gr.Textbox(label='INSTALL STATUS', interactive=False)
                            model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

        # Event Handlers & Binding
        ref_btn.click(update_models_list, None, outputs=rvc_model)
        show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
        show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

        is_webui = gr.Number(value=1, visible=False)
        generate_btn.click(
            song_cover_pipeline,
            inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                    inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                    protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                    output_format],
            outputs=[ai_cover]
        )
        clear_btn.click(
            lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                     reverb_dry, reverb_damping, output_format, ai_cover]
        )

    app.queue()
    app.launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )