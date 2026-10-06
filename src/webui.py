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

CUSTOM_CSS = """
/* Font & Global Setup */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    background-color: #F8F9FD !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 24px !important;
    animation: fadeInUp 0.6s ease-out;
}

/* Keyframes Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
    0% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.08); }
    100% { opacity: 0.3; transform: scale(1); }
}

@keyframes shine {
    0% { left: -100%; }
    100% { left: 200%; }
}

/* Fantasy Purple Header Banner */
.app-header {
    padding: 28px 32px;
    margin-bottom: 24px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1E1035 0%, #3B1C66 45%, #6B21A8 80%, #9333EA 100%);
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 36px -8px rgba(107, 33, 168, 0.3);
    border: 1px solid rgba(192, 132, 252, 0.35);
}

.app-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.5) 0%, rgba(0,0,0,0) 70%);
    pointer-events: none;
    animation: glowPulse 4s infinite ease-in-out;
}

.app-header h1 {
    margin: 0 0 6px;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 12px;
    position: relative;
    z-index: 1;
}

.app-header p {
    margin: 0;
    font-size: 14px;
    color: #E9D5FF !important;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* Tab Navigation with Animated Indicator */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    border-bottom: 2px solid #E9D5FF !important;
    gap: 8px;
    margin-bottom: 20px !important;
}

.tab-nav button {
    font-weight: 700 !important;
    font-size: 15px !important;
    color: #6B7280 !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 12px 24px !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button:hover {
    color: #7C3AED !important;
    background: rgba(233, 213, 255, 0.35) !important;
}

.tab-nav button.selected {
    color: #7C3AED !important;
    border-bottom: 3px solid #7C3AED !important;
    background: #FFFFFF !important;
    box-shadow: 0 -4px 16px rgba(124, 58, 237, 0.12) !important;
}

/* Cards & Panels with Hover Lift */
.block, .form, .panel, accordion {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1.5px solid #E9D5FF !important;
    box-shadow: 0 4px 16px -2px rgba(124, 58, 237, 0.04) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.block:hover {
    border-color: #C084FC !important;
    box-shadow: 0 8px 28px -4px rgba(124, 58, 237, 0.15) !important;
    transform: translateY(-2px);
}

/* Inputs, Textboxes, Dropdowns */
input[type="text"], textarea, select, .gr-box, .gr-input {
    border-radius: 10px !important;
    border: 1.5px solid #E9D5FF !important;
    background-color: #FAFAFE !important;
    color: #1F2937 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

input[type="text"]:focus, textarea:focus, select:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.18) !important;
    background-color: #FFFFFF !important;
}

/* Primary Button with Shine Effect & Glow */
button.primary-action, .primary-btn, button.primary {
    background: linear-gradient(135deg, #7C3AED 0%, #9333EA 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35) !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button.primary-action::after, .primary-btn::after, button.primary::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: none;
}

button.primary-action:hover::after, .primary-btn:hover::after, button.primary:hover::after {
    animation: shine 0.85s ease-in-out;
}

button.primary-action:hover, .primary-btn:hover, button.primary:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 25px rgba(124, 58, 237, 0.5) !important;
    background: linear-gradient(135deg, #6D28D9 0%, #7E22CE 100%) !important;
}

button.primary-action:active, .primary-btn:active, button.primary:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Secondary Button Styling */
button.secondary-btn, .clear-btn {
    border: 1.5px solid #D8B4FE !important;
    color: #7C3AED !important;
    background: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button.secondary-btn:hover, .clear-btn:hover {
    background: #F3E8FF !important;
    border-color: #A855F7 !important;
    transform: translateY(-1px);
}

/* Highlighted Result Box */
.result-box {
    background: linear-gradient(180deg, #FFFFFF 0%, #FBF8FF 100%) !important;
    border: 2px solid #C084FC !important;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.12) !important;
    transition: all 0.3s ease !important;
}

footer {
    visibility: hidden;
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

    # no filter
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            if model['name'] not in installed_models:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    # filter based on tags and query
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if model['name'] not in installed_models and query in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    # filter based on only tags
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if model['name'] not in installed_models and all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])

    # filter based on only query
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
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    with gr.Blocks(title='AICoverGen Unofficial Edition | AI Cover Studio', css=CUSTOM_CSS) as app:
        gr.Markdown(
            '<div class="app-header">'
            '<h1>✨ AICoverGen Unofficial Edition</h1>'
            '<p>เนรมิตเพลงคัฟเวอร์ด้วย AI เสียงที่คุณต้องการ จัดแต่งง่ายและจบในที่เดียว</p>'
            '</div>'
        )

        with gr.Tabs():
            # MAIN STUDIO TAB
            with gr.Tab('🎵 สตูดิโอสร้างเพลง (Main Studio)'):
                with gr.Row():
                    # LEFT SIDE: INPUTS & SETTINGS
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.Markdown('### 1️⃣ เลือกโมเดลเสียง & แหล่งที่มาเพลง')
                            with gr.Row():
                                rvc_model = gr.Dropdown(
                                    voice_models, label='🎭 เลือกโมเดลเสียง AI (Voice Model)',
                                    info='กดรีเฟรชเมื่อมีการเพิ่มโมเดลใหม่', scale=4
                                )
                                ref_btn = gr.Button('🔄', elem_classes=['secondary-btn'], scale=1)

                            with gr.Column() as yt_link_col:
                                song_input = gr.Textbox(
                                    label='🔗 ลิงก์ YouTube หรือ Path ไฟล์ในเครื่อง',
                                    placeholder='วางลิงก์ YouTube หรือระบุ Path ไฟล์ .wav / .mp3'
                                )
                                show_file_upload_button = gr.Button('📁 เปลี่ยนเป็นอัปโหลดไฟล์ตรง', elem_classes=['secondary-btn'])

                            with gr.Column(visible=False) as file_upload_col:
                                local_file = gr.File(label='📄 ไฟล์เสียงที่อัปโหลด')
                                song_input_file = gr.UploadButton(
                                    '📤 เลือกไฟล์เสียงจากอุปกรณ์', file_types=['audio'], variant='primary', elem_classes=['primary-btn']
                                )
                                show_yt_link_button = gr.Button('🔗 กลับไปใช้ลิงก์/Path', elem_classes=['secondary-btn'])
                                song_input_file.upload(
                                    process_file_upload, inputs=[song_input_file],
                                    outputs=[local_file, song_input]
                                )

                        with gr.Group():
                            gr.Markdown('### 2️⃣ ปรับระดับคีย์เสียง (Pitch Control)')
                            with gr.Row():
                                pitch = gr.Slider(
                                    -3, 3, value=0, step=1, label='🎼 คีย์เสียงร้อง AI (Octaves)',
                                    info='+1 เสียงหญิง | -1 เสียงชาย'
                                )
                                pitch_all = gr.Slider(
                                    -12, 12, value=0, step=1, label='🎶 คีย์เพลงทั้งหมด (Semitones)',
                                    info='ปรับพร้อมกันทั้งดนตรีและเสียงร้อง'
                                )

                        # Accordion Settings Categorization
                        with gr.Accordion('⚙️ ปรับแต่งเสียง AI แบบละเอียด (Voice Tuning)', open=False):
                            with gr.Row():
                                index_rate = gr.Slider(0, 1, value=0.5, label='🎯 Index Rate (ความคล้ายโมเดล)')
                                filter_radius = gr.Slider(0, 7, value=3, step=1, label='🧹 Filter Radius (ลดเสียงพร่า)')
                            with gr.Row():
                                rms_mix_rate = gr.Slider(0, 1, value=0.25, label='🔊 RMS Mix Rate (คงระดับเสียงเดิม)')
                                protect = gr.Slider(0, 0.5, value=0.33, label='🛡️ Protect Breath (รักษาลมหายใจ)')
                            with gr.Row():
                                f0_method = gr.Dropdown(
                                    ['rmvpe', 'mangio-crepe'], value='rmvpe', label='🔍 F0 Method (ตรวจจับระดับเสียง)'
                                )
                                crepe_hop_length = gr.Slider(
                                    32, 320, value=128, step=1, visible=False, label='⏱️ Crepe Hop Length'
                                )
                                f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)

                        with gr.Accordion('🎚️ มิกเซอร์ระดับเสียง (Audio Mixer)', open=False):
                            with gr.Row():
                                main_gain = gr.Slider(-20, 20, value=0, step=1, label='🎤 เสียงร้องหลัก AI (dB)')
                                backup_gain = gr.Slider(-20, 20, value=0, step=1, label='👥 เสียงร้องประสาน (dB)')
                                inst_gain = gr.Slider(-20, 20, value=0, step=1, label='🎸 เสียงดนตรี (dB)')

                        with gr.Accordion('🔮 เอฟเฟกต์มิติเสียง (Reverb Studio)', open=False):
                            with gr.Row():
                                reverb_rm_size = gr.Slider(0, 1, value=0.15, label='🏛️ Room Size')
                                reverb_wet = gr.Slider(0, 1, value=0.2, label='💧 Wet (เสียงก้อง)')
                            with gr.Row():
                                reverb_dry = gr.Slider(0, 1, value=0.8, label='🎙 Dry (เสียงตรง)')
                                reverb_damping = gr.Slider(0, 1, value=0.7, label='🔇 Damping (ซับความถี่สูง)')

                        with gr.Accordion('🎼 ตั้งค่าการส่งออก (Export Option)', open=False):
                            output_format = gr.Radio(
                                ['mp3', 'wav'], value='mp3', label='รูปแบบไฟล์ผลลัพธ์ (Format)'
                            )
                            keep_files = gr.Checkbox(
                                label='💾 บันทึกไฟล์เสียงแยก (ร้อง/ดนตรี) ลงในโฟลเดอร์ song_output'
                            )

                    # RIGHT SIDE: OUTPUT & ACTION
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=['result-box']):
                            gr.Markdown('### 🎧 เครื่องมือประมวลผล & ผลลัพธ์')
                            generate_btn = gr.Button(
                                '⚡ เริ่มสร้าง AI Cover', variant='primary', elem_classes=['primary-action']
                            )
                            clear_btn = gr.ClearButton(
                                value='🧹 ล้างค่าทั้งหมด', components=[song_input, rvc_model, keep_files, local_file],
                                elem_classes=['clear-btn']
                            )
                            ai_cover = gr.Audio(label='🎵 เพลง AI Cover ที่เสร็จสมบูรณ์', buttons=['download'])

            # MODEL MANAGEMENT HUB TAB
            with gr.Tab('📦 จัดการโมเดลเสียง (Model Hub)'):
                with gr.Tabs():
                    with gr.Tab('🔗 ดาวน์โหลดผ่าน Direct URL'):
                        gr.Markdown('#### 🌐 โหลดโมเดลผ่านลิงก์ ZIP โดยตรง')
                        with gr.Row():
                            model_zip_link = gr.Textbox(label='🔗 ลิงก์ไฟล์ ZIP โมเดล')
                            model_name = gr.Textbox(label='🏷️ ตั้งชื่อโมเดลใหม่')
                        download_btn = gr.Button('⬇️ ดาวน์โหลดโมเดล', variant='primary', elem_classes=['primary-action'])
                        dl_output_message = gr.Textbox(label='📌 สถานะ', interactive=False)
                        download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                        gr.Markdown('##### 💡 ตัวอย่างลิงก์โมเดลสำเร็จรูป')
                        gr.Examples(
                            [
                                ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                                ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                                ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']
                            ],
                            [model_zip_link, model_name],
                            [],
                            download_online_model,
                        )

                    with gr.Tab('🌐 ค้นหาจากคลังสาธารณะ (Public Hub)'):
                        gr.Markdown('#### 🔍 เลือกโมเดลฟรีจากคลังระบบ')
                        with gr.Row():
                            pub_zip_link = gr.Textbox(label='🔗 URL โมเดลที่เลือก')
                            pub_model_name = gr.Textbox(label='🏷️ ชื่อโมเดลที่เลือก')
                        download_pub_btn = gr.Button('⬇️ ติดตั้งโมเดลที่เลือก', variant='primary', elem_classes=['primary-action'])
                        pub_dl_output_message = gr.Textbox(label='📌 สถานะการติดตั้ง', interactive=False)

                        filter_tags = gr.CheckboxGroup(value=[], label='🏷️ กรองตามแท็ก', choices=[])
                        search_query = gr.Textbox(label='🔍 ค้นหาชื่อ/คำอธิบาย')
                        load_public_models_button = gr.Button('🔄 ดึงรายการโมเดลสาธารณะทั้งหมด', elem_classes=['secondary-btn'])

                        public_models_table = gr.DataFrame(
                            value=[], headers=['ชื่อโมเดล', 'รายละเอียด', 'เครดิต', 'URL', 'แท็ก'],
                            label='📋 โมเดลที่พร้อมติดตั้ง', interactive=False
                        )

                        public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                        load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                        search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

                    with gr.Tab('📤 อัปโหลดจากเครื่อง (.zip)'):
                        gr.Markdown('#### 📦 เพิ่มโมเดลด้วยไฟล์ ZIP จากเครื่องของคุณ')
                        with gr.Row():
                            zip_file = gr.File(label='📂 เลือกไฟล์โมเดล ZIP', file_types=['.zip'])
                            local_model_name = gr.Textbox(label='🏷 ตั้งชื่อโมเดล')
                        model_upload_button = gr.Button('📤 ติดตั้งโมเดลเข้าสู่ระบบ', variant='primary', elem_classes=['primary-action'])
                        local_upload_output_message = gr.Textbox(label='📌 สถานะ', interactive=False)
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
        css=CUSTOM_CSS,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )