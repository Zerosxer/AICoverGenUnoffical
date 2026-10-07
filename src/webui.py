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
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&family=Orbitron:wght@600;800&display=swap');

/* --- Global Modern Flat Theme --- */
:root {
    --bg-dark: #090613;
    --panel-bg: rgba(18, 11, 31, 0.85);
    --border-color: rgba(168, 85, 247, 0.2);
    --accent-purple: #a855f7;
    --accent-cyan: #06b6d4;
    --text-main: #f3e8ff;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    font-family: 'Kanit', sans-serif !important;
    color: var(--text-main) !important;
    max-width: 98% !important; /* ขยายเกือบเต็มจอ */
    margin: 0 auto !important;
    padding: 10px !important;
}

/* Header แบนเรียบ มีสไตล์ */
.app-header {
    padding: 16px 24px;
    margin-bottom: 16px;
    background: linear-gradient(90deg, #2e1065 0%, #0f172a 100%);
    border-bottom: 2px solid var(--accent-purple);
    border-radius: 4px;
}

.app-header h1 {
    font-family: 'Orbitron', 'Kanit', sans-serif !important;
    color: #ffffff !important;
    font-size: 24px !important;
    margin: 0 !important;
}

/* ลบกล่องซ้อนกรอบซ้ำซ้อน (Remove Nested Box Shadows & Heavy Borders) */
.gr-group, .gr-box, .gr-form, .gr-panel {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important; /* ปรับขอบให้เหลี่ยมขึ้น ไม่มนแปลกๆ */
    box-shadow: none !important;
    padding: 12px !important;
    margin: 4px 0 !important;
}

/* ปรับพวกช่อง Input ให้เรียบกลืนไปกับแผงควบคุม */
input[type="text"], textarea, select, .gr-dropdown {
    background: rgba(5, 3, 10, 0.8) !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    border-radius: 4px !important;
    color: #ffffff !important;
}

input[type="text"]:focus, .gr-dropdown:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 8px rgba(6, 182, 212, 0.4) !important;
}

/* แท็บด้านบน */
.tabs > .tab-nav {
    border-bottom: 2px solid var(--border-color) !important;
    gap: 4px !important;
}

.tabs > .tab-nav > button {
    background: transparent !important;
    color: #a78bfa !important;
    border: none !important;
    border-radius: 4px 4px 0 0 !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}

.tabs > .tab-nav > button.selected {
    background: rgba(168, 85, 247, 0.2) !important;
    color: #ffffff !important;
    border-bottom: 2px solid var(--accent-purple) !important;
}

/* ปุ่มกดหลัก */
button.primary-action {
    background: linear-gradient(90deg, #7e22ce 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
}

button.primary-action:hover {
    opacity: 0.9;
}

button.secondary-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #cbd5e1 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
}

button.secondary-btn:hover {
    background: rgba(168, 85, 247, 0.15) !important;
    color: #ffffff !important;
}

footer { visibility: hidden !important; }
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
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    # ใช้ Soft Theme ที่ปลอดภัยจาก AttributeError
    custom_theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="cyan",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="rgba(22, 13, 38, 0.75)",
        block_border_color="rgba(168, 85, 247, 0.25)",
    )

    with gr.Blocks(title='AICoverGen Studio', css=CUSTOM_CSS, theme=custom_theme) as app:
        
        # Banner Header
        gr.Markdown(
            '<div class="app-header">'
            '<h1>🔮 AICoverGen Studio</h1>'
            '<p>ระบบเนรมิตเพลงคัฟเวอร์ด้วยเสียงสังเคราะห์ AI ระดับมืออาชีพ</p>'
            '</div>'
        )

        with gr.Tabs():
            # ---------------- MAIN STUDIO TAB ----------------
            with gr.Tab('🎼 สตูดิโอสร้างเพลง (Main Studio)'):
                # ปรับสัดส่วน 1:1 (Scale 1 และ Scale 1 เท่ากันทั้งซ้ายและขวา)
                with gr.Row():
                    
                    # === LEFT COLUMN: INPUTS & SETTINGS ===
                    with gr.Column(scale=1):
                        
                        # Section 1: Voice & Song Selection
                        with gr.Group():
                            gr.Markdown('### 1️⃣ เลือกโมเดลเสียง & แหล่งข้อมูลเพลง')
                            with gr.Row():
                                rvc_model = gr.Dropdown(
                                    voice_models, label='🎭 เลือกโมเดลเสียง AI (Voice Model)',
                                    info='กดรีเฟรชเมื่อเพิ่มโมเดลใหม่', scale=4
                                )
                                ref_btn = gr.Button('🔄', elem_classes=['secondary-btn'], scale=1)

                            with gr.Column() as yt_link_col:
                                song_input = gr.Textbox(
                                    label='🔗 ลิงก์ YouTube หรือ Path ไฟล์เพลง',
                                    placeholder='วาง URL YouTube หรือ Path ไฟล์ .wav/.mp3'
                                )
                                show_file_upload_button = gr.Button('📁 สลับไปใช้วิธีอัปโหลดไฟล์ตรง', elem_classes=['secondary-btn'])

                            with gr.Column(visible=False) as file_upload_col:
                                local_file = gr.File(label='📄 ไฟล์เสียงที่เลือก')
                                song_input_file = gr.UploadButton(
                                    '📤 เลือกไฟล์เสียงจากอุปกรณ์', file_types=['audio'], variant='primary', elem_classes=['primary-action']
                                )
                                show_yt_link_button = gr.Button('🔗 สลับกลับไปใช้ลิงก์/Path', elem_classes=['secondary-btn'])
                                song_input_file.upload(
                                    process_file_upload, inputs=[song_input_file],
                                    outputs=[local_file, song_input]
                                )

                        # Section 2: Pitch Tuning
                        with gr.Group():
                            gr.Markdown('### 2️⃣ ปรับระดับคีย์เสียง (Pitch Control)')
                            with gr.Row():
                                pitch = gr.Slider(
                                    -3, 3, value=0, step=1, label='🎼 คีย์เสียงร้อง AI (Octaves)',
                                    info='+1 เสียงหญิง | -1 เสียงชาย'
                                )
                                pitch_all = gr.Slider(
                                    -12, 12, value=0, step=1, label='🎶 คีย์รวมทั้งเพลง (Semitones)',
                                    info='ปรับพร้อมกันทั้งร้องและดนตรี'
                                )

                        # Section 3: Advanced Options Accordion
                        with gr.Accordion('⚙️ ปรับแต่งเพิ่มเติม (Advanced Settings)', open=False):
                            with gr.Tab('🎙️ Voice Tuning'):
                                with gr.Row():
                                    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate')
                                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius')
                                with gr.Row():
                                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate')
                                    protect = gr.Slider(0, 0.5, value=0.33, label='Protect Breath')
                                with gr.Row():
                                    f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='F0 Method')
                                    crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe Hop')
                                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)

                            with gr.Tab('🎚️ Audio Mixer'):
                                with gr.Row():
                                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงร้องหลัก AI (dB)')
                                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงร้องประสาน (dB)')
                                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงดนตรีประกอบ (dB)')

                            with gr.Tab('🔮 Reverb Studio'):
                                with gr.Row():
                                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room Size')
                                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Wet')
                                with gr.Row():
                                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Dry')
                                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping')

                            with gr.Tab('💾 Export'):
                                output_format = gr.Radio(['mp3', 'wav'], value='mp3', label='ฟอร์แมตไฟล์ผลลัพธ์')
                                keep_files = gr.Checkbox(label='บันทึกไฟล์แยกชิ้นส่วน (ร้อง/ดนตรี)')

                    # === RIGHT COLUMN: GENERATION & OUTPUT ===
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=['output-card']):
                            gr.Markdown('### ⚡ ประมวลผล & ผลลัพธ์ (Generation)')
                            
                            generate_btn = gr.Button(
                                '⚡ เริ่มสร้าง AI Cover', variant='primary', elem_classes=['primary-action']
                            )
                            clear_btn = gr.ClearButton(
                                value='🧹 ล้างค่าทั้งหมด', components=[song_input, rvc_model, keep_files, local_file],
                                elem_classes=['secondary-btn']
                            )
                            
                            gr.Markdown('---')
                            ai_cover = gr.Audio(label='🎧 ผลงาน AI Cover ที่เสร็จสมบูรณ์', buttons=['download'])

            # ---------------- MODEL HUB TAB ----------------
            with gr.Tab('📦 ศูนย์จัดการโมเดลเสียง (Model Hub)'):
                with gr.Tabs():
                    with gr.Tab('🔗 ดาวน์โหลดผ่าน Direct URL'):
                        gr.Markdown('#### 🌐 ดาวน์โหลดโมเดลผ่านลิงก์ ZIP')
                        with gr.Row():
                            model_zip_link = gr.Textbox(label='🔗 ลิงก์ไฟล์ ZIP โมเดล')
                            model_name = gr.Textbox(label='🏷️ ตั้งชื่อโมเดล')
                        download_btn = gr.Button('⬇️ เริ่มดาวน์โหลด', variant='primary', elem_classes=['primary-action'])
                        dl_output_message = gr.Textbox(label='📌 รายงานสถานะ', interactive=False)
                        download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                        gr.Markdown('##### 💡 ลิงก์โมเดลตัวอย่าง')
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

                    with gr.Tab('🌐 ค้นหาคลังโมเดลสาธารณะ (Public Hub)'):
                        gr.Markdown('#### 🔍 เลือกโมเดลจากคลังระบบออนไลน์')
                        with gr.Row():
                            pub_zip_link = gr.Textbox(label='🔗 URL โมเดลที่เลือก')
                            pub_model_name = gr.Textbox(label='🏷️ ชื่อโมเดลที่เลือก')
                        download_pub_btn = gr.Button('⬇️ ติดตั้งโมเดลที่เลือก', variant='primary', elem_classes=['primary-action'])
                        pub_dl_output_message = gr.Textbox(label='📌 สถานะการติดตั้ง', interactive=False)

                        filter_tags = gr.CheckboxGroup(value=[], label='🏷️ กรองตามหมวดหมู่/แท็ก', choices=[])
                        search_query = gr.Textbox(label='🔍 ค้นหาตามชื่อหรือคำอธิบาย')
                        load_public_models_button = gr.Button('🔄 โหลดรายการคลังโมเดลทั้งหมด', elem_classes=['secondary-btn'])

                        public_models_table = gr.DataFrame(
                            value=[], headers=['ชื่อโมเดล', 'รายละเอียด', 'เครดิต', 'URL', 'แท็ก'],
                            label='📋 รายการโมเดลที่พร้อมติดตั้ง', interactive=False
                        )

                        public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                        load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                        search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

                    with gr.Tab('📤 อัปโหลดจากเครื่อง (.zip)'):
                        gr.Markdown('#### 📦 เพิ่มโมเดลด้วยไฟล์ ZIP ในเครื่อง')
                        with gr.Row():
                            zip_file = gr.File(label='📂 เลือกไฟล์โมเดล .zip', file_types=['.zip'])
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