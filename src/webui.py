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
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

/* ตั้งค่าฟอนต์และพื้นหลังหลัก */
* {
    font-family: 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Header Banner ดีไซน์โมเดิร์น ไล่เฉดสีสมูท */
.app-header {
    padding: 24px 30px;
    margin-bottom: 20px;
    border-radius: 16px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
    color: #FFFFFF !important;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
}

.app-header h1 {
    margin: 0 0 6px 0 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}

.app-header p {
    margin: 0 !important;
    font-size: 14px !important;
    color: #E0E7FF !important;
    opacity: 0.9;
}

/* ปรับแต่งกล่องคอนเทนเนอร์ และ Card ต่างๆ */
.block, .gr-group, .gr-box, .gr-panel {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04) !important;
    padding: 16px !important;
}

/* แก้ไขปัญหาสี Text / Label จมอ่านยาก */
label span, .gr-form label, label, .text-gray-500 {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* แก้ไข Input Field ให้ชัดเจน สบายตา อ่านง่าย */
input[type="text"], textarea, .gr-input, select, .gr-dropdown {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: #6366F1 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ปุ่มกดหลัก (Primary Button) */
button.primary-action, button.primary {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.2s ease !important;
}

button.primary-action:hover, button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
}

/* ปุ่มรอง (Secondary Button) */
button.secondary-btn {
    background: #F1F5F9 !important;
    color: #475569 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

button.secondary-btn:hover {
    background: #E2E8F0 !important;
    color: #0F172A !important;
}

/* ปรับแต่ง Tabs ให้คมชัดสวยงาม */
.tabs {
    border-bottom: 2px solid #E2E8F0 !important;
}

.tab-nav button {
    font-weight: 500 !important;
    color: #64748B !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}

.tab-nav button.selected {
    color: #4F46E5 !important;
    border-bottom: 2px solid #4F46E5 !important;
    font-weight: 600 !important;
}

/* ซ่อน Footer และปรับการแสดงผลองค์ประกอบเล็กๆ */
footer {
    display: none !important;
}

.section-title {
    font-size: 15px;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
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

    # ธีม Soft ของ Gradio ร่วมกับ Custom CSS
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title='AICoverGen Studio', css=CUSTOM_CSS, theme=custom_theme) as app:
        
        # Header Banner
        gr.Markdown(
            '<div class="app-header">'
            '<h1>✨ AICoverGen Studio</h1>'
            '<p>ระบบสร้างและคัฟเวอร์เพลงด้วยเสียง AI อัจฉริยะ | รองรับ RVC v2 Models</p>'
            '</div>'
        )

        with gr.Tabs():
            # ---------------- MAIN STUDIO TAB ----------------
            with gr.Tab('🎼 สตูดิโอสร้างเพลง (Main Studio)'):
                with gr.Row(equal_height=False):
                    
                    # ฝั่งซ้าย: ตั้งค่าและเลือกเพลง (60%)
                    with gr.Column(scale=3):
                        
                        # Section 1: โมเดลและไฟล์เพลง
                        with gr.Group():
                            gr.Markdown('<div class="section-title">1️⃣ เลือกโมเดลเสียง & แหล่งข้อมูลเพลง</div>')
                            
                            with gr.Row():
                                rvc_model = gr.Dropdown(
                                    voice_models, label='🎭 โมเดลเสียง AI (Voice Model)',
                                    info='เลือกโมเดลเสียงที่ต้องการใช้ร้อง', scale=5
                                )
                                ref_btn = gr.Button('🔄 รีเฟรช', elem_classes=['secondary-btn'], scale=1)

                            with gr.Column() as yt_link_col:
                                song_input = gr.Textbox(
                                    label='🔗 ลิงก์ YouTube หรือ Path ไฟล์เพลงในเครื่อง',
                                    placeholder='วาง URL ของ YouTube (เช่น https://www.youtube.com/watch?v=...) หรือ Path ไฟล์ .mp3 / .wav'
                                )
                                show_file_upload_button = gr.Button('📁 สลับไปใช้วิธีอัปโหลดไฟล์ตรง', elem_classes=['secondary-btn'])

                            with gr.Column(visible=False) as file_upload_col:
                                local_file = gr.File(label='📄 ไฟล์เสียงที่เลือกไว้')
                                song_input_file = gr.UploadButton(
                                    '📤 อัปโหลดไฟล์เสียงจากอุปกรณ์', file_types=['audio'], variant='primary', elem_classes=['primary-action']
                                )
                                show_yt_link_button = gr.Button('🔗 สลับกลับไปใช้ URL / Path', elem_classes=['secondary-btn'])
                                song_input_file.upload(
                                    process_file_upload, inputs=[song_input_file],
                                    outputs=[local_file, song_input]
                                )

                        # Section 2: ปรับระดับคีย์เสียง
                        with gr.Group():
                            gr.Markdown('<div class="section-title">2️⃣ ปรับระดับคีย์เสียง (Pitch Control)</div>')
                            with gr.Row():
                                pitch = gr.Slider(
                                    -3, 3, value=0, step=1, label='🎼 คีย์เสียงร้อง AI (Octaves)',
                                    info='+1 เปลี่ยนเสียงชายเป็นหญิง | -1 เปลี่ยนเสียงหญิงเป็นชาย'
                                )
                                pitch_all = gr.Slider(
                                    -12, 12, value=0, step=1, label='🎶 คีย์รวมทั้งเพลง (Semitones)',
                                    info='ปรับเปลี่ยนคีย์พร้อมกันทั้งเสียงร้องและเสียงดนตรีประกอบ'
                                )

                        # Section 3: ตั้งค่าระดับสูง
                        with gr.Accordion('⚙️ การปรับแต่งเสียงขั้นสูง (Advanced Settings)', open=False):
                            with gr.Tab('🎙️ Voice Tuning'):
                                with gr.Row():
                                    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate (ความเหมือนต้นฉบับ)')
                                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius (ลดเสียงเพี้ยน)')
                                with gr.Row():
                                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate (ความดังตามต้นฉบับ)')
                                    protect = gr.Slider(0, 0.5, value=0.33, label='Protect Breath (ปกป้องเสียงลมหายใจ)')
                                with gr.Row():
                                    f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='F0 Method (อัลกอริทึมจับ Pitch)')
                                    crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe Hop Length')
                                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)

                            with gr.Tab('🎚️ Audio Mixer'):
                                with gr.Row():
                                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงร้องหลัก AI (dB)')
                                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงร้องประสาน (dB)')
                                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงดนตรีประกอบ (dB)')

                            with gr.Tab('🔮 Reverb Studio'):
                                with gr.Row():
                                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room Size (ขนาดห้อง)')
                                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Wet Level (เสียงเอฟเฟกต์)')
                                with gr.Row():
                                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Dry Level (เสียงคลีน)')
                                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping (การซับเสียง)')

                            with gr.Tab('💾 Export Settings'):
                                output_format = gr.Radio(['mp3', 'wav'], value='mp3', label='นามสกุลไฟล์ผลลัพธ์ (Format)')
                                keep_files = gr.Checkbox(label='บันทึกไฟล์แยกชิ้นส่วน (แยกเสียงร้องและดนตรีเก็บไว้)')

                    # ฝั่งขวา: ปุ่มประมวลผล & แสดงผลลัพธ์ (40%)
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown('<div class="section-title">⚡ การประมวลผล & ผลลัพธ์</div>')
                            
                            generate_btn = gr.Button(
                                '⚡ เริ่มสร้าง AI Cover', variant='primary', size='lg', elem_classes=['primary-action']
                            )
                            clear_btn = gr.ClearButton(
                                value='🧹 ล้างค่าทั้งหมด', components=[song_input, rvc_model, keep_files, local_file],
                                elem_classes=['secondary-btn']
                            )
                            
                            gr.Markdown('---')
                            ai_cover = gr.Audio(label='🎧 ผลงานเพลง AI Cover ที่เสร็จสมบูรณ์', type='filepath')

            # ---------------- MODEL HUB TAB ----------------
            with gr.Tab('📦 ศูนย์จัดการโมเดลเสียง (Model Hub)'):
                with gr.Tabs():
                    with gr.Tab('🔗 ดาวน์โหลดผ่าน Direct URL'):
                        gr.Markdown('<div class="section-title">🌐 ดาวน์โหลดโมเดลผ่านลิงก์ ZIP</div>')
                        with gr.Row():
                            model_zip_link = gr.Textbox(label='🔗 ลิงก์ไฟล์ ZIP ของโมเดล', placeholder='https://...')
                            model_name = gr.Textbox(label='🏷️ ตั้งชื่อโมเดล', placeholder='เช่น Lisa, Gura')
                        download_btn = gr.Button('⬇️ เริ่มดาวน์โหลดและติดตั้ง', variant='primary', elem_classes=['primary-action'])
                        dl_output_message = gr.Textbox(label='📌 รายงานสถานะ', interactive=False)
                        download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                        gr.Markdown('##### 💡 ตัวอย่างลิงก์โมเดลทดสอบ')
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

                    with gr.Tab('🌐 คลังโมเดลสาธารณะ (Public Hub)'):
                        gr.Markdown('<div class="section-title">🔍 ค้นหาและเลือกติดตั้งโมเดลออนไลน์</div>')
                        with gr.Row():
                            pub_zip_link = gr.Textbox(label='🔗 URL โมเดลที่เลือก', interactive=False)
                            pub_model_name = gr.Textbox(label='🏷️ ชื่อโมเดลที่เลือก', interactive=False)
                        download_pub_btn = gr.Button('⬇️ ติดตั้งโมเดลที่เลือก', variant='primary', elem_classes=['primary-action'])
                        pub_dl_output_message = gr.Textbox(label='📌 สถานะการติดตั้ง', interactive=False)

                        with gr.Row():
                            filter_tags = gr.CheckboxGroup(value=[], label='🏷️ กรองตามหมวดหมู่/แท็ก', choices=[])
                            search_query = gr.Textbox(label='🔍 ค้นหาด้วยชื่อหรือคำอธิบาย', placeholder='พิมพ์ชื่อโมเดล...')
                        
                        load_public_models_button = gr.Button('🔄 โหลด/รีเฟรชรายการโมเดลทั้งหมด', elem_classes=['secondary-btn'])

                        public_models_table = gr.DataFrame(
                            value=[], headers=['ชื่อโมเดล', 'รายละเอียด', 'เครดิต', 'URL', 'แท็ก'],
                            label='📋 รายการโมเดลที่พร้อมติดตั้ง (คลิกเพื่อเลือก)', interactive=False
                        )

                        public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                        load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                        search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

                    with gr.Tab('📤 อัปโหลดจากเครื่อง (.zip)'):
                        gr.Markdown('<div class="section-title">📦 เพิ่มโมเดลด้วยไฟล์ ZIP ในเครื่องคอมพิวเตอร์</div>')
                        with gr.Row():
                            zip_file = gr.File(label='📂 เลือกไฟล์โมเดล .zip', file_types=['.zip'])
                            local_model_name = gr.Textbox(label='🏷️ ตั้งชื่อโมเดล')
                        model_upload_button = gr.Button('📤 ติดตั้งโมเดลเข้าสู่ระบบ', variant='primary', elem_classes=['primary-action'])
                        local_upload_output_message = gr.Textbox(label='📌 สถานะการติดตั้ง', interactive=False)
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