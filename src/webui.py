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
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.app-header {padding: 22px 26px; margin-bottom: 18px; border-radius: 16px;
    background: linear-gradient(120deg, #38206f, #6545ad 62%, #985fba); color: white;}
.app-header h1 {margin: 0 0 6px; font-size: 28px; color: #fff !important;}
.app-header p {margin: 0; opacity: .9; color: #fff !important;}
.primary-action {min-height: 48px; font-size: 16px !important; font-weight: 700 !important;}
footer {visibility: hidden;}
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

    with gr.Blocks(title='AICoverGen | สร้าง AI Cover') as app:
        gr.Markdown(
            '<div class="app-header"><h1>🎙️ AICoverGen</h1>'
            '<p>สร้างเพลงคัฟเวอร์ด้วย AI Voice แยกตั้งค่าเป็นหมวด ใช้งานได้ง่ายในไม่กี่ขั้นตอน</p></div>'
        )

        with gr.Tab('สร้างเพลง'):
            with gr.Tabs():
                with gr.Tab('1 · เพลงและเสียงร้อง'):
                    gr.Markdown('เลือกเสียงร้องต้นฉบับและโมเดลเสียงที่ต้องการ')
                    with gr.Row():
                        with gr.Column(scale=2):
                            rvc_model = gr.Dropdown(
                                voice_models, label='โมเดลเสียง AI',
                                info='กดรีเฟรชหลังเพิ่มโมเดลลงในโฟลเดอร์ rvc_models'
                            )
                            ref_btn = gr.Button('รีเฟรชรายการโมเดล')
                        with gr.Column(scale=3):
                            with gr.Column() as yt_link_col:
                                song_input = gr.Textbox(
                                    label='ลิงก์ YouTube หรือพาธไฟล์เสียง',
                                    placeholder='วางลิงก์ YouTube หรือพาธไฟล์ .wav / .mp3',
                                    info='หรือเลือกอัปโหลดไฟล์เสียงจากเครื่อง'
                                )
                                show_file_upload_button = gr.Button('อัปโหลดไฟล์จากเครื่อง')
                            with gr.Column(visible=False) as file_upload_col:
                                local_file = gr.File(label='ไฟล์เสียงที่เลือก')
                                song_input_file = gr.UploadButton(
                                    'เลือกไฟล์เสียง', file_types=['audio'], variant='primary'
                                )
                                show_yt_link_button = gr.Button('กลับไปใช้ลิงก์หรือพาธไฟล์')
                                song_input_file.upload(
                                    process_file_upload, inputs=[song_input_file],
                                    outputs=[local_file, song_input]
                                )
                    with gr.Row():
                        pitch = gr.Slider(
                            -3, 3, value=0, step=1, label='ปรับคีย์เสียงร้อง AI (อ็อกเทฟ)',
                            info='โดยทั่วไปใช้ +1 เมื่อต้องการเสียงหญิง หรือ -1 สำหรับเสียงชาย'
                        )
                        pitch_all = gr.Slider(
                            -12, 12, value=0, step=1, label='ปรับคีย์เพลงทั้งหมด (เซมิโทน)',
                            info='เปลี่ยนคีย์ทั้งเสียงร้องและดนตรี อาจลดคุณภาพเสียงเล็กน้อย'
                        )
                    show_file_upload_button.click(
                        swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file]
                    )
                    show_yt_link_button.click(
                        swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file]
                    )

                with gr.Tab('2 · ปรับเสียง AI'):
                    gr.Markdown('ปรับคุณภาพการแปลงเสียง · ค่าเริ่มต้นเหมาะกับการใช้งานทั่วไป')
                    with gr.Row():
                        index_rate = gr.Slider(
                            0, 1, value=0.5, label='ความคล้ายโทนเสียง (Index rate)',
                            info='ค่าสูงขึ้นจะยึดโทนเสียงของโมเดลมากขึ้น'
                        )
                        filter_radius = gr.Slider(
                            0, 7, value=3, step=1, label='ลดเสียงพร่า (Filter radius)',
                            info='ตั้งแต่ 3 ขึ้นไปจะช่วยกรองการสั่นของระดับเสียง'
                        )
                    with gr.Row():
                        rms_mix_rate = gr.Slider(
                            0, 1, value=0.25, label='รักษาความดังเดิม (RMS mix rate)',
                            info='0 ใช้ความดังต้นฉบับ · 1 ใช้ความดังมาตรฐาน'
                        )
                        protect = gr.Slider(
                            0, 0.5, value=0.33, label='ปกป้องลมหายใจและเสียงพยัญชนะ',
                            info='ตั้ง 0.5 เพื่อปิดการปกป้อง'
                        )
                    with gr.Row():
                        f0_method = gr.Dropdown(
                            ['rmvpe', 'mangio-crepe'], value='rmvpe',
                            label='วิธีตรวจจับระดับเสียง',
                            info='RMVPE ชัดเจน · Mangio-Crepe ให้เสียงนุ่มขึ้น'
                        )
                        crepe_hop_length = gr.Slider(
                            32, 320, value=128, step=1, visible=False,
                            label='Crepe hop length',
                            info='ค่าน้อยลงเพิ่มความละเอียดแต่ใช้เวลาประมวลผลมากขึ้น'
                        )
                        f0_method.change(
                            show_hop_slider, inputs=f0_method, outputs=crepe_hop_length
                        )
                    keep_files = gr.Checkbox(
                        label='เก็บไฟล์เสียงระหว่างประมวลผล',
                        info='เก็บไฟล์เสียงร้อง/ดนตรีที่แยกแล้วไว้ใน song_output ใช้พื้นที่เพิ่ม'
                    )

                with gr.Tab('3 · มิกซ์เสียงและส่งออก'):
                    gr.Markdown('กำหนดระดับเสียง เอฟเฟกต์ และรูปแบบไฟล์ผลลัพธ์')
                    gr.Markdown('#### ระดับเสียง (dB)')
                    with gr.Row():
                        main_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงร้องหลัก AI')
                        backup_gain = gr.Slider(-20, 20, value=0, step=1, label='เสียงร้องประสาน')
                        inst_gain = gr.Slider(-20, 20, value=0, step=1, label='ดนตรี')
                    gr.Markdown('#### Reverb สำหรับเสียงร้อง AI')
                    with gr.Row():
                        reverb_rm_size = gr.Slider(0, 1, value=0.15, label='ขนาดห้อง')
                        reverb_wet = gr.Slider(0, 1, value=0.2, label='เสียงก้อง (Wet)')
                        reverb_dry = gr.Slider(0, 1, value=0.8, label='เสียงตรง (Dry)')
                        reverb_damping = gr.Slider(0, 1, value=0.7, label='ลดความถี่สูง')
                    output_format = gr.Radio(
                        ['mp3', 'wav'], value='mp3', label='รูปแบบไฟล์',
                        info='MP3 ขนาดเล็ก · WAV คุณภาพสูงและไฟล์ใหญ่'
                    )

            with gr.Row():
                clear_btn = gr.ClearButton(
                    value='ล้างผลลัพธ์', components=[song_input, rvc_model, keep_files, local_file]
                )
                generate_btn = gr.Button(
                    'สร้าง AI Cover', variant='primary', elem_classes=['primary-action']
                )
            ai_cover = gr.Audio(label='เพลง AI Cover ที่สร้างเสร็จ', buttons=['download'])

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                               inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                       inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                       protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                                       output_format],
                               outputs=[ai_cover])
            clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
                            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                                     reverb_dry, reverb_damping, output_format, ai_cover])

        # Download tab
        with gr.Tab('ดาวน์โหลดโมเดล'):

            with gr.Tab('ดาวน์โหลดจากลิงก์'):
                gr.Markdown('ดาวน์โหลดไฟล์ ZIP ที่มีไฟล์โมเดล `.pth` และไฟล์ดัชนี `.index` (ถ้ามี)')
                with gr.Row():
                    model_zip_link = gr.Textbox(label='ลิงก์ไฟล์โมเดล ZIP')
                    model_name = gr.Textbox(label='ตั้งชื่อโมเดล', info='ต้องไม่ซ้ำกับโมเดลที่มีอยู่')

                with gr.Row():
                    download_btn = gr.Button('ดาวน์โหลดโมเดล', variant='primary', scale=19)
                    dl_output_message = gr.Textbox(label='สถานะ', interactive=False, scale=20)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('ตัวอย่างลิงก์')
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

            with gr.Tab('คลังโมเดลสาธารณะ'):

                gr.Markdown('กดโหลดรายการ ค้นหาหรือกรองด้วยแท็ก แล้วเลือกแถวโมเดลเพื่อเติมลิงก์ดาวน์โหลดอัตโนมัติ')

                with gr.Row():
                    pub_zip_link = gr.Textbox(label='ลิงก์ดาวน์โหลด')
                    pub_model_name = gr.Textbox(label='ชื่อโมเดล')

                with gr.Row():
                    download_pub_btn = gr.Button('ดาวน์โหลดโมเดล', variant='primary', scale=19)
                    pub_dl_output_message = gr.Textbox(label='สถานะ', interactive=False, scale=20)

                filter_tags = gr.CheckboxGroup(value=[], label='กรองตามแท็ก', choices=[])
                search_query = gr.Textbox(label='ค้นหาโมเดล')
                load_public_models_button = gr.Button(value='โหลดรายการโมเดล', variant='primary')

                public_models_table = gr.DataFrame(
                    value=[], headers=['ชื่อโมเดล', 'รายละเอียด', 'เครดิต', 'URL', 'แท็ก'],
                    label='โมเดลที่ติดตั้งได้', interactive=False
                )
                public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

        # Upload tab
        with gr.Tab('เพิ่มโมเดลจากเครื่อง'):
            gr.Markdown(
                'บีบอัดไฟล์โมเดล `.pth` (และ `.index` หากมี) เป็น ZIP '
                'จากนั้นเลือกไฟล์และตั้งชื่อโมเดลที่ไม่ซ้ำ'
            )

            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(label='ไฟล์โมเดล ZIP', file_types=['.zip'])

                local_model_name = gr.Textbox(label='ชื่อโมเดล')

            with gr.Row():
                model_upload_button = gr.Button('เพิ่มโมเดล', variant='primary', scale=19)
                local_upload_output_message = gr.Textbox(label='สถานะ', interactive=False, scale=20)
                model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.queue()
    app.launch(
        share=args.share_enabled,
        css=CUSTOM_CSS,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
