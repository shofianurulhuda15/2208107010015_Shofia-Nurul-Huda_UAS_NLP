import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import base64

def voice_chat(audio):
    if audio is None:
        return None, "Error: Tidak ada audio yang direkam.", None
    
    sr, audio_data = audio

    # Simpan sebagai .wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    # Kirim ke endpoint FastAPI
    try:
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            response = requests.post("http://localhost:8000/voice-chat", files=files)
        
        print(f"[DEBUG] Response status code: {response.status_code}")
        print(f"[DEBUG] Response content: {response.text}")
        
        if response.status_code == 200:
            # Parse JSON response
            data = response.json()
            audio_base64 = data.get("audio")
            audio_filename = data.get("audio_filename", "response.wav")
            transcript = data.get("transcript", "Error: Transkrip tidak tersedia")
            response_text = data.get("response_text", "Error: Respons teks tidak tersedia")
            
            # Simpan file audio dari base64
            output_audio_path = os.path.join(tempfile.gettempdir(), audio_filename)
            with open(output_audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            
            return output_audio_path, transcript, response_text
        else:
            error_msg = response.json().get("error", f"Request failed with status {response.status_code}")
            return None, f"Error: {error_msg}", f"Error: {error_msg}"
    except Exception as e:
        print(f"[ERROR] Failed to process request: {e}")
        return None, f"Error: Gagal memproses permintaan: {str(e)}", f"Error: Gagal memproses permintaan: {str(e)}"

# Custom CSS dengan nuansa cute dan font Poppins
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
    --main-bg-color: #fff8fa;
    --panel-bg-color: #fff2f6;
    --primary-color: #ff85a2;
    --secondary-color: #a2d2ff;
    --accent-color: #ffc8dd;
    --text-color: #5e5b6a;
    --border-radius: 15px;
    --box-shadow: 0 4px 12px rgba(255, 133, 162, 0.15);
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    padding: 30px !important;
    background-color: var(--main-bg-color) !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-color) !important;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    color: var(--primary-color) !important;
}

.input-section, .output-section {
    background-color: var(--panel-bg-color) !important;
    border: 2px dashed var(--accent-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 20px !important;
    margin-bottom: 25px !important;
    box-shadow: var(--box-shadow) !important;
    transition: all 0.3s ease !important;
}

.input-section:hover, .output-section:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 8px 15px rgba(255, 133, 162, 0.25) !important;
}

.gr-button {
    font-family: 'Poppins', sans-serif !important;
    border-radius: 25px !important;
    padding: 8px 25px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(255, 133, 162, 0.3) !important;
}

.gr-button-primary {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
}

.gr-button-primary:hover {
    background: #ff7090 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(255, 133, 162, 0.4) !important;
}

.gr-button-secondary {
    background: var(--secondary-color) !important;
    color: white !important;
    border: none !important;
}

.gr-button-secondary:hover {
    background: #8bd3ff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(255, 133, 162, 0.4) !important;
}

.gr-textbox, .gr-audio {
    border-radius: 12px !important;
    border: 2px solid var(--accent-color) !important;
    padding: 10px !important;
    font-family: 'Poppins', sans-serif !important;
}

.gr-audio .audio-container {
    background: var(--panel-bg-color) !important;
    border-radius: 12px !important;
}

.footer {
    text-align: center !important;
    color: var(--primary-color) !important;
    font-size: 1em !important;
    margin-top: 30px !important;
    font-weight: 500 !important;
}

.error-message {
    color: #e74c3c !important;
    font-weight: 500 !important;
    background-color: #fdeaec !important;
    padding: 10px !important;
    border-radius: 10px !important;
    border-left: 4px solid #e74c3c !important;
    animation: fadeIn 0.5s ease-in !important;
    text-align: center !important;
}

.animate-pulse {
    animation: pulse 2s infinite !important;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
"""

# UI Gradio yang disesuaikan
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🌸 Voice Assistant Indonesia
        Berbicara dengan asisten AI dalam Bahasa Indonesia! 🎤 Rekam pertanyaanmu dan dengar jawabannya! 💖
        """,
        elem_classes="header"
    )
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel", elem_classes="input-section"):
            gr.Markdown("### 🎀 Rekam Pertanyaanmu")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Tekan untuk merekam suaramu! 🎙️",
                waveform_options={"show_controls": True, "waveform_color": "#ff85a2"}
            )
            transcript_output = gr.Textbox(
                label="✨ Transkrip Input",
                placeholder="Transkrip akan muncul di sini... ✨",
                interactive=False,
                lines=3
            )
            with gr.Row():
                submit_btn = gr.Button("💌 Kirim", variant="primary")
                clear_btn = gr.Button("🧹 Hapus", variant="secondary")
        
        with gr.Column(scale=1, variant="panel", elem_classes="output-section"):
            gr.Markdown("### 🎧 Dengar Jawabannya")
            audio_output = gr.Audio(
                type="filepath",
                label="🔊 Balasan suara dari asisten! 🎵",
                interactive=False,
                waveform_options={"show_controls": True, "waveform_color": "#a2d2ff"}
            )
            response_text_output = gr.Textbox(
                label="💬 Respons Teks",
                placeholder="Jawaban teks akan muncul di sini... 🌈",
                interactive=False,
                lines=5
            )
    
    status_message = gr.Markdown("", visible=False, elem_classes="error-message")
    
    def clear_inputs():
        return None, "", None, "", ""
    
    def process_audio(audio):
        if audio is None:
            return None, "Silakan rekam suara terlebih dahulu!", None, "⚠️ Silakan rekam suara terlebih dahulu!"
        
        # Tampilkan pesan sedang diproses
        status_message_content = """
        <div class="animate-pulse" style="text-align: center; color: #ff85a2;">
            🎵 Sedang memproses suara Anda... 🎵
        </div>
        """
        
        # Proses audio
        audio_path, transcript, response_text = voice_chat(audio)
        
        if audio_path:
            return audio_path, transcript, response_text, ""
        else:
            return None, transcript, response_text, f"⚠️ {transcript}"
    
    submit_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[audio_output, transcript_output, response_text_output, status_message],
        show_progress="full",
        queue=True
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[audio_input, transcript_output, audio_output, response_text_output, status_message]
    )
    
    gr.Markdown(
        """
        <div class="footer">
            🎀 Made with Love by Shofia Nurul Huda 🎀
        </div>
        """
    )

demo.launch()