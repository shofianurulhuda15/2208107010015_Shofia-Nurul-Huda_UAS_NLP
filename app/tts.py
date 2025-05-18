import os
import uuid
import tempfile
import subprocess
import wave
import re
from g2p_id import G2P
from num2words import num2words

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke folder utilitas TTS
COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")

# Path ke file model TTS
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")

# Path ke file konfigurasi
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")

# Nama speaker yang digunakan
COQUI_SPEAKER = "ardi"

# Inisialisasi g2p-id
g2p = G2P()

def transcribe_text_to_speech(text: str) -> str:
    """
    Fungsi untuk mengonversi teks menjadi suara menggunakan TTS engine yang ditentukan.
    Args:
        text (str): Teks yang akan diubah menjadi suara.
    Returns:
        str: Path ke file audio hasil konversi.
    """
    path = _tts_with_coqui(text)
    return path

# === ENGINE 1: Coqui TTS ===
def _tts_with_coqui(text: str) -> str:
    # Create a more permanent temp directory (not in /tmp which may be cleaned up)
    output_dir = os.path.join(tempfile.gettempdir(), "voice_assistant_tts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create unique filename
    output_path = os.path.join(output_dir, f"tts_{uuid.uuid4()}.wav")

    # Langkah 1: Konversi angka ke teks (misalnya, "17" menjadi "tujuh belas")
    def convert_numbers_to_words(text):
        def replace_number(match):
            num_str = match.group(0)
            # Hapus tanda titik atau koma untuk mendapatkan angka murni
            clean_num = num_str.replace(".", "").replace(",", "")
            try:
                # Konversi angka ke kata dalam Bahasa Indonesia
                return num2words(int(clean_num), lang='id')
            except ValueError:
                return num_str  # Kembalikan asli jika bukan angka valid
        
        # Ganti semua angka dalam teks, termasuk yang punya pemisah ribuan
        pattern = r'\b\d{1,3}(?:[,.]\d{3})*\b'
        return re.sub(pattern, replace_number, text)

    processed_text = convert_numbers_to_words(text)
    print(f"[INFO] Text after number conversion: {processed_text}")

    # Langkah 2: Coba konversi ke fonem IPA menggunakan g2p-id
    try:
        phonemes = g2p(processed_text)
        print(f"[INFO] Phonemes generated: {phonemes}")
        input_text = phonemes
    except Exception as e:
        print(f"[WARNING] G2P conversion failed: {e}. Falling back to processed text.")
        input_text = processed_text  # Fallback ke teks asli jika G2P gagal

    # Langkah 3: Gunakan input teks (fonem IPA atau teks asli) untuk Coqui TTS
    cmd = [
        "tts",
        "--text", input_text,
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--speaker_idx", COQUI_SPEAKER,
        "--out_path", output_path
    ]
    
    try:
        # Atur encoding UTF-8 untuk subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        print(f"[INFO] Running TTS command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env
        )
        print(f"[INFO] TTS stdout: {result.stdout}")
        
        # Validasi file WAV benar-benar valid
        if not os.path.exists(output_path):
            print(f"[ERROR] TTS output file not created: {output_path}")
            return "[ERROR] TTS failed to create output file"
            
        if os.path.getsize(output_path) == 0:
            print(f"[ERROR] TTS output file is empty: {output_path}")
            return "[ERROR] TTS created empty file"
            
        with wave.open(output_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            framerate = wav_file.getframerate()
            frames = wav_file.getnframes()
            print(f"[INFO] Valid WAV: {channels} ch, {framerate} Hz, {frames} frames")
            
            # Sanity check to make sure the file has actual audio content
            if frames < 100:  # Arbitrary small number to detect essentially empty files
                print(f"[WARNING] WAV file has very few frames: {frames}")
                
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TTS subprocess failed: {e}")
        print(f"[ERROR] TTS stderr: {e.stderr}")
        return "[ERROR] Failed to synthesize speech"
    except wave.Error as e:
        print(f"[ERROR] Invalid WAV file generated: {e}")
        return "[ERROR] Invalid WAV file"
    except FileNotFoundError as e:
        print(f"[ERROR] File not found during TTS: {e}")
        return "[ERROR] File not found"
    except Exception as e:
        print(f"[ERROR] Unexpected error in TTS: {str(e)}")
        return f"[ERROR] {str(e)}"

    print(f"[INFO] Output audio saved to: {output_path}")
    return output_path