import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("[ERROR] GEMINI_API_KEY not found in environment variables")

MODEL = "gemini-2.0-flash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)

system_instruction = """ 
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian. 
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice. 
 
Your answers must: 
- Be written in polite and easily understandable Indonesian. 
- Be short and to the point (maximum 2–3 sentences). 
- Avoid repeating the user's question; respond directly with the answer. 
 
Example tone: 
User: Cuaca hari ini gimana? 
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat. 
 
User: Kamu tahu siapa presiden Indonesia? 
Assistant: Presiden Indonesia saat ini adalah Joko Widodo. 
 
If you're unsure about an answer, be honest and say that you don't know. 
""" 

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name=MODEL)
    print(f"[INFO] Successfully configured Gemini API with model: {MODEL}")
except Exception as e:
    print(f"[ERROR] Failed to configure Gemini API: {e}")
    model = None

def save_chat_history(chat):
    try:
        # Convert chat history to a serializable format
        history = []
        for message in chat.history:
            history.append({
                "role": message.role,
                "parts": [part.text for part in message.parts if hasattr(part, 'text')]
            })
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        print(f"[INFO] Chat history saved to {CHAT_HISTORY_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save chat history: {e}")

def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_FILE) and os.path.getsize(CHAT_HISTORY_FILE) > 0:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            print(f"[INFO] Chat history loaded from {CHAT_HISTORY_FILE}")
            return model.start_chat(history=history)
    except Exception as e:
        print(f"[ERROR] Failed to load chat history: {e}")
    
    print("[INFO] Starting new chat session")
    return model.start_chat()

try:
    chat = load_chat_history()
except Exception as e:
    print(f"[ERROR] Failed to initialize chat: {e}")
    chat = None

def generate_response(prompt: str) -> str:
    if not model or not chat:
        return "[ERROR] Gemini API not properly initialized"
        
    try:
        print(f"[INFO] Processing user prompt: '{prompt}'")
        
        if not chat.history:
            print("[INFO] Sending system instruction to new chat")
            chat.send_message(system_instruction)
        
        response = chat.send_message(prompt)
        response_text = response.text.strip()
        
        print(f"[INFO] Gemini response: '{response_text}'")
        save_chat_history(chat)
        
        return response_text
    except Exception as e:
        print(f"[ERROR] Failed to generate response: {e}")
        return f"[ERROR] {e}"

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
import os
import tempfile
import logging
from typing import Optional
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice-assistant")

app = FastAPI(title="Voice AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Voice AI Assistant API is running"}

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    """
    Process voice chat workflow:
    1. Receive audio file from frontend
    2. Convert speech to text using Whisper
    3. Generate response using Gemini
    4. Convert response to speech
    5. Return audio file, transcript, and response text
    """
    logger.info(f"Received request from {request.client.host} with file: {file.filename}")
    logger.info(f"Request headers: {request.headers}")
    
    contents = await file.read()
    file_size = len(contents)
    logger.info(f"File received: {file_size} bytes")
    
    if not contents or file_size == 0:
        logger.error("Empty file received")
        return JSONResponse(
            status_code=400,
            content={"error": "Empty file", "transcript": "", "response_text": ""}
        )
    
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    logger.info("Starting speech-to-text processing")
    transcript = transcribe_speech_to_text(contents, file_ext=os.path.splitext(file.filename)[1])
    if transcript.startswith("[ERROR]"):
        logger.error(f"STT error: {transcript}")
        return JSONResponse(
            status_code=500,
            content={"error": transcript, "transcript": transcript, "response_text": ""}
        )
    
    logger.info(f"Transcribed: {transcript}")
    
    logger.info("Generating LLM response")
    response_text = generate_response(transcript)
    if response_text.startswith("[ERROR]"):
        logger.error(f"LLM error: {response_text}")
        return JSONResponse(
            status_code=500,
            content={"error": response_text, "transcript": transcript, "response_text": response_text}
        )
    
    logger.info(f"LLM Response: {response_text}")
    
    logger.info("Starting text-to-speech processing")
    audio_path = transcribe_text_to_speech(response_text)
    if not audio_path or audio_path.startswith("[ERROR]"):
        logger.error(f"TTS error: {audio_path}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate speech: {audio_path}", "transcript": transcript, "response_text": response_text}
        )
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found at: {audio_path}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Audio file not found at: {audio_path}", "transcript": transcript, "response_text": response_text}
        )
    
    file_size = os.path.getsize(audio_path)
    logger.info(f"Audio file saved at: {audio_path} ({file_size} bytes)")
    
    # Baca file audio dan konversi ke base64 agar bisa dikirim dalam JSON
    with open(audio_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
    
    logger.info(f"Sending response with audio, transcript, and response text")
    return {
        "audio": audio_data,
        "audio_filename": "response.wav",
        "transcript": transcript,
        "response_text": response_text
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)