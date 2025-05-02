import os
import time
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv
import requests
import deepspeech
import numpy as np
from tts import TTS
import ffmpeg
import tempfile
import logging

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configurações
LEVELS = {
    "introductory": "Simple phrases, basic vocabulary",
    "beginner": "Short sentences, everyday topics",
    "preintermediate": "Moderate complexity, varied topics",
    "intermediate": "Complex sentences, broader topics",
    "advanced": "Fluent, nuanced responses"
}
DAILY_LIMIT = 20  # Interações ou minutos por dia
INACTIVITY_TIMEOUT = 120  # 2 minutos
DEEPSPEECH_MODEL = "deepspeech-0.9.3-models.pbmm"
DEEPSPEECH_SCORER = "deepspeech-0.9.3-models.scorer"
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

# Estado do usuário
user_data = {}  # {user_id: {"level": str, "interactions": int, "start_time": datetime, "last_active": float, "reset_date": date}}

# Inicializar DeepSpeech
ds = deepspeech.Model(DEEPSPEECH_MODEL)
ds.enableExternalScorer(DEEPSPEECH_SCORER)

# Inicializar TTS
tts = TTS(model_name=TTS_MODEL)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data[user_id] = {
        "level": None,
        "interactions": 0,
        "start_time": None,
        "last_active": time.time(),
        "reset_date": datetime.now().date()
    }
    await update.message.reply_text(
        "Hi, welcome here. I'm Mr. Teachwell, your e-teacher! Please choose your level so we get started: /introductory, /beginner, /preintermediate, /intermediate, /advanced"
    )

async def set_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    command = update.message.text.lstrip("/").lower()
    if command not in LEVELS:
        await update.message.reply_text("Invalid level. Use: /introductory, /beginner, /preintermediate, /intermediate, /advanced")
        return

    if user_id not in user_data:
        user_data[user_id] = {
            "level": command,
            "interactions": 0,
            "start_time": None,
            "last_active": time.time(),
            "reset_date": datetime.now().date()
        }
    else:
        user_data[user_id]["level"] = command
        user_data[user_id]["last_active"] = time.time()

    await update.message.reply_text(f"Level set to {command.capitalize()}. Let's practice! Send a message or voice note.")

async def check_limits(user_id):
    now = datetime.now()
    user = user_data.get(user_id)
    if not user:
        return False

    # Reset diário
    if user["reset_date"] != now.date():
        user["interactions"] = 0
        user["start_time"] = None
        user["reset_date"] = now.date()

    # Checar interações
    if user["interactions"] >= DAILY_LIMIT:
        return False

    # Checar tempo
    if user["start_time"]:
        elapsed = (now - user["start_time"]).total_seconds() / 60
        if elapsed >= DAILY_LIMIT:
            return False

    return True

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = time.time()

    if user_id not in user_data or user_data[user_id]["level"] is None:
        await update.message.reply_text("Please set your level: /introductory, /beginner, /preintermediate, /intermediate, /advanced")
        return

    user = user_data[user_id]

    # Checar inatividade
    if now - user["last_active"] > INACTIVITY_TIMEOUT:
        await update.message.reply_text("Session timed out. Let's continue!")
        user["last_active"] = now
        return

    # Checar limites
    if not await check_limits(user_id):
        await update.message.reply_text(
            "You've used your daily limit (20 interactions or 20 minutes). Come back tomorrow!"
        )
        return

    # Atualizar estado
    user["interactions"] += 1
    user["last_active"] = now
    if user["start_time"] is None:
        user["start_time"] = datetime.now()

    # Processar mensagem
    text = update.message.text
    try:
        # Detectar português
        if any(word in text.lower() for word in ["oi", "olá", "como", "está", "por", "favor"]):
            await update.message.reply_text("Please use English only!")
            return

        # Chamar Gemini API
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"Respond in English, short (max 50 words), appropriate for {user['level']} level: {text}"
                            }
                        ]
                    }
                ]
            },
            params={"key": GEMINI_API_KEY}
        )
        response.raise_for_status()
        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Responder
        await update.message.reply_text(answer)

        # Gerar voz (TTS)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            tts.tts_to_file(text=answer, file_path=temp_wav.name)
            await update.message.reply_voice(voice=open(temp_wav.name, "rb"))
            os.unlink(temp_wav.name)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text("Sorry, something went wrong. Try again!")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = time.time()

    if user_id not in user_data or user_data[user_id]["level"] is None:
        await update.message.reply_text("Please set your level: /introductory, /beginner, /preintermediate, /intermediate, /advanced")
        return

    user = user_data[user_id]

    # Checar inatividade
    if now - user["last_active"] > INACTIVITY_TIMEOUT:
        await update.message.reply_text("Session timed out. Let's continue!")
        user["last_active"] = now
        return

    # Checar limites
    if not await check_limits(user_id):
        await update.message.reply_text(
            "You've used your daily limit (20 interactions or 20 minutes). Come back tomorrow!"
        )
        return

    # Atualizar estado
    user["interactions"] += 1
    user["last_active"] = now
    if user["start_time"] is None:
        user["start_time"] = datetime.now()

    # Processar voz
    try:
        voice_file = await update.message.voice.get_file()
        voice_path = await voice_file.download_to_drive()

        # Converter OGG pra WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            stream = ffmpeg.input(str(voice_path))
            stream = ffmpeg.output(stream, temp_wav.name, format="wav", acodec="pcm_s16le", ar="16000")
            ffmpeg.run(stream)

            # DeepSpeech (voz → texto)
            audio = np.frombuffer(open(temp_wav.name, "rb").read(), dtype=np.int16)
            text = ds.stt(audio)

            # Detectar português
            if any(word in text.lower() for word in ["oi", "olá", "como", "está", "por", "favor"]):
                await update.message.reply_text("Please speak in English!")
                os.unlink(temp_wav.name)
                os.unlink(str(voice_path))
                return

            # Chamar Gemini API
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": f"Respond in English, short (max 50 words), appropriate for {user['level']} level: {text}"
                                }
                            ]
                        }
                    ]
                ],
                params={"key": GEMINI_API_KEY}
            )
            response.raise_for_status()
            answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            # Responder em texto
            await update.message.reply_text(answer)

            # Gerar voz (TTS)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_tts:
                tts.tts_to_file(text=answer, file_path=temp_wav_tts.name)
                await update.message.reply_voice(voice=open(temp_wav_tts.name, "rb"))
                os.unlink(temp_wav_tts.name)

            # Limpar arquivos
            os.unlink(temp_wav.name)
            os.unlink(str(voice_path))

    except Exception as e:
        logger.error(f"Error processing voice: {e}")
        await update.message.reply_text("Sorry, something went wrong. Try again!")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("introductory", set_level))
    application.add_handler(CommandHandler("beginner", set_level))
    application.add_handler(CommandHandler("preintermediate", set_level))
    application.add_handler(CommandHandler("intermediate", set_level))
    application.add_handler(CommandHandler("advanced", set_level))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_error_handler(error_handler)

    # Iniciar com webhook
    application.run_webhook(
        listen="0.0.0.0",
        port=8080,
        url_path="/webhook",
        webhook_url="https://mrteachwell.koyeb.app/webhook"
    )

if __name__ == "__main__":
    main()