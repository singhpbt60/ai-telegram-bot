import asyncio
import io
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from huggingface_hub import InferenceClient
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "Lightricks/LTX-Video")
VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", "fal-ai")
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}

HELP = (
    "नमस्ते! मैं AI Image/Video generator bot हूँ.\n\n"
    "• सिर्फ़ prompt भेजें -> image\n"
    "• /image <prompt> -> image\n"
    "• /video <prompt> -> छोटा video (1-3 मिनट लग सकते हैं)\n\n"
    "Prompt English में लिखने पर result बेहतर आता है."
)


def make_image(prompt: str) -> bytes:
    client = InferenceClient(token=HF_TOKEN)
    img = client.text_to_image(prompt, model=IMAGE_MODEL)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_video(prompt: str) -> bytes:
    client = InferenceClient(provider=VIDEO_PROVIDER, api_key=HF_TOKEN)
    return client.text_to_video(prompt, model=VIDEO_MODEL)


def friendly_error(e: Exception) -> str:
    m = str(e)
    if "402" in m or "credit" in m.lower():
        return "Free credits खत्म हो गए हैं. कुछ देर बाद try करें."
    if "429" in m:
        return "Rate limit लग गई. थोड़ी देर बाद try करें."
    if "401" in m or "403" in m:
        return "Hugging Face token गलत है या permission नहीं है."
    if "503" in m or "loading" in m.lower():
        return "Model load हो रहा है. 1 मिनट बाद दोबारा try करें."
    return f"Error: {m[:300]}"


def allowed(update: Update) -> bool:
    return not ALLOWED or (update.effective_user and update.effective_user.id in ALLOWED)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return await update.message.reply_text(f"Access नहीं है. आपकी ID: {update.effective_user.id}")
    await update.message.reply_text(HELP)


async def do_image(update: Update, prompt: str):
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        data = await asyncio.to_thread(make_image, prompt)
        await update.message.reply_photo(photo=data, caption=prompt[:200], write_timeout=120)
    except Exception as e:
        log.exception("image failed")
        await update.message.reply_text(friendly_error(e))


async def do_video(update: Update, prompt: str):
    await update.message.reply_text("Video बन रहा है, कृपया इंतज़ार करें...")
    await update.message.chat.send_action(ChatAction.UPLOAD_VIDEO)
    try:
        data = await asyncio.to_thread(make_video, prompt)
        await update.message.reply_video(video=data, caption=prompt[:200], write_timeout=180)
    except Exception as e:
        log.exception("video failed")
        await update.message.reply_text(friendly_error(e))


async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    prompt = " ".join(context.args).strip()
    if not prompt:
        return await update.message.reply_text("ऐसे लिखें: /image a cat astronaut on the moon")
    await do_image(update, prompt)


async def video_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    prompt = " ".join(context.args).strip()
    if not prompt:
        return await update.message.reply_text("ऐसे लिखें: /video a sailing boat at sunset")
    await do_video(update, prompt)


async def text_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await do_image(update, update.message.text.strip())


def start_health_server():
    port = os.getenv("PORT")
    if not port:
        return

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    threading.Thread(target=HTTPServer(("0.0.0.0", int(port)), H).serve_forever, daemon=True).start()


def main():
    start_health_server()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("image", image_cmd))
    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_msg))
    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
