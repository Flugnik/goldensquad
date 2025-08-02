"""ростейший Telegram-бот олотой команды."""
import os, logging
from fastapi import FastAPI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "dummy")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Golden Squad Bot")

@app.on_event("startup")
async def on_startup() -> None:
    bot = Application.builder().token(BOT_TOKEN).build()
    bot.add_handler(CommandHandler("start", start))
    await bot.initialize()
    await bot.start()
    logging.info("Telegram-bot started")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🟡 Golden Squad: бот запущен!")

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}
