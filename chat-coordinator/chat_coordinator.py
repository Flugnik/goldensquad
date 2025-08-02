# chat_coordinator.py - Чистый координатор без DNA логики
import asyncio
import aiohttp
import json
import re
import os
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from telegram import Update, Message
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoldenSquadChat:
    def __init__(self):
        # Подключение к агентам Золотой команды
        self.agents = {
            "leon": "http://localhost:8000",      # Шеф-редактор
            "alina": "http://localhost:8001",     # Копирайтер
            "adrian": "http://localhost:8002"     # Дизайнер
        }

        # Русские имена агентов
        self.agent_names = {
            "leon": "Леон",
            "alina": "Алина",
            "adrian": "Адриан"
        }

        # Контекст пользователей
        self.user_contexts = {}

        logger.info(
            "✅ Golden Squad Chat инициализирован с агентами: Леон, Алина, Адриан")

    def parse_mentions(self, message: str) -> Dict[str, str]:
        """Парсит @-упоминания агентов"""
        mentions = {}

        agent_patterns = {
            r'@(?:leon|леон|лион|Леон|шеф|Шеф|chief|главный)': 'leon',
            r'@(?:alina|алина|Алина|copywriter|копирайтер|Копирайтер)': 'alina',
            r'@(?:adrian|адриан|Адриан|designer|дизайнер|Дизайнер)': 'adrian',
            r'@(?:team|команда|Команда|все|всем|всей команде)': 'team'
        }

        for pattern, agent in agent_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                if agent == 'team':
                    for a in ['leon', 'alina', 'adrian']:
                        mentions[a] = message
                else:
                    mentions[agent] = message

        return mentions

    async def call_agent(self, agent: str, message: str) -> str:
        """Отправляет запрос агенту - агент сам определяет клиента"""
        url = self.agents.get(agent)
        if not url:
            return f"❌ Агент {agent} недоступен"

        try:
            if agent == "leon":
                data = {"message": message}
                endpoint = "/chat"

            elif agent == "alina":
                data = {"task_from_chief": message}
                endpoint = "/write_post"

            elif agent == "adrian":
                data = {
                    "task_from_chief": message,
                    "post_text": message[:100],
                    "topic": "user_request"
                }
                endpoint = "/create_image"

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{url}{endpoint}", json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return f"**{self.agent_names[agent]}:** {result.get('response', str(result))}"
                    else:
                        return f"❌ **{self.agent_names[agent]}:** Ошибка обработки запроса"

        except Exception as e:
            logger.error(f"Ошибка вызова агента {agent}: {e}")
            return f"❌ **{self.agent_names[agent]}:** Временно недоступен"


# Создаем экземпляр чата
golden_chat = GoldenSquadChat()

# Telegram Bot Handler


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик сообщений Telegram"""
    user_id = update.effective_user.id
    message = update.message.text

    logger.info(f"Получено сообщение от {user_id}: {message[:50]}...")

    mentions = golden_chat.parse_mentions(message)

    if not mentions:
        active_agent = golden_chat.user_contexts.get(user_id, "leon")
        response = await golden_chat.call_agent(active_agent, message)
        golden_chat.user_contexts[user_id] = active_agent
    else:
        responses = []
        for agent in mentions.keys():
            response = await golden_chat.call_agent(agent, message)
            responses.append(response)
        response = "\n\n".join(responses)

    try:
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(response)

# FastAPI для мониторинга
app = FastAPI(title="Golden Squad Chat Coordinator", version="2.0")


@app.get("/health")
async def health():
    agent_status = {}
    for agent, url in golden_chat.agents.items():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    agent_status[agent] = "healthy" if response.status == 200 else "unhealthy"
        except:
            agent_status[agent] = "unreachable"

    return {
        "status": "ok",
        "service": "chat-coordinator",
        "version": "2.0",
        "agents": agent_status
    }

# Основная функция


async def main():
    TELEGRAM_TOKEN = os.getenv(
        "TELEGRAM_BOT_TOKEN", "8135072706:AAELQci6TsMPUhp6EBKnpqj6PjVtJsXilt8")

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Запускаю Chat Coordinator Золотой команды v2.0...")
    logger.info(f"📊 Подключены агенты: {list(golden_chat.agents.keys())}")

    await application.run_polling(drop_pending_updates=True)

# chat_coordinator.py - финальная версия БЕЗ Telegram threading
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Golden Squad Chat Coordinator v2.0 стартует...")
    logger.info(f"📊 Подключены агенты: {list(golden_chat.agents.keys())}")

    # Запускаем только FastAPI - стабильно и надежно
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
