# telegram_bot.py - Версия 5.5. "Золотая команда" с полной цепочкой: Леон → Ева → Адриан → designer-fal
# Управляет полным циклом с генерацией визуала через Fal.ai.
import logging
import os
import uuid
import asyncio
import httpx
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
from enum import IntEnum

# Загрузка переменных окружения
load_dotenv()

# Импорт системы логирования
try:
    from utils.supabase_logger import logger
except ImportError:
    logger = None
    logging.warning("⚠️ Модуль supabase_logger не найден. Логирование в Supabase отключено.")

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s| %(name)s| %(levelname)s| %(message)s',
    level=logging.INFO
)

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_MSG_LIMIT = 4000

# Адреса наших сервисов в Docker-сети
CHIEF_EDITOR_URL = "http://chief-editor:8000"
COPYWRITER_URL = "http://copywriter:8001"
DESIGNER_URL = "http://designer:8002"

# Путь к результатам
RESULTS_DIR = os.getenv('RESULTS_DIR', '/app/results')
LATEST_VISUAL_PATH = os.path.join(RESULTS_DIR, 'latest_visual.txt')
PENDING_PROMPT_PATH = os.path.join(RESULTS_DIR, 'pending_prompt.txt')

# Класс состояния пользователя
class UserState(IntEnum):
    IDLE = 0
    AWAITING_TOPIC_CHOICE = 1
    WAITING_FOR_COPY = 2
    WAITING_FOR_VISUAL = 3


# Универсальная функция для вызова API агентов с логированием
async def call_api(service_url: str, endpoint: str, payload: dict, topic_id: str = None, user_id: str = None):
    start_time = asyncio.get_event_loop().time()
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(f"{service_url}/{endpoint}", json=payload)
            response.raise_for_status()
            result = response.json()
        elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

        # Логируем успешный вызов
        if logger:
            try:
                logger.log_event(
                    agent_name="telegram-bot",
                    event_type=f"api_call_{endpoint}",
                    input_data={"service": service_url, "endpoint": endpoint, "payload_keys": list(payload.keys())},
                    output_data={"response_keys": list(result.keys()) if isinstance(result, dict) else str(result)[:200]},
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage="api_communication",
                    status="success"
                )
            except Exception as e:
                logging.error(f"❌ Не удалось записать лог успешного API вызова: {e}")

        return result

    except httpx.RequestError as e:
        elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        error_msg = f"Сетевая ошибка при обращении к {service_url}: {str(e)}"
        logging.error(error_msg)

        if logger:
            try:
                logger.log_event(
                    agent_name="telegram-bot",
                    event_type=f"api_call_{endpoint}",
                    input_data={"service": service_url, "endpoint": endpoint, "payload": payload},
                    output_data={"error": str(e)},
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage="api_communication",
                    status="error",
                    error_message=str(e)
                )
            except Exception as log_e:
                logging.error(f"❌ Не удалось записать лог ошибки API: {log_e}")

        return {"error": error_msg}

    except Exception as e:
        elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        error_msg = f"Внутренняя ошибка: {str(e)}"
        logging.error(error_msg)

        if logger:
            try:
                logger.log_event(
                    agent_name="telegram-bot",
                    event_type=f"api_call_{endpoint}",
                    input_data={"service": service_url, "endpoint": endpoint, "payload": payload},
                    output_data={"error": str(e)},
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage="api_communication",
                    status="error",
                    error_message=str(e)
                )
            except Exception as log_e:
                logging.error(f"❌ Не удалось записать лог критической ошибки: {log_e}")

        return {"error": error_msg}


# Основной класс бота
class GoldenTeamBot:
    def __init__(self):
        if not BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в переменных окружения!")
        self.app = Application.builder().token(BOT_TOKEN).build()
        self._register_handlers()

        # Логируем инициализацию
        if logger:
            try:
                logger.log_event(
                    agent_name="telegram-bot",
                    event_type="bot_initialized",
                    input_data={"version": "5.5"},
                    output_data={"services": [CHIEF_EDITOR_URL, COPYWRITER_URL, DESIGNER_URL]},
                    elapsed_ms=0,
                    stage="initialization",
                    status="success"
                )
            except Exception as e:
                logging.error(f"❌ Не удалось записать лог инициализации бота: {e}")

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("create", self.create))
        self.app.add_handler(CallbackQueryHandler(self.on_callback))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        # Логируем старт
        if logger:
            try:
                logger.log_event(
                    agent_name="telegram-bot",
                    event_type="command_start",
                    input_data={"user_id": user_id},
                    output_data={"response": "start_message_sent"},
                    elapsed_ms=0,
                    stage="user_interaction",
                    status="success"
                )
            except Exception as e:
                logging.error(f"❌ Не удалось записать лог команды start: {e}")

        await update.message.reply_text(
            "🏆 **Система 'Золотая команда' v5.5 активна!**\n"
            "Используйте `/create <тема бизнеса>` для начала.",
            parse_mode='Markdown'
        )

    async def create(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        business_theme = " ".join(context.args)
        user_id = str(update.effective_user.id)

        if not business_theme:
            await update.message.reply_text("❌ Укажите тему бизнеса. Пример: `/create свиноводство`")
            return

        # Сохраняем данные
        context.chat_data['business_theme'] = business_theme
        context.chat_data['user_id'] = user_id
        context.chat_data['topic_id'] = str(uuid.uuid4())
        context.chat_data['state'] = UserState.AWAITING_TOPIC_CHOICE  # <-- УСТАНОВКА СОСТОЯНИЯ

        await update.message.reply_text("🔍 Генерирую темы...")

        # 1. Генерация тем через Шефа
        payload = {"business_theme": business_theme}
        topic_result = await call_api(CHIEF_EDITOR_URL, "generate_topics", payload, topic_id=context.chat_data['topic_id'], user_id=user_id)

        if "error" in topic_result:
            await update.message.reply_text(f"❌ Ошибка: {topic_result['error']}")
            return

        topics = topic_result.get("topics", [])
        if not topics:
            await update.message.reply_text("❌ Не удалось сгенерировать темы.")
            return

        context.chat_data['generated_topics'] = topics

        # Отправляем клавиатуру с темами
        keyboard = []
        for i, topic in enumerate(topics):
            keyboard.append([InlineKeyboardButton(topic[:60], callback_data=f"topic:{i}")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("📋 Выберите тему:", reply_markup=reply_markup)

    async def on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()  # Важно: ответить на callback
        data = query.data
        user_id = str(update.effective_user.id)

        # Проверяем состояние
        if context.chat_data.get('state') != UserState.AWAITING_TOPIC_CHOICE:
            await query.edit_message_text("Эта кнопка больше не активна.", reply_markup=None)
            return

        if not data.startswith("topic:"):
            return

        await query.edit_message_text("⚙️ Команда приступает к работе...", reply_markup=None)
        topic_index = int(data.split(":", 1)[1])
        selected_topic = context.chat_data.get('generated_topics', [])[topic_index]
        business_theme = context.chat_data.get('business_theme', '')
        topic_id = context.chat_data.get('topic_id', str(uuid.uuid4))

        # Обновляем состояние
        context.chat_data['selected_topic'] = selected_topic
        context.chat_data['state'] = UserState.WAITING_FOR_COPY

        # 2. Текст от Копирайтера
        text_payload = {"topic": selected_topic, "business_theme": business_theme}
        text_result = await call_api(COPYWRITER_URL, "generate_text", text_payload, topic_id=topic_id, user_id=user_id)

        if "error" in text_result:
            await query.message.reply_text(f"❌ Ошибка на шаге 2 (текст): {text_result['error']}")
            return

        final_text = text_result.get('text', '')
        if not final_text:
            await query.message.reply_text("❌ Не удалось получить текст.")
            return

        # 3. Промпт от Дизайнера
        visual_payload = {"topic": selected_topic, "text": final_text}
        visual_result = await call_api(DESIGNER_URL, "make_visual", visual_payload, topic_id=topic_id, user_id=user_id)

        if "error" in visual_result:
            await query.message.reply_text(f"❌ Ошибка на шаге 3 (промпт): {visual_result['error']}")
            return

        prompt = visual_result.get('visual_result', '')
        if not prompt:
            await query.message.reply_text("❌ Не удалось получить промпт для визуала.")
            return

        # 4. Генерация визуала через designer-fal
        await query.message.reply_text("🖼️ Генерирую визуал через Fal.ai...")

        # Очищаем старый результат
        if os.path.exists(LATEST_VISUAL_PATH):
            os.remove(LATEST_VISUAL_PATH)

        # Передаём промпт в файл
        try:
            with open(PENDING_PROMPT_PATH, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as e:
            await query.message.reply_text(f"❌ Не удалось передать промпт в designer-fal: {e}")
            return

        # Ждём готовности изображения
        image_url = None
        timeout = 120
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if os.path.exists(LATEST_VISUAL_PATH):
                try:
                    with open(LATEST_VISUAL_PATH, "r", encoding="utf-8") as f:
                        image_url = f.read().strip()
                    if image_url:
                        break
                except Exception as e:
                    logging.error(f"❌ Ошибка чтения LATEST_VISUAL_PATH: {e}")
            await asyncio.sleep(2)

        if not image_url:
            await query.message.reply_text("❌ Не удалось получить визуал за отведённое время.")
        else:
            await query.message.reply_photo(photo=image_url, caption="🎨 Готовый визуал")

        # Отправляем текст
        for i in range(0, len(final_text), TELEGRAM_MSG_LIMIT):
            await query.message.reply_text(final_text[i:i + TELEGRAM_MSG_LIMIT])

        # Логируем завершение
        if logger:
            try:
                logger.log_event(
                    agent_name="telegram-bot",
                    event_type="workflow_completed",
                    input_data={"selected_topic": selected_topic, "user_id": user_id},
                    output_data={"text_length": len(final_text), "visual_created": bool(image_url)},
                    elapsed_ms=0,
                    topic_id=topic_id,
                    stage="workflow_completion",
                    status="success",
                    business_theme=business_theme
                )
            except Exception as e:
                logging.error(f"❌ Не удалось записать лог завершения workflow: {e}")

        await query.message.reply_text("🎉 **Готово! Команда завершила работу.**")

    def run(self):
        logging.info("🚀 Telegram-бот v5.5 ('Золотая команда' с полной цепочкой) запущен!")
        self.app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    GoldenTeamBot().run()