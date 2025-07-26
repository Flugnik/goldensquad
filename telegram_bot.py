# telegram_bot.py - Версия 4.0. "Золотая команда" в полном сборе.
# Управляет полным циклом: Леон -> Ева -> Адриан.

import logging
import os
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Настройка ---
logging.basicConfig(format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', level=logging.INFO)
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_MSG_LIMIT = 4000

# Адреса наших сервисов в Docker-сети
CHIEF_EDITOR_URL = "http://chief-editor:8000"
COPYWRITER_URL = "http://copywriter:8001"
DESIGNER_URL = "http://designer:8002" # Новый участник

class UserState:
    IDLE, AWAITING_TOPIC_CHOICE = range(2)

# --- Функция для вызова API агентов (без изменений) ---
async def call_api(service_url: str, endpoint: str, payload: dict):
    """Универсальная функция для вызова API наших агентов."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{service_url}/{endpoint}", json=payload, timeout=180.0) # Увеличим таймаут для генерации
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logging.error(f"Ошибка вызова API {e.request.url!r}: {e}")
        return {"error": f"Сетевая ошибка при обращении к агенту."}
    except Exception as e:
        logging.error(f"Неизвестная ошибка при вызове API: {e}")
        return {"error": f"Произошла внутренняя ошибка."}

# --- Основной класс бота ---
class GoldenTeamBot:
    def __init__(self):
        self.app = Application.builder().token(BOT_TOKEN).build()
        self._register_handlers()
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🏆 **Система 'Золотая команда' активна!**\nИспользуйте /create <тема бизнеса> для начала.", parse_mode='Markdown')

    async def create(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        business_theme = " ".join(context.args)
        if not business_theme:
            await update.message.reply_text("Пожалуйста, укажите тему бизнеса. Например: /create Элитные корма для собак")
            return

        context.chat_data['business_theme'] = business_theme
        message = await update.message.reply_text(f"✅ Принято. Связываюсь с Шефом Леоном...")
        
        payload = {"business_theme": business_theme, "count": 5}
        result = await call_api(CHIEF_EDITOR_URL, "generate_topics", payload)

        if "error" not in result and "topics" in result:
            context.chat_data['generated_topics'] = result['topics']
            context.chat_data['state'] = UserState.AWAITING_TOPIC_CHOICE
            keyboard = [[InlineKeyboardButton(topic[:60], callback_data=f"topic_idx:{i}")] for i, topic in enumerate(result['topics'])]
            await message.edit_text("🎯 **Леон подготовил идеи. Выберите тему для поста:**", reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await message.edit_text(f"❌ **Ошибка:** {result.get('error', 'Не удалось получить темы от Шефа.')}")

    async def on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        if data.startswith("topic_idx:"):
            if context.chat_data.get('state') != UserState.AWAITING_TOPIC_CHOICE:
                await query.edit_message_text("Эта кнопка больше не активна.", reply_markup=None)
                return

            await query.edit_message_text("Отличный выбор! Команда приступает к работе...", reply_markup=None)
            
            topic_index = int(data.split(":", 1)[1])
            selected_topic = context.chat_data.get('generated_topics', [])[topic_index]
            business_theme = context.chat_data.get('business_theme', '')

            # --- ЭТАП 1: Получаем задание от Шефа ---
            await query.message.reply_text(f"⏳ **Шаг 1/3:** Леон ставит задачу Копирайтеру...")
            task_payload = {"topic": selected_topic, "business_theme": business_theme}
            task_result = await call_api(CHIEF_EDITOR_URL, "create_task", task_payload)

            if "error" in task_result:
                await query.message.reply_text(f"❌ **Ошибка на шаге 1:** {task_result.get('error')}")
                return
            task_for_copywriter = task_result.get('task_for_copywriter')

            # --- ЭТАП 2: Получаем текст от Копирайтера ---
            await query.message.reply_text(f"⏳ **Шаг 2/3:** Ева пишет текст...")
            write_payload = {"task_from_chief": task_for_copywriter}
            post_result = await call_api(COPYWRITER_URL, "write_post", write_payload)

            if "error" in post_result:
                await query.message.reply_text(f"❌ **Ошибка на шаге 2:** {post_result.get('error')}")
                return
            final_text = post_result.get('post_text')

            # --- ЭТАП 3: Получаем изображение от Дизайнера ---
            await query.message.reply_text(f"⏳ **Шаг 3/3:** Адриан создает визуал...")
            image_payload = {
                "task_from_chief": task_for_copywriter,
                "post_text": final_text,
                "topic": selected_topic
            }
            image_result = await call_api(DESIGNER_URL, "create_image", image_payload)

            if "error" in image_result:
                await query.message.reply_text(f"❌ **Ошибка на шаге 3:** {image_result.get('error')}")
                # Даже если картинка не сгенерировалась, покажем текст
                await query.message.reply_text("Но текст уже готов:")
                for i in range(0, len(final_text), TELEGRAM_MSG_LIMIT):
                    await query.message.reply_text(final_text[i:i + TELEGRAM_MSG_LIMIT])
                return

            # --- ФИНАЛ: Показываем готовый пост ---
            final_image_url = image_result.get('image_url')
            await query.message.reply_text("🎉 **Готово! Команда завершила работу.**")
            # Отправляем сначала изображение, потом текст
            await query.message.reply_photo(photo=final_image_url, caption=f"**Тема:** {selected_topic}")
            for i in range(0, len(final_text), TELEGRAM_MSG_LIMIT):
                await query.message.reply_text(final_text[i:i + TELEGRAM_MSG_LIMIT])

            context.chat_data.clear()

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("create", self.create))
        self.app.add_handler(CallbackQueryHandler(self.on_callback))

    def run(self):
        print("🚀 Telegram-бот v4.0 ('Золотая команда' в сборе) запущен!")
        self.app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    GoldenTeamBot().run()
