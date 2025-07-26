# telegram_bot_fixed.py - Версия с надежной JSON-коммуникацией

import asyncio
import logging
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Настройка (без изменений) ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
BOT_TOKEN = "8135072706:AAELQci6TsMPUhp6EBKnpqj6PjVtJsXilt8"

# --- Анимация (без изменений) ---
class ProgressIndicator:
    @staticmethod
    async def show_progress(query, steps, title):
        for i, step in enumerate(steps):
            progress_bar = "▓" * (i + 1) + "░" * (len(steps) - i - 1)
            message = f"⏳ **{title}**\n\n{step}\n\n**Прогресс:** [{progress_bar}] {i + 1}/{len(steps)}"
            await query.edit_message_text(message, parse_mode='Markdown')
            await asyncio.sleep(1)

# --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Функция вызова Docker с JSON ---

async def call_chief_editor(command, *args):
    """Вызывает метод у Шеф-редактора и получает ответ в формате JSON."""
    base_command = ["docker-compose", "exec", "-T", "chief-editor", "python", "-c"]
    
    # Мы просим AI-агента обернуть свой ответ в JSON
    python_code = f"""
import asyncio
import json
from chief_editor import ChiefEditor

editor = ChiefEditor()
result = asyncio.run(editor.{command}({', '.join(map(repr, args))}))
# Печатаем результат как JSON-строку для надежной передачи
print(json.dumps(result, ensure_ascii=False))
"""
    try:
        print(f"🚀 Выполняю команду в Docker: {command}")
        process = await asyncio.create_subprocess_exec(
            *base_command, python_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            response_str = stdout.decode('utf-8').strip()
            print(f"✅ Получен JSON от Docker: {response_str}")
            try:
                # Используем безопасный json.loads вместо eval()
                return json.loads(response_str)
            except json.JSONDecodeError:
                return {"error": "Ответ от AI-агента не является корректным JSON."}
        else:
            error_message = stderr.decode('utf-8').strip()
            print(f"❌ Ошибка Docker: {error_message}")
            return {"error": error_message or "Неизвестная ошибка выполнения."}
    except Exception as e:
        print(f"❌ Критическая ошибка вызова Docker: {e}")
        return {"error": str(e)}

# --- Обработчики команд (без изменений, т.к. логика вынесена) ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏆 **Система 'Золотая команда' активна!**\nИспользуйте /create для создания поста.", parse_mode='Markdown')

async def create_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /create теперь работает с надежным JSON."""
    message = await update.message.reply_text("🧠 **Шеф-редактор придумывает темы...**\nПожалуйста, подождите 10-15 секунд.", parse_mode='Markdown')
    topics = await call_chief_editor("generate_topics")

    if isinstance(topics, list) and topics:
        keyboard = [
            [InlineKeyboardButton(topic[:60] + '...' if len(topic) > 60 else topic, callback_data=f"select_topic:{topic}")]
            for topic in topics
        ]
        await message.edit_text("🎯 **Шеф-редактор предлагает следующие темы:**", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        error_info = topics.get('error', 'Некорректный формат ответа') if isinstance(topics, dict) else str(topics)
        await message.edit_text(f"❌ **Не удалось сгенерировать темы.**\n`{error_info}`", parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок."""
    query = update.callback_query
    await query.answer()

    if query.data.startswith("select_topic:"):
        selected_topic = query.data.split(":", 1)[1]
        steps = [f"Создаю текст на тему: '{selected_topic[:30]}...'", "Генерирую идею для изображения...", "✅ Пост готов!"]
        await ProgressIndicator.show_progress(query, steps, "Создание поста")
        result = await call_chief_editor("create_post_by_topic", selected_topic)

        if isinstance(result, dict) and "error" not in result:
            await query.edit_message_text(f"✅ **Пост на тему '{selected_topic}' создан!**\n📁 Текст сохранен в: `{result.get('text_path')}`", parse_mode='Markdown')
        else:
            await query.edit_message_text(f"❌ **Ошибка при создании поста.**\n`{result.get('error', 'Проверьте логи Docker')}`", parse_mode='Markdown')

def main():
    print("🤖 Запуск обновленного Telegram-бота...")
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("create", create_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    print("🚀 Бот готов к работе!")
    application.run_polling()

if __name__ == "__main__":
    main()
