# designer.py - Версия 2.0. Адриан, "Маэстро Визуала" с API на FastAPI.
# Готов принимать задания от команды и создавать визуальные образы.

import asyncio
import os
import sys
import fal_client
import requests
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Модели данных для API (Pydantic) ---
class CreateImageRequest(BaseModel):
    task_from_chief: str
    post_text: str
    topic: str # Добавляем тему для сохранения файла

# --- Основной класс агента (логика остается прежней) ---
class Designer:
    def __init__(self):
        self.generation_model = "fal-ai/flux-pro"
        self.personality = self.load_personality()
        print(f"✅ Дизайнер v2.0 ({self.personality['name']}) инициализирован. Инструмент: {self.generation_model}", file=sys.stderr)
        self.fal_api_key = os.getenv('FAL_KEY')
        self.constitution = self.personality.get('constitution')

    def load_personality(self):
        """Загружает "слепок личности" Дизайнера."""
        return {
            "name": "Адриан, 'Маэстро Визуала'",
            "constitution": """Ты — Дизайнер Адриан, «Маэстро Визуала». Твоя миссия — переводить стратегию и текст в мощный визуальный язык. Ты создаешь не картинки, а образы, которые усиливают эмоциональное воздействие.

Твои принципы:
1.  **Визуальный сторителлинг:** Ты не иллюстрируешь, а рассказываешь историю.
2.  **Эстетический интеллект:** У тебя врожденное чувство стиля.
3.  **Глубина и метафора:** Ты ищешь не буквальные, а метафорические образы.
4.  **Техническое мастерство:** Ты в совершенстве владеешь передовыми инструментами генерации."""
        }

    async def create_image(self, task_from_chief, post_text):
        """Создает визуал на основе задания от Шефа и текста от Копирайтера."""
        generation_prompt = f"""
        **Задание от Шефа-редактора:** {task_from_chief}
        **Текст от Копирайтера:** "{post_text}"

        **Твоя задача, Маэстро:**
        Проанализируй суть и эмоцию этого текста. Найди центральную метафору. Создай фотореалистичное изображение, которое станет визуальным воплощением этой идеи.
        """
        print(f"🎨 Маэстро Адриан получил задание. Отправляю промпт в {self.generation_model}...", file=sys.stderr)
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: fal_client.run(self.generation_model, arguments={"prompt": generation_prompt})
            )
            image_url = result['images'][0]['url']
            print(f"✅ Изображение сгенерировано: {image_url}", file=sys.stderr)
            return {"image_url": image_url}
        except Exception as e:
            return {"error": str(e)}

    async def save_approved_image(self, image_url, topic):
        """Скачивает и сохраняет одобренное изображение."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/ready_for_publish/image_{topic.replace(' ', '_')}_{timestamp}.png"
        # ... (логика сохранения файла без изменений) ...
        return {"image_path": filename}

# --- Создание экземпляров FastAPI и нашего агента ---
app = FastAPI()
designer = Designer()

# --- API Endpoints (точки входа) ---
@app.post("/create_image")
async def api_create_image(request: CreateImageRequest):
    """API для создания изображения."""
    result = await designer.create_image(request.task_from_chief, request.post_text)
    # В будущем здесь будет логика согласования, а пока просто возвращаем URL
    return result

# --- Точка запуска сервера ---
if __name__ == "__main__":
    print("🚀 Запускаю сервер Дизайнера Адриана...")
    uvicorn.run(app, host="0.0.0.0", port=8002) # Используем порт 8002
