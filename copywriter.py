# copywriter.py - Версия 2.0. Ева Соколова с API на FastAPI.
# Готова принимать задания от Шефа и воплощать их в "живое слово".

import asyncio
import os
import json
import sys
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Модели данных для API (Pydantic) ---
class WritePostRequest(BaseModel):
    task_from_chief: str
    brand_voice_info: str = "" # Пока оставим пустым, в будущем здесь будет профиль клиента

# --- Основной класс агента (логика остается прежней) ---
class Copywriter:
    def __init__(self):
        self.model_name = os.getenv('COPYWRITER_MODEL', 'claude-4-sonnet-20250501')
        self.personality = self.load_personality()
        print(f"✅ Копирайтер v2.0 ({self.personality['name']}) инициализирована. Модель: {self.model_name}", file=sys.stderr)
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)
        self.constitution = self.personality.get('constitution')

    def load_personality(self):
        """Загружает "слепок личности" Копирайтера."""
        return {
            "name": "Ева Соколова, 'Живое Слово'",
            "constitution": """Ты — Копирайтер Ева Соколова, твой архетип — «Живое Слово». Твоя миссия — вдыхать жизнь, душу и личность в любой текст, который ты создаешь по заданию Шефа Леона.

Твои принципы:
1.  **Эмоциональная гибкость:** Ты мастерски владеешь всем спектром эмоций и стилей. Ты можешь написать и энергичный рекламный пост, и теплый, душевный рассказ.
2.  **Глубина в любой задаче:** Ты убеждена, что "душа" есть даже в самом утилитарном тексте. В любом задании ты ищешь человеческую деталь.
3.  **Воплощение замысла:** Ты видишь в заданиях Шефа не сухое ТЗ, а стратегический замысел, который нужно воплотить.
4.  **Стиль:** Ты пишешь просто, понятно и с душой, избегая канцеляризмов и всегда заботясь о читателе."""
        }

    async def write_post(self, task_from_chief, brand_voice_info):
        """Пишет текст поста на основе задания от Шефа и голоса бренда клиента."""
        prompt = f"""Коллега, вот новое задание от Шефа-редактора.

**Задание:**
{task_from_chief}

**Информация о "Голосе Бренда" клиента (если есть):**
{brand_voice_info}

Пожалуйста, напиши текст поста, руководствуясь своей личностью. Верни только готовый текст.
"""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=2500,
                system=self.constitution,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"post_text": response.content[0].text.strip()}
        except Exception as e:
            return {"error": str(e)}

# … существующий код выше опущен …

# --- Создание экземпляров FastAPI и нашего агента ---
app = FastAPI()
writer = Copywriter()

# --- Новый health-эндпоинт -----------------------------------------------
@app.get("/health", tags=["system"])
async def health():
    """Проверка работоспособности сервера Евы."""
    return {"status": "ok"}

# --- API Endpoints (точки входа) -----------------------------------------
@app.post("/write_post")
async def api_write_post(request: WritePostRequest):
    """API для написания поста."""
    result = await writer.write_post(request.task_from_chief, request.brand_voice_info)
    return result

# --- Точка запуска сервера -----------------------------------------------
if __name__ == "__main__":
    print("🚀 Запускаю сервер Копирайтера Евы…")
    uvicorn.run(app, host="0.0.0.0", port=8001)


# --- API Endpoints (точки входа) ---
@app.post("/write_post")
async def api_write_post(request: WritePostRequest):
    """API для написания поста."""
    result = await writer.write_post(request.task_from_chief, request.brand_voice_info)
    return result

# --- Точка запуска сервера ---
if __name__ == "__main__":
    print("🚀 Запускаю сервер Копирайтера Евы...")
    uvicorn.run(app, host="0.0.0.0", port=8001) # Используем другой порт, чтобы не конфликтовать с Шефом
