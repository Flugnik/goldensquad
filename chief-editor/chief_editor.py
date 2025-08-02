# chief_editor.py — Версия 4.3. Шеф Леон с OpenRouter интеграцией
# Независимый микросервис с Google Gemini 2.5 Flash через OpenRouter API

import traceback
import sys
import asyncio
import json
import os
import re
import logging
from typing import List, Optional
import aiohttp

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# --------------------------------------------------------------------------- #
#                     Конфигурация и константы                                #
# --------------------------------------------------------------------------- #
class Config:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    MAX_JSON_SIZE: int = 10000
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("chief_editor")

# --------------------------------------------------------------------------- #
#                     Pydantic-модели для входящих запросов                   #
# --------------------------------------------------------------------------- #
class TopicsRequest(BaseModel):
    business_theme: str = Field(..., min_length=2, max_length=120)
    count: int = Field(5, ge=1, le=10)

class TextDraftRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120)
    business_theme: Optional[str] = ""

# --------------------------------------------------------------------------- #
#                              Логика Шефа Леона                              #
# --------------------------------------------------------------------------- #
class ChiefEditor:
    """
    Шеф-редактор Леон с OpenRouter интеграцией для Google Gemini 2.5 Flash
    """
    def __init__(self) -> None:
        # Используем Google Gemini 2.5 Flash через OpenRouter
        self.model_name = "google/gemini-2.5-flash"
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY не найден в переменных окружения!")

        if not self.openrouter_api_key.startswith('sk-or-'):
            logger.warning("API ключ может быть некорректным (ожидается sk-or-*)")

        self.personality = self._load_personality()
        self.constitution = self.personality["constitution"]

        logger.info(
            f"✅ Шеф-редактор v4.3 ({self.personality['name']}) "
            f"инициализирован с OpenRouter моделью: {self.model_name}"
        )

    # ---------- личность --------------------------------------------------- #
    @staticmethod
    def _load_personality() -> dict:
        return {
            "name": "Леон Мураками",
            "constitution": (
                "Ты — Шеф-редактор Леон Мураками. "
                "Твоя задача — стратегически управлять контент-планом, "
                "формируя идеи и постановки задач для команды. "
                "Пиши чётко, по-делу и по-деловому."
            ),
        }

    # ---------- вспомогательные методы ------------------------------------- #
    @staticmethod
    def _extract_json(text: str) -> Optional[List[str]]:
        """Пытается вытащить JSON-массив из ответа модели."""
        try:
            safe_text = text[:Config.MAX_JSON_SIZE]
            match = re.search(r"\[.*?]", safe_text, re.S)
            if not match:
                logger.warning("JSON-массив не найден в ответе Gemini")
                return None
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.error("Ошибка JSONDecode при разборе ответа Gemini")
            return None

    async def _ask_openrouter(self, prompt: str, max_tokens: int = 1024) -> str:
        """Обращение к OpenRouter API с Google Gemini 2.5 Flash"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://goldensquad.ai",  # Для рейтинга OpenRouter
            "X-Title": "Golden Squad - Chief Editor"    # Название приложения
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.constitution},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{Config.OPENROUTER_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if 'choices' not in data or not data['choices']:
                        raise ValueError("Некорректный ответ от OpenRouter API")
                    
                    return data["choices"][0]["message"]["content"].strip()
                    
        except aiohttp.ClientError as e:
            logger.error(f"OpenRouter API connection error: {e}")
            raise HTTPException(status_code=503, detail=f"Ошибка соединения с OpenRouter: {str(e)}")
        except Exception as e:
            logger.error(f"OpenRouter API unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка API: {str(e)}")

    # ---------- публичные методы ------------------------------------------- #
    async def generate_topics(self, business_theme: str, count: int) -> List[str]:
        """Генерирует темы для постов через Google Gemini 2.5 Flash"""
        prompt = (
            f"Сформулируй {count} свежих и релевантных тем для постов по направлению "
            f"«{business_theme}». Ответ верни **строго** как JSON-массив строк без "
            f"дополнительного текста."
        )
        
        raw_response = await self._ask_openrouter(prompt)
        topics = self._extract_json(raw_response)
        
        if not topics:
            raise ValueError("Gemini вернул ответ без корректного JSON-массива.")
        
        return topics[:count]

    async def create_task_for_copywriter(self, topic: str, business_theme: str) -> str:
        """Создает ТЗ для копирайтера через Google Gemini 2.5 Flash"""
        prompt = (
            "Сформулируй краткое, вдохновляющее ТЗ для копирайтера.\n\n"
            f"Тема поста: «{topic}».\n"
            f"Контекст бизнеса: «{business_theme or '—'}».\n\n"
            "Формат: 2–3 предложения. Чётко обозначь цель и ключевой посыл."
        )
        
        return await self._ask_openrouter(prompt, max_tokens=256)

# --------------------------------------------------------------------------- #
#                           Конфигурация FastAPI-сервера                      #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Golden Squad — Chief Editor API", 
    version="4.3",
    description="Микросервис Шеф-редактора с OpenRouter + Google Gemini 2.5 Flash"
)

editor = ChiefEditor()

@app.get("/health", tags=["system"])
async def health():
    """Health check с проверкой OpenRouter API"""
    try:
        # Пробный запрос для проверки соединения
        test_response = await editor._ask_openrouter("test", max_tokens=5)
        return {
            "status": "ok", 
            "agent": "chief-editor",
            "name": "Леон Мураками",
            "model": editor.model_name,
            "openrouter_api": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "agent": "chief-editor", 
            "name": "Леон Мураками",
            "model": editor.model_name,
            "openrouter_api": "disconnected",
            "error": str(e)
        }

@app.post("/generate_topics", tags=["workflow"])
async def api_generate_topics(request: TopicsRequest):
    """Генерация тем для постов"""
    try:
        topics = await editor.generate_topics(request.business_theme, request.count)
        return {"topics": topics}
    except Exception as exc:
        logger.error(f"Error generating topics: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/create_task", tags=["workflow"])
async def api_create_task(request: TextDraftRequest):
    """Создание задания для копирайтера"""
    try:
        task = await editor.create_task_for_copywriter(request.topic, request.business_theme)
        return {"task_for_copywriter": task}
    except Exception as exc:
        logger.error(f"Error creating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

# --------------------------------------------------------------------------- #
#                               Точка входа                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("🚀 Запускаю сервер Шефа-редактора Леона с OpenRouter + Gemini 2.5 Flash...")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level=Config.LOG_LEVEL)
