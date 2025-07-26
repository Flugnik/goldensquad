# chief_editor.py — Версия 4.1. Шеф Леон с API на FastAPI.
# Независимый микросервис, готовый к коммуникации с командой.

import asyncio
import json
import os
import re
import sys
from typing import List, Optional

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

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
    def __init__(self) -> None:
        self.model_name = os.getenv("CHIEF_EDITOR_MODEL", "claude-4-sonnet-20250501")
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")

        if not self.claude_api_key:
            raise RuntimeError("CLAUDE_API_KEY не найден в переменных окружения!")

        self.client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)
        self.personality = self._load_personality()
        self.constitution = self.personality["constitution"]

        print(
            f"✅ Шеф-редактор v4.1 ({self.personality['name']}) "
            f"инициализирован. Модель: {self.model_name}",
            file=sys.stderr,
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
        """Пытаемся вытащить JSON-массив из ответа модели."""
        try:
            match = re.search(r"\[.*?]", text, re.S)
            return json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            return None

    async def _ask_claude(self, prompt: str, max_tokens: int = 1024) -> str:
        """Единая точка общения с Anthropic API."""
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=self.constitution,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ---------- публичные методы ------------------------------------------- #
    async def generate_topics(self, business_theme: str, count: int) -> List[str]:
        prompt = (
            f"Сформулируй {count} свежих и релевантных тем для постов по направлению "
            f"«{business_theme}». Ответ верни **строго** как JSON-массив строк без "
            f"дополнительного текста."
        )
        raw = await self._ask_claude(prompt)
        topics = self._extract_json(raw)
        if not topics:
            raise ValueError("Claude вернул ответ без корректного JSON-массива.")
        return topics[:count]

    async def create_task_for_copywriter(self, topic: str, business_theme: str) -> str:
        prompt = (
            "Сформулируй краткое, вдохновляющее ТЗ для копирайтера.\n\n"
            f"Тема поста: «{topic}».\n"
            f"Контекст бизнеса: «{business_theme or '—'}».\n\n"
            "Формат: 2–3 предложения. Чётко обозначь цель и ключевой посыл."
        )
        return await self._ask_claude(prompt, max_tokens=256)


# --------------------------------------------------------------------------- #
#                           Конфигурация FastAPI-сервера                      #
# --------------------------------------------------------------------------- #
app = FastAPI(title="Golden Squad — Chief Editor API", version="4.1")
editor = ChiefEditor()


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@app.post("/generate_topics", tags=["workflow"])
async def api_generate_topics(req: TopicsRequest):
    try:
        topics = await editor.generate_topics(req.business_theme, req.count)
        return {"topics": topics}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/create_task", tags=["workflow"])
async def api_create_task(req: TextDraftRequest):
    try:
        task = await editor.create_task_for_copywriter(req.topic, req.business_theme)
        return {"task_for_copywriter": task}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --------------------------------------------------------------------------- #
#                               Точка входа                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("🚀 Запускаю сервер Шефа-редактора Леона…")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
