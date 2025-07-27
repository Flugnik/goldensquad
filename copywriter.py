# copywriter.py - Версия 2.0. Алина Сомова с API на FastAPI.
# Готова принимать задания от Шефа и воплощать их в "живое слово".

import asyncio
import os
import sys
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Модели данных для API (Pydantic) ---
class WritePostRequest(BaseModel):
    task_from_chief: str
    brand_voice_info: str = ""  # В будущем тут профиль клиента

# --- Основной класс агента (логика с constitution) ---
class Copywriter:
    def __init__(self):
        self.model_name = os.getenv('COPYWRITER_MODEL', 'claude-sonnet-4-20250514')
        self.constitution = self.build_constitution()
        print("=== [constitution загрузка] ===", file=sys.stderr)
        for k, v in self.constitution.items():
            print(f"{k}: {v[:150]} ...", file=sys.stderr)
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)

    def build_constitution(self):
        # Однократная загрузка всех нужных текстов из файлов
        return {
            "professional_core": self.load_file('personas/copywriter/Alina_Persona.txt'),
            "client_dna": self.load_file('client_profiles/nikolay_dna.txt'),
            "search_methods": self.load_file('personas/copywriter/Poisk-informacii.txt')
        }

    def load_file(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def build_system_prompt(self, task_from_chief, brand_voice_info):
        # Единый промпт с constitution и задачей
        return f"""
[ПРОФЕССИОНАЛЬНОЕ ЯДРО КОПИРАЙТЕРА:]
{self.constitution['professional_core']}

[ДНК-клиента:]
{self.constitution['client_dna']}

[Методы поиска информации:]
{self.constitution['search_methods']}

[ЗАДАНИЕ ОТ ШЕФА:]
{task_from_chief}

[ГОЛОС БРЕНДА/ДОПОЛНИТЕЛЬНО:]
{brand_voice_info}
""".strip()

    async def write_post(self, task_from_chief, brand_voice_info):
        prompt = self.build_system_prompt(task_from_chief, brand_voice_info)
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=2500,
                system="",
                messages=[{"role": "user", "content": prompt}]
            )
            return {"post_text": response.content[0].text.strip()}
        except Exception as e:
            return {"error": str(e)}

# --- Создание экземпляров FastAPI и агента ---
app = FastAPI()
writer = Copywriter()

# --- Health-эндпоинт ---
@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}

# --- API-тoчка для написания поста ---
@app.post("/write_post")
async def api_write_post(request: WritePostRequest):
    result = await writer.write_post(request.task_from_chief, request.brand_voice_info)
    return result

# --- Точка запуска сервера ---
if __name__ == "__main__":
    print("🚀 Запускаю сервер Копирайтера Алины...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
