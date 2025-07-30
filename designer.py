# designer.py - Версия 3.1. Адриан с полной constitution-архитектурой, императивами и Supabase логированием.
import asyncio
import os
import sys
import json
import traceback
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from typing import Optional

# Загрузка переменных окружения
load_dotenv()

# Импорт системы логирования
from utils.supabase_logger import log_agent_action, logger

# --- Модели данных для API (Pydantic) ---
class CreateVisualRequest(BaseModel):
    task_from_chief: str
    post_text: str = ""
    topic: str = ""
    topic_id: str = ""

# --- Основной класс агента ---
class DesignerAgent:
    def __init__(self):
        self.model_name = os.getenv("DESIGNER_MODEL", "claude-3-7-sonnet-20250219")
        self.client = anthropic.AsyncClient(api_key=os.getenv("CLAUDE_API_KEY"))
        self.constitution = self.load_constitution()
        if not self.constitution:
            raise RuntimeError("❌ Не удалось загрузить constitution. Проверьте пути к файлам.")

    def load_constitution(self):
        constitution = {}
        constitution_status = {}

        try:
            # Загрузка constitution.json
            with open("constitution.json", "r", encoding="utf-8-sig") as f:
                constitution_data = json.load(f)
            constitution["professional_core"] = constitution_data.get("professional_core", "")
            constitution["agent_dna"] = constitution_data.get("agent_dna", {})
            constitution["client_dna"] = constitution_data.get("client_dna", "")
            constitution["imperatives"] = constitution_data.get("imperatives", {})

            constitution_status["constitution.json"] = "loaded"

            # Загрузка topics_archive.json
            try:
                with open("topics_archive.json", "r", encoding="utf-8-sig") as f:
                    constitution["topics_archive"] = json.load(f)
                constitution_status["topics_archive.json"] = "loaded"
            except Exception as e:
                print(f"⚠️ topics_archive.json не загружен: {e}")
                constitution["topics_archive"] = {}
                constitution_status["topics_archive.json"] = "error"

            # Логируем успешную загрузку
            try:
                logger.log_event(
                    agent_name="designer",
                    event_type="constitution_loaded",
                    input_data={"files_to_load": list(constitution.keys())},
                    output_data={"constitution_status": constitution_status},
                    elapsed_ms=0,
                    stage="initialization",
                    status="success",
                    constitution_files=constitution_status
                )
            except Exception as e:
                print(f"⚠️ Не удалось записать лог инициализации дизайнера: {e}", file=sys.stderr)

            print(f"✅ Дизайнер v3.1 (Адриан) инициализирован. Модель: {self.model_name}")
            return constitution

        except Exception as e:
            print(f"❌ Ошибка при загрузке constitution: {e}")
            traceback.print_exc(file=sys.stderr)
            return {}

    def build_system_prompt(self, task_from_chief, post_text, topic=""):
        base_prompt = f"""[ПРОФЕССИОНАЛЬНОЕ ЯДРО ДИЗАЙНЕРА:]{self.constitution['professional_core']}[ДНК ДИЗАЙНЕРА:]{json.dumps(self.constitution['agent_dna'], ensure_ascii=False, indent=2) if self.constitution['agent_dna'] else 'Файл не загружен'}[ДНК КЛИЕНТА:]{self.constitution['client_dna']}[ЗАДАНИЕ ОТ ШЕФА:]{task_from_chief}[ТЕКСТ ОТ КОПИРАЙТЕРА:]{post_text}ОБЯЗАТЕЛЬНО:- Ты должен строго соблюдать императивы дизайнера (см. constitution).- Твой ответ должен включать блоки отчетности:- 📋 РАБОЧИЕ ФАЙЛЫ: (указать использованные материалы)- 🔍 ПРОВЕРКА ВИЗУАЛА: (анализ соответствия ДНК клиента и задачам)- 🔎 ИСТОЧНИКИ ВИЗУАЛА: (описание использованных генераторов/референсов)- 📚 ВЫВОДЫ ДЛЯ РАЗВИТИЯ: (инсайты и предложения по улучшению)- Генерируй промпт для генерации картинки ИСКЛЮЧИТЕЛЬНО НА АНГЛИЙСКОМ ЯЗЫКЕ для external visual tools.""".strip()
        return base_prompt

    @log_agent_action(agent_name="designer", event_type="make_visual", stage="visual_creation")
    async def make_visual(self, task_from_chief, post_text, topic="", topic_id=None):
        prompt = self.build_system_prompt(task_from_chief, post_text, topic)
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=2500,
                system="",
                messages=[{"role": "user", "content": prompt}]
            )
            # Извлекаем текст ответа
            raw_content = response.content[0].text.strip()

            # Ищем блок с промптом (предположим, он после "PROMPT:" или "Visual prompt:")
            # Упрощённый парсинг — можно улучшить
            lines = raw_content.split('\n')
            visual_prompt = None
            for line in lines:
                if line.strip().lower().startswith("prompt:") or "английском языке" in line.lower():
                    visual_prompt = line.split(":", 1)[1].strip()
                    break

            # Если не нашли — возвращаем весь ответ
            if not visual_prompt:
                visual_prompt = raw_content[:500]  # первые 500 символов

            # Возвращаем результат
            result = {
                "visual_result": visual_prompt,
                "full_response": raw_content,
                "topic_id": topic_id
            }
            return result

        except Exception as e:
            error_msg = f"Ошибка при генерации визуала: {str(e)}"
            print(error_msg, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

            # Логируем ошибку
            try:
                logger.log_event(
                    agent_name="designer",
                    event_type="make_visual",
                    input_data={"task_from_chief": task_from_chief, "topic": topic},
                    output_data={"error": str(e)},
                    elapsed_ms=0,
                    topic_id=topic_id,
                    stage="visual_creation",
                    status="error",
                    error_message=str(e)
                )
            except Exception as log_e:
                print(f"⚠️ Не удалось записать лог ошибки make_visual: {log_e}", file=sys.stderr)

            # Возвращаем ошибку, но не падаем
            return {"error": "Не удалось сгенерировать промпт для визуала", "details": str(e)}

# --- FastAPI приложение ---
app = FastAPI(title="Designer Agent (Adrian)")

# Инициализация агента
adrian = DesignerAgent()

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "designer", "name": "Adrian"}

@app.post("/make_visual")
async def api_make_visual(request: CreateVisualRequest):
    result = await adrian.make_visual(
        task_from_chief=request.task_from_chief,
        post_text=request.post_text,
        topic=request.topic,
        topic_id=request.topic_id
    )
    return result

# --- Точка входа ---
if __name__ == "__main__":
    print("🚀 Запускаю сервер Дизайнера Адриана v3.1 с Supabase логированием...")
    uvicorn.run("designer:app", host="0.0.0.0", port=8002, reload=False)