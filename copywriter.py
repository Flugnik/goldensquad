# copywriter.py - Версия 3.2. Алина с OpenRouter интеграцией и исправленной моделью

import asyncio
import os
import sys
import json
import traceback
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Импорт системы логирования
from utils.supabase_logger import log_agent_action, logger

# --- Константы ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Модели данных для API (Pydantic) ---
class WritePostRequest(BaseModel):
    task_from_chief: str
    brand_voice_info: str = ""
    topic_id: str = ""

# --- Основной класс агента ---
class Copywriter:
    def __init__(self):
        # Модель изменена на anthropic/claude-3-5-sonnet-20241022 (OpenRouter совместимая)
        self.model_name = os.getenv('COPYWRITER_MODEL', 'anthropic/claude-sonnet-4')
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY не найден в переменных окружения!")
            
        self.constitution = self.build_constitution()
        
        print("=== [constitution загрузка] ===", file=sys.stderr)
        for k, v in self.constitution.items():
            if isinstance(v, dict):
                print(f"{k}: JSON с {len(v)} ключами", file=sys.stderr)
            elif isinstance(v, list):
                print(f"{k}: список из {len(v)} элементов", file=sys.stderr)
            else:
                print(f"{k}: {str(v)[:150]} ...", file=sys.stderr)
        
        print(f"✅ Копирайтер v3.2 (Алина) инициализирован. Модель: {self.model_name}", file=sys.stderr)

    def build_constitution(self):
        constitution = {
            "professional_core": self.load_file('personas/copywriter/Alina_Persona.txt'),
            "agent_dna": self.load_json('personas/copywriter/Alina_DNA.json'),
            "client_dna": self.load_file('client_profiles/nikolay_dna.txt'),
            "search_methods": self.load_file('personas/copywriter/Poisk-informacii.txt'),
            "imperatives": self.load_json('personas/copywriter/Alina_MI.json')
        }
        
        constitution_status = {
            "professional_core": bool(constitution['professional_core']),
            "agent_dna": bool(constitution['agent_dna']),
            "client_dna": bool(constitution['client_dna']),
            "search_methods": bool(constitution['search_methods']),
            "imperatives": bool(constitution['imperatives'])
        }
        
        try:
            logger.log_event(
                agent_name="copywriter",
                event_type="constitution_loaded",
                input_data={"files_to_load": list(constitution.keys())},
                output_data={"constitution_status": constitution_status},
                elapsed_ms=0,
                stage="initialization",
                status="success",
                constitution_files=constitution_status
            )
        except Exception as e:
            print(f"⚠️ Не удалось записать лог инициализации: {e}", file=sys.stderr)
        
        return constitution

    def load_file(self, path):
        try:
            with open(path, encoding='utf-8-sig') as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ Файл не найден: {path}", file=sys.stderr)
            return ""

    def load_json(self, path):
        try:
            with open(path, encoding='utf-8-sig') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ JSON файл не найден: {path}", file=sys.stderr)
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON {path}: {e}", file=sys.stderr)
            return {}

    def build_system_prompt(self, task_from_chief, brand_voice_info):
        base_prompt = f"""
[ПРОФЕССИОНАЛЬНОЕ ЯДРО КОПИРАЙТЕРА:]
{self.constitution['professional_core']}

[ДНК АГЕНТА АЛИНА:]
{json.dumps(self.constitution['agent_dna'], ensure_ascii=False, indent=2) if self.constitution['agent_dna'] else 'Файл не загружен'}

[ДНК КЛИЕНТА:]
{self.constitution['client_dna']}

[МЕТОДЫ ПОИСКА ИНФОРМАЦИИ:]
{self.constitution['search_methods']}

[ЗАДАНИЕ ОТ ШЕФА:]
{task_from_chief}

[ГОЛОС БРЕНДА/ДОПОЛНИТЕЛЬНО:]
{brand_voice_info}

ОБЯЗАТЕЛЬНО: Твой ответ должен включать блоки отчетности согласно императивам:
- 📋 РАБОЧИЕ ФАЙЛЫ: (указать использованные разделы)
- 🔍 ПРОВЕРКА СООТВЕТСТВИЯ: (анализ соответствия ДНК клиента)
- 🔎 ИСТОЧНИКИ ИНФОРМАЦИИ: (если искал дополнительную информацию)
- 📚 ВЫВОДЫ ДЛЯ РАЗВИТИЯ: (инсайты для улучшения)
""".strip()
        
        return base_prompt

    async def _ask_openrouter(self, prompt: str, max_tokens: int = 3000) -> str:
        """Безопасный вызов через OpenRouter с обработкой ошибок."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://goldensquad.ai"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error_msg = f"Ошибка при вызове OpenRouter: {str(e)}"
            print(error_msg, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            try:
                logger.log_event(
                    agent_name="copywriter",
                    event_type="openrouter_api_error",
                    input_data={"model": self.model_name},
                    output_data={"error": str(e)},
                    status="error",
                    error_message=str(e)
                )
            except Exception as log_e:
                print(f"⚠️ Не удалось записать лог ошибки API: {log_e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="Не удалось связаться с ИИ-моделью")

    @log_agent_action(agent_name="copywriter", event_type="write_post", stage="content_creation")
    async def write_post(self, task_from_chief, brand_voice_info, topic_id=None):
        prompt = self.build_system_prompt(task_from_chief, brand_voice_info)
        
        try:
            post_content = await self._ask_openrouter(prompt, max_tokens=3000)
            
            reporting_blocks = [
                "📋 РАБОЧИЕ ФАЙЛЫ:", 
                "🔍 ПРОВЕРКА СООТВЕТСТВИЯ:",
                "🔎 ИСТОЧНИКИ ИНФОРМАЦИИ:",
                "📚 ВЫВОДЫ ДЛЯ РАЗВИТИЯ:"
            ]
            
            has_reporting = any(block in post_content for block in reporting_blocks)
            present_blocks = [block for block in reporting_blocks if block in post_content]
            
            result = {
                "post_text": post_content,
                "reporting_analysis": {
                    "blocks_present": present_blocks,
                    "total_blocks": len(present_blocks),
                    "compliance_score": len(present_blocks) / len(reporting_blocks)
                }
            }
            
            if not has_reporting:
                result["warning"] = "Алина не включила блоки отчетности согласно императивам"
            
            try:
                logger.log_event(
                    agent_name="copywriter",
                    event_type="imperatives_check",
                    input_data={"required_blocks": reporting_blocks},
                    output_data={"present_blocks": present_blocks, "compliance_score": result["reporting_analysis"]["compliance_score"]},
                    elapsed_ms=0,
                    topic_id=topic_id,
                    stage="quality_control",
                    status="success" if has_reporting else "warning",
                    imperatives_check=has_reporting,
                    reporting_blocks_present=has_reporting
                )
            except Exception as e:
                print(f"⚠️ Не удалось записать лог проверки императивов: {e}", file=sys.stderr)
            
            return result
            
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"error": str(e)}

# --- FastAPI настройка ---
app = FastAPI(title="Golden Squad — Copywriter Alina API", version="3.2")
alina = Copywriter()

@app.get("/health", tags=["system"])
async def health():
    constitution_status = {
        "professional_core": bool(alina.constitution['professional_core']),
        "agent_dna": bool(alina.constitution['agent_dna']),
        "client_dna": bool(alina.constitution['client_dna']),
        "search_methods": bool(alina.constitution['search_methods']),
        "imperatives": bool(alina.constitution['imperatives'])
    }
    
    try:
        logger.log_event(
            agent_name="copywriter",
            event_type="health_check",
            input_data={"endpoint": "/health"},
            output_data={"constitution_status": constitution_status, "model": alina.model_name},
            elapsed_ms=0,
            stage="monitoring",
            status="success"
        )
    except Exception as e:
        print(f"⚠️ Не удалось записать лог health-check: {e}", file=sys.stderr)
    
    return {
        "status": "ok",
        "agent": "copywriter",
        "name": "Alina",
        "model": alina.model_name,
        "constitution_loaded": constitution_status,
        "supabase_logging": "enabled"
    }

@app.get("/imperatives", tags=["debug"])
async def show_imperatives():
    if not alina.constitution['imperatives']:
        return {"error": "Императивы не загружены"}
    
    imperatives_data = {
        "total_imperatives": len(alina.constitution['imperatives'].get('imperatives', [])),
        "activation_triggers": alina.constitution['imperatives'].get('activation_triggers', []),
        "required_files": alina.constitution['imperatives'].get('required_files', []),
        "critical_imperatives": [
            imp['title'] for imp in alina.constitution['imperatives'].get('imperatives', [])
            if imp.get('priority') == 'critical'
        ]
    }
    
    try:
        logger.log_event(
            agent_name="copywriter",
            event_type="imperatives_request",
            input_data={"endpoint": "/imperatives"},
            output_data=imperatives_data,
            elapsed_ms=0,
            stage="debugging",
            status="success"
        )
    except Exception as e:
        print(f"⚠️ Не удалось записать лог запроса императивов: {e}", file=sys.stderr)
    
    return imperatives_data

@app.post("/write_post")
async def api_write_post(request: WritePostRequest):
    topic_id = request.topic_id or None
    
    result = await alina.write_post(
        request.task_from_chief, 
        request.brand_voice_info,
        topic_id=topic_id
    )
    
    if topic_id:
        result["topic_id"] = topic_id
    
    return result

if __name__ == "__main__":
    print("🚀 Запускаю сервер Копирайтера Алины v3.2 с OpenRouter интеграцией...")
    uvicorn.run("copywriter:app", host="0.0.0.0", port=8001, reload=False)
