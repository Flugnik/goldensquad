# copywriter.py - Версия 3.0. Алина с полной интеграцией constitution и императивов /mi
# Готова принимать задания от Шефа и воплощать их в "живое слово" с отчетностью.

import asyncio
import os
import sys
import json
import traceback
import anthropic
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Модели данных для API (Pydantic) ---
class WritePostRequest(BaseModel):
    task_from_chief: str
    brand_voice_info: str = ""  # В будущем тут профиль клиента

# --- Основной класс агента (логика с полной constitution) ---
class Copywriter:
    def __init__(self):
        self.model_name = os.getenv('COPYWRITER_MODEL', 'claude-sonnet-4-20250514')
        self.constitution = self.build_constitution()
        
        print("=== [constitution загрузка] ===", file=sys.stderr)
        for k, v in self.constitution.items():
            if isinstance(v, dict):
                print(f"{k}: JSON с {len(v)} ключами", file=sys.stderr)
            elif isinstance(v, list):
                print(f"{k}: список из {len(v)} элементов", file=sys.stderr)
            else:
                print(f"{k}: {str(v)[:150]} ...", file=sys.stderr)
        
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)

    def build_constitution(self):
        # Полная загрузка всех компонентов архитектуры Алины
        return {
            "professional_core": self.load_file('personas/copywriter/Alina_Persona.txt'),
            "agent_dna": self.load_json('personas/copywriter/Alina_DNA.json'),
            "client_dna": self.load_file('client_profiles/nikolay_dna.txt'),
            "search_methods": self.load_file('personas/copywriter/Poisk-informacii.txt'),
            "imperatives": self.load_json('personas/copywriter/Alina_MI.json')
        }

    def load_file(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ Файл не найден: {path}", file=sys.stderr)
            return ""

    def load_json(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ JSON файл не найден: {path}", file=sys.stderr)
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON {path}: {e}", file=sys.stderr)
            return {}

    def build_system_prompt(self, task_from_chief, brand_voice_info):
        # Базовый промпт с constitution
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

    def extract_imperatives_summary(self):
        """Краткое извлечение ключевых императивов для системного промпта"""
        if not self.constitution['imperatives'] or 'imperatives' not in self.constitution['imperatives']:
            return "Императивы не загружены"
        
        critical_imperatives = [
            imp for imp in self.constitution['imperatives']['imperatives'] 
            if imp.get('priority') == 'critical'
        ]
        
        return "\n".join([f"- {imp['title']}: {imp['text'][:100]}..." for imp in critical_imperatives[:3]])

    async def write_post(self, task_from_chief, brand_voice_info):
        prompt = self.build_system_prompt(task_from_chief, brand_voice_info)
        
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=3000,  # Увеличено для отчетности
                system="",
                messages=[{"role": "user", "content": prompt}]
            )
            
            post_content = response.content[0].text.strip()
            
            # Проверяем наличие блоков отчетности
            has_reporting = any(block in post_content for block in [
                "📋 РАБОЧИЕ ФАЙЛЫ:", 
                "🔍 ПРОВЕРКА СООТВЕТСТВИЯ:",
                "🔎 ИСТОЧНИКИ ИНФОРМАЦИИ:",
                "📚 ВЫВОДЫ ДЛЯ РАЗВИТИЯ:"
            ])
            
            result = {"post_text": post_content}
            
            if not has_reporting:
                result["warning"] = "Алина не включила блоки отчетности согласно императивам"
            
            return result
            
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return {"error": str(e)}

# --- Создание экземпляров FastAPI и агента ---
app = FastAPI(title="Golden Squad — Copywriter Alina API", version="3.0")
writer = Copywriter()

# --- Health-эндпоинт с диагностикой ---
@app.get("/health", tags=["system"])
async def health():
    constitution_status = {
        "professional_core": bool(writer.constitution['professional_core']),
        "agent_dna": bool(writer.constitution['agent_dna']),
        "client_dna": bool(writer.constitution['client_dna']),
        "search_methods": bool(writer.constitution['search_methods']),
        "imperatives": bool(writer.constitution['imperatives'])
    }
    
    return {
        "status": "ok",
        "constitution_loaded": constitution_status,
        "model": writer.model_name
    }

# --- Диагностический эндпоинт для проверки императивов ---
@app.get("/imperatives", tags=["debug"])
async def show_imperatives():
    if not writer.constitution['imperatives']:
        return {"error": "Императивы не загружены"}
    
    return {
        "total_imperatives": len(writer.constitution['imperatives'].get('imperatives', [])),
        "activation_triggers": writer.constitution['imperatives'].get('activation_triggers', []),
        "required_files": writer.constitution['imperatives'].get('required_files', []),
        "critical_imperatives": [
            imp['title'] for imp in writer.constitution['imperatives'].get('imperatives', [])
            if imp.get('priority') == 'critical'
        ]
    }

# --- API-точка для написания поста ---
@app.post("/write_post")
async def api_write_post(request: WritePostRequest):
    result = await writer.write_post(request.task_from_chief, request.brand_voice_info)
    return result

# --- Точка запуска сервера ---
if __name__ == "__main__":
    print("🚀 Запускаю сервер Копирайтера Алины v3.0...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
