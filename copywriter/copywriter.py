# copywriter.py - Версия 2.1. Исправленная с OpenRouter API и улучшенной обработкой ошибок

import asyncio
import os
import sys
import logging
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import openai

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# --- Модели данных для API (Pydantic) ---
class WritePostRequest(BaseModel):
    task_from_chief: str = Field(min_length=10, description="Task description from chief editor")
    brand_voice_info: str = Field(default="", description="Brand voice and additional context")

# --- Основной класс агента ---
class Copywriter:
    def __init__(self):
        self.model_name = os.getenv('COPYWRITER_MODEL', 'anthropic/claude-sonnet-4')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2500'))
        self.api_key = self.read_secret('openrouter_api_key')
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=os.getenv('API_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        self.constitution = self.load_constitution()
        logger.info("Copywriter initialized successfully")

    def read_secret(self, secret_name: str) -> str:
        """Читает секрет из Docker Secrets или переменных окружения"""
        try:
            with open(f'/run/secrets/{secret_name}', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback для локальной разработки
            env_var = secret_name.upper().replace('_', '_')
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError(f"Neither secret file nor env var found for {secret_name}")
            return api_key

    @lru_cache(maxsize=1)
    def load_constitution(self):
        """Загружает constitution с кэшированием"""
        constitution = {
            "professional_core": self.load_file('personas/copywriter/Alina_Persona.txt', required=True),
            "client_dna": self.load_file('client_profiles/nikolay_dna.txt', required=False),
            "search_methods": self.load_file('personas/copywriter/Poisk-informacii.txt', required=False)
        }
        
        logger.info("Constitution loaded successfully")
        for k, v in constitution.items():
            logger.debug(f"{k}: {len(v)} characters loaded")
        
        return constitution

    def load_file(self, path: str, required: bool = False) -> str:
        """Загружает файл с улучшенной обработкой ошибок"""
        try:
            with open(path, encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Loaded file: {path} ({len(content)} chars)")
                return content
        except FileNotFoundError:
            if required:
                logger.error(f"Required file not found: {path}")
                raise FileNotFoundError(f"Critical file missing: {path}")
            else:
                logger.warning(f"Optional file not found: {path}")
                return ""

    def build_system_prompt(self) -> str:
        """Строит system prompt из constitution"""
        return f"""Ты профессиональный копирайтер Алина Сомова.

{self.constitution['professional_core']}

ДНК клиента:
{self.constitution['client_dna']}

Методы работы:
{self.constitution['search_methods']}""".strip()

    async def write_post(self, task_from_chief: str, brand_voice_info: str = ""):
        """Создает пост на основе задания от шефа"""
        logger.info(f"Processing writing task: {task_from_chief[:100]}...")
        
        user_prompt = f"""
ЗАДАНИЕ ОТ ШЕФА:
{task_from_chief}

ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ О БРЕНДЕ:
{brand_voice_info}

Напиши пост согласно твоей профессиональной конституции."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self.build_system_prompt(),
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated post ({len(result)} chars)")
            return {"post_text": result}
            
        except openai.AuthenticationError:
            logger.error("Authentication failed with OpenRouter API")
            raise HTTPException(status_code=401, detail="API authentication failed")
        except openai.RateLimitError:
            logger.error("Rate limit exceeded")
            raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later")
        except openai.APITimeoutError:
            logger.error("API timeout")
            raise HTTPException(status_code=504, detail="API request timeout")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Создание экземпляров FastAPI и агента ---
app = FastAPI(title="Copywriter Eva API", version="2.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

writer = Copywriter()

# --- Health-эндпоинт с проверкой API ---
@app.get("/health", tags=["system"])
async def health():
    try:
        # Проверяем доступность API
        await writer.client.models.list()
        return {"status": "ok", "api_connection": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "degraded", "api_connection": "unhealthy", "error": str(e)}

# --- API-точка для написания поста ---
@app.post("/write_post", tags=["copywriting"])
@limiter.limit("10/minute")
async def api_write_post(request: WritePostRequest):
    """Создает пост на основе задания от главного редактора"""
    result = await writer.write_post(request.task_from_chief, request.brand_voice_info)
    return result

# --- Точка запуска сервера ---
if __name__ == "__main__":
    logger.info("🚀 Запускаю сервер Копирайтера Алины...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
