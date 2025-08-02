# designer.py - Версия 2.1. Исправленный с OpenRouter API и улучшенной архитектурой

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import openai
import fal_client

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# --- Модели данных для API (Pydantic) ---
class CreateImageRequest(BaseModel):
    task_from_chief: str = Field(min_length=10, description="Task from chief editor")
    post_text: str = Field(min_length=5, description="Text from copywriter")
    topic: str = Field(min_length=2, description="Topic for file naming")

class SaveImageRequest(BaseModel):
    image_url: str = Field(description="URL of approved image")
    topic: str = Field(description="Topic for file naming")

# --- Основной класс агента ---
class Designer:
    def __init__(self):
        self.fal_model = os.getenv('FAL_MODEL', 'fal-ai/flux-pro')
        self.openrouter_model = os.getenv('DESIGNER_MODEL', 'google/gemini-2.5-flash')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '1000'))
        
        # Инициализация API клиентов
        self.fal_api_key = self.read_secret('fal_api_key')
        self.openrouter_key = self.read_secret('openrouter_api_key')
        
        # Настройка FAL клиента
        os.environ['FAL_KEY'] = self.fal_api_key
        
        # Настройка OpenRouter клиента
        self.openrouter_client = openai.AsyncOpenAI(
            api_key=self.openrouter_key,
            base_url=os.getenv('API_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        
        self.personality = self.load_personality()
        self.ensure_directories()
        logger.info(f"✅ Дизайнер v2.0 (Адриан, 'Маэстро Визуала') инициализирован. Инструмент: {self.fal_model}")

    def read_secret(self, secret_name: str) -> str:
        """Читает секрет из Docker Secrets или переменных окружения"""
        try:
            with open(f'/run/secrets/{secret_name}', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback для локальной разработки
            env_map = {
                'fal_api_key': 'FAL_API_KEY',
                'openrouter_api_key': 'OPENROUTER_API_KEY'
            }
            env_var = env_map.get(secret_name, secret_name.upper())
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Neither secret file nor env var found for {secret_name}")
            return api_key

    def load_personality(self):
        """Загружает личность дизайнера"""
        return {
            "name": "Adrian, Visual Maestro",
            "constitution": """You are Adrian, the Visual Maestro. Your mission is to translate strategy and text into powerful visual language. You create not just images, but visual stories that amplify emotional impact.

Your principles:
1. Visual Storytelling - You don't illustrate, you tell stories
2. Aesthetic Intelligence - You have innate sense of style  
3. Depth and Metaphor - You seek metaphorical, not literal images
4. Technical Mastery - You master cutting-edge generation tools"""
        }

    def ensure_directories(self):
        """Создает необходимые директории"""
        Path("results/ready_for_publish").mkdir(parents=True, exist_ok=True)
        Path("images").mkdir(exist_ok=True)

    async def generate_english_prompt(self, task_from_chief: str, post_text: str) -> str:
        """Генерирует английский промпт для изображения через OpenRouter"""
        system_prompt = f"""You are Adrian, a professional visual designer. Create a detailed English prompt for image generation based on the given task and text.

{self.personality['constitution']}

Generate a concise, visual-focused English prompt (max 200 words) that captures the essence and emotion of the content."""

        user_prompt = f"""TASK FROM CHIEF: {task_from_chief}
COPYWRITER TEXT: {post_text}

Create an English image generation prompt that:
1. Captures the core metaphor and emotion
2. Describes visual elements, style, lighting
3. Avoids text/letters in the image
4. Uses photorealistic style"""

        try:
            response = await self.openrouter_client.chat.completions.create(
                model=self.openrouter_model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            english_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated English prompt: {english_prompt[:100]}...")
            return english_prompt
            
        except Exception as e:
            logger.error(f"Error generating English prompt: {e}")
            # Fallback промпт
            return f"Professional image based on: {post_text[:100]}, photorealistic style, high quality"

    async def create_image(self, task_from_chief: str, post_text: str) -> dict:
        """Создает изображение на основе задания от Шефа и текста Копирайтера"""
        logger.info(f"Creating image for task: {task_from_chief[:50]}...")
        
        try:
            # Генерируем английский промпт через OpenRouter
            english_prompt = await self.generate_english_prompt(task_from_chief, post_text)
            
            # Генерируем изображение через FAL
            logger.info(f"Sending to FAL model: {self.fal_model}")
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: fal_client.run(
                    self.fal_model,
                    arguments={
                        "prompt": english_prompt,
                        "image_size": "landscape_4_3",
                        "num_inference_steps": 28,
                        "guidance_scale": 3.5,
                        "num_images": 1,
                        "enable_safety_checker": True
                    }
                )
            )
            
            if not result.get('images'):
                raise ValueError("No images generated by FAL API")
                
            image_url = result['images'][0]['url']
            logger.info(f"Image generated successfully: {image_url}")
            
            return {
                "image_url": image_url,
                "english_prompt": english_prompt,
                "fal_model": self.fal_model
            }
            
        except Exception as e:
            logger.error(f"Error creating image: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    async def save_approved_image(self, image_url: str, topic: str) -> dict:
        """Скачивает и сохраняет одобренное изображение"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"results/ready_for_publish/image_{clean_topic.replace(' ', '_')}_{timestamp}.png"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(filename, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        logger.info(f"Image saved: {filename}")
                        return {"image_path": filename, "status": "saved"}
                    else:
                        raise HTTPException(status_code=400, detail="Failed to download image")
                        
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

# --- Создание экземпляров FastAPI и агента ---
app = FastAPI(title="Designer Adrian API", version="2.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

designer = Designer()

# --- Health endpoint (ИСПРАВЛЕН!) ---
@app.get("/health", tags=["system"])
async def health():
    """Health check endpoint для мониторинга состояния сервиса"""
    try:
        # Проверяем подключение к OpenRouter
        await designer.openrouter_client.models.list()
        
        # Проверяем директории
        Path("results/ready_for_publish").mkdir(parents=True, exist_ok=True)
        
        return {
            "status": "ok",
            "agent": "designer", 
            "name": "Адриан Маэстро Визуала",
            "fal_model": designer.fal_model,
            "openrouter_model": designer.openrouter_model,
            "openrouter_connection": "healthy"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded", 
            "error": str(e),
            "agent": "designer"
        }

# --- API Endpoints ---
@app.post("/create_image", tags=["design"])
@limiter.limit("3/minute")
async def api_create_image(request: CreateImageRequest):
    """Создает изображение на основе задания и текста"""
    result = await designer.create_image(request.task_from_chief, request.post_text)
    return result

@app.post("/save_image", tags=["design"])
async def api_save_image(request: SaveImageRequest):
    """Сохраняет одобренное изображение"""
    result = await designer.save_approved_image(request.image_url, request.topic)
    return result

# --- Точка запуска сервера ---
if __name__ == "__main__":
    logger.info("🚀 Запускаю сервер Дизайнера Адриана...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
