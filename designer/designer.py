import os
from dotenv import load_dotenv
from pathlib import Path
import httpx
import cachetools
from yarl import URL
import ipaddress
import hashlib
import imghdr

# --- Определяем DEBUG_MODE ДО всего остального ---
DEBUG_MODE = os.getenv("DEBUG", "true").lower() == "true"

# --- Загружаем .env из корня проекта ---
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

print(f"✅ Загружено .env из: {env_path}")
print(f"🔍 FAL_API_KEY: {'YES' if os.getenv('FAL_API_KEY') else 'NO'}")
print(f"🔍 OPENROUTER_API_KEY: {'YES' if os.getenv('OPENROUTER_API_KEY') else 'NO'}")
print(f"🔧 DEBUG_MODE: {DEBUG_MODE}")

# --- Оригинальный код ---
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
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

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)

# --- Константы ---
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_HOSTS = ["fal.media", "cdn.fal.ai", "i.imgur.com", "example-assets.com"]

# --- Модели данных ---
class CreateImageRequest(BaseModel):
    task_from_chief: str = Field(min_length=10, description="Task from chief editor")
    post_text: str = Field(min_length=5, description="Text from copywriter")
    topic: str = Field(min_length=2, description="Topic for file naming")

class SaveImageRequest(BaseModel):
    image_url: str = Field(description="URL of approved image")
    topic: str = Field(description="Topic for file naming")


# --- Безопасный менеджер секретов ---
class SecretManager:
    def __init__(self, allow_env_fallback: bool = DEBUG_MODE):
        self.allow_env_fallback = allow_env_fallback

    def get(self, secret_name: str) -> str:
        secret_path = f"/run/secrets/{secret_name}"
        if os.path.exists(secret_path):
            with open(secret_path, 'r') as f:
                return f.read().strip()

        if self.allow_env_fallback:
            env_map = {
                'fal_api_key': 'FAL_API_KEY',
                'openrouter_api_key': 'OPENROUTER_API_KEY'
            }
            env_var = env_map.get(secret_name, secret_name.upper())
            value = os.getenv(env_var)
            if value:
                logger.warning(f"Using env var {env_var} for {secret_name} (DEV MODE)")
                return value

        raise RuntimeError(f"Secret {secret_name} not found in /run/secrets or env")

# --- Валидатор URL (защита от SSRF) ---
def is_safe_url(url_str: str) -> bool:
    try:
        url = URL(url_str)
        if url.scheme not in ('http', 'https'):
            return False
        host = url.host.lower()
        if not host:
            return False

        # Проверка на loopback и private IP
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_loopback or ip.is_private:
                return False
        except ValueError:
            pass  # Это домен

        # Проверка разрешенных доменов
        return any(host.endswith(allowed) for allowed in ALLOWED_HOSTS)
    except Exception:
        return False


# --- Prompt Generator Service ---
class PromptGenerator:
    def __init__(self, client: httpx.AsyncClient, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.personality = self._load_personality()
        self.cache = cachetools.TTLCache(maxsize=100, ttl=300)  # 5 минут

    def _load_personality(self) -> Dict[str, str]:
        return {
            "name": "Adrian, Visual Maestro",
            "constitution": (
                "You are Adrian, the Visual Maestro. Your mission is to translate strategy and text "
                "into powerful visual language. You create not just images, but visual stories that "
                "amplify emotional impact. Principles: Visual Storytelling, Aesthetic Intelligence, "
                "Depth and Metaphor, Technical Mastery."
            )
        }

    def _get_cache_key(self, task: str, text: str) -> str:
        content = f"{task[:100]}::{text[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate(self, task: str, text: str) -> str:
        cache_key = self._get_cache_key(task, text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        system_prompt = f"""{self.personality['constitution']}
Create a concise, visual-focused English prompt (max 150 words) that captures the essence and emotion.
Avoid text in the image. Use photorealistic style."""

        user_prompt = f"""TASK: {task}
TEXT: {text}
Create an image generation prompt describing visual elements, style, lighting, metaphor."""

        try:
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            prompt = data["choices"][0]["message"]["content"].strip()
            logger.info(f"Generated prompt: {prompt[:80]}...")
            self.cache[cache_key] = prompt
            return prompt
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Photorealistic image based on: {text[:100]}"


# --- Асинхронный FAL Image Generator ---
class FALImageGenerator:
    def __init__(self, api_key: str, model: str = "fal-ai/flux-pro"):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://api.fal.ai/v1/run/{model}"

    async def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "image_size": os.getenv("IMAGE_SIZE", "landscape_4_3"),
            "num_inference_steps": int(os.getenv("FAL_STEPS", "28")),
            "guidance_scale": float(os.getenv("FAL_GUIDANCE", "3.5")),
            "num_images": 1,
            "enable_safety_checker": True
        }

        timeout = httpx.Timeout(connect=10, read=60, write=30, pool=15)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                if not result.get("images"):
                    raise ValueError("No images returned from FAL")
                image_url = result["images"][0]["url"]
                logger.info(f"FAL generated image: {image_url}")
                return image_url
            except httpx.HTTPStatusError as e:
                logger.error(f"FAL API HTTP error {e.response.status_code}: {e.response.text}")
                raise HTTPException(status_code=502, detail="Image generation failed (FAL)")
            except Exception as e:
                logger.error(f"FAL API error: {e}")
                raise HTTPException(status_code=500, detail="Image generation failed")


# --- Image Saver Service ---
class ImageSaver:
    def __init__(self, save_dir: Path = Path("results/ready_for_publish")):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, topic: str) -> str:
        clean = "".join(c for c in topic if c.isalnum() or c in " -_").strip()
        return clean.replace(" ", "_")

    def _validate_path(self, filename: str) -> Path:
        full_path = (self.save_dir / filename).resolve()
        try:
            full_path.relative_to(self.save_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return full_path

    async def save_from_url(self, image_url: str, topic: str) -> Dict[str, str]:
        if not is_safe_url(image_url):
            raise HTTPException(status_code=400, detail="Disallowed image URL (SSRF protection)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = self._sanitize_filename(topic)
        filename = f"image_{safe_topic}_{timestamp}.png"
        file_path = self._validate_path(filename)

        timeout = httpx.Timeout(connect=10, read=30)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                response = await client.get(image_url, stream=True)
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_IMAGE_SIZE:
                    raise HTTPException(status_code=413, detail="Image too large")

                total_size = 0
                async with aiofiles.open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        total_size += len(chunk)
                        if total_size > MAX_IMAGE_SIZE:
                            await f.close()
                            file_path.unlink(missing_ok=True)
                            raise HTTPException(status_code=413, detail="Image too large")
                        await f.write(chunk)

                # Проверка формата
                if imghdr.what(file_path) not in ("png", "jpeg", "jpg", "webp"):
                    file_path.unlink()
                    raise HTTPException(status_code=400, detail="Invalid image format")

                logger.info(f"Image saved: {file_path}")
                return {"image_path": str(file_path), "status": "saved"}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to download image")


# --- Основной сервис Designer ---
class DesignerService:
    def __init__(self):
        self.secret_manager = SecretManager()
        self.fal_api_key = self.secret_manager.get("fal_api_key")
        self.openrouter_api_key = self.secret_manager.get("openrouter_api_key")

        # Конфигурация
        self.fal_model = os.getenv("FAL_MODEL", "fal-ai/flux-pro")
        self.openrouter_model = os.getenv("DESIGNER_MODEL", "google/gemini-2.5-flash")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))

        # Сервисы
        self.prompt_generator = PromptGenerator(
            client=httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
                timeout=httpx.Timeout(30.0)
            ),
            model=self.openrouter_model,
            max_tokens=self.max_tokens
        )

        self.image_generator = FALImageGenerator(
            api_key=self.fal_api_key,
            model=self.fal_model
        )

        self.image_saver = ImageSaver()

        logger.info(f"✅ Designer Service initialized. Model: {self.fal_model}")

    async def create_image(self, task: str, text: str) -> Dict[str, Any]:
        try:
            prompt = await self.prompt_generator.generate(task, text)
            image_url = await self.image_generator.generate(prompt)
            return {
                "image_url": image_url,
                "english_prompt": prompt,
                "model": self.fal_model
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Image creation failed: {e}")
            raise HTTPException(status_code=500, detail="Image generation failed")

    async def save_image(self, image_url: str, topic: str) -> Dict[str, str]:
        return await self.image_saver.save_from_url(image_url, topic)

    async def health_check(self) -> Dict[str, Any]:
        try:
            # Проверка OpenRouter
            test_response = await self.prompt_generator.client.get(
                "https://openrouter.ai/api/v1/models",
                timeout=5.0
            )
            test_response.raise_for_status()

            # Проверка директории
            import os # добавить импорт вверху файла

            return {
                "status": "ok",
                "agent": "designer",
                "name": "Adrian, Visual Maestro",
                "fal_model": self.fal_model,
                "openrouter_model": self.openrouter_model,
                "storage": "ready"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "agent": "designer"
            }


# --- Инициализация приложения ---
app = FastAPI(title="Designer Adrian API", version="3.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Создание сервиса
designer = DesignerService()


# --- Health Endpoint ---
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
    result = await designer.create_image(request.task_from_chief, request.post_text)
    return result


@app.post("/save_image", tags=["design"])
async def api_save_image(request: SaveImageRequest):
    result = await designer.save_image(request.image_url, request.topic)
    return result


# --- Запуск ---
if __name__ == "__main__":
    logger.info("🚀 Starting Designer Adrian API v3.0...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")