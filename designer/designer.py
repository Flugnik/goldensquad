# designer.py — версия 3.2 (исправления + улучшения)
"""
Основное:
• Добавлены отсутствующие импорты, константы и Pydantic-модели.
• Инициализирован глобальный logger до первого использования.
• MAX_IMAGE_SIZE вынесена в константу/ENV.
• Исправлены мелкие аннотации и форматирование.
"""

import os
import re                                 # ⬅️ был пропущен
import ipaddress
import asyncio
import hashlib
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import aiofiles
import cachetools
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from PIL import Image                    # вместо imghdr
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from yarl import URL

# ─── Настройка окружения ────────────────────────────────────────────────────
DEBUG_MODE = os.getenv("DEBUG", "true").lower() == "true"

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# ─── Логирование ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Константы ──────────────────────────────────────────────────────────────
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10000000"))  # 10 МБ по умолчанию

ALLOWED_HOSTS = {
    "fal.media",
    "cdn.fal.ai",
    "i.imgur.com",
    "example-assets.com",
}

# ─── Pydantic-модели запросов ───────────────────────────────────────────────
class CreateImageRequest(BaseModel):
    task_from_chief: str = Field(..., min_length=1)
    post_text: str = Field(..., min_length=1)


class SaveImageRequest(BaseModel):
    image_url: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)


# ─── Безопасный менеджер секретов ───────────────────────────────────────────
class SecretManager:
    def __init__(self, allow_env_fallback: bool = DEBUG_MODE):
        self.allow_env_fallback = allow_env_fallback

    def get(self, secret_name: str) -> str:
        secret_path = f"/run/secrets/{secret_name}"
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                return f.read().strip()

        if self.allow_env_fallback:
            env_map = {
                "fal_api_key": "FAL_API_KEY",
                "openrouter_api_key": "OPENROUTER_API_KEY",
            }
            env_var = env_map.get(secret_name, secret_name.upper())
            value = os.getenv(env_var)
            if value:
                if DEBUG_MODE:
                    logger.warning(
                        "Using env var %s for %s (DEV MODE)", env_var, secret_name
                    )
                return value

        raise RuntimeError(
            f"Secret {secret_name} not found in /run/secrets or environment"
        )


class PromptGenerator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None
        self._cache = cachetools.TTLCache(maxsize=100, ttl=300)
        self._cache_lock = asyncio.Lock()
        self.personality = self._load_personality()  # ← Вызов метода экземпляра

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _load_personality(self) -> Dict[str, str]:
        import json
        from pathlib import Path
        try:
            file_path = Path(__file__).parent / "personality_adrian.json"
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return {
                "name": "Adrian, Visual Maestro",
                "constitution": (
                    "You are Adrian, the Visual Maestro. Your mission is to translate "
                    "strategy and text into powerful visual language. You create not just "
                    "images, but visual stories that amplify emotional impact. Principles: "
                    "Visual Storytelling, Aesthetic Intelligence, Depth and Metaphor, "
                    "Technical Mastery."
                ),
            }


import json
from pathlib import Path
from typing import Dict

def _load_personality(self) -> Dict[str, str]:
    """
    Загружает профиль личности Adrian из файла personality_adrian.json,
    который должен лежать в той же папке, что и скрипт.
    Если файл не найден или ошибка — возвращает дефолтную личность.
    """
    personality_path = Path(__file__).parent / "personality_adrian.json"
    try:
        with open(personality_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        # Фолбек на стандартную личность
        return {
            "name": "Adrian, Visual Maestro",
            "constitution": (
                "You are Adrian, the Visual Maestro. Your mission is to translate "
                "strategy and text into powerful visual language. You create not just "
                "images, but visual stories that amplify emotional impact. Principles: "
                "Visual Storytelling, Aesthetic Intelligence, Depth and Metaphor, "
                "Technical Mastery."
            ),
        }


    # ---------- main --------------------------------------------------------
    async def generate(self, task: str, text: str) -> str:
        cache_key = self._get_cache_key(task, text)
        async with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        system_prompt = (
            f"{self.personality['constitution']}\n"
            "Create a concise, visual-focused English prompt (max 150 words) that "
            "captures the essence and emotion. Avoid text in the image. "
            "Use photorealistic style."
        )

        user_prompt = (
            f"TASK: {task}\nTEXT: {text}\n"
            "Create an image generation prompt describing visual elements, style, "
            "lighting, metaphor."
        )

        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()
            prompt = data["choices"][0]["message"]["content"].strip()
            logger.info("Generated prompt: %s…", prompt[:80])
            async with self._cache_lock:
                self._cache[cache_key] = prompt
            return prompt
        except Exception as e:
            logger.error("OpenRouter API error: %s", e)
            return f"Photorealistic image based on: {text[:100]}"


# ─── Image Saver ────────────────────────────────────────────────────────────
class ImageSaver:
    def __init__(self, save_dir: Path = Path("results/ready_for_publish")):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(topic: str) -> str:
        clean = "".join(c for c in topic if c.isalnum() or c in " -_").strip()
        return re.sub(r"[-_\s]+", "_", clean)

    def _validate_path(self, filename: str) -> Path:
        full_path = (self.save_dir / filename).resolve()
        if self.save_dir.resolve() not in full_path.parents:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return full_path

    async def save_from_url(self, image_url: str, topic: str) -> Dict[str, str]:
        if not is_safe_url(image_url):
            raise HTTPException(
                status_code=400, detail="Disallowed image URL (SSRF protection)"
            )

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
                            raise HTTPException(
                                status_code=413, detail="Image too large"
                            )
                        await f.write(chunk)

                if not validate_image_format(file_path):
                    file_path.unlink()
                    raise HTTPException(status_code=400, detail="Invalid image format")

                logger.info("Image saved: %s", file_path)
                return {"image_path": str(file_path), "status": "saved"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Download failed: %s", e)
                raise HTTPException(status_code=500, detail="Failed to download image")


def validate_image_format(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as img:
            return img.format in {"PNG", "JPEG", "JPG", "WEBP"}
    except Exception:
        return False


# ─── URL-валидация ──────────────────────────────────────────────────────────
def is_safe_url(url_str: str) -> bool:
    try:
        parsed = urlparse(url_str.lower())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False

        try:  # IP-адрес?
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_loopback or ip.is_private:
                return False
        except ValueError:
            pass  # hostname – это домен

        return parsed.hostname in ALLOWED_HOSTS
    except Exception:
        return False


# ─── FAL Image Generator ────────────────────────────────────────────────────
class FALImageGenerator:
    def __init__(self, api_key: str, model: str = "fal-ai/flux-pro"):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://api.fal.ai/v1/run/{model}"

    async def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image_size": os.getenv("IMAGE_SIZE", "landscape_4_3"),
            "num_inference_steps": int(os.getenv("FAL_STEPS", "28")),
            "guidance_scale": float(os.getenv("FAL_GUIDANCE", "3.5")),
            "num_images": 1,
            "enable_safety_checker": True,
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
                logger.info("FAL generated image: %s", image_url)
                return image_url
            except httpx.HTTPStatusError as e:
                logger.error(
                    "FAL API HTTP error %s: %s", e.response.status_code, e.response.text
                )
                raise HTTPException(
                    status_code=502, detail="Image generation failed (FAL)"
                )
            except Exception as e:
                logger.error("FAL API error: %s", e)
                raise HTTPException(status_code=500, detail="Image generation failed")


# ─── Designer Service ───────────────────────────────────────────────────────
class DesignerService:
    def __init__(self):
        self.secret_manager = SecretManager()
        self.fal_model = os.getenv("FAL_MODEL", "fal-ai/flux-pro")
        self.openrouter_model = os.getenv("DESIGNER_MODEL", "google/gemini-2.5-flash")

        self.prompt_generator: Optional[PromptGenerator] = None
        self.image_generator: Optional[FALImageGenerator] = None
        self.image_saver: Optional[ImageSaver] = None

    # ---------- lifecycle ---------------------------------------------------
    async def startup(self):
        try:
            self.fal_api_key = self.secret_manager.get("fal_api_key")
            self.openrouter_api_key = self.secret_manager.get("openrouter_api_key")

            self.prompt_generator = PromptGenerator(
                self.openrouter_api_key, self.openrouter_model
            )
            await self.prompt_generator.__aenter__()

            self.image_generator = FALImageGenerator(
                self.fal_api_key, self.fal_model
            )
            self.image_saver = ImageSaver()

            logger.info("✅ Designer Service initialized. Model: %s", self.fal_model)
        except Exception as e:
            logger.critical("❌ Failed to initialize DesignerService: %s", e)
            raise

    async def shutdown(self):
        if self.prompt_generator:
            await self.prompt_generator.__aexit__(None, None, None)

    # ---------- business methods -------------------------------------------
    async def create_image(self, task: str, text: str) -> Dict[str, Any]:
        if not self.prompt_generator:
            raise HTTPException(status_code=503, detail="Service not ready")
        try:
            prompt = await self.prompt_generator.generate(task, text)
            image_url = await self.image_generator.generate(prompt)
            return {
                "image_url": image_url,
                "english_prompt": prompt,
                "model": self.fal_model,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Image creation failed: %s", e)
            raise HTTPException(status_code=500, detail="Image generation failed")

    async def save_image(self, image_url: str, topic: str) -> Dict[str, Any]:
        if not self.image_saver:
            raise HTTPException(status_code=503, detail="Service not ready")
        return await self.image_saver.save_from_url(image_url, topic)


# ─── FastAPI приложение ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Designer Adrian API", version="3.2")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

designer = DesignerService()


@app.on_event("startup")
async def startup_event():
    await designer.startup()


@app.on_event("shutdown")
async def shutdown_event():
    await designer.shutdown()


# ---------- health ---------------------------------------------------------
@app.get("/health", tags=["system"])
async def health():
    try:
        if not designer.prompt_generator:
            raise RuntimeError("Prompt generator not initialized")

        await designer.prompt_generator._client.get("/models", timeout=5.0)

        Path("results/ready_for_publish").mkdir(parents=True, exist_ok=True)

        return {
            "status": "ok",
            "agent": "designer",
            "name": "Adrian, Visual Maestro",
            "fal_model": designer.fal_model,
            "openrouter_model": designer.openrouter_model,
            "storage": "ready",
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {
            "status": "degraded",
            "error": str(e),
            "agent": "designer",
        }


# ---------- endpoints ------------------------------------------------------
@app.post("/create_image", tags=["design"])
@limiter.limit("3/minute")
async def api_create_image(request: CreateImageRequest):
    return await designer.create_image(request.task_from_chief, request.post_text)


@app.post("/save_image", tags=["design"])
async def api_save_image(request: SaveImageRequest):
    return await designer.save_image(request.image_url, request.topic)


# ─── Local run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if DEBUG_MODE:
        logger.info("🔧 Debug mode: ON")
    logger.info("🚀 Starting Designer Adrian API v3.2…")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
