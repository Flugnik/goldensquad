import os
import logging
import asyncio
import ipaddress
from typing import Dict
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
import openai
import uvicorn
import aiofiles
from contextlib import asynccontextmanager

# === Settings & Constants ===
COPYWRITER_MODEL_ID = os.getenv("COPYWRITER_MODEL_ID", "anthropic/claude-sonnet-4")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2500"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
DEFAULT_PORT = int(os.getenv("PORT", "8001"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost").split(",") if o.strip()]
TRUSTED_PROXIES = os.getenv("TRUSTED_PROXIES", "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16").split(",")
PERSONA_PATH = os.getenv("PERSONA_PATH", "personas/copywriter/Alina_Persona.txt")
CLIENT_DNA_PATH = os.getenv("CLIENT_DNA_PATH", "client_profiles/nikolay_dna.txt")
METHODS_PATH = os.getenv("METHODS_PATH", "personas/copywriter/Poisk-informacii.txt")
RATE_LIMIT_STORAGE = os.getenv("RATE_LIMIT_STORAGE", "redis://redis:6379")
REQUIRE_ORIGIN = os.getenv("REQUIRE_ORIGIN", "1") == "1"
SERVICE_KEY = os.getenv("SERVICE_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NETWORKS = []

def init_networks():
    global NETWORKS
    NETWORKS = [ipaddress.ip_network(net.strip()) for net in TRUSTED_PROXIES if net.strip()]

# === SecretManager ===
class SecretManager:
    def __init__(self, allow_env_fallback: bool = True):
        self.allow_env_fallback = allow_env_fallback
        self._secrets: Dict[str, str] = {}

    async def async_load(self, name: str) -> str:
        path = f"/run/secrets/{name}"
        if os.path.exists(path):
            st = os.stat(path)
            if (st.st_uid not in (os.getuid(), 0)) or (st.st_mode & 0o077):
                raise PermissionError(f"Unsafe permissions for secret {path}")
            async with aiofiles.open(path, mode="r") as f:
                self._secrets[name] = (await f.read()).strip()
                return self._secrets[name]
        if self.allow_env_fallback:
            env_var = name.upper()
            if val := os.getenv(env_var):
                logger.warning(f"DEV-mode: using {env_var} from ENV")
                self._secrets[name] = val
                return val
        raise RuntimeError(f"Secret {name} not found")

    def get(self, name: str) -> str:
        if name in self._secrets:
            return self._secrets[name]
        path = f"/run/secrets/{name}"
        if os.path.exists(path):
            st = os.stat(path)
            if (st.st_uid not in (os.getuid(), 0)) or (st.st_mode & 0o077):
                raise PermissionError(f"Unsafe permissions for secret {path}")
            with open(path) as f:
                self._secrets[name] = f.read().strip()
                return self._secrets[name]
        if self.allow_env_fallback:
            env_var = name.upper()
            if val := os.getenv(env_var):
                logger.warning(f"DEV-mode: using {env_var} from ENV")
                self._secrets[name] = val
                return val
        raise RuntimeError(f"Secret {name} not found")

# === PromptBuilder (метод system_prompt, корректный cache_clear) ===
class PromptBuilder:
    def __init__(self, persona_path: str, client_dna_path: str, methods_path: str, reload: bool = False):
        self.persona_path = persona_path
        self.client_dna_path = client_dna_path
        self.methods_path = methods_path
        self.reload = reload
        self._cache = {}

    def _read(self, path: str, required: bool = False) -> str:
        try:
            with open(path, encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            if required:
                logger.error(f"Required profile file missing: {path}")
                raise
            logger.warning(f"Optional profile file missing: {path}")
            return ""

    def system_prompt(self) -> str:
        if self.reload or "system_prompt" not in self._cache:
            pc = self._read(self.persona_path, True)
            cdna = self._read(self.client_dna_path)
            sm = self._read(self.methods_path)
            prompt = f"Ты — профессиональный копирайтер Алина Сомова.\n{pc}\n"
            if cdna:
                prompt += f"ДНК клиента:\n{cdna}\n"
            if sm:
                prompt += f"Методы работы:\n{sm}\n"
            self._cache["system_prompt"] = prompt.strip()
        return self._cache["system_prompt"]

    def cache_clear(self):
        for attr in list(self._cache.keys()):
            try:
                del self._cache[attr]
            except AttributeError:
                pass

# === LLMClient с connection pool ===
class LLMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = MAX_TOKENS):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        http_opts = {
            "timeout": 90,
            "connect_timeout": 10,
            "limits": openai.httpx.Limits(max_keepalive=20)
        }
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1"),
            http_options=http_opts
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=16),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.Timeout,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIError,
        )),
    )
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("OpenAI completion call failed", exc_info=True)
            raise

    async def close(self) -> None:
        await self.client.aclose()

# === Pydantic v2 модель ===
class WritePostRequest(BaseModel):
    task_from_chief: str = Field(min_length=10)
    brand_voice_info: str = Field(default="", max_length=1000)

    @field_validator('task_from_chief', mode="after")
    @classmethod
    def non_empty_task(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Task description must not be empty")
        return v

    @field_validator('brand_voice_info', mode="after")
    @classmethod
    def non_empty_brand(cls, v: str) -> str:
        if v and not v.strip():
            raise ValueError("Brand voice info must not be empty if provided")
        return v

# === IP extraction с кешем сетей ===
def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        for ip in (i.strip() for i in xff.split(",")):
            try:
                addr = ipaddress.ip_address(ip)
                if not any(addr in net for net in NETWORKS):
                    return ip
            except ValueError:
                continue
    return request.client.host if request.client else "unknown"

# === Limiter с Redis backend ===
limiter = Limiter(key_func=client_ip, storage_uri=RATE_LIMIT_STORAGE)
app = FastAPI(title="Copywriter AI", version="3.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Middleware для ограничения тела запроса
class LimitBodySizeMiddleware:
    def __init__(self, app, max_body_size: int):
        self.app = app
        self.max_body_size = max_body_size
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            received = await receive()
            body = received.get('body', b'')
            if len(body) > self.max_body_size:
                response = JSONResponse({"detail": "Request body too large"}, 413)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

app.add_middleware(LimitBodySizeMiddleware, max_body_size=1_000_000)

@app.middleware("http")
async def csrf_guard(request: Request, call_next):
    if request.method == "POST":
        if SERVICE_KEY and request.headers.get("x-api-key") == SERVICE_KEY:
            return await call_next(request)
        if REQUIRE_ORIGIN:
            origin = request.headers.get("origin") or ""
            referer = request.headers.get("referer") or ""
            if not origin and not referer:
                return JSONResponse({"detail": "CSRF check: Origin required"}, 403)
            if not any(origin.startswith(o) for o in ALLOWED_ORIGINS) and not any(referer.startswith(o) for o in ALLOWED_ORIGINS):
                return JSONResponse({"detail": "CSRF check failed"}, 403)
    return await call_next(request)

# Lifespan контекст со всеми инициализациями
secret_manager = SecretManager()
prompt_builder = PromptBuilder(PERSONA_PATH, CLIENT_DNA_PATH, METHODS_PATH, reload=bool(os.getenv("DEV_MODE", "0")) == "1")
llm_client = None

class Copywriter:
    def __init__(self, sm: SecretManager, pb: PromptBuilder, llm: LLMClient):
        self.secret_manager = sm
        self.prompt_builder = pb
        self.llm_client = llm
        self.model_id = llm.model

    async def check_model_status(self) -> bool:
        try:
            models = await self.llm_client.client.models.list()
            return any(m.id == self.model_id for m in models.data)
        except Exception as e:
            logger.exception("Failed to check model status")
            return False

    async def write_post(self, task_from_chief: str, brand_voice_info: str = "") -> Dict[str, str]:
        if self.prompt_builder.reload:
            self.prompt_builder.cache_clear()
        sys_prompt = self.prompt_builder.system_prompt()
        user_prompt = (
            f"ЗАДАНИЕ ОТ ШЕФА:\n{task_from_chief}\n\n"
            f"ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ О БРЕНДЕ:\n{brand_voice_info}\n\n"
            "Напиши пост согласно твоей профессиональной конституции."
        )
        try:
            text = await self.llm_client.generate(sys_prompt, user_prompt)
            logger.info("Post generated, %s chars", len(text))
            return {"post_text": text}
        except FileNotFoundError as ex:
            logger.error("Profile or constitution not found", exc_info=True)
            raise HTTPException(status_code=404, detail="Profile/constitution not found") from ex
        except (openai.APIConnectionError, openai.Timeout, openai.APITimeoutError, openai.RateLimitError, openai.APIError) as ex:
            logger.error("LLM unavailable", exc_info=True)
            raise HTTPException(status_code=503, detail="LLM temporarily unavailable") from ex
        except Exception as exc:
            logger.exception("LLM generation failed (unexpected)")
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    async def close(self) -> None:
        await self.llm_client.close()

copywriter: Copywriter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, copywriter
    # Инициализация/parallel preload secret(s)
    await asyncio.gather(secret_manager.async_load("openrouter_api_key"))
    api_key = secret_manager.get("openrouter_api_key")
    llm_client = LLMClient(api_key, COPYWRITER_MODEL_ID, MAX_TOKENS)
    copywriter = Copywriter(secret_manager, prompt_builder, llm_client)
    init_networks()  # Подсети для rate limit
    yield
    await copywriter.close()

app.router.lifespan_context = lifespan

# === Эндпоинты с Depends и без глобальных race-condition! ===
def get_writer() -> Copywriter:
    if not copywriter:
        raise HTTPException(status_code=503, detail="Writer not ready")
    return copywriter

@app.get("/health", tags=["system"])
async def health(writer: Copywriter = Depends(get_writer)):
    ok = await writer.check_model_status()
    if not ok:
        return {"status": "degraded", "api_connection": "unhealthy", "model_id": writer.model_id}
    return {"status": "ok", "api_connection": "healthy", "model_id": writer.model_id}

@app.get("/model_status", tags=["system"])
async def model_status(writer: Copywriter = Depends(get_writer)):
    ok = await writer.check_model_status()
    if not ok:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return {"status": "ok", "model_id": writer.model_id}

@app.post("/write_post", tags=["copywriting"])
@limiter.limit(RATE_LIMIT)
async def api_write_post(request: WritePostRequest, writer: Copywriter = Depends(get_writer)):
    return await writer.write_post(request.task_from_chief, request.brand_voice_info)

if __name__ == "__main__":
    uvicorn.run(
        "copywriter:app",
        host="0.0.0.0",
        port=DEFAULT_PORT,
        log_level="info",
        reload=bool(os.getenv("DEV_MODE", "0")) == "1"
    )
