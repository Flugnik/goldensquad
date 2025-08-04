# Dockerfile - Версия 3.0 (Production-Ready)
FROM python:3.11-slim

# Предотвращение интерактивных диалогов при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Системные утилиты: только необходимые
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Создание непривилегированного пользователя ДО копирования кода
RUN adduser --system --no-create-home --group --uid 1000 designer

# Копируем только requirements.txt для эффективного кэширования слоев
COPY --chown=designer:designer requirements.txt .

# Устанавливаем зависимости от имени пользователя (не root)
USER designer
RUN pip install --user --no-cache-dir -r requirements.txt

# Копируем исходный код
USER root
COPY --chown=designer:designer . .
USER designer

# Создаем директории для изображений
RUN mkdir -p results/ready_for_publish images

# HEALTHCHECK: использует curl с таймаутом, совместим с непривилегированным пользователем
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -s http://localhost:8002/health -m 8 || exit 1

# Запуск через python -m (лучше практика)
CMD ["python", "-m", "designer"]