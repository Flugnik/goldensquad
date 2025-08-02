FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Системные утилиты
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY . .

# Директории для изображений
RUN mkdir -p results/ready_for_publish images

# Непривилегированный пользователь
RUN useradd -m -u 1000 designer && chown -R designer:designer /app
USER designer

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-u", "designer.py"]
