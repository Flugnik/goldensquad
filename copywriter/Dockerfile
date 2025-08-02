FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Системные утилиты (curl для health-check)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY . .

# Непривилегированный пользователь
RUN useradd -m -u 1000 copywriter && chown -R copywriter:copywriter /app
USER copywriter

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-u", "copywriter.py"]
