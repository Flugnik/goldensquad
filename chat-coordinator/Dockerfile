FROM python:3.11-slim

WORKDIR /app

# Системные утилиты
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY . .

# Безопасный пользователь
RUN useradd -m -u 1000 chatuser && chown -R chatuser:chatuser /app
USER chatuser

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["python", "chat_coordinator.py"]

