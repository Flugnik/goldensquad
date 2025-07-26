# copywriter.Dockerfile — финальная версия
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# 1. Устанавливаем curl для healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 2. Python-зависимости
COPY copywriter_requirements.txt .
RUN pip install --no-cache-dir -r copywriter_requirements.txt

# 3. Исходный код
COPY . .

CMD ["python", "-u", "copywriter.py"]




