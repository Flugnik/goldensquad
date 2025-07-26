# designer.Dockerfile - Финальная проверенная версия
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY designer_requirements.txt .
RUN pip install --no-cache-dir -r designer_requirements.txt

COPY . .
CMD ["python", "-u", "designer.py"]
