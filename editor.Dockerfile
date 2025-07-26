# editor.Dockerfile - Финальная проверенная версия
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY editor_requirements.txt .
RUN pip install --no-cache-dir -r editor_requirements.txt

COPY . .
CMD ["python", "-u", "chief_editor.py"]
