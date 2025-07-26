# Указываем базовый образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# Копируем файл с зависимостями и устанавливаем их
COPY bot_requirements.txt .
RUN pip install --no-cache-dir -r bot_requirements.txt

# Копируем остальной код проекта в контейнер
COPY . .

# Команда, которая будет выполняться при запуске контейнера
CMD ["python", "-u", "telegram_bot.py"]
