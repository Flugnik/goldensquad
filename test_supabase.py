import asyncio
import os
from dotenv import load_dotenv  # Загрузка переменных из .env

# Загружаем переменные окружения из .env файла
load_dotenv()

# Проверяем переменные окружения
print("SUPABASE_URL:", os.getenv('SUPABASE_URL'))
print("SUPABASE_KEY:", "****" + str(os.getenv('SUPABASE_KEY'))[-10:] if os.getenv('SUPABASE_KEY') else "НЕ НАЙДЕН")

from utils.supabase_logger import logger

async def test_connection():
    try:
        # Тестовая запись
        record_id = logger.log_event(
            agent_name="test-agent",
            event_type="connection_test",
            input_data={"test": "connection"},
            output_data={"status": "connected"},
            elapsed_ms=100
        )
        print(f"✅ Подключение к Supabase успешно! ID записи: {record_id}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
