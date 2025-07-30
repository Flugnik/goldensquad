# utils/supabase_logger.py
# Централизованный модуль логирования для Golden Squad с интеграцией Supabase

import os
import json
import uuid
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from supabase import create_client, Client
import asyncio
from functools import wraps

class SupabaseLogger:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')  # service_role key
        
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError("SUPABASE_URL и SUPABASE_KEY должны быть установлены в переменных окружения!")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        print(f"✅ SupabaseLogger инициализирован. URL: {self.supabase_url[:50]}...", file=sys.stderr)

    def log_event(self, 
                 agent_name: str,
                 event_type: str,
                 input_data: Dict[str, Any],
                 output_data: Dict[str, Any],
                 elapsed_ms: int,
                 topic_id: Optional[str] = None,
                 stage: Optional[str] = None,
                 business_theme: Optional[str] = None,
                 status: str = "success",
                 error_message: Optional[str] = None,
                 imperatives_check: bool = False,
                 constitution_files: Optional[Dict] = None,
                 reporting_blocks_present: bool = False,
                 warnings: Optional[List] = None) -> str:
        """
        Записывает событие в Supabase лог
        Возвращает ID записи
        """
        
        try:
            # Генерируем ID если не передан
            if not topic_id:
                topic_id = str(uuid.uuid4())
            
            log_entry = {
                "topic_id": topic_id,
                "request_id": str(uuid.uuid4()),
                "agent_name": agent_name,
                "event_type": event_type,
                "stage": stage,
                "input_data": input_data,
                "output_data": output_data,
                "elapsed_ms": elapsed_ms,
                "business_theme": business_theme,
                "status": status,
                "error_message": error_message,
                "imperatives_check": imperatives_check,
                "constitution_files_loaded": constitution_files or {},
                "reporting_blocks_present": reporting_blocks_present,
                "warnings": warnings or []
            }
            
            # Записываем в Supabase
            result = self.client.table('agent_workflow_log').insert(log_entry).execute()
            
            if result.data:
                record_id = result.data[0]['id']
                print(f"📝 Лог записан: {agent_name}/{event_type} -> {record_id}", file=sys.stderr)
                return record_id
            else:
                print(f"⚠️ Ошибка записи лога в Supabase: {result}", file=sys.stderr)
                return ""
                
        except Exception as e:
            print(f"❌ Критическая ошибка логирования: {e}", file=sys.stderr)
            # В случае ошибки Supabase - можно записать в локальный файл как fallback
            self._fallback_log(agent_name, event_type, input_data, output_data, elapsed_ms)
            return ""

    def _fallback_log(self, agent_name: str, event_type: str, input_data: Dict, output_data: Dict, elapsed_ms: int):
        """Fallback логирование в локальный файл если Supabase недоступен"""
        try:
            fallback_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "event_type": event_type,
                "input_data": input_data,
                "output_data": output_data,
                "elapsed_ms": elapsed_ms
            }
            
            with open("logs/fallback_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"❌ Даже fallback логирование не удалось: {e}", file=sys.stderr)

    def log_topic_status(self, topic_text: str, business_theme: str, status: str = "pending", client_id: str = "nikolay"):
        """Записывает тему в архив тем"""
        try:
            topic_entry = {
                "topic_text": topic_text,
                "business_theme": business_theme,
                "status": status,
                "client_id": client_id,
                "generated_by": "chief-editor"
            }
            
            result = self.client.table('topics_archive').insert(topic_entry).execute()
            
            if result.data:
                return result.data[0]['id']
            return None
            
        except Exception as e:
            print(f"❌ Ошибка записи темы в архив: {e}", file=sys.stderr)
            return None

    def get_topics_archive(self, business_theme: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Получает архив тем для анализа"""
        try:
            query = self.client.table('topics_archive').select("*").order('created_at', desc=True).limit(limit)
            
            if business_theme:
                query = query.eq('business_theme', business_theme)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            print(f"❌ Ошибка чтения архива тем: {e}", file=sys.stderr)
            return []

# Глобальный экземпляр логгера
logger = SupabaseLogger()

# Декоратор для авто-логирования методов агентов
def log_agent_action(agent_name: str, event_type: str, stage: Optional[str] = None):
    """
    Декоратор для автоматического логирования действий агентов
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            topic_id = kwargs.get('topic_id') or str(uuid.uuid4())
            
            try:
                # Выполняем основную функцию
                result = await func(*args, **kwargs)
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                # Определяем input/output данные
                input_data = {"args": str(args[1:]), "kwargs": kwargs}  # args[0] это self
                output_data = {"result": str(result)[:1000]}  # ограничиваем размер
                
                # Проверяем наличие блоков отчётности
                reporting_blocks_present = False
                if isinstance(result, dict) and "post_text" in result:
                    text = result["post_text"]
                    reporting_blocks_present = any(block in text for block in [
                        "📋 РАБОЧИЕ ФАЙЛЫ:",
                        "🔍 ПРОВЕРКА СООТВЕТСТВИЯ:",
                        "🔎 ИСТОЧНИКИ ИНФОРМАЦИИ:",
                        "📚 ВЫВОДЫ ДЛЯ РАЗВИТИЯ:"
                    ])
                
                # Записываем успешное выполнение
                logger.log_event(
                    agent_name=agent_name,
                    event_type=event_type,
                    input_data=input_data,
                    output_data=output_data,
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage=stage,
                    status="success",
                    reporting_blocks_present=reporting_blocks_present
                )
                
                return result
                
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                # Записываем ошибку
                logger.log_event(
                    agent_name=agent_name,
                    event_type=event_type,
                    input_data={"args": str(args[1:]), "kwargs": kwargs},
                    output_data={"error": str(e)},
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage=stage,
                    status="error",
                    error_message=str(e)
                )
                
                raise  # Пробрасываем исключение дальше
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Для синхронных функций
            start_time = time.time()
            topic_id = kwargs.get('topic_id') or str(uuid.uuid4())
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                logger.log_event(
                    agent_name=agent_name,
                    event_type=event_type,
                    input_data={"args": str(args[1:]), "kwargs": kwargs},
                    output_data={"result": str(result)[:1000]},
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage=stage,
                    status="success"
                )
                
                return result
                
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                logger.log_event(
                    agent_name=agent_name,
                    event_type=event_type,
                    input_data={"args": str(args[1:]), "kwargs": kwargs},
                    output_data={"error": str(e)},
                    elapsed_ms=elapsed_ms,
                    topic_id=topic_id,
                    stage=stage,
                    status="error",
                    error_message=str(e)
                )
                
                raise
        
        # Возвращаем соответствующий wrapper
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
