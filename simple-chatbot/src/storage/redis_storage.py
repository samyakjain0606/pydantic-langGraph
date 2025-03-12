import json
import redis
from datetime import timedelta
from typing import List, Dict, Optional
from src.core.models import Message
from src.core.config import settings

class RedisStorage:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True
        )
        self.expiry_time = timedelta(days=1)

    def save_chat_session(self, session_id: str, messages: List[Message]):
        """Save chat session to Redis with 1-day expiry."""
        messages_data = [
            {"content": msg.content, "type": msg.type}
            for msg in messages
        ]
        self.redis_client.setex(
            f"chat:{session_id}",
            self.expiry_time,
            json.dumps(messages_data)
        )
        # Add to sessions list
        self.redis_client.sadd("chat_sessions", session_id)

    def get_chat_session(self, session_id: str) -> List[Message]:
        """Retrieve chat session from Redis."""
        data = self.redis_client.get(f"chat:{session_id}")
        if not data:
            return []
        
        messages_data = json.loads(data)
        return [
            Message(content=msg["content"], type=msg["type"])
            for msg in messages_data
        ]

    def get_all_sessions(self) -> List[str]:
        """Get all active chat session IDs."""
        return list(self.redis_client.smembers("chat_sessions"))

    def delete_chat_session(self, session_id: str):
        """Delete a chat session."""
        self.redis_client.delete(f"chat:{session_id}")
        self.redis_client.srem("chat_sessions", session_id) 