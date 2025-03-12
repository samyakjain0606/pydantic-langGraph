from typing import List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Represents a single message in the conversation."""
    content: str
    type: str = Field(..., description="Type of message: 'user' or 'assistant'")

class Conversation(BaseModel):
    """Represents a conversation with its messages and metadata."""
    id: str
    messages: List[Message] = Field(default_factory=list)
    title: Optional[str] = None

class ChatState(BaseModel):
    """Represents the current state of the chat in the LangGraph workflow."""
    conversation_id: str
    messages: List[Message]
    metadata: dict = Field(default_factory=dict) 