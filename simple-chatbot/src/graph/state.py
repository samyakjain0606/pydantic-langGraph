from langgraph.graph import MessagesState
from pydantic import Field
from typing import Optional, List, Dict, Any

from src.core.models import Message
from src.core.config import settings

class ChatbotState(MessagesState):
    """State for the chatbot workflow."""
    conversation_id: str = Field(..., description="ID of the current conversation")
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    current_message: Optional[Dict[str, Any]] = Field(default=None)
    model_config: Dict[str, Any] = Field(
        default_factory=lambda: settings.AVAILABLE_MODELS[settings.DEFAULT_MODEL],
        description="Configuration for the selected language model"
    )

    def add_message(self, message: Message):
        """Add a message to the state, converting it to a dict."""
        message_dict = {
            "content": message.content,
            "type": message.type,
        }
        self.messages.append(message_dict)

    def get_messages(self) -> List[Message]:
        """Get messages as Message objects."""
        return [
            Message(
                content=msg["content"],
                type=msg["type"]
            )
            for msg in self.messages
        ]

    def update_model_config(self, model_name: str):
        """Update the model configuration."""
        self.model_config = settings.AVAILABLE_MODELS[model_name]

    class Config:
        arbitrary_types_allowed = True 