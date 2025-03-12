from typing import Dict, Any
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.graph.state import ChatbotState
from src.core.config import settings
from src.graph.nodes import (
    message_handler_node,
    conversation_node,
    storage_node,
    should_store
)

@lru_cache(maxsize=1)
def create_chat_graph() -> StateGraph:
    """Create the chat workflow graph."""
    
    # Initialize the graph with our state
    graph = StateGraph(ChatbotState)
    
    # Add nodes
    graph.add_node("message_handler", message_handler_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("storage", storage_node)
    
    # Define the flow
    # Start with message handling
    graph.add_edge(START, "message_handler")
    
    # Then process conversation
    graph.add_edge("message_handler", "conversation")
    
    # Add conditional edge for storage
    graph.add_conditional_edges(
        "conversation",
        should_store,
        {
            "store": "storage",  # Store if needed
            "skip": END  # Skip storage
        }
    )
    graph.add_edge("storage", END)
    
    return graph

# Create a compiled version of the graph
graph = create_chat_graph().compile()

# Function to create initial state
def create_initial_state(conversation_id: str, model_name: str = None) -> Dict[str, Any]:
    """Create initial state for the graph."""
    model_config = settings.AVAILABLE_MODELS[model_name] if model_name else settings.AVAILABLE_MODELS[settings.DEFAULT_MODEL]
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "metadata": {},
        "model_config": model_config
    } 