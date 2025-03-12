import uuid
import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage

from src.graph.graph import graph, create_initial_state
from src.core.models import Message
from src.storage.dynamodb import DynamoDBStorage
from src.storage.redis_storage import RedisStorage
from src.core.logger import get_logger
from src.core.config import settings

# Initialize storage and logger
storage = DynamoDBStorage()
redis_storage = RedisStorage()
logger = get_logger(__name__)

def init_session_state():
    """Initialize session state variables."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = redis_storage.get_all_sessions()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False

def load_chat_session(session_id: str):
    """Load a chat session from Redis."""
    st.session_state.conversation_id = session_id
    st.session_state.messages = redis_storage.get_chat_session(session_id)
    st.session_state.chat_started = True

def save_current_session():
    """Save current session to Redis."""
    redis_storage.save_chat_session(
        st.session_state.conversation_id,
        st.session_state.messages
    )

def display_messages():
    """Display chat messages."""
    for message in st.session_state.messages:
        if message.type == "assistant_reasoning":
            # Display reasoning in a different style
            with st.chat_message("assistant", avatar="🤔"):
                st.markdown("""
                <div style='padding: 10px; border-radius: 10px; border-left: 5px solid #9e9e9e;'>
                    <p style='color: #666; font-style: italic; margin: 0; font-size: 0.8em;'>
                        Thinking process:
                    </p>
                    <p style='margin: 5px 0 0 0; font-size: 0.9em;'>
                        {content}
                    </p>
                </div>
                """.format(content=message.content), unsafe_allow_html=True)
        else:
            # Display regular messages
            with st.chat_message("user" if message.type == "user" else "assistant"):
                st.write(message.content)

def cleanup_old_messages():
    """Remove old messages if we've exceeded the maximum history length."""
    if len(st.session_state.messages) > settings.MAX_HISTORY_LENGTH:
        # Keep the most recent messages
        st.session_state.messages = st.session_state.messages[-settings.MAX_HISTORY_LENGTH:]
        logger.info(f"Cleaned up messages to maximum length of {settings.MAX_HISTORY_LENGTH}")

async def process_message(user_message: str):
    """Process user message through the graph."""
    logger.info("Processing new message")
    
    # Create and add message
    message = Message(content=user_message, type="user")
    st.session_state.messages.append(message)
    
    # Cleanup old messages if needed
    cleanup_old_messages()
    
    # Get all current messages for context and create state
    state_dict = create_initial_state(st.session_state.conversation_id, st.session_state.selected_model)
    state_dict["messages"] = [
        {
            "content": msg.content,
            "type": msg.type,
        }
        for msg in st.session_state.messages
    ]
    
    try:
        response = await graph.ainvoke(state_dict)
        logger.info("Received response from graph")
        
        # Add AI response to session state
        if "messages" in response and response["messages"]:
            for msg_dict in response["messages"]:
                new_message = Message(
                    content=msg_dict["content"],
                    type=msg_dict["type"]
                )
                st.session_state.messages.append(new_message)
                logger.info(f"Added response message to session state: {new_message}")
        
        # Save to Redis after processing
        save_current_session()
        # Update chat sessions list
        st.session_state.chat_sessions = redis_storage.get_all_sessions()
        
    except Exception as e:
        logger.error(f"Error from graph: {str(e)}")
        st.error("Sorry, I encountered an error. Please try again.")

def main():
    st.set_page_config(layout="wide")
    
    # Initialize session
    init_session_state()
    
    # Sidebar for chat history
    with st.sidebar:
        st.title("Chat History")
        
        # New Chat button
        if st.button("New Chat"):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.selected_model = None
            st.session_state.chat_started = False
            st.rerun()
        
        st.divider()
        
        # Display chat sessions
        for session_id in st.session_state.chat_sessions:
            # Get first message or use session ID as fallback
            messages = redis_storage.get_chat_session(session_id)
            session_name = messages[0].content[:30] + "..." if messages and len(messages[0].content) > 30 else (messages[0].content if messages else f"Chat {session_id[:8]}")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(session_name, key=f"session_{session_id}"):
                    load_chat_session(session_id)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}"):
                    redis_storage.delete_chat_session(session_id)
                    st.session_state.chat_sessions = redis_storage.get_all_sessions()
                    if session_id == st.session_state.conversation_id:
                        st.session_state.conversation_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.selected_model = None
                        st.session_state.chat_started = False
                    st.rerun()
    
    # Main chat area
    st.title("💬 Sumo Logic Chatbot")
    
    if not st.session_state.chat_started and not st.session_state.selected_model:
        # Show model selection in the center when starting a new chat
        st.markdown("### Select a Model to Start Chat")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Model selection cards
            for model_name, model_info in settings.AVAILABLE_MODELS.items():
                with st.container(border=True):
                    st.markdown(f"#### {model_name}")
                    st.markdown(f"*{model_info['description']}*")
                    if st.button("Select", key=f"select_{model_name}"):
                        st.session_state.selected_model = model_name
                        st.session_state.chat_started = True
                        st.rerun()
                st.markdown("---")
    else:
        # Display current model
        st.caption(f"Using: {st.session_state.selected_model}")

        # Display existing messages
        display_messages()

        # Chat input
        if user_input := st.chat_input("Type your message here..."):
            # Immediately display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Show spinner while processing
            with st.chat_message("assistant"):
                with st.spinner("Processing message..."):
                    # Process message
                    asyncio.run(process_message(user_input))
            
            # Rerun to update the display with the new messages
            st.rerun()

if __name__ == "__main__":
    main() 