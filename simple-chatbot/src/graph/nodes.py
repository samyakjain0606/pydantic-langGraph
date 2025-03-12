from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrockConverse
from typing_extensions import Literal

from src.core.config import settings
from src.core.models import Message
from src.storage.dynamodb import DynamoDBStorage
from src.core.logger import get_logger

# Initialize storage and logger
storage = DynamoDBStorage()
logger = get_logger(__name__)

def get_bedrock_chat(model_config: Dict[str, str]):
    """Get the Bedrock chat model."""
    think_params= {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 16000
        }
    }
    return ChatBedrockConverse(
        model_id=model_config["model_id"],
        region_name=settings.AWS_REGION,
        max_tokens=16001,
        additional_model_request_fields=think_params
        # model_kwargs={"temperature": 0.7}
    )

async def message_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming messages and prepare for conversation."""
    logger.info("Entering message_handler_node")
    logger.debug(f"Input state: {state}")
    
    # Get the last message
    last_message = state["messages"][-1]
    logger.info(f"Processing message: {last_message}")
    
    # Convert to LangChain format for the model
    if last_message["type"] == "user":
        lc_message = HumanMessage(content=last_message["content"])
    else:
        lc_message = AIMessage(content=last_message["content"])
    
    logger.info("Exiting message_handler_node")
    return {"current_message": lc_message}

async def conversation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the conversation and generate a response."""
    logger.info("Entering conversation_node")
    logger.debug(f"Input state: {state}")
    
    # Get all messages for context
    lc_messages = []
    for msg in state["messages"]:
        logger.debug(f"Converting message: {msg}")
        message_type = msg.get("type", "")
        logger.info(f"Message type: {message_type}")
        
        if message_type == "user":
            logger.info("Converting to HumanMessage")
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif message_type == "assistant":
            logger.info("Converting to AIMessage")
            lc_messages.append(AIMessage(content=msg["content"]))
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    logger.info(f"Converted messages for model: {lc_messages}")
    
    # Get response from selected model
    chat = get_bedrock_chat(state["model_config"])
    try:
        response = await chat.ainvoke(lc_messages)
        logger.info(f"Received response from model: {response}")
        
        # Handle Claude 3.7 response format
        if isinstance(response.content, list) and state["model_config"]["model_id"] == "us.anthropic.claude-3-7-sonnet-20250219-v1:0":
            # Extract reasoning and final response
            reasoning = None
            final_response = None
            
            for content in response.content:
                if content.get("type") == "reasoning_content":
                    reasoning = content["reasoning_content"]["text"]
                elif content.get("type") == "text":
                    final_response = content["text"]
            
            # Create messages for both reasoning and response
            messages = []
            if reasoning:
                messages.append({
                    "content": reasoning,
                    "type": "assistant_reasoning"
                })
            if final_response:
                messages.append({
                    "content": final_response,
                    "type": "assistant"
                })
            
            return {"messages": messages}
        else:
            # Handle regular response format
            return {"messages": [{
                "content": response.content,
                "type": "assistant"
            }]}
            
    except Exception as e:
        logger.error(f"Error from model: {str(e)}")
        raise

async def storage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store the conversation in DynamoDB."""
    logger.info("Entering storage_node")
    logger.debug(f"Input state: {state}")
    
    from src.core.models import Conversation, Message
    
    # Convert dict messages to Message objects
    messages = [
        Message(
            content=msg["content"],
            type=msg["type"]
        )
        for msg in state["messages"]
    ]
    
    # Create conversation object
    conversation = Conversation(
        id=state["conversation_id"],
        messages=messages
    )
    
    # Save to DynamoDB
    try:
        storage.save_conversation(conversation)
        logger.info("Successfully saved conversation to DynamoDB")
    except Exception as e:
        logger.error(f"Error saving to DynamoDB: {str(e)}")
        raise
    
    logger.info("Exiting storage_node")
    return {}

def should_continue(state: Dict[str, Any]) -> Literal["continue", "__end__"]:
    """Determine if the conversation should continue."""
    logger.info("Checking if conversation should continue")
    
    # Check if the last message indicates end of conversation
    last_message = state["messages"][-1]["content"].lower()
    
    if "goodbye" in last_message or "bye" in last_message:
        logger.info("Conversation ending")
        return "__end__"
    
    logger.info("Conversation continuing")
    return "continue"

def should_store(state: Dict[str, Any]) -> Literal["store", "skip"]:
    """Determine if we should store the conversation."""
    logger.info("Checking if conversation should be stored")
    
    # Get number of messages
    num_messages = len(state["messages"])
    logger.info(f"Current message count: {num_messages}")
    
    # Store if we've reached the batch size or it's a goodbye message
    if num_messages >= settings.STORAGE_BATCH_SIZE:
        logger.info("Storing conversation - reached batch size")
        return "store"
        
    # Check if it's a goodbye message
    last_message = state["messages"][-1]["content"].lower()
    if "goodbye" in last_message or "bye" in last_message:
        logger.info("Storing conversation - goodbye message")
        return "store"
    
    logger.info("Skipping storage")
    return "skip" 