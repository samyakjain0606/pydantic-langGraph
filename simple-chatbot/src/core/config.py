import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration."""
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
    
    # DynamoDB Configuration
    DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "chatbot_conversations")
    
    # LLM Configuration
    AVAILABLE_MODELS: Dict[str, Dict] = {
        "Claude 3.7 Sonnet": {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "description": "Latest Claude 3 Sonnet model - Latest Model with Best quality responses"
        },
        "Claude 3.5 Sonnet v2": {
            "provider": "bedrock",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "description": "Claude 3.5 Sonnet v2 - Best in terms of quality and speed"
        },
        # "Deepseek R1": {
        #     "provider": "bedrock",
        #     "model_id": "deepseek.r1-v1:0",
        #     "description": "Deepseek R1 - Best in terms of quality and speed"
        # }
    }
    DEFAULT_MODEL = "Claude 3.5 Sonnet v2"
    
    # Memory Configuration
    MAX_HISTORY_LENGTH = 50  # Maximum number of messages to keep in memory
    
    # Storage Configuration
    STORAGE_BATCH_SIZE = 20  # Number of messages before storing to DynamoDB

    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

settings = Settings() 