import boto3
from typing import List, Optional
import logging

from src.core.config import settings
from src.core.models import Conversation, Message

logger = logging.getLogger(__name__)

class DynamoDBStorage:
    """Handles conversation storage in DynamoDB."""
    
    def __init__(self):
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.table = self.dynamodb.Table(settings.DYNAMODB_TABLE_NAME)
        self._batch_count = 0
    
    def create_table_if_not_exists(self):
        """Creates the DynamoDB table if it doesn't exist."""
        try:
            self.dynamodb.create_table(
                TableName=settings.DYNAMODB_TABLE_NAME,
                KeySchema=[
                    {
                        'AttributeName': 'id',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'id',
                        'AttributeType': 'S'
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            print(f"Table {settings.DYNAMODB_TABLE_NAME} created successfully")
        except self.dynamodb.meta.client.exceptions.ResourceInUseException:
            print(f"Table {settings.DYNAMODB_TABLE_NAME} already exists")
    
    def save_conversation(self, conversation: Conversation):
        """Saves a conversation to DynamoDB."""
        # Convert the conversation to a dictionary
        item = {
            'id': conversation.id,
            'messages': [
                {
                    'content': msg.content,
                    'type': msg.type
                }
                for msg in conversation.messages
            ],
            'title': conversation.title
        }
        
        try:
            self.table.put_item(Item=item)
            self._batch_count = 0  # Reset batch count after successful save
            logger.info("Successfully saved conversation batch to DynamoDB")
        except Exception as e:
            logger.error(f"Failed to save conversation batch: {e}")
            raise
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieves a conversation from DynamoDB."""
        response = self.table.get_item(Key={'id': conversation_id})
        if 'Item' not in response:
            return None
            
        item = response['Item']
        
        # Convert messages back to Message objects
        messages = [
            Message(
                content=msg['content'],
                type=msg['type']
            )
            for msg in item['messages']
        ]
        
        return Conversation(
            id=item['id'],
            messages=messages,
            title=item.get('title')
        ) 