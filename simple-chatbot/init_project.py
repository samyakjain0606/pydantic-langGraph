from src.storage.dynamodb import DynamoDBStorage
from src.core.config import settings

def init_project():
    """Initialize the project by setting up required resources."""
    print("Initializing project...")
    
    # Initialize DynamoDB
    print("Setting up DynamoDB...")
    storage = DynamoDBStorage()
    storage.create_table_if_not_exists()
    
    print("Project initialization complete!")

if __name__ == "__main__":
    init_project() 