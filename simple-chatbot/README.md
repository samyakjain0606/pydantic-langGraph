# Simple LangGraph Chatbot

A simple chatbot implementation using LangGraph, AWS Bedrock Claude 3.5, and DynamoDB.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your AWS credentials
5. Run the initialization script:
   ```bash
   python init_project.py
   ```

## Project Structure 

## Running the Application

1. Make sure you have initialized the project first
2. Run the Streamlit app:
   ```bash
   python run_app.py
   ```
3. Open your browser and navigate to http://localhost:8501 