from setuptools import setup, find_packages

setup(
    name="simple-langgraph-chatbot",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "langgraph",
        "langgraph-checkpoint==2.0.10",
        "langgraph-sdk==0.1.51",
        "boto3==1.35.90",
        "pydantic==2.10.5",
        "pydantic-settings==2.4.0",
        "pydantic_core==2.27.2",
        "python-dotenv==1.0.1",
        "langchain-aws==0.2.12",
        "langchain-core",
        "streamlit==1.39.0",
        "streamlit-keyup==0.2.4",
    ],
) 