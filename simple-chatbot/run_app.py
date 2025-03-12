import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the Streamlit application."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add src directory to Python path
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, project_root)
    
    # Get the app path
    filename = os.path.join(project_root, "src", "ui", "streamlit_app.py")
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    
    # Run Streamlit
    subprocess.run(["streamlit", "run", filename], env=env)

if __name__ == "__main__":
    main() 