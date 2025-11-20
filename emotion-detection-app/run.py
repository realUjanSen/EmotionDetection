import os
import subprocess

def main():
    # Current directory (emotion-detection-app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Venv is in the parent directory: ../venv
    venv_python = os.path.abspath(os.path.join(current_dir, "..", "venv", "Scripts", "python.exe"))
    
    # App is in current directory
    app_path = "app.py"
    
    if not os.path.exists(venv_python):
        print(f"Error: Python interpreter not found at {venv_python}")
        return

    print(f"Starting app using: {venv_python}")
    try:
        subprocess.run([venv_python, app_path])
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
