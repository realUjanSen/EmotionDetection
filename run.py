import os
import subprocess

def main():
    # Path to emotion-detection-app directory
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emotion-detection-app')
    os.chdir(app_dir)

    # Venv is in the root directory: ./venv
    venv_python = os.path.abspath(os.path.join(app_dir, '..', 'venv', 'Scripts', 'python.exe'))

    # App is in emotion-detection-app directory
    app_path = 'app.py'

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
