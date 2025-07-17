from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY') or 'your_secret_key'
    MYSQL_HOST = os.getenv('MYSQL_HOST') or 'localhost'
    MYSQL_USER = os.getenv('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD') or ''
    MYSQL_DB = os.getenv('MYSQL_DB') or 'emotionusers'
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit for uploads
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}