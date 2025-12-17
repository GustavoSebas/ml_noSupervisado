import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Variables de entorno o defaults
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "gestion_tareas")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

def get_engine():
    return create_engine(DATABASE_URL, echo=False)
