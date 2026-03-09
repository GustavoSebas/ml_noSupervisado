# app/db.py
import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

def get_engine():
    pwd = quote_plus(DB_PASS)

    # Si DB_HOST empieza con /cloudsql/, significa que estamos en Cloud Run 
    # y debemos conectarnos a través del socket UNIX de Cloud SQL.
    if DB_HOST.startswith("/cloudsql/"):
        url = (
            f"postgresql+psycopg2://{DB_USER}:{pwd}"
            f"@/{DB_NAME}?host={DB_HOST}"
        )
    else:
        # Conexión estándar TCP (desarrollo local, etc.)
        url = (
            f"postgresql+psycopg2://{DB_USER}:{pwd}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

    return create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=280
    )
