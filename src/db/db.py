# app/db.py
import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# ✅ Defaults = tus datos de Railway (proxy)
DB_HOST = os.getenv("DB_HOST", "centerbeam.proxy.rlwy.net")
DB_PORT = int(os.getenv("DB_PORT", "14209"))
DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "EDXzpjNKykmenwtSNlnlXnpGmckkIgRx")

def get_engine():
    pwd = quote_plus(DB_PASS)

    url = (
        f"mysql+pymysql://{DB_USER}:{pwd}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?charset=utf8mb4"
    )

    return create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=280,
        connect_args={"ssl": {}}, 
    )
