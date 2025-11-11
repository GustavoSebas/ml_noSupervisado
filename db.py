from sqlalchemy import create_engine

# --- MYSQL interno de Railway ---
MYSQL_HOST = "mysql.railway.internal"
MYSQL_PORT = 3306
MYSQL_DB   = "railway"
MYSQL_USER = "root"
MYSQL_PASSWORD = "EDXzpjNKykmenwtSNlnlXnpGmckkIgRx"  # tu pass de Railway

def get_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=5)