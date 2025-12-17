# Stage 1: Builder
# Usamos una imagen slim para empezar
FROM python:3.10-slim AS builder

# Instalar dependencias del sistema necesarias para compilar (pandas, numpy, scikit-learn lo suelen necesitar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Crear virtualenv para aislar dependencias
RUN python -m venv /opt/venv
# Asegurar que usamos el pip del virtualenv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Instalar dependencias
# --no-cache-dir para reducir tamaño
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
# Empezamos con una imagen limpia
FROM python:3.10-slim

WORKDIR /app

# Copiar el entorno virtual desde el builder
COPY --from=builder /opt/venv /opt/venv

# Habilitar el path del entorno virtual
ENV PATH="/opt/venv/bin:$PATH"

# Copiar el código fuente
COPY src/ ./src/
COPY main.py .

# Exponer el puerto
EXPOSE 8000

# Usuario no root para seguridad (opcional pero recomendado)
RUN useradd -m myuser
USER myuser

# Comando de arranque
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
