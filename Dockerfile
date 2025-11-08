# Dockerfile para ejecutar una API FastAPI cuyo entrypoint es ./app/main.py
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala herramientas de compilación necesarias (opcional pero útil)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia todo el contenido del proyecto al contenedor
COPY . /app

# Actualiza pip e instala dependencias:
# si existe requirements.txt se usa, si no instala fastapi y uvicorn por defecto
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; \
       else pip install --no-cache-dir fastapi uvicorn; fi

# Crear un usuario no root y usarlo
RUN useradd -m appuser || true
USER appuser

EXPOSE 8000

# Ejecuta la app FastAPI (asumiendo que el objeto ASGI se llama `app` en app/main.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]