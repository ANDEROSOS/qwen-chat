FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY app.py .

# Variables de entorno
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache

# Crear directorio de cache
RUN mkdir -p /app/cache

# Puerto
EXPOSE 7860

# Los modelos se descargan en segundo plano al iniciar (lazy loading)
# Esto permite que el servidor responda inmediatamente

# Comando de inicio
CMD ["python", "app.py"]
