FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar PyTorch CPU-only (mucho mas pequeno ~200MB vs 3GB)
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Instalar resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el codigo
COPY app.py .

# Variables de entorno
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache

# Crear directorio de cache
RUN mkdir -p /app/cache

# Puerto
EXPOSE 7860

# Comando de inicio
CMD ["python", "app.py"]
