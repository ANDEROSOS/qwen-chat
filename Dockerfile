FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-descargar modelos (esto tarda pero solo se hace una vez al construir)
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel; \
    print('Descargando Qwen2.5-1.5B-Instruct...'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    print('Descargando modelo de embeddings...'); \
    AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
    AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
    print('Modelos descargados!')"

# Puerto
EXPOSE 7860

# Comando de inicio
CMD ["python", "app.py"]
