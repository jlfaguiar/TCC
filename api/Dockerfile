FROM python:3.10-slim

# Instala dependências essenciais
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia apenas arquivos essenciais
COPY requirements.txt .

# Instala dependências sem cache
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia código e modelos
COPY app.py .
COPY pipeline_functions.py .
COPY bertimbau-basic-sentimento/ bertimbau-basic-sentimento/
COPY llm-models/openhermes.gguf llm-models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
