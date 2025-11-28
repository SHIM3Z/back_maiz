FROM python:3.12-slim

WORKDIR /app

# 1. ACTUALIZA APT Y INSTALA LIBRERÍAS DE GRÁFICOS
# Esto es necesario para libGL.so.1 (OpenCV, Torchvision) y libgthread
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar e instalar dependencias de Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

COPY . /app/

EXPOSE 8000

# Usar variable de entorno PORT (Render) si existe, sino usar 8000 (local)
# Formato shell permite expansión de variables ${PORT:-8000}
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

LABEL authors="shim3z"
