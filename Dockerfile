FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload" , "--host", "0.0.0.0", "--port", "8000"]

LABEL authors="shim3z"
