import io

import cv2
import numpy as np
from fastapi import HTTPException

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64

from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = "runs/segment/train/weights/best.pt"  # Actualiza con la ruta de tu modelo
model = YOLO(MODEL_PATH)

origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "https://ldzcc7vk-4200.brs.devtunnels.ms"# Dirección de tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permite solicitudes desde el frontend
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


# @app.post("/process_image/")
# async def process_image(file: UploadFile = File(...)):
#     # Verifica si se recibe un archivo
#     if not file:
#         return JSONResponse(content={"error": "No file provided"}, status_code=400)
#
#     # Lee la imagen desde el archivo subido
#     try:
#         image_data = await file.read()
#         image = Image.open(BytesIO(image_data))
#
#         # Procesa la imagen (ejemplo: convierte a escala de grises)
#         image = image.convert("L")
#
#         # Guarda la imagen procesada en memoria en formato PNG
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#
#         # Codifica la imagen procesada a base64 para enviarla en formato JSON
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#
#         # Devuelve la imagen en formato JSON
#         return JSONResponse(content={"image": img_str})
#
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen del archivo
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Realizar la predicción
        results = model.predict(image, conf=0.6)[0]  # Asegúrate de que 'model' esté definido

        # Procesar la imagen anotada y convertirla al esquema de color RGB
        annotated_image = results.plot()  # Esto devuelve un array NumPy
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Conversión a RGB

        # Convertir el array NumPy a una imagen PIL
        annotated_image = Image.fromarray(np.uint8(annotated_image)).convert("RGB")

        # Guardar la imagen en un buffer en formato JPEG
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        buffered.seek(0)

        # Convertir a Base64 para enviarlo al frontend
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(f"imagen: {img_str}")

        return JSONResponse(content={"image": img_str})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
