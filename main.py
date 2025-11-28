import io

import cv2
import numpy as np
from fastapi import HTTPException

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import base64

from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = "runs/segment/train/weights/best.pt"  # Actualiza con la ruta de tu modelo

# Cargar modelo de forma lazy para evitar timeout en inicio
model = None

def get_model():
    global model
    if model is None:
        print("🔄 Cargando modelo YOLO por primera vez...")
        model = YOLO(MODEL_PATH)
        print("✅ Modelo YOLO cargado exitosamente en memoria")
    else:
        print("♻️ Reutilizando modelo YOLO ya cargado (no se recarga)")
    return model

origins = [
    "https://front-maiz.onrender.com"# Dirección de tu frontend
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
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        import time
        start_time = time.time()

        # Leer la imagen del archivo
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Realizar la predicción
        current_model = get_model()
        results = current_model.predict(image, conf=0.6)[0]

        # Procesar la imagen anotada y convertirla al esquema de color RGB
        annotated_image = results.plot()  # Esto devuelve un array NumPy
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Conversión a RGB

        # Convertir el array NumPy a una imagen PIL
        annotated_image = Image.fromarray(np.uint8(annotated_image)).convert("RGB")

        # Guardar la imagen en un buffer en formato JPEG
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG", quality=85)
        buffered.seek(0)

        # Convertir a Base64 para enviarlo al frontend
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Extraer metadata de las detecciones
        num_detections = 0
        if hasattr(results, 'boxes') and results.boxes is not None:
            num_detections = len(results.boxes)

        processing_time = round(time.time() - start_time, 2)

        print(f"✅ Imagen procesada: {num_detections} detecciones en {processing_time}s")

        return JSONResponse(content={
            "image": img_str,
            "detections": num_detections,
            "processing_time_seconds": processing_time,
            "model_confidence_threshold": 0.6
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


# Endpoint alternativo: Devuelve imagen binaria directa (más eficiente, -33% de tamaño)
@app.post("/process_image/binary")
async def process_image_binary(file: UploadFile = File(...)):
    """
    Endpoint alternativo que devuelve la imagen procesada como binario JPEG.
    Ventajas: 33% más pequeño, más rápido.
    Desventajas: No incluye metadata adicional.
    """
    try:
        # Leer la imagen del archivo
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Realizar la predicción
        current_model = get_model()
        results = current_model.predict(image, conf=0.6)[0]

        # Procesar la imagen anotada
        annotated_image = results.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(np.uint8(annotated_image)).convert("RGB")

        # Guardar en buffer
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG", quality=85)
        buffered.seek(0)

        print(f"✅ Imagen procesada (binario)")

        # Devolver imagen binaria directamente
        return StreamingResponse(
            buffered,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=processed_image.jpg",
                "X-Detections": str(len(results.boxes) if hasattr(results, 'boxes') and results.boxes is not None else 0)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
