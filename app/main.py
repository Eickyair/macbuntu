from fastapi import FastAPI
from datetime import datetime
import platform
import uvicorn
import pickle
model = 

app = FastAPI(title="API ligera", version="0.1")

@app.get("/", tags=["root"])
async def read_root():
    return {"message": "API ligera funcionando", "status": "ok"}

@app.get("/health", tags=["health"])
async def health():
    """
    Endpoint de salud mínimo.
    Devuelve estado, timestamp UTC y nombre del host.
    """
    return {
        "status": "healthy",
        "time": datetime.utcnow().isoformat() + "Z",
        "host": platform.node(),
    }

@app.post("/predict", tags=["prediction"])
async def predict(data: dict):
    """
    Endpoint de predicción.
    Recibe datos y devuelve una predicción simulada.
    """
    return {
        "prediction": "resultado_ejemplo",
        "input_received": data,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)