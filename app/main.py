from fastapi import FastAPI
from datetime import datetime
import uvicorn
import pickle

app = FastAPI(title="API ligera", version="0.1")

@app.get("/health", tags=["health"])
async def health():
    """
    Endpoint de salud mínimo.
    Devuelve estado, timestamp UTC y nombre del host.
    """
    return {
        "status": "ok"
    }
@app.get("/info", tags=["info"])
async def info():
    """
    Endpoint de información.
    Devuelve información básica sobre la API.
    """
    return {
        "team": "Equipo de Desarrollo",
        "model" : "AdaBoostClassifier",
        "base_estimator" : "DecisionTreeClassifier(max_depth=1)",
        "n_estimators": 50,
        "preprocessing": {
            "pclass": 5,
            "sex": "male",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 10.5,
            "embarked": "S"
        }
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

