from fastapi import FastAPI
import pickle
import uvicorn
import sys
import pandas as pd
import time


from app.constans import MODEL_PATH, PREPROCESSOR_PATH,CSV_PATH_TRAIN
from model.adaboost_custom import (
    SimpleAdaBoost,
    TitanicPipeline,
    TitanicOutliersTransformer,
    ExpectedColumns,
    TitanicCategoriesTransformer,
    TitanicImputationTransformer,
    TitanicScalingTransformer,
)

setattr(sys.modules.get("__main__"), "SimpleAdaBoost", SimpleAdaBoost)
setattr(sys.modules.get("__main__"), "TitanicPipeline", TitanicPipeline)
setattr(sys.modules.get("__main__"), "TitanicOutliersTransformer", TitanicOutliersTransformer)
setattr(sys.modules.get("__main__"), "ExpectedColumns", ExpectedColumns)
setattr(sys.modules.get("__main__"), "TitanicCategoriesTransformer", TitanicCategoriesTransformer)
setattr(sys.modules.get("__main__"), "TitanicImputationTransformer", TitanicImputationTransformer)
setattr(sys.modules.get("__main__"), "TitanicScalingTransformer", TitanicScalingTransformer)


model = pickle.load(open(MODEL_PATH, "rb"))
preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))

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
        "team": "macbuntu",
        "model" : "AdaBoostClassifier",
        "base_estimator" : "DecisionTreeClassifier(max_depth=1)",
        "n_estimators": 100,
        "preprocessing": {
            "pclass": "Selected by ExpectedColumns; if missing imputed with most_frequent; one-hot encoded (OneHotEncoder drop='first', handle_unknown='error') producing N-1 binary columns; unseen categories raise an error.",
            "sex": "Selected by ExpectedColumns; if missing imputed with most_frequent; one-hot encoded (drop='first') producing binary column(s); no scaling applied.",
            "age": "Outlier rule: values > 100 -> set to NaN; imputed with median(age); then scaled with StandardScaler -> (age - mean)/scale.",
            "sibsp": "If missing imputed with median(sibsp); then scaled with StandardScaler.",
            "parch": "If missing imputed with median(parch); then scaled with StandardScaler.",
            "fare": "Capped at upper = Q3 + 1.5*IQR for outliers; if missing imputed with median(fare); then scaled with StandardScaler.",
            "embarked": "If missing imputed with most_frequent; one-hot encoded (drop='first', handle_unknown='error') producing N-1 binary columns; unseen categories raise an error."
        }
    }

@app.post("/predict", tags=["prediction"])
async def predict(data: dict):
    """
        Endpoint de predicción.
    """
    try:
        features = pd.DataFrame([data["features"]])
        processed = preprocessor.transform(features)
        prediction = model.predict(processed)
        return {
            "prediction": int(prediction),
        }
    except Exception as e:
        return {
            "error" : "Error in prediction"
        }
# train endpoint
@app.get("/train", tags=["training"])
async def train():
    try:
        start_time = time.time()

        data = pd.read_csv(CSV_PATH_TRAIN)
        n_samples = len(data)
        preprocessorData = preprocessor.transform(data)
        y = data["survived"].map(preprocessor.transformTarget)
        model.fit(preprocessorData, y)
        elapsed_time = time.time() - start_time

        return {
            "status": "success",
            "message": "Model trained successfully",
            "training_samples": n_samples,
            "elapsed_time_seconds": round(elapsed_time, 2)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error during training",
            "error": str(e)
        }


def start():
    uvicorn.run(app, host="0.0.0.0", port=8000)
