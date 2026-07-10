from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from Schema.user_input import UserInput

# Import ML model (Pickle)
with open(
    "E:\Python\Virtual Environment\Insurance_Predictor\Model\model.pkl", "rb"
) as f:
    model = pickle.load(f)

MODEL_VERSION = "1.0.0"

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API"}


@app.get("/health")
def health_check():
    return {"status": "Ok", "version": MODEL_VERSION, "model_loaded": model is not None}


@app.post("/predict")
def predict_premium(data: UserInput):

    input_df = pd.DataFrame(
        [
            {
                "bmi": data.bmi,
                "age_group": data.age_group,
                "lifestyle_risk": data.lifestyle_risk,
                "city_tier": data.city_tier,
                "income_lpa": data.income_lpa,
                "occupation": data.occupation,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    return JSONResponse(status_code=200, content={"predicted_category": prediction})
