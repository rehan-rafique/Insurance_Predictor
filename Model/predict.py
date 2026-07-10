import sys
import os

# Add the Insurance_Predictor directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd

# Import ML model (Pickle)
# with open(
#     "E:\Python\Virtual Environment\Insurance_Predictor\Model\model.pkl", "rb"
# ) as f:
#     model = pickle.load(f)

# Load model using relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "Model", "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

MODEL_VERSION = "1.0.0"


def predict_output(user_input: dict):
    input_df = pd.DataFrame.from_dict([user_input])
    output = model.predict(input_df)[0]
    return output
