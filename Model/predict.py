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

# Get class labels from the trained model
class_labels = model.classes_.tolist()


def predict_output(user_input: dict):
    df = pd.DataFrame.from_dict([user_input])
    # Predict the class
    predicted_class = model.predict(df)[0]

    # Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    # Create mapping
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs,
    }
    output = model.predict(input_df)[0]
    return output
