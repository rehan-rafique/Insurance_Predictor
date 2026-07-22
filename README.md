# Insurance Premium Category Predictor

A machine learning web application that predicts a user's **insurance premium category** (Low / Medium / High) based on personal, lifestyle, and demographic inputs. The project is built with a **FastAPI** backend serving a **scikit-learn Random Forest** model, and a **Streamlit** frontend for interactive use — fully containerized with Docker.

## Overview

Insurance providers price premiums based on risk factors such as age, BMI, smoking status, income, and location. This project trains a classification model on historical insurance data to predict which premium tier (Low, Medium, or High) a new applicant is likely to fall into, and exposes that model through a REST API and a simple web UI.

## Features

- **REST API** built with FastAPI for real-time premium category predictions
- **Interactive UI** built with Streamlit for non-technical users
- **Engineered features** computed automatically from raw user input (BMI, age group, lifestyle risk, city tier)
- **Input validation** using Pydantic models with strict typing and constraints
- **Prediction confidence scores** and per-class probabilities returned alongside the predicted category
- **Dockerized** for consistent deployment across environments
- **Health check endpoint** for monitoring model/service status

## Tech Stack

| Layer          | Technology                     |
|----------------|---------------------------------|
| Backend API    | FastAPI, Pydantic               |
| Model          | scikit-learn (Random Forest)    |
| Data Handling  | pandas, numpy                   |
| Frontend       | Streamlit                       |
| Serving        | Uvicorn                         |
| Containerization | Docker                        |

## Project Structure

```
Insurance_Predictor/
├── Config/
│   └── city_tier.py         # Tier 1 / Tier 2 city lists used for city_tier feature
├── FastAPI/
│   └── app.py                # API routes: /, /health, /predict
├── Frontend/
│   └── frontend.py           # Streamlit UI that consumes the API
├── Model/
│   ├── model.pkl              # Trained scikit-learn pipeline
│   └── predict.py             # Model loading and prediction logic
├── Schema/
│   └── user_input.py          # Pydantic schema with computed/engineered fields
├── insurance.csv              # Training dataset
├── insurance.ipynb            # Data exploration, feature engineering, and model training
├── requirements.txt
├── Dockerfile
└── README.md
```

## How It Works

1. The user submits raw details (age, weight, height, income, smoker status, city, occupation) either through the Streamlit UI or directly to the API.
2. The `UserInput` Pydantic schema (`Schema/user_input.py`) validates the raw input and automatically derives four engineered features:
   - **BMI** — calculated from weight and height
   - **Age Group** — young / adult / middle_aged / senior
   - **Lifestyle Risk** — low / medium / high, based on smoking status and BMI
   - **City Tier** — 1, 2, or 3, based on a predefined list of Indian cities
3. The engineered features are passed to a trained **Random Forest** classification pipeline (`Model/predict.py`), which returns:
   - The predicted premium category
   - A confidence score
   - Class probabilities for all categories
4. The FastAPI backend (`FastAPI/app.py`) exposes this logic as a REST API, which the Streamlit frontend calls to display results.

## Dataset

The model is trained on `insurance.csv`, containing records with the following columns:

`age, weight, height, income_lpa, smoker, city, occupation, insurance_premium_category`

Feature engineering and model training steps (including a Random Forest pipeline with one-hot encoding for categorical variables) are documented in `insurance.ipynb`.

## Getting Started

### Prerequisites
- Python 3.12+
- pip

### Installation

```bash
git clone https://github.com/rehan-rafique/Insurance_Predictor.git
cd Insurance_Predictor
pip install -r requirements.txt
```

### Running the API

```bash
uvicorn FastAPI.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### Running the Frontend

In a separate terminal:

```bash
streamlit run Frontend/frontend.py
```

> **Note:** `Frontend/frontend.py` currently points to a hardcoded API URL. Update the `API_URL` variable to `http://localhost:8000/predict` (or your deployed API address) before running locally.

## API Reference

### `GET /`
Returns a welcome message confirming the API is running.

### `GET /health`
Returns the service status, model version, and whether the model loaded successfully.

### `POST /predict`
Predicts the insurance premium category for a given user.

**Request body:**

```json
{
  "age": 30,
  "weight": 65.0,
  "height": 1.7,
  "income_lpa": 10.0,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

**Response:**

```json
{
  "predicted_category": "Medium"
}
```

## Running with Docker

Build and run the API in a container:

```bash
docker build -t insurance-predictor .
docker run -p 8000:8000 insurance-predictor
```

The API will be accessible at `http://localhost:8000`.

## Future Improvements

- Add automated tests for the API and prediction logic
- Deploy the frontend and backend together via Docker Compose
- Support batch predictions via CSV upload
- Add model retraining pipeline with versioning

## Author

Built by Rehan Rafique, AI/ML Engineer in training focused on AI-powered automation tools.
