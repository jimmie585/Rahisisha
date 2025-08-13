from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from functools import lru_cache
from typing import List, Dict

# === PATHS ===
BASE_DIR = r"C:\Users\ADMIN\Downloads\student_app"
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUBJECT_DIR = os.path.join(MODEL_DIR, "subjects")  # <- per-subject files live here

os.makedirs(SUBJECT_DIR, exist_ok=True)

def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load {path}: {e}")

# === LIGHTWEIGHT MODELS (load at startup) ===
try:
    score_regressor = load_model(os.path.join(MODEL_DIR, "score_predictor.pkl"))
    score_encoder   = load_model(os.path.join(MODEL_DIR, "score_encoder.pkl"))
    cluster_model   = load_model(os.path.join(MODEL_DIR, "cluster_predictor.pkl"))
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# === FASTAPI TYPES ===
class StudentInput(BaseModel):
    admission: str           # used for labeling the response
    cohort: str
    workspace: str
    workspace_id: str

student_app = FastAPI(title="🎓 Student Performance Prediction API")

@student_app.get("/")
def root():
    return {"message": "✅ Student score prediction API is live."}

def preprocess_input(data: StudentInput) -> pd.DataFrame:
    # Must match the encoder training columns exactly
    return pd.DataFrame([{
        "cohort": data.cohort,
        "workspace": data.workspace,
        "workspace_id": data.workspace_id
    }]).astype(str)

def available_subjects() -> List[str]:
    # Detect subjects by file pattern: SUBJECT_model.pkl
    names = []
    for fname in os.listdir(SUBJECT_DIR):
        if fname.endswith("_model.pkl"):
            names.append(fname.replace("_model.pkl", ""))
    return sorted(set(names))

@lru_cache(maxsize=64)
def get_subject_artifacts(subject: str):
    """
    Lazy-load one subject's model+encoder on demand.
    Keep a small cache so repeated calls are fast without blowing RAM.
    """
    mpath = os.path.join(SUBJECT_DIR, f"{subject}_model.pkl")
    epath = os.path.join(SUBJECT_DIR, f"{subject}_encoder.pkl")
    if not os.path.exists(mpath) or not os.path.exists(epath):
        raise FileNotFoundError(f"No artifacts found for subject '{subject}' in {SUBJECT_DIR}")
    model = joblib.load(mpath)    # loading one small file at a time prevents MemoryError
    enc   = joblib.load(epath)
    return model, enc

def recommendation_text(avg_score: float) -> str:
    if avg_score >= 70:
        return "🟢 Excellent - Maintain performance"
    if avg_score >= 50:
        return "🟡 Good - Could improve with guidance"
    return "🔴 Needs intervention - Below passing threshold"

@student_app.get("/subjects")
def list_subjects():
    return {"subjects": available_subjects()}

@student_app.post("/predict/")
def predict_scores(data: StudentInput):
    try:
        input_df = preprocess_input(data)
        X = score_encoder.transform(input_df)

        avg_score = float(score_regressor.predict(X)[0])
        cluster   = int(cluster_model.predict(X)[0])

        # Subject-wise predictions (lazy load each)
        subjects = available_subjects()
        subject_predictions: List[Dict] = []
        for subject in subjects:
            try:
                model, enc = get_subject_artifacts(subject)
                Xs = enc.transform(input_df)
                score = float(model.predict(Xs)[0])
                subject_predictions.append({
                    "subject": subject,
                    "score": round(score, 2),
                    "status": "Pass" if score >= 50 else "Fail"
                })
            except Exception as se:
                # Skip problematic subject, but keep the rest working
                subject_predictions.append({
                    "subject": subject,
                    "error": f"Could not predict: {se}"
                })

        return {
            "student": data.admission,
            "cluster": cluster,
            "average_score": round(avg_score, 2),
            "recommendation": recommendation_text(avg_score),
            "subjects": subject_predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
