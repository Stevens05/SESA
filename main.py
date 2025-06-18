from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib, shutil, os
import tempfile
import librosa
import python_multipart
from utils.audio_features import extract_features
from prometheus_fastapi_instrumentator import Instrumentator
from database import engine
from models import Base
from pydantic import BaseModel
from sqlalchemy.orm import Session
from models import Prediction


Base.metadata.create_all(bind=engine)

app = FastAPI()
model = joblib.load("model/best_multiclass_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")

# main.py
class PredictionInput(BaseModel):
    predicted_class: str
    true_class: str

from database import SessionLocal
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Instrumentation Prometheus
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse("index.html")

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...),
                        db: Session = Depends(get_db)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    features = extract_features(tmp_path)
    X = scaler.transform([features])
    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]

    real_label = file.filename.split("_")[0]

    db_pred = Prediction(
        predicted_label=label,
        true_label=real_label
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    
    os.remove(tmp_path)
    return {"message": "Prediction saved",
            "prediction": label,
            "true_label": real_label}

    # raise HTTPException(
    #     status_code=404,
    #     detail=f"file {tmp_path} is not valid"
    # )