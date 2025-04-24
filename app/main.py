from fastapi import FastAPI, UploadFile, File
import joblib, shutil, os
import tempfile
from utils.audio_features import extract_features

app = FastAPI()
model = joblib.load("model/best_multiclass_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    features = extract_features(tmp_path)
    X = scaler.transform([features])
    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]
    
    os.remove(tmp_path)
    return {"prediction": label}
