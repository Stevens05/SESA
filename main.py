from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib, shutil, os
import tempfile
import librosa
import python_multipart
from utils.audio_features import extract_features

app = FastAPI()
model = joblib.load("model/best_multiclass_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "world"}

@app.post("/predict/")
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
