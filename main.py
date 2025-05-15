from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib, shutil, os
import tempfile
import librosa
import python_multipart
from utils.audio_features import extract_features
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
model = joblib.load("model/best_multiclass_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")


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

    # raise HTTPException(
    #     status_code=404,
    #     detail=f"file {tmp_path} is not valid"
    # )
