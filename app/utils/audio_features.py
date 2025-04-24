import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Mel-frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    # RMS energy
    rms = librosa.feature.rms(y=y).mean()

    return np.hstack([mfccs_mean, chroma_mean, zcr, centroid, rms])