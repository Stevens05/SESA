import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Mel-frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    delta_mfcc = librosa.feature.delta(mfccs)

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Spectral centroid : brightness of sound
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Bandwidth : Spread of frequencies around the centroid
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Rolloff : Frequency below which a certain % of the total spectral energy lies
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Flatness : Tonal vs. noisy quality (close to 1: noisy)
    flatness = librosa.feature.spectral_flatness(y=y)

    # Contrast : Difference in energy between peaks and valleys in spectrum
    contrast = librosa.feature.spectral_contrast(y=y)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    

    return np.hstack([mfccs.mean(axis=1), chroma.mean(axis=1), zcr.std(), zcr.mean(), centroid.mean(), centroid.std(), rms.mean(), rms.std(),
                      bandwidth.mean(), bandwidth.std(), rolloff.mean(), rolloff.std(), flatness.mean(), flatness.std(), tempo,
                      contrast.mean(), contrast.std(), delta_mfcc.mean(axis=1), mel_spec.mean(), mel_spec.std(), duration])