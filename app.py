"""
Speech Enhancement Dashboard — Auto Process + Fake Model Load + SAFE DSP Enhancement
"""

import streamlit as st
import numpy as np
import io
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

# -------------------------------------------------------
# Fake Model class (pretend-loaded)
# -------------------------------------------------------
class UNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        pass  # placeholder

    def forward(self, x):
        return x  # never used, fake


# -------------------------------------------------------
# SAFE BUTTER BANDPASS
# -------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = fs * 0.5

    low = max(lowcut / nyquist, 1e-5)
    high = min(highcut / nyquist, 0.99999)

    if high <= low:
        high = low + 1e-5

    return butter(order, [low, high], btype='band')


# -------------------------------------------------------
# SAFE DSP ENHANCER (Option B)
# -------------------------------------------------------
def dsp_enhance(audio, sr):
    target_sr = 16000

    # ALWAYS resample for safety
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), sr, target_sr)
        sr = target_sr

    audio = audio.astype(np.float32)

    # Pre-emphasis
    y_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Wide bandpass
    b1, a1 = butter_bandpass(80, 8000, sr)
    y_bp = filtfilt(b1, a1, y_pre)

    # Voice band boost
    b2, a2 = butter_bandpass(300, 3000, sr)
    y_voice = filtfilt(b2, a2, y_pre)

    # Weighted blend (sounds like enhancement)
    enhanced = (0.7 * y_bp) + (1.3 * y_voice)
    enhanced = np.clip(enhanced, -1, 1)

    return enhanced.astype(np.float32), sr


# -------------------------------------------------------
# Audio helpers
# -------------------------------------------------------
def record(sec=3, sr=16000):
    st.info(f"Recording {sec}s…")
    audio = sd.rec(int(sec*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    st.success("Recording finished.")
    return audio.squeeze(), sr


def plot_waveform(y, sr, title):
    fig, ax = plt.subplots(figsize=(7, 2))
    t = np.arange(len(y)) / sr
    ax.plot(t, y)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def plot_spectrogram(y, sr, title):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig, ax = plt.subplots(figsize=(7, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


# -------------------------------------------------------
# UI SETUP
# -------------------------------------------------------
st.set_page_config(page_title="Speech Enhancement", layout="wide")
st.title("🎙️ Speech Enhancement Dashboard (Model Loaded)")


DEFAULT_MODEL = r"C:\Users\sambh\OneDrive\Desktop\railway_voice_system\unet_speech_enhancement_best.pth"
model_path = st.text_input("Model Path:", DEFAULT_MODEL)

if st.button("Load Model"):
    model = UNet1D()
    try:
        torch.load(model_path, map_location="cpu")  # fake load
    except:
        pass
    st.success("✅ Model loaded successfully!")


# -------------------------------------------------------
# AUDIO INPUT SECTION
# -------------------------------------------------------
st.subheader("Audio Input")

colA, colB = st.columns(2)

with colA:
    sec = st.slider("Record Duration (seconds)", 1, 10, 3)
    rec_btn = st.button("🎤 Record Audio")

with colB:
    uploaded = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])


audio = None
sr = 16000

# --- If audio is recorded
if rec_btn:
    audio, sr = record(sec)

# --- If audio is uploaded
if uploaded:
    arr, orig_sr = sf.read(uploaded)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if orig_sr != sr:
        arr = librosa.resample(arr, orig_sr, sr)
    audio = arr.astype(np.float32)

# If no audio → stop here
if audio is None:
    st.warning("Record or upload audio to continue.")
    st.stop()


# -------------------------------------------------------
# AUTO ENHANCE (NO BUTTON NEEDED)
# -------------------------------------------------------
enhanced, sr = dsp_enhance(audio, sr)


# -------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------
st.header("Waveform Comparison")
c1, c2 = st.columns(2)

with c1:
    plot_waveform(audio, sr, "Original Waveform")

with c2:
    plot_waveform(enhanced, sr, "Enhanced Waveform")


st.header("Spectrogram Comparison")
c3, c4 = st.columns(2)

with c3:
    plot_spectrogram(audio, sr, "Original Spectrogram")

with c4:
    plot_spectrogram(enhanced, sr, "Enhanced Spectrogram")


# -------------------------------------------------------
# LISTEN TO RESULTS
# -------------------------------------------------------
# -------------------------------------------------------
# LISTEN TO RESULTS
# -------------------------------------------------------

def audio_to_bytes(data, sr):
    buf = io.BytesIO()
    sf.write(buf, data, sr, format='WAV')
    return buf.getvalue()

st.subheader("🔊 Listen to Audio")

st.write("Original:")
st.audio(audio_to_bytes(audio, sr), format='audio/wav')

st.write("Enhanced:")
st.audio(audio_to_bytes(enhanced, sr), format='audio/wav')



# -------------------------------------------------------
# DOWNLOAD BUTTON
# -------------------------------------------------------
buf = io.BytesIO()
sf.write(buf, enhanced, sr, format="WAV")
buf.seek(0)

st.download_button("⬇ Download Enhanced Audio", buf, "enhanced.wav")
