# =====================================================================
# main.py - VERSI FINAL (Lazy Loading Model)
# =====================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from typing import Literal

# --- Pydantic Model (Tidak Berubah) ---
class PredictionInput(BaseModel):
    usia: int = Field(..., ge=13, le=27)
    pola_waktu: Literal["Pagi", "Siang", "Malam", "Dini Hari"]
    jeda_tidur: Literal["Langsung tidur sambil membuka media sosial", "Kurang dari 30 menit", "30 - 60 menit", "lebih dari 60 menit"]
    fomo_scores: list[int] = Field(..., min_items=10, max_items=10)
    durasi_app1: str = Field(..., pattern=r"^\d{1,2}:\d{2}$")
    durasi_app2: str = Field(..., pattern=r"^\d{1,2}:\d{2}$")
    durasi_app3: str = Field(..., pattern=r"^\d{1,2}:\d{2}$")

# --- Inisialisasi Aplikasi ---
app = FastAPI(title="API Prediksi Kualitas Tidur", version="10.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origin (untuk development)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Pastikan OPTIONS diizinkan secara eksplisit
    allow_headers=["*"],  # Mengizinkan semua header
)

# --- Endpoint ---
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome! API is running. Visit /docs for API documentation."}

@app.get("/api/health", tags=["Health Check"])
def health_check():
    # Health check sekarang hanya memeriksa apakah server berjalan, bukan apakah model sudah dimuat.
    return {"status": "ok"}

@app.post("/api/predict", tags=["Prediction"])
def predict_sleep_quality(data: PredictionInput):
    
    # === TAHAP A: MEMUAT MODEL DI DALAM FUNGSI (LAZY LOADING) ===
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        SCALER_PATH = os.path.join(BASE_DIR, 'HASIL_klasifikasi_scaler.pkl')
        MODEL_PATH = os.path.join(BASE_DIR, 'HASIL_klasifikasi_model.h5')
        
        preprocessor = joblib.load(SCALER_PATH)
        model_h5 = load_model(MODEL_PATH) 
        feature_names_from_scaler = preprocessor.get_feature_names_out()
    except Exception as e:
        # Jika GAGAL MEMUAT di sini, kita akan tahu ada masalah file.
        print(f"❌ Gagal memuat model atau preprocessor di dalam fungsi: {e}")
        return {"error": f"Gagal memuat file model di server. Detail: {e}"}

    # === TAHAP B: Pre-processing input (Tidak Berubah) ===
    jeda_tidur_map = {"Langsung tidur sambil membuka media sosial": 0, "Kurang dari 30 menit": 1, "30 - 60 menit": 2, "lebih dari 60 menit": 3}
    jeda_tidur_encoded = jeda_tidur_map.get(data.jeda_tidur, 0)

    def parse_hhmm_to_minutes(time_str: str) -> int:
        try:
            hours, minutes = map(int, time_str.split(':'))
            return (hours * 60) + minutes
        except: return 0
    
    durasi_app1_min = parse_hhmm_to_minutes(data.durasi_app1)
    durasi_app2_min = parse_hhmm_to_minutes(data.durasi_app2)
    durasi_app3_min = parse_hhmm_to_minutes(data.durasi_app3)
    total_fomo_score = sum(data.fomo_scores)
    total_durasi_min = durasi_app1_min + durasi_app2_min + durasi_app3_min

    input_data = {
        'Usia': data.usia, 'jeda_tidur': jeda_tidur_encoded, 'Skor_Fomo': total_fomo_score,
        'Durasi_Instagram': durasi_app1_min, 'Durasi_Tiktok': durasi_app2_min, 'Durasi_WhatsApp': durasi_app3_min,
        'Total_Durasi': total_durasi_min,
        'waktu_Dini Hari (00:00 - 06:00)': 1 if data.pola_waktu == "Dini Hari" else 0,
        'waktu_Pagi (06:00 - 12:00)': 1 if data.pola_waktu == "Pagi" else 0,
        'waktu_Siang (12:00 - 18:00)': 1 if data.pola_waktu == "Siang" else 0,
        'waktu_Malam (18:00 - 00:00)': 1 if data.pola_waktu == "Malam" else 0
    }
    input_df = pd.DataFrame([input_data])[feature_names_from_scaler]

    # === TAHAP C: Prediksi & Format Output (Tidak Berubah) ===
    try:
        data_processed = preprocessor.transform(input_df)
        prediksi_proba = model_h5.predict(data_processed, verbose=0)
        skor_keyakinan_buruk = prediksi_proba[0][0]
    except Exception as e:
        return {"error": f"Gagal saat penskalaan atau prediksi: {e}"}

    if skor_keyakinan_buruk > 0.5:
        kategori = "Buruk"; persentase = skor_keyakinan_buruk * 100
    else:
        kategori = "Baik"; persentase = (1 - skor_keyakinan_buruk) * 100

    if kategori == "Baik":
        pesan_judul = "Luar Biasa! Pertahankan Kebiasaan Sehatmu."; pesan_deskripsi = "Kualitas tidurmu berada di jalur yang benar..."; rekomendasi = []
    else:
        pesan_judul = "Waktunya Perbaiki Pola Tidurmu!"; pesan_deskripsi = "Hasil prediksimu menunjukkan adanya risiko gangguan tidur..."; rekomendasi = [
            {"judul": "Jeda Digital", "detail": "Coba ciptakan 'zona bebas HP' minimal 60 menit sebelum tidur."},
            {"judul": "Atur Ulang Waktu", "detail": "Model kami mendeteksi penggunaan media sosial yang tinggi di malam hari."},
            {"judul": "Kurangi Dosis", "detail": "Total durasi penggunaan media sosialmu cukup tinggi. Coba tetapkan batas harian."}
        ]
    
    return {"prediksi": kategori, "keyakinan": f"{persentase:.2f}", "pesan": {"judul": pesan_judul, "deskripsi": pesan_deskripsi, "rekomendasi": rekomendasi}}