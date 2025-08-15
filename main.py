from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from typing import Literal # Digunakan untuk validasi input string yang spesifik

# ---------------------------------------------------------------------
# BAGIAN 1: MODEL INPUT (PYDANTIC) DIMODIFIKASI
# Menerima input yang lebih ramah pengguna dari frontend
# ---------------------------------------------------------------------
class PredictionInput(BaseModel):
    usia: int = Field(..., ge=13, le=27, description="Usia antara 13-27 tahun")
    pola_waktu: Literal["Pagi", "Siang", "Malam", "Dini Hari"]
    jeda_tidur: Literal[
        "Langsung tidur sambil membuka media sosial",
        "Kurang dari 30 menit",
        "30 - 60 menit",
        "lebih dari 60 menit"
    ]
    fomo_scores: list[int] = Field(..., min_items=10, max_items=10)
    # Durasi sekarang diterima sebagai string "Jam:Menit"
    durasi_app1: str = Field(..., pattern=r"^\d{1,2}:\d{2}$")
    durasi_app2: str = Field(..., pattern=r"^\d{1,2}:\d{2}$")
    durasi_app3: str = Field(..., pattern=r"^\d{1,2}:\d{2}$")

# ---------------------------------------------------------------------
# BAGIAN 2: INISIALISASI & MEMUAT MODEL
# Tidak ada perubahan di bagian ini
# ---------------------------------------------------------------------
app = FastAPI(title="API Prediksi Kualitas Tidur Final", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the GenZ Sleep Prediction API! Visit /docs for API documentation."}

try:
    preprocessor = joblib.load('HASIL_klasifikasi_scaler.pkl')
    model_h5 = load_model('HASIL_klasifikasi_model.h5')
    feature_names_from_scaler = preprocessor.get_feature_names_out()
    print("✅ Model dan preprocessor berhasil dimuat.")
    print(f"✅ Urutan fitur yang dikenali: {list(feature_names_from_scaler)}")
except Exception as e:
    preprocessor = None
    model_h5 = None
    feature_names_from_scaler = []
    print(f"❌ Gagal memuat model atau preprocessor: {e}")

# ---------------------------------------------------------------------
# BAGIAN 3: ENDPOINT PREDIKSI (LOGIKA UTAMA DIMODIFIKASI)
# ---------------------------------------------------------------------
@app.post("/api/predict", tags=["Prediction"])
def predict_sleep_quality(data: PredictionInput):
    if not all([preprocessor, model_h5]):
        return {"error": "Model atau preprocessor tidak tersedia di server."}

    # === TAHAP A: PRE-PROCESSING INPUT DARI FRONTEND ===

    # 1. Encode `jeda_tidur` dari teks menjadi angka 0-3
    jeda_tidur_map = {
        "Langsung tidur sambil membuka media sosial": 0,
        "Kurang dari 30 menit": 1,
        "30 - 60 menit": 2,
        "lebih dari 60 menit": 3
    }
    jeda_tidur_encoded = jeda_tidur_map.get(data.jeda_tidur, 0) # Default ke 0 jika ada kesalahan

    # 2. Parse durasi "HH:MM" menjadi total menit untuk setiap aplikasi
    def parse_hhmm_to_minutes(time_str: str) -> int:
        try:
            hours, minutes = map(int, time_str.split(':'))
            return (hours * 60) + minutes
        except:
            return 0 # Default ke 0 jika format salah
    
    durasi_app1_min = parse_hhmm_to_minutes(data.durasi_app1)
    durasi_app2_min = parse_hhmm_to_minutes(data.durasi_app2)
    durasi_app3_min = parse_hhmm_to_minutes(data.durasi_app3)

    # 3. Hitung total dari semua nilai yang sudah diproses
    total_fomo_score = sum(data.fomo_scores)
    total_durasi_min = durasi_app1_min + durasi_app2_min + durasi_app3_min

    # === TAHAP B: PERSIAPAN DATA UNTUK MODEL (Sama seperti sebelumnya) ===

    # Buat dictionary lengkap dengan data yang sudah bersih
    input_data = {
        'Usia': data.usia,
        'jeda_tidur': jeda_tidur_encoded, # Menggunakan nilai yang sudah di-encode
        'Skor_Fomo': total_fomo_score,
        'Durasi_Instagram': durasi_app1_min, # Menggunakan nilai menit
        'Durasi_Tiktok': durasi_app2_min,   # Menggunakan nilai menit
        'Durasi_WhatsApp': durasi_app3_min, # Menggunakan nilai menit
        'Total_Durasi': total_durasi_min,
        'waktu_Dini Hari (00:00 - 06:00)': 1 if data.pola_waktu == "Dini Hari" else 0,
        'waktu_Pagi (06:00 - 12:00)': 1 if data.pola_waktu == "Pagi" else 0,
        'waktu_Siang (12:00 - 18:00)': 1 if data.pola_waktu == "Siang" else 0,
        'waktu_Malam (18:00 - 00:00)': 1 if data.pola_waktu == "Malam" else 0
    }
    input_df = pd.DataFrame([input_data])
    
    # Pastikan urutan kolom benar
    try:
        input_df_ordered = input_df[feature_names_from_scaler]
    except KeyError as e:
        return {"error": f"Ketidakcocokan nama kolom saat mengurutkan: {e}"}

    # === TAHAP C: PREDIKSI & FORMAT OUTPUT (Sama seperti sebelumnya) ===
    
    try:
        data_processed = preprocessor.transform(input_df_ordered)
        prediksi_proba = model_h5.predict(data_processed, verbose=0)
        skor_keyakinan_buruk = prediksi_proba[0][0]
    except Exception as e:
        return {"error": f"Gagal saat penskalaan atau prediksi: {e}"}

    if skor_keyakinan_buruk > 0.5:
        kategori = "Buruk"
        persentase = skor_keyakinan_buruk * 100
    else:
        kategori = "Baik"
        persentase = (1 - skor_keyakinan_buruk) * 100

    if kategori == "Baik":
        pesan_judul = "Luar Biasa! Pertahankan Kebiasaan Sehatmu."
        pesan_deskripsi = "Kualitas tidurmu berada di jalur yang benar..."
        rekomendasi = []
    else:
        pesan_judul = "Waktunya Perbaiki Pola Tidurmu!"
        pesan_deskripsi = "Hasil prediksimu menunjukkan adanya risiko gangguan tidur..."
        rekomendasi = [
            {"judul": "Jeda Digital", "detail": "Coba ciptakan 'zona bebas HP' minimal 60 menit sebelum tidur."},
            {"judul": "Atur Ulang Waktu", "detail": "Model kami mendeteksi penggunaan media sosial yang tinggi di malam hari."},
            {"judul": "Kurangi Dosis", "detail": "Total durasi penggunaan media sosialmu cukup tinggi. Coba tetapkan batas harian."}
        ]

    return {
        "prediksi": kategori,
        "keyakinan": f"{persentase:.2f}",
        "pesan": {"judul": pesan_judul, "deskripsi": pesan_deskripsi, "rekomendasi": rekomendasi}
    }

# Endpoint Health Check tidak berubah
@app.get("/api/health", tags=["Health Check"])
def health_check():
    return {"status": "ok", "model_loaded": model_h5 is not None, "preprocessor_loaded": preprocessor is not None}