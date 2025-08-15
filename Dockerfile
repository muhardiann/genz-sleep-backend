# Gunakan base image resmi Python yang ringan
FROM python:3.11-slim

# Tetapkan direktori kerja di dalam kontainer
WORKDIR /code

# Salin file requirements.txt terlebih dahulu.
# Langkah ini dioptimalkan untuk Docker caching.
COPY ./requirements.txt /code/requirements.txt

# Install semua library dari daftar belanjaan kita
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin semua sisa kode proyek ke dalam direktori kerja
COPY . /code/

# Perintah untuk menjalankan aplikasi saat kontainer dimulai
# Menggunakan Gunicorn sebagai server produksi yang memanggil Uvicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]