# Gunakan base image resmi Python
FROM python:3.9-slim

# Set direktori kerja di dalam kontainer
WORKDIR /code

# Salin file requirements terlebih dahulu untuk caching
COPY ./requirements.txt /code/requirements.txt

# Install dependensi
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin semua sisa kode proyek ke dalam direktori kerja
COPY . /code/

# Jalankan aplikasi menggunakan gunicorn saat kontainer dimulai
# Hugging Face Spaces menyediakan port melalui variabel $PORT
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:7860"]