import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# Memuat model CNN
model = load_model('corn_disease_model.h5')

# Definisikan label kelas
class_names = ['Blight', 'Common Rust', 'Gray Leaf', 'Healthy']

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image = image.resize((224, 224))  
    image_array = np.array(image) / 255.0  
    return np.expand_dims(image_array, axis=0)  

# CSS untuk styling 
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        font-family: 'Arial', sans-serif;
        color: #333333;
    }

    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        padding: 10px;
        border-bottom: 2px solid #4caf50;
        margin-bottom: 20px;
    }

    .result-healthy {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32; /* Hijau */
        margin-top: 20px;
        text-align: center;
    }

    .result-sick {
        font-size: 24px;
        font-weight: bold;
        color: #d32f2f; /* Merah */
        margin-top: 20px;
        text-align: center;
    }

    footer {
        font-size: 14px;
        color: #2e7d32;
        text-align: center;
        padding: 10px;
        margin-top: 20px;
        border-top: 1px solid #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul aplikasi
st.markdown("<h1 class='main-title'>Prediksi Penyakit 🌽</h1>", unsafe_allow_html=True)

# Pilihan untuk mengunggah gambar atau mengambil foto
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Atau ambil foto menggunakan kamera")

# Cek apakah pengguna mengunggah gambar atau mengambil foto
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    
elif camera_input is not None:
    image = Image.open(camera_input)
    st.image(image, caption='Foto yang diambil.', use_column_width=True)

if uploaded_file is not None or camera_input is not None:
    # Proses gambar
    processed_image = preprocess_image(image)

    # Prediksi
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_probability = predictions[0][predicted_class_index] * 100  # Akurasi untuk kelas prediksi terpilih

    # Tampilkan hasil prediksi dengan warna sesuai dengan kondisi
    predicted_class_name = class_names[predicted_class_index]
    if predicted_class_name == "Healthy":
        st.markdown(f"<div class='result-healthy'>Hasil Prediksi: {predicted_class_name} 🌿 ({predicted_probability:.2f}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-sick'>Peringatan! Hasil Prediksi: {predicted_class_name} ⚠️ ({predicted_probability:.2f}%)</div>", unsafe_allow_html=True)

    # Tampilkan probabilitas untuk setiap kelas
    st.subheader("Probabilitas:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")

# Footer
st.markdown("<footer>Dikembangkan oleh Alif Naywa • © 2024</footer>", unsafe_allow_html=True)
