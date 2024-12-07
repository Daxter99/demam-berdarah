import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import pickle
import time

# Fungsi untuk mengambil data suhu dari ThingSpeak
def get_suhu_from_thingspeak(api_url, api_key, results=2):
    params = {
        "api_key": api_key,
        "results": results
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        feeds = data.get("feeds", [])
        if feeds:
            # Mengambil nilai suhu terakhir
            suhu = feeds[-1].get("field1", None)
            return float(suhu) if suhu else None
        else:
            st.warning("Tidak ada data tersedia dari ThingSpeak.")
            return None
    else:
        st.error(f"Gagal mengambil data dari ThingSpeak. Status code: {response.status_code}")
        return None

# URL API ThingSpeak
API_URL = "https://api.thingspeak.com/channels/2778023/fields/1.json"
API_KEY = "GN8KSJHYFO2YJD9K"

# Fungsi untuk hitungan mundur
def countdown_timer(seconds):
    countdown_placeholder = st.empty()  # Placeholder untuk memperbarui tulisan
    for i in range(seconds, 0, -1):
        countdown_placeholder.markdown(f"### Mengambil Suhu dalam **{i} detik**...")
        time.sleep(1)
    countdown_placeholder.markdown("### Menyegarkan halaman...")

# Fungsi untuk memuat model dan menghindari pemuatan ulang yang berulang-ulang
@st.cache
def load_model():
    return pickle.load(open('model_dd4.pkl', 'rb'))

# Load model
model = load_model()

# Tampilan utama aplikasi Streamlit
st.write("""
# Demam Berdarah Prediction
"Klasifikasi Tingkat Demam Berdarah Menggunakan Metode Naive Bayes Untuk Deteksi Dini"
""")

# Menampilkan gambar
img = Image.open('dd.jpg')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

# Sidebar untuk input parameter
st.sidebar.header('Input Parameters')

# Mendapatkan data suhu dari ThingSpeak
st.sidebar.subheader("Data Suhu dari ThingSpeak")
suhu_thingspeak = get_suhu_from_thingspeak(API_URL, API_KEY)

if suhu_thingspeak is not None:
    st.sidebar.write(f"**Suhu Terakhir (ThingSpeak):** {suhu_thingspeak}°C")
else:
    st.sidebar.write("Tidak dapat mengambil data suhu dari ThingSpeak.")

# Upload File CSV
upload_file = st.sidebar.file_uploader('Upload CSV File', type=['csv'])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        Umur = st.sidebar.slider('Umur', 20, 80)
        Jenis_Kelamin = st.sidebar.selectbox('Jenis Kelamin', ('L', 'P'))
        
        # Gunakan suhu dari ThingSpeak jika ada
        Suhu = suhu_thingspeak if suhu_thingspeak is not None else st.sidebar.slider('Suhu (Manual)', 30, 45)
        
        Ruam_Kulit = st.sidebar.selectbox('Ruam Kulit', ('YES', 'NO'))
        Manifestasi_perdarahan = st.sidebar.selectbox('Manifestasi perdarahan', ('YES', 'NO'))
        Kegagalan_Sirkulasi = st.sidebar.selectbox('Kegagalan Sirkulasi', ('YES', 'NO'))
        Syok_berat = st.sidebar.selectbox('Syok berat', ('YES', 'NO'))
        Uji_Tokniket = st.sidebar.selectbox('Uji Tokniket', ('YES', 'NO'))
        Kebocoran_plasma = st.sidebar.selectbox('Kebocoran Plasma', ('YES', 'NO'))
        Pendarahan_Spontan = st.sidebar.selectbox('Pendarahan Spontan', ('YES', 'NO'))
        Trombositopenia = st.sidebar.slider('Trombositopenia', 45000, 150000)
        Peningkatan_Hematokrit = st.sidebar.slider('Peningkatan Hematokrit', 0, 30)
        
        data = {
            'Umur': Umur,
            'Jenis Kelamin': Jenis_Kelamin,
            'Suhu': Suhu,
            'Ruam Kulit': Ruam_Kulit,
            'Manifestasi perdarahan': Manifestasi_perdarahan,
            'Kegagalan Sirkulasi': Kegagalan_Sirkulasi,
            'Syok berat': Syok_berat,
            'Uji Tokniket': Uji_Tokniket,
            'Kebocoran Plasma': Kebocoran_plasma,
            'Pendarahan spontan': Pendarahan_Spontan,
            'Trombositopenia': Trombositopenia,
            'Peningkatan Hematokrit': Peningkatan_Hematokrit
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    
    inputan = input_user()

# Preprocessing data inputan
ddPrediction_raw = pd.read_csv('dataset4.csv')
ddPredictions = ddPrediction_raw.drop(columns=['Label'])
df = pd.concat([inputan, ddPredictions], axis=0)

encode = ['Jenis Kelamin', 'Ruam Kulit', 'Manifestasi perdarahan', 'Kegagalan Sirkulasi', 'Syok berat', 'Uji Tokniket', 'Kebocoran Plasma', 'Pendarahan spontan']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

# Menampilkan data inputan yang digunakan untuk prediksi
st.subheader('Input Parameters')
st.write(df)

# Prediksi
prediksi = model.predict(df)
prediksi_proba = model.predict_proba(df)

# Mapping kategori prediksi
kategori_prediksi = {
    0: "Demam Dengue",
    1: "Demam Berdarah Dengue Tingkat 1",
    2: "Demam Berdarah Dengue Tingkat 2",
    3: "Demam Berdarah Dengue Tingkat 3",
    4: "Demam Berdarah Dengue Tingkat 4"
}

# Menampilkan hasil prediksi
st.subheader('Prediction Result (Demam Berdarah Prediction)')
prediksi_kategori = kategori_prediksi.get(prediksi[0], "Kategori Tidak Dikenal")
st.markdown(f"<h2 style='font-size: 40px; color: #ff6347;'>Anda Terindikasi : {prediksi_kategori}</h2>", unsafe_allow_html=True)

# Mencari probabilitas tertinggi
probabilitas_tertinggi = np.max(prediksi_proba)
st.write(f"**Probabilitas:** {probabilitas_tertinggi:.4f}")
