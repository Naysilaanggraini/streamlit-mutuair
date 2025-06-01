import streamlit as st
import pandas as pd
import numpy as np
import pickle

# SET PAGE CONFIG HARUS DI BARIS PERTAMA STREAMLIT
st.set_page_config(page_title="Hybrid Model Prediksi", layout="wide")

# CSS Styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: #f0f4f8;
        font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: background 0.5s ease;
    }

    h1, .css-10trblm {
        font-weight: 900 !important;
        color: #0d47a1 !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        letter-spacing: 1.5px;
        font-size: 2.8rem !important;
        margin-bottom: 1rem;
    }

    p, .desc {
        font-family: 'Times New Roman', Times, serif !important;
        text-align: center !important;
        font-size: 1.15rem;
        color: #000000;  /* diganti dari abu ke biru keabu-abuan */
        margin: 0.8rem auto 1.5rem auto;
        max-width: 700px;
        line-height: 1.5;
    }

    .css-1d391kg {
        background: #0a2647 !important;
        color: #000000 !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 24px rgba(0, 114, 255, 0.3);
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.03em;
    }
    .css-1d391kg label, .css-1d391kg div, .css-1d391kg span {
        color: #cfd8e8 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(0, 180, 219, 0.5);
        border: none;
        transition: background 0.4s ease, transform 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0083b0 0%, #005f73 100%);
        cursor: pointer;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 95, 115, 0.6);
    }

    input[type=number] {
        border: 2px solid #00b4db !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.6rem !important;
        color: #004466 !important;
        font-weight: 700;
        font-size: 1rem;
        background-color: #e8f4fc !important;
        transition: border-color 0.3s ease;
    }
    input[type=number]:focus {
        border-color: #0083b0 !important;
        outline: none !important;
        box-shadow: 0 0 8px #0083b0;
    }

    /* Box hasil prediksi diubah dari hijau ke biru */
    .stAlert {
        border-radius: 16px !important;
        background: linear-gradient(135deg, #c1dfff, #4ca1ff) !important;
        color: #003366 !important;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
        padding: 1rem !important;
    }

    img {
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.3);
        transition: transform 0.3s ease;
    }
    img:hover {
        transform: scale(1.05);
    }

    hr {
        border: 0;
        height: 3px;
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
        margin: 3rem 0;
        border-radius: 3px;
        box-shadow: 0 2px 10px rgba(0, 132, 176, 0.6);
    }

    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a2647;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #00b4db;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load scaler dan model
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    dt_model = pickle.load(open('dt_model.pkl', 'rb'))
    svm_model = pickle.load(open('svm_model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model atau scaler: {e}")
    st.stop()

# Fungsi prediksi hybrid (voting)
def hybrid_predict(input_df):
    # Skalakan input
    X_scaled = scaler.transform(input_df)
    
    # Prediksi dari kedua model
    dt_pred = dt_model.predict(X_scaled)
    svm_pred = svm_model.predict(X_scaled)
    
    # Voting
    final_pred = []
    for dt, svm in zip(dt_pred, svm_pred):
        votes = [dt, svm]
        pred = max(set(votes), key=votes.count)  # majority voting
        final_pred.append(pred)
    
    return final_pred

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Prediksi", "Tentang"])

# -------------------------
# Page 1: Home
if page == "Home":
    st.title("🌊 HyQual (Hybrid Quality)")
    st.image("AIR JAWA TIMUR.jpg", use_column_width=True)
    st.markdown("""
    HyQual adalah aplikasi prediksi mutu air berbasis kecerdasan buatan yang 
    menggabungkan kekuatan **Decision Tree** dan **Support Vector Machine (SVM)** dalam satu model hybrid.  
    Dirancang untuk memantau kualitas air laut secara **akurat**, **cepat**, dan **mudah diakses**.
    """)

    st.markdown("""
    ### 🔹 Keunggulan HyQual  
    - 🔬 **Prediksi Akurat**: Menggunakan model hybrid DT–SVM untuk klasifikasi mutu air berdasarkan parameter lingkungan seperti pH, DO, BOD, TSS, dan lainnya.  
    - 📈 **Analisis Data Otomatis**: Proses prediksi berlangsung otomatis setelah data diunggah.  
    - 🧭 **Mendukung Pengambilan Keputusan**: Menyediakan hasil yang dapat digunakan oleh pengelola wilayah pesisir dan peneliti.  
    - 💡 **Antarmuka Sederhana & Informatif**: Dirancang agar mudah digunakan oleh berbagai kalangan.
    """)

    st.info("Silakan pilih halaman 'Prediksi' untuk mengunggah data dan melihat hasil prediksi.")

# -------------------------
# Page 2: Prediksi
elif page == "Prediksi":
    st.title("Prediksi Mutu Air")
    st.subheader("Input Manual")
    col1, col2 = st.columns(2)
    with col1:
        temperatur = st.number_input("Temperatur (°C)", min_value=0.0, max_value=50.0, step=0.1)
        tds = st.number_input("TDS (mg/L)", min_value=0.0, step=1.0)
        tss = st.number_input("TSS (mg/L)", min_value=0.0, step=1.0)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    with col2:
        bod = st.number_input("BOD (mg/L)", min_value=0.0, step=0.1)
        cod = st.number_input("COD (mg/L)", min_value=0.0, step=0.1)
        do = st.number_input("DO (mg/L)", min_value=0.0, step=0.1)
        curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=0.1)
    
    if st.button("Prediksi"):
        # Pastikan fitur scaler sudah ada nama kolom
        if not hasattr(scaler, 'feature_names_in_'):
            st.error("Scaler tidak memiliki atribut feature_names_in_. Silakan cek kembali scaler Anda.")
            st.stop()
        
        input_array = np.array([[temperatur, tds, tss, ph, bod, cod, do, curah_hujan]])
        input_df = pd.DataFrame(input_array, columns=scaler.feature_names_in_)
        
        pred = hybrid_predict(input_df)[0]
        
        # Interpretasi hasil prediksi
        if pred == 2:
            mutuair_kategori = 'Mutu air: 🟢 **BAIK**'
        elif pred == 3:
            mutuair_kategori = 'Mutu air: 🟡 **CUKUP**'
        elif pred == 4:
            mutuair_kategori = 'Mutu air: 🔴 **KURANG BAIK**'
        elif pred == 5:
            mutuair_kategori = 'Mutu air: 🔴 **BURUK**'
        else:
            mutuair_kategori = 'Kategori mutu tidak dikenali'
       
        st.success(f"Kategori mutu air yang diprediksi: {pred}")
        st.info(mutuair_kategori)

# -------------------------
# Page 3: Tentang
elif page == "Tentang":
    st.title("Tentang Aplikasi")

    # Buat dua kolom
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Nama Aplikasi**")
        st.markdown("HyQual (Hybrid Quality)")

    with col2:
        st.markdown("**Pengembang**")
        st.markdown("Tim Data Science")

    st.markdown("---")

    st.markdown("""
    **Deskripsi**  
    HyQual adalah aplikasi prediksi mutu air berbasis kecerdasan buatan yang menggabungkan model hybrid  
    Decision Tree dan Support Vector Machine (SVM). Aplikasi ini dirancang untuk mendukung pengelolaan  
    wilayah pesisir secara akurat dan berkelanjutan dengan memberikan prediksi kualitas air yang tepat.
    """)

    st.markdown("---")

    st.markdown("""
    **Kontak**  
    📧 [email@example.com](mailto:email@example.com)  
    🌐 Website: [www.hyqualapp.com](https://www.hyqualapp.com)  
    """)
