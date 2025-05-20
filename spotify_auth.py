import streamlit as st
import joblib
import requests
import base64
import random
import numpy as np
from tensorflow.keras.models import load_model

# Spotify API credentials
CLIENT_ID = "f6d135fb142841acbc6422b34998da1b"
CLIENT_SECRET = "d1fe71664ba1449da6a8eb113fa47417"

# Playlist ID berdasarkan mood
mood_playlist_map = {
    "senang": "2tUhGmB8Vn6Fe8zRdvGBpE",
    "sedih": "6yYA6aUGp8qUTgQWWYkPkP",
    "marah": "1GXRoQWlxTNQiMNkOe7RqA",
    "galau": "5YhjEIWz3B8QGIeJFrV9wN",
    "bosan": "64Qo67THdNCiGxlVAD74IY"
}

# Emoji per mood
mood_emoji = {
    "senang": "😄",
    "sedih": "😢",
    "marah": "😠",
    "galau": "💔",
    "bosan": "🥱"
}

# Mapping mood ke nilai numerik
label_to_mood = {
    0: "bosan",
    1: "galau",
    2: "marah",
    3: "sedih",
    4: "senang"
}

# Fungsi ambil token dari Spotify
def get_access_token():
    token_url = "https://accounts.spotify.com/api/token"
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {
        "grant_type": "client_credentials"
    }

    response = requests.post(token_url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("access_token", None)
    else:
        st.error(f"Gagal mendapatkan token: {response.status_code}")
        st.text(response.text)
        return None

# Fungsi ambil lagu dari playlist berdasarkan mood
def cari_lagu_dari_playlist(mood, token):
    playlist_id = mood_playlist_map.get(mood.lower())
    if not playlist_id:
        st.warning("Mood tidak dikenali.")
        return []

    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=50"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json()
        items = data.get("items", [])
        if items:
            random_tracks = random.sample(items, min(3, len(items)))
            hasil = []
            for track_item in random_tracks:
                track = track_item.get("track", {})
                title = track.get("name")
                artist = track.get("artists", [{}])[0].get("name")
                url = track.get("external_urls", {}).get("spotify")
                hasil.append((title, artist, url))
            return hasil
        else:
            st.warning("Playlist kosong.")
    else:
        st.error(f"❌ Spotify API Error: {res.status_code}")
        st.text(res.text)
    return []

# Load model & vectorizer berdasarkan pilihan
def load_model_and_vectorizer(pilihan):
    if pilihan == "Random Forest + TF-IDF":
        model = joblib.load("random_forest_best_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
    elif pilihan == "MLP + BoW":
        model = load_model("mlp_model_v1.h5")
        vectorizer = joblib.load("bows_vectorizer.joblib")
    else:
        model, vectorizer = None, None
    return model, vectorizer

# ===============================
# UI STREAMLIT
# ===============================

# Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
    background: linear-gradient(to right, #a1c4fd, #fcb69f);
    color: #333;
    }
    .card {
        background-color: #ffffffcc;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin-top: 25px;
    }
    .footer {
        text-align: center;
        margin-top: 60px;
        color: #444;
        font-size: 0.9em;
    }
    a {
        color: #e94e77;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 style='text-align:center;'>🎧 Mood Detection + Spotify Rekomendasi 🎶</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Prediksi mood dari tweet kamu & dapatkan 3 lagu yang cocok!</h4>", unsafe_allow_html=True)

# Dropdown untuk model
model_choice = st.selectbox("📊 Pilih Model:", ["Random Forest + TF-IDF", "MLP + BoW"])

# Input teks
text_input = st.text_area("📝 Masukkan tweet kamu di sini:")

# Prediksi mood & tampilkan rekomendasi
if st.button("🔍 Prediksi Mood"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        model, vectorizer = load_model_and_vectorizer(model_choice)
        if model is None or vectorizer is None:
            st.error("Model atau vectorizer tidak ditemukan.")
        else:
            text_vec = vectorizer.transform([text_input])

            if model_choice == "MLP + BoW":
                mood_code = np.argmax(model.predict(text_vec), axis=1)[0]
            else:
                mood_code = model.predict(text_vec)[0]

            mood = label_to_mood.get(mood_code, "unknown")
            emoji = mood_emoji.get(mood, "🎭")

            # Tampilkan hasil prediksi
            st.markdown(f"""
                <div class='card'>
                    <h3>🎯 Mood Terdeteksi: {emoji} <b>{mood.upper()}</b></h3>
            """, unsafe_allow_html=True)

            token = get_access_token()
            if token:
                hasil_lagu = cari_lagu_dari_playlist(mood, token)
                if hasil_lagu:
                    for i, (title, artist, url) in enumerate(hasil_lagu, 1):
                        st.markdown(f"""
                            <p>🎵 <b>Lagu #{i}</b>: {title}<br>
                            👤 Oleh: {artist}<br>
                            🔗 <a href="{url}" target="_blank">▶️ Dengarkan di Spotify</a></p>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.warning("Lagu tidak ditemukan di playlist.")
            else:
                st.error("Gagal mendapatkan token dari Spotify.")

# Footer
st.markdown("""
    <div class='footer'>
        Dibuat dengan ❤️ oleh kelompok 3, menggunakan Streamlit & Spotify API 🎶
    </div>
""", unsafe_allow_html=True)
