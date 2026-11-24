import streamlit as st

st.set_page_config(page_title="Spotify Insight", page_icon=":material/headphones:", layout="wide")

# Menu navigasi: ikon Material bawaan Streamlit (tampil sama di semua OS, tidak bergantung font emoji)
nav = st.navigation([
    st.Page("pages/beranda.py", title="Beranda", icon=":material/home:", default=True),
    st.Page("pages/kualitas_data.py", title="Kualitas Data", icon=":material/cleaning_services:"),
    st.Page("pages/eksplorasi_audio.py", title="Eksplorasi Audio", icon=":material/graphic_eq:"),
    st.Page("pages/clustering_mood.py", title="Clustering Mood", icon=":material/bubble_chart:"),
    st.Page("pages/kemiripan_genre_artis.py", title="Kemiripan Genre & Artis", icon=":material/hub:"),
    st.Page("pages/prediksi_hit.py", title="Prediksi Hit", icon=":material/trending_up:"),
    st.Page("pages/pola_asosiasi.py", title="Pola Asosiasi", icon=":material/link:"),
    st.Page("pages/deteksi_anomali.py", title="Deteksi Anomali", icon=":material/report:"),
    st.Page("pages/rekomendasi_lagu.py", title="Rekomendasi Lagu", icon=":material/queue_music:"),
    st.Page("pages/insight.py", title="Insight", icon=":material/lightbulb:"),
])
nav.run()
