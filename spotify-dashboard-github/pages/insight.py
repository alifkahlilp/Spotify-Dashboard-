import streamlit as st
from utils import page, load_art

page("Insight & Kesimpulan", "💡", "Ringkasan temuan utama dari seluruh analisis.")
art = load_art(); hit = art["hit"]; mc = art["method_compare"]

cards = [
    ("🧹", "Data perlu dibersihkan", "21% baris adalah lagu yang sama di beberapa genre, 14% popularity-nya 0, dan songwriter = singer-songwriter. "
     "Imputasi KNN untuk tempo sedikit lebih baik dari mean/median."),
    ("🎧", "Fitur audio ≠ popularitas", "Semua fitur audio hanya lemah berkorelasi dengan popularity (|r| ≤ 0,13). Lagu explicit sedikit lebih populer."),
    ("🌌", "5 mood, batasnya kabur", f"K-Means menemukan Party & Latin, Ambient / Sleep, Keras & Intens, Spoken word, Akustik Mellow. "
     f"K-Prototype hampir sama (ARI {mc['ari_kproto_kmeans']:.2f}); FCM: {mc['fcm_ambigu']:.0%} lagu adalah campuran mood."),
    ("🪐", "Galaksi punya 4 pulau", "UMAP memisahkan lagu instrumental, spoken word, dan secara mengejutkan musik Brasil (pagode, samba, sertanejo)."),
    ("🧬", "8 kelompok genre", "Genre dengan profil fitur audio termirip: latino–reggaeton, indie–indie-pop, punk–punk-rock, edm–house. Jaringan artis membentuk komunitas lintas bahasa."),
    ("🔮", "Hit sulit ditebak dari fitur audio", f"ROC-AUC {hit['auc']:.2f} dari audio, naik ke {hit['auc_genre']:.2f} dengan genre. SHAP: instrumentalness, "
     "acousticness, dan durasi panjang menurunkan peluang hit; loudness dan danceability menaikkan."),
    ("🔗", "Pola asosiasi", "danceable + explicit → 1,9× lebih mungkin populer; instrumental + ceria → 2,3× lebih mungkin tidak populer."),
    ("🚨", "Lagu salah genre", "2% lagu per genre tidak cocok dengan genrenya, terutama rekaman live, instrumental, dan spoken word."),
]
for i in range(0, len(cards), 2):
    cols = st.columns(2)
    for c, (ic, h, t) in zip(cols, cards[i:i + 2]):
        c.markdown(f'<div class="kpi" style="margin-bottom:12px"><div style="font-size:1.6rem">{ic}</div>'
                   f'<div style="font-weight:700;font-size:1.1rem;margin:.2rem 0 .3rem">{h}</div>'
                   f'<div style="color:#c9ccd3">{t}</div></div>', unsafe_allow_html=True)

st.subheader("Keterbatasan & pengembangan")
st.markdown("""- Popularity adalah snapshot saat data diambil, dan versi kompilasi menurunkan popularity lagu yang sama.
- Label genre berasal dari playlist pencarian Spotify, sehingga satu lagu bisa punya beberapa genre.
- Pengembangan: tambah data lirik/tanggal rilis, model gradient boosting, atau rekomendasi hibrid (audio + genre).""")
