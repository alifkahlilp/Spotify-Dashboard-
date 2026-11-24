import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import page, load_art, load_songs, kpi_row, insight, style, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Kualitas Data", "🧹", "Cek nilai kosong, duplikat, nilai tidak valid, dan bandingkan metode imputasi sebelum analisis.")
art = load_art(); songs = load_songs()

q = art["quality"]
kpi_row([("Baris mentah", f"{art['raw_shape'][0]:,}", f"{art['raw_shape'][1]} kolom"),
         ("Lagu di >1 genre", f"{q.Jumlah[2]:,}", f"{q['% data'][2]}% baris"),
         ("Popularity = 0", f"{q.Jumlah[5]:,}", f"{q['% data'][5]}% baris"),
         ("Nilai tidak valid", f"{q.Jumlah[3] + q.Jumlah[4]:,}", "tempo 0 & time_signature 0/1")])
st.write("")

tab1, tab2, tab3 = st.tabs(["📋 Ringkasan masalah", "🧪 Perbandingan imputasi", "📦 Outlier"])
with tab1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.dataframe(q, hide_index=True, width="stretch",
                     column_config={"% data": st.column_config.ProgressColumn("% data", min_value=0, max_value=25, format="%.2f%%")})
        st.markdown("""**Keputusan cleaning**
1. Buang 1 baris tanpa nama lagu/artis.
2. `tempo = 0` dan `time_signature` 0/1 → NaN → imputasi **KNN**.
3. Analisis lagu memakai data **unik per `track_id`**; klasifikasi genre memakai semua baris.
4. `songwriter` dan `singer-songwriter` berisi lagu yang sama → sebaiknya digabung.""")
    with c2:
        fig = px.bar(q.sort_values("Jumlah"), x="Jumlah", y="Masalah", orientation="h", log_x=True, text="Jumlah",
                     title="Baris bermasalah", color_discrete_sequence=[GREEN])
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(style(fig, 380, False), width="stretch")
    st.subheader("Contoh: satu lagu, banyak versi")
    judul = st.text_input("Cari judul lagu", "Blinding Lights")
    ex = songs[songs.track_name.str.lower() == judul.lower()][["track_id", "artists", "album_name", "track_genre", "popularity"]]
    st.dataframe(ex.sort_values("popularity", ascending=False), hide_index=True, width="stretch")
    insight("Lagu yang sama bisa muncul dengan <b>track_id berbeda</b> (versi album, kompilasi, single). "
            "Versi kompilasi sering ber-popularity 0, sehingga popularity 0 belum tentu berarti lagu tidak dikenal.")

with tab2:
    st.markdown("10% nilai tempo yang valid **disembunyikan**, lalu diisi ulang dengan tiap metode, dan hasilnya dibandingkan dengan nilai asli.")
    imp = art["imputation"]
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(imp, hide_index=True, width="stretch")
        metode = st.radio("Lihat sebaran hasil metode", imp.Metode.tolist(), index=2, horizontal=True)
        best = imp.sort_values("MAE (BPM)").iloc[0]
        insight(f"<b>{best.Metode}</b> paling akurat (MAE {best['MAE (BPM)']:.1f} BPM), tapi selisihnya kecil karena "
                "tempo hanya lemah berhubungan dengan fitur audio lain.")
    with c2:
        pv = art["imputation_preview"]
        fig = go.Figure()
        fig.add_histogram(x=pv["asli"], name="nilai asli", opacity=.6, marker_color="#9aa0aa", nbinsx=40)
        fig.add_histogram(x=pv[metode], name=metode, opacity=.75, marker_color=GREEN, nbinsx=40)
        fig.update_layout(barmode="overlay", title="Tempo asli vs imputasi", xaxis_title="tempo (BPM)")
        st.plotly_chart(style(fig, 400), width="stretch")

with tab3:
    c1, c2 = st.columns([1, 3])
    col = c1.selectbox("Kolom", ["duration_min", "loudness", "tempo", "speechiness", "liveness"])
    genres = c1.multiselect("Genre", sorted(songs.track_genre.unique()), ["classical", "pop", "heavy-metal", "comedy", "sleep"])
    d = songs[songs.track_genre.isin(genres)] if genres else songs
    q1, q3 = d[col].quantile([.25, .75]); iqr = q3 - q1
    n_out = ((d[col] < q1 - 1.5 * iqr) | (d[col] > q3 + 1.5 * iqr)).sum()
    c1.metric("Outlier (aturan 1,5×IQR)", f"{n_out:,}", f"{n_out / len(d):.1%} dari data terpilih", delta_color="off")
    fig = px.box(d, x="track_genre" if genres else None, y=col, color="track_genre" if genres else None, points="outliers",
                 title=f"Boxplot {col}")
    c2.plotly_chart(style(fig, 450, False), width="stretch")
