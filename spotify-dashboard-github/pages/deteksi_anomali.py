import plotly.express as px
import streamlit as st
from utils import page, load_songs, insight, style, kpi_row, spotify_embed, CF, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Deteksi Anomali", "🚨", "Isolation Forest per genre: lagu yang nilai fitur audionya (energy, loudness, dll.) paling menyimpang dari lagu lain di genre yang sama.")
songs = load_songs()

kpi_row([("Lagu ditandai anomali", f"{songs.anomali.sum():,}", "2% per genre"),
         ("Penyebab tersering", songs[songs.anomali].paling_beda.value_counts().index[0], "fitur paling menyimpang")])
st.write("")

tab1, tab2 = st.tabs(["🎯 Per genre", "🌍 Paling aneh secara keseluruhan"])
with tab1:
    c1, c2 = st.columns([1, 3])
    g = c1.selectbox("Genre", sorted(songs.track_genre.unique()), index=sorted(songs.track_genre.unique()).index("sleep"))
    x = c1.selectbox("Sumbu X", CF, index=1)
    y = c1.selectbox("Sumbu Y", CF, index=2)
    d = songs[songs.track_genre == g].copy()
    d["status"] = d.anomali.map({True: "anomali", False: "normal"})
    fig = px.scatter(d, x=x, y=y, color="status", size="anomali_skor", size_max=14, hover_name="track_name",
                     hover_data={"artists": True, "paling_beda": True, "anomali_skor": ":.3f", "status": False},
                     color_discrete_map={"anomali": NEG, "normal": "#3A4150"}, title=f"Anomali di {g}")
    c2.plotly_chart(style(fig, 500), width="stretch")
    top = d.sort_values("anomali_skor", ascending=False).head(10)
    st.dataframe(top[["track_name", "artists", "paling_beda", "popularity", "anomali_skor"]].round(3), hide_index=True, width="stretch")
    with st.expander("▶️ Dengarkan lagu paling aneh di genre ini"):
        spotify_embed(top.track_id.iloc[0])

with tab2:
    c1, c2 = st.columns([1.6, 1])
    w = songs.sort_values("anomali_skor", ascending=False).drop_duplicates("track_id").head(25)
    c1.dataframe(w[["track_name", "artists", "track_genre", "paling_beda", "popularity", "anomali_skor"]].round(3),
                 hide_index=True, width="stretch", height=600)
    vc = songs[songs.anomali].paling_beda.value_counts().head(10).iloc[::-1]
    fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Penyebab anomali",
                 color_discrete_sequence=[GREEN])
    fig.update_layout(yaxis_title=None, xaxis_title="lagu")
    c2.plotly_chart(style(fig, 420, False), width="stretch")
    rate = songs.groupby("track_genre").anomali_skor.mean().sort_values()
    c2.markdown(f"Genre paling **homogen**: {', '.join(rate.head(3).index)}  \nGenre paling **beragam**: {', '.join(rate.tail(3).index)}")

insight("Anomali paling sering berasal dari rekaman <b>live</b>, lagu <b>instrumental</b> di genre vokal, dan <b>spoken word</b>. "
        "Contoh menarik: <i>Through The Fire And Flames</i> (DragonForce) terlabel classical.")
