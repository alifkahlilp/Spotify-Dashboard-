import plotly.express as px
import streamlit as st
from utils import page, load_tracks, load_art, sidebar_filters, kpi_row, insight, style, spotify_embed, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Spotify Insight", "🎧", "Dashboard analisis 114.000 lagu Spotify: kualitas data, mood, clustering, peta genre, "
     "prediksi hit, pola asosiasi, anomali, dan rekomendasi lagu.")

tracks = load_tracks()
art = load_art()
t = sidebar_filters(tracks, "home")
if t.empty:
    st.warning("Tidak ada lagu yang cocok dengan filter."); st.stop()

artists = t.artists.str.split(";").explode()
kpi_row([
    ("Lagu unik", f"{len(t):,}", f"dari {art['raw_shape'][0]:,} baris mentah"),
    ("Artis", f"{artists.nunique():,}", "setelah memecah kolaborasi"),
    ("Genre", f"{t.track_genre.nunique()}", "label genre Spotify"),
    ("Rata-rata popularity", f"{t.popularity.mean():.1f}", "skala 0–100"),
    ("Lagu explicit", f"{t.explicit.mean():.1%}", "lirik eksplisit"),
    ("Durasi median", f"{t.duration_min.median():.2f} mnt", "per lagu"),
])
st.write("")

if True:
    g = t.groupby("track_genre").agg(lagu=("track_id", "count"), popularity=("popularity", "mean")).reset_index()
    fam = art["genre_family"]
    from utils import FAMILY_NAME, FAMILY_COLOR
    g["kelompok"] = g.track_genre.map(fam).map(FAMILY_NAME)
    pecah = {"progressive-house": "progres-<br>sive-<br>house", "singer-songwriter": "singer-<br>song-<br>writer", "songwriter": "song-<br>writer",
             "electronic": "electro-<br>nic", "grindcore": "grind-<br>core", "alternative": "alter-<br>native", "reggaeton": "reggae-<br>ton"}
    g["nama"] = [pecah.get(n, n.replace("-", "-<br>", 1) if len(n) > 8 else n) for n in g.track_genre]
    fig = px.treemap(g, path=["kelompok", "nama"], values="lagu", color="kelompok", color_discrete_map=FAMILY_COLOR,
                     custom_data=["popularity"], title="Peta genre")
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value} lagu<extra></extra>", marker=dict(line=dict(color="#0B0D12", width=1)), tiling=dict(pad=2),
                      textfont=dict(color="#0B0D12"), root_color="rgba(0,0,0,0)", opacity=.9)
    fig = style(fig, 640)
    fig.update_layout(uniformtext=dict(minsize=11, mode="show"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Ukuran kotak = jumlah lagu unik per genre. Warna = 8 kelompok genre (lihat halaman Kemiripan Genre & Artis).")
if True:
    top = g.sort_values("popularity", ascending=False).head(12).iloc[::-1]
    fig = px.bar(top, x="popularity", y="track_genre", orientation="h", title="Genre terpopuler", text_auto=".1f",
                 color_discrete_sequence=[GREEN])
    fig.update_layout(yaxis_title=None, xaxis_title="rata-rata popularity")
    st.plotly_chart(style(fig, 420, legend=False), width="stretch")

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("🔥 Lagu terpopuler")
    top_s = t.nlargest(10, "popularity")[["track_name", "artists", "track_genre", "popularity"]]
    st.dataframe(top_s, hide_index=True, width="stretch",
                 column_config={"popularity": st.column_config.ProgressColumn("popularity", min_value=0, max_value=100, format="%d")})
    with st.expander("▶️ Putar lagu #1"):
        spotify_embed(t.nlargest(1, "popularity").track_id.iloc[0])
with c2:
    st.subheader("🎤 Artis paling populer")
    a = t.assign(artist=t.artists.str.split(";")).explode("artist")
    min_n = st.slider("Minimal jumlah lagu", 1, 30, 10)
    ta = (a.groupby("artist").agg(lagu=("track_id", "count"), popularity=("popularity", "mean"))
           .query("lagu >= @min_n").sort_values("popularity", ascending=False).head(10).reset_index())
    st.dataframe(ta.round(1), hide_index=True, width="stretch",
                 column_config={"popularity": st.column_config.ProgressColumn("popularity", min_value=0, max_value=100, format="%.1f")})

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(t, x="popularity", nbins=50, title="Sebaran popularity", color_discrete_sequence=[GREEN])
    st.plotly_chart(style(fig, 330, False), width="stretch")
with c2:
    m = t.mood.value_counts().reset_index()
    fig = px.pie(m, names="mood", values="count", hole=.55, title="Komposisi mood",
                 color="mood", color_discrete_map=MOOD_WARNA)
    st.plotly_chart(style(fig, 330), width="stretch")

top4 = ", ".join(g.sort_values("popularity", ascending=False).track_genre.head(4))
insight(f"Mood terbanyak: <b>{m.iloc[0]['mood']}</b> ({m.iloc[0]['count'] / m['count'].sum():.0%}). "
        f"Genre dengan popularity tertinggi: <b>{top4}</b>. Gunakan menu di kiri untuk menjelajah tiap analisis.")
