import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from utils import page, load_tracks, load_art, insight, style, radar_fig, spotify_embed, RADAR, CF
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Rekomendasi Lagu Mirip", "🎯", "Cari lagu, lalu dapatkan lagu dengan nilai 9 fitur audio paling mirip (cosine similarity), dan putar langsung.")
tracks = load_tracks(); art = load_art()


@st.cache_resource(show_spinner="Menyiapkan indeks kemiripan…")
def build_index():
    pool = tracks[tracks.popularity >= 20].drop_duplicates(["track_name", "artists"]).reset_index(drop=True)
    Z = art["scaler"].transform(pool[CF])
    nn = NearestNeighbors(metric="cosine").fit(Z)
    return pool, Z, nn


pool, Z, nn = build_index()
tab1, tab2 = st.tabs(["🎵 Rekomendasi", "⚖️ Bandingkan dua lagu"])

with tab1:
    c1, c2 = st.columns([1, 1])
    q = c1.text_input("Cari judul atau artis", "Blinding Lights")
    m = pool[pool.track_name.str.contains(q, case=False, regex=False) | pool.artists.str.contains(q, case=False, regex=False)].nlargest(50, "popularity")
    if m.empty:
        st.info("Lagu tidak ditemukan."); st.stop()
    lab = m.track_name + " — " + m.artists.str.slice(0, 40) + " (" + m.track_genre + ")"
    pick = c1.selectbox("Pilih lagu", lab.tolist())
    i = m.index[lab == pick][0]
    gf = c2.multiselect("Batasi ke genre (opsional)", sorted(pool.track_genre.unique()))
    n = c2.slider("Jumlah rekomendasi", 5, 20, 10)
    dist, idx = nn.kneighbors(Z[[i]], n_neighbors=min(len(pool), 400))
    rec = pool.iloc[idx[0][1:]].assign(kemiripan=1 - dist[0][1:])
    if gf:
        rec = rec[rec.track_genre.isin(gf)]
    rec = rec.head(n)
    ref = pool.loc[i]
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(f"#### 🎧 {ref.track_name}\n{ref.artists} · _{ref.track_genre}_ · popularity {ref.popularity}")
        spotify_embed(ref.track_id, 152)
        if len(rec):
            st.plotly_chart(radar_fig({f"acuan: {ref.track_name}": ref[RADAR], f"#1: {rec.iloc[0].track_name}": rec.iloc[0][RADAR]},
                                      "Acuan vs rekomendasi #1", h=380), width="stretch")
    with c2:
        st.dataframe(rec[["track_name", "artists", "track_genre", "popularity", "kemiripan"]].round(3), hide_index=True,
                     width="stretch", column_config={"kemiripan": st.column_config.ProgressColumn("kemiripan", min_value=0.8, max_value=1, format="%.3f")})
        if len(rec):
            sel = st.selectbox("Putar rekomendasi", (rec.track_name + " — " + rec.artists.str.slice(0, 40)).tolist())
            spotify_embed(rec.track_id.iloc[(rec.track_name + " — " + rec.artists.str.slice(0, 40)).tolist().index(sel)], 152)
    insight("Lagu dengan fitur audio mirip sering datang dari <b>genre berbeda</b>. Pakai filter genre bila ingin rekomendasi yang terasa lebih 'sejenis'.")

with tab2:
    c1, c2 = st.columns(2)
    def picker(col, key, default):
        q = col.text_input("Cari lagu", default, key=f"q{key}")
        m = pool[pool.track_name.str.contains(q, case=False, regex=False) | pool.artists.str.contains(q, case=False, regex=False)].nlargest(30, "popularity")
        if m.empty:
            col.info("Tidak ditemukan"); return None
        lab = m.track_name + " — " + m.artists.str.slice(0, 40)
        return m.loc[m.index[lab == col.selectbox("Pilih", lab.tolist(), key=f"s{key}")][0]]
    a = picker(c1, "a", "Blinding Lights"); b = picker(c2, "b", "Someone Like You")
    if a is not None and b is not None:
        st.plotly_chart(radar_fig({f"{a.track_name} — {a.artists}": a[RADAR], f"{b.track_name} — {b.artists}": b[RADAR]}, "Radar dua lagu", h=480),
                        width="stretch")
        cmp = pd.DataFrame({a.track_name: a[RADAR + ["tempo", "loudness", "popularity"]], b.track_name: b[RADAR + ["tempo", "loudness", "popularity"]]}).astype(float)
        cmp["selisih"] = cmp.iloc[:, 0] - cmp.iloc[:, 1]
        st.dataframe(cmp.round(3), width="stretch")
