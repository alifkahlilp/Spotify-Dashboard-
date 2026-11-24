import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from utils import page, load_tracks, load_art, insight, style, kpi_row, CF, CL_WARNA, MOOD_WARNA, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Clustering Mood Lagu", "🌌", "Mengelompokkan lagu berdasarkan 9 fitur audio: K-Means, K-Prototype, Fuzzy C-Means, dan peta galaksi UMAP.")
tracks = load_tracks(); art = load_art()
scaler = art["scaler"]


@st.cache_data(show_spinner="Menjalankan K-Means…")
def run_kmeans(k: int):
    X = scaler.transform(tracks[CF])
    km = KMeans(k, n_init=3, random_state=42).fit(X)
    return km.labels_


tabs = st.tabs(["🔢 Pilih k & profil", "🌌 Galaksi lagu (UMAP)", "📈 Parallel coordinates", "⚖️ Perbandingan metode", "🍩 Cluster × mood"])

with tabs[0]:
    el = art["elbow"]
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=el.k, y=el.inertia, mode="lines+markers", line_color=GREEN))
    fig.update_layout(title="Metode elbow", xaxis_title="k")
    c1.plotly_chart(style(fig, 300, False), width="stretch")
    fig = go.Figure(go.Scatter(x=el.k, y=el.silhouette, mode="lines+markers", line_color=PALETTE[1]))
    fig.update_layout(title="Silhouette score", xaxis_title="k")
    c2.plotly_chart(style(fig, 300, False), width="stretch")

    k = st.slider("Jumlah cluster (k)", 2, 10, 5)
    lab = run_kmeans(k)
    t = tracks.assign(cl=lab)
    prof = t.groupby("cl")[CF].mean()
    z = (prof - tracks[CF].mean()) / tracks[CF].std()
    fig = px.imshow(z, text_auto=False, color_continuous_scale=DIVERGING, zmin=-2.5, zmax=2.5, aspect="auto",
                    labels=dict(color="z-score", y="cluster"), title="Profil cluster")
    fig.update_traces(text=prof.round(2).values, texttemplate="%{text}")
    st.plotly_chart(style(fig, 120 + 55 * k), width="stretch")
    summ = t.groupby("cl").agg(lagu=("track_id", "count"), popularity=("popularity", "mean"),
                               genre_dominan=("track_genre", lambda s: ", ".join(s.value_counts().head(5).index)))
    summ["ciri tertinggi"] = z.apply(lambda r: ", ".join(r.nlargest(2).index), axis=1)
    summ["ciri terendah"] = z.apply(lambda r: ", ".join(r.nsmallest(2).index), axis=1)
    st.dataframe(summ.round(1), width="stretch")
    if k == 5:
        insight("Dengan k = 5, cluster bisa dinamai: <b>Party & Latin</b>, <b>Ambient / Sleep</b>, <b>Keras & Intens</b>, "
                "<b>Spoken word / Live</b>, <b>Akustik Mellow</b>. Silhouette rendah (~0,18) menandakan batas antar mood landai.")

with tabs[1]:
    u = art["umap"]
    c1, c2 = st.columns([1, 4])
    color = c1.radio("Warnai berdasarkan", ["cluster", "mood", "genre pilihan", "popularity"])
    if color == "genre pilihan":
        gsel = c1.multiselect("Genre", sorted(u.track_genre.unique()), ["pop", "hip-hop", "classical", "heavy-metal", "sleep", "samba", "k-pop"])
        d = u[u.track_genre.isin(gsel)]; ckey = "track_genre"
    else:
        d = u; ckey = color
    cari = c1.text_input("Sorot lagu/artis", "")
    fig = px.scatter(d, x="umap_x", y="umap_y", color=ckey, hover_name="track_name",
                     hover_data={"artists": True, "track_genre": True, "popularity": True, "umap_x": False, "umap_y": False},
                     color_discrete_map=CL_WARNA if ckey == "cluster" else (MOOD_WARNA if ckey == "mood" else None),
                     color_continuous_scale=SEQUENTIAL, opacity=.65, title="Galaksi lagu")
    fig.update_traces(marker_size=3)
    if cari:
        h = u[u.track_name.str.contains(cari, case=False, regex=False) | u.artists.str.contains(cari, case=False, regex=False)]
        fig.add_scatter(x=h.umap_x, y=h.umap_y, mode="markers", marker=dict(size=11, color="white", line=dict(color="black", width=1)),
                        name=f"'{cari}' ({len(h)})", text=h.track_name, hoverinfo="text")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    c2.plotly_chart(style(fig, 680), width="stretch")
    insight("Ada 4 'pulau': lagu vokal umum, lagu <b>instrumental</b> (sleep, study, techno), <b>spoken word</b>, dan "
            "<b>musik Brasil</b> (pagode, samba, sertanejo) yang kombinasi fitur audionya cukup khas sehingga terpisah sendiri.")

with tabs[2]:
    s = tracks.sample(3000, random_state=42).copy()
    s["loudness"] = (s.loudness - tracks.loudness.min()) / (tracks.loudness.max() - tracks.loudness.min())
    s["tempo"] = s.tempo / tracks.tempo.max()
    order = list(CL_WARNA)
    s["cid"] = s.cluster.map({c: i for i, c in enumerate(order)})
    fig = go.Figure(go.Parcoords(line=dict(color=s.cid, colorscale=[[i / 4, CL_WARNA[c]] for i, c in enumerate(order)]),
                                 dimensions=[dict(label=c, values=s[c], range=[0, 1]) for c in CF]))
    fig.update_layout(title="Profil audio per cluster")
    st.plotly_chart(style(fig, 520), width="stretch")
    st.markdown(" ".join(f'<span style="color:{v}">■</span> {k} &nbsp;' for k, v in CL_WARNA.items()), unsafe_allow_html=True)

with tabs[3]:
    mc = art["method_compare"]
    kpi_row([("K-Prototype vs K-Means (ARI)", f"{mc['ari_kproto_kmeans']:.2f}", "1 = identik"),
             ("Fuzzy Partition Coefficient", f"{mc['fpc']:.2f}", "1 = tegas, 0,20 = sangat kabur"),
             ("Lagu ambigu (FCM)", f"{mc['fcm_ambigu']:.0%}", "keanggotaan tertinggi < 0,5")])
    fig = px.histogram(x=mc["fcm_maxu"], nbins=40, title="Keanggotaan cluster (FCM)",
                       color_discrete_sequence=[GREEN])
    fig.update_layout(xaxis_title="keanggotaan tertinggi", yaxis_title="lagu")
    st.plotly_chart(style(fig, 360, False), width="stretch")
    insight("Menambah fitur kategorik (key, mode, explicit) lewat K-Prototype hampir tidak mengubah hasil. "
            "Fuzzy C-Means menunjukkan hampir semua lagu adalah <b>campuran beberapa mood</b>, jadi cluster lebih tepat dibaca sebagai gradasi.")

with tabs[4]:
    c1, c2 = st.columns([1.2, 1])
    sb = tracks.groupby(["cluster", "mood"]).size().reset_index(name="lagu")
    fig = px.sunburst(sb, path=["cluster", "mood"], values="lagu", color="cluster", color_discrete_map=CL_WARNA, title="Mood di tiap cluster")
    c1.plotly_chart(style(fig, 520), width="stretch")
    ct = pd.crosstab(tracks.cluster, tracks.mood, normalize="index").mul(100).round(1)
    c2.dataframe(ct, width="stretch")
    pop = tracks.groupby("cluster").popularity.mean().sort_values()
    fig = px.bar(x=pop.values, y=pop.index, orientation="h", color=pop.index, color_discrete_map=CL_WARNA, title="Popularity per cluster")
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    c2.plotly_chart(style(fig, 300, False), width="stretch")
