import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu
from utils import page, load_tracks, load_songs, sidebar_filters, insight, style, radar_fig, CF, MOOD_WARNA, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Eksplorasi Fitur Audio", "🎧", "Hubungan antar fitur audio, peta mood, sidik jari genre, dan apa yang berkaitan dengan popularity.")
tracks = load_tracks(); songs = load_songs()
t = sidebar_filters(tracks, "eda")
if len(t) < 50:
    st.warning("Terlalu sedikit lagu untuk dianalisis, longgarkan filter."); st.stop()

tabs = st.tabs(["🔗 Korelasi", "🎯 Scatter bebas", "😊 Peta mood", "🧬 Sidik jari genre", "🎻 Distribusi per genre", "🔞 Explicit"])

with tabs[0]:
    num = CF + ["duration_min", "popularity"]
    corr = t[num].corr()
    c1, c2 = st.columns([1.5, 1])
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1, title="Korelasi antar fitur")
    c1.plotly_chart(style(fig, 560), width="stretch")
    cp = corr["popularity"].drop("popularity").sort_values()
    fig = px.bar(x=cp.values, y=cp.index, orientation="h", title="Korelasi ke popularity",
                 color=cp.values > 0, color_discrete_map={True: GREEN, False: NEG})
    fig.update_layout(yaxis_title=None, xaxis_title="r")
    c2.plotly_chart(style(fig, 560, False), width="stretch")
    strong = corr.where(~np.eye(len(corr), dtype=bool)).stack().abs().sort_values(ascending=False)
    a, b = strong.index[0]
    insight(f"Pasangan paling kuat: <b>{a} – {b}</b> (r = {corr.at[a, b]:.2f}). Semua fitur audio hanya lemah berkorelasi dengan "
            f"popularity (|r| maks {cp.abs().max():.2f}), jadi popularitas tidak bisa dijelaskan dari fitur audio saja.")

with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("Sumbu X", CF + ["popularity", "duration_min"], index=7)
    y = c2.selectbox("Sumbu Y", CF + ["popularity", "duration_min"], index=1)
    col = c3.selectbox("Warna", ["cluster", "mood", "explicit", "popularity", "track_genre"])
    n = c4.slider("Jumlah titik", 1000, 20000, 6000, 1000)
    s = t.sample(min(n, len(t)), random_state=1)
    fig = px.scatter(s, x=x, y=y, color=col, opacity=.6, hover_name="track_name", hover_data=["artists", "track_genre"],
                     color_continuous_scale=SEQUENTIAL, title=f"{y} vs {x}")
    fig.update_traces(marker_size=4)
    st.plotly_chart(style(fig, 560), width="stretch")

with tabs[2]:
    c1, c2 = st.columns([1.4, 1])
    fig = px.density_heatmap(t, x="valence", y="energy", nbinsx=40, nbinsy=40, color_continuous_scale=SEQUENTIAL,
                             title="Peta mood")
    for xx, yy, lbl in [(.75, .97, "Senang & energik"), (.25, .97, "Tegang / marah"), (.75, .03, "Tenang & positif"), (.25, .03, "Sedih / muram")]:
        fig.add_annotation(x=xx, y=yy, text=f"<b>{lbl}</b>", showarrow=False, font=dict(color="white", size=13))
    fig.add_hline(.5, line_color="white", line_width=1); fig.add_vline(.5, line_color="white", line_width=1)
    c1.plotly_chart(style(fig, 520), width="stretch")
    m = t.groupby("mood").agg(lagu=("track_id", "count"), popularity=("popularity", "mean")).reset_index()
    m["persen"] = m.lagu / m.lagu.sum() * 100
    fig = px.bar(m.sort_values("persen"), x="persen", y="mood", orientation="h", color="mood", color_discrete_map=MOOD_WARNA,
                 text=m.sort_values("persen").popularity.round(1).map(lambda v: f"pop {v}"), title="Porsi tiap mood")
    c2.plotly_chart(style(fig, 520, False), width="stretch")

with tabs[3]:
    g = st.multiselect("Bandingkan genre (2–5)", sorted(songs.track_genre.unique()), ["classical", "edm", "hip-hop", "sad", "k-pop"], max_selections=5)
    if g:
        from utils import RADAR
        prof = songs[songs.track_genre.isin(g)].groupby("track_genre")[RADAR].mean()
        c1, c2 = st.columns([1.2, 1])
        c1.plotly_chart(radar_fig({k: prof.loc[k] for k in g}, "Sidik jari genre"), width="stretch")
        c2.dataframe(prof.T.round(2), width="stretch")
        gm = songs.groupby("track_genre")[["valence", "energy", "danceability", "acousticness"]].mean()
        c2.markdown(f"😀 Paling bahagia: **{', '.join(gm.valence.nlargest(3).index)}**  \n"
                    f"😢 Paling muram: **{', '.join(gm.valence.nsmallest(3).index)}**  \n"
                    f"⚡ Paling energik: **{', '.join(gm.energy.nlargest(3).index)}**  \n"
                    f"💃 Paling danceable: **{', '.join(gm.danceability.nlargest(3).index)}**")

with tabs[4]:
    c1, c2 = st.columns([1, 3])
    feat = c1.selectbox("Fitur", CF + ["popularity", "duration_min"], index=7, key="dist_f")
    order = songs.groupby("track_genre")[feat].median().sort_values()
    pick = c1.radio("Genre yang ditampilkan", ["16 genre tersebar merata", "Pilih sendiri"])
    if pick == "Pilih sendiri":
        sel = c1.multiselect("Genre", order.index.tolist(), ["sleep", "classical", "pop", "edm", "salsa"])
    else:
        sel = list(order.index[np.linspace(0, len(order) - 1, 16).astype(int)])
    d = songs[songs.track_genre.isin(sel)]
    fig = px.violin(d, y="track_genre", x=feat, orientation="h", color="track_genre", box=True, points=False,
                    category_orders={"track_genre": [x for x in order.index if x in sel]}, title=f"{feat} per genre")
    c2.plotly_chart(style(fig, 640, False), width="stretch")

with tabs[5]:
    c1, c2 = st.columns([1.3, 1])
    fig = px.box(t, x="explicit", y="popularity", color="explicit", points=False, title="Explicit vs non-explicit",
                 color_discrete_map={True: NEG, False: GREEN})
    c1.plotly_chart(style(fig, 420, False), width="stretch")
    e = t.groupby("explicit").popularity.agg(["count", "mean", "median"]).round(1)
    c2.dataframe(e, width="stretch")
    if t.explicit.nunique() == 2:
        p = mannwhitneyu(t[t.explicit].popularity, t[~t.explicit].popularity).pvalue
        c2.metric("Uji Mann-Whitney (p-value)", f"{p:.1e}", "signifikan" if p < .05 else "tidak signifikan", delta_color="off")
        insight(f"Lagu explicit rata-rata {e.loc[True, 'mean'] - e.loc[False, 'mean']:+.1f} poin popularity dibanding non-explicit.")
