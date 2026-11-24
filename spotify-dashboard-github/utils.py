"""Fungsi bersama: tema, pemuatan data/hasil analisis, filter sidebar, kartu KPI."""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# ---------------------------------------------------------------- design tokens
GREEN = "#1ED760"            # aksen utama (brand)
BG, SURFACE, BORDER = "#0B0D12", "#151821", "#262A36"
TEXT, MUTED = "#E8EAF0", "#9AA3B2"
FONT = "Inter, 'Plus Jakarta Sans', system-ui, sans-serif"
HEAD_FONT = "'Plus Jakarta Sans', Inter, sans-serif"

# Palet kategorik untuk latar gelap: hue berjauhan, kecerahan setara, aman untuk buta warna merah-hijau
PALETTE = ["#4CC9F0", "#F7B538", "#F2668B", "#2EC4B6", "#9B8CFF", "#FF8C42", "#B5E48C", "#E0AAFF", "#7D8597"]
# Divergen (z-score): biru = di bawah rata-rata, netral gelap = rata-rata, kuning = di atas rata-rata
DIVERGING = [[0, "#3A86FF"], [.25, "#5E8FCF"], [.5, "#1B1E27"], [.75, "#D9A441"], [1, "#FFC94A"]]
SEQUENTIAL = [[0, "#13261C"], [.5, "#159E4A"], [1, "#B9F6CA"]]

CF = ["danceability", "energy", "loudness", "speechiness", "acousticness",
      "instrumentalness", "liveness", "valence", "tempo"]
RADAR = ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness"]
CL_WARNA = {"Party & Latin": "#F7B538", "Ambient / Sleep": "#4CC9F0", "Keras & Intens": "#F2668B",
            "Spoken word / Live": "#9B8CFF", "Akustik Mellow": "#2EC4B6"}
FAMILY = {1: ("Tenang & Instrumental", "#4CC9F0"), 2: ("Dansa & Ceria", "#F7B538"), 3: ("Elektronik", "#9B8CFF"),
          4: ("Extreme Metal", "#F2668B"), 5: ("Rock & Keras", "#FF8C42"), 6: ("Klasik Vokal", "#E0AAFF"),
          7: ("Pop & Akustik", "#2EC4B6"), 8: ("Komedi", "#B5E48C")}
FAMILY_NAME = {k: v[0] for k, v in FAMILY.items()}
FAMILY_COLOR = {v[0]: v[1] for v in FAMILY.values()}
NEG = "#F2668B"   # warna negatif / menurunkan
MOOD_WARNA = {"Senang & energik": "#F7B538", "Tegang / marah": "#F2668B",
              "Tenang & positif": "#2EC4B6", "Sedih / muram": "#4CC9F0"}

import plotly.io as pio
_t = go.layout.Template(pio.templates["plotly_dark"])
_t.layout.update(colorway=PALETTE, font=dict(family=FONT, size=13, color=TEXT),
                 title=dict(font=dict(family=HEAD_FONT, size=17, color=TEXT), x=0, xanchor="left"),
                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                 xaxis=dict(gridcolor="#1F232E", zerolinecolor="#2B303D", linecolor=BORDER),
                 yaxis=dict(gridcolor="#1F232E", zerolinecolor="#2B303D", linecolor=BORDER),
                 hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
                 legend=dict(itemsizing="constant", font=dict(color=MUTED)),
                 coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED))))
pio.templates["spotify"] = _t
pio.templates.default = "spotify"

CSS = f"""
<style>
.block-container {{padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1400px;}}
h1, h2, h3, h4 {{font-family: {HEAD_FONT}; letter-spacing: -.01em;}}
.hero {{padding: 1.2rem 1.5rem; border-radius: 18px; margin-bottom: 1.1rem;
       background: radial-gradient(120% 140% at 0% 0%, rgba(30,215,96,.22) 0%, rgba(76,201,240,.08) 45%, rgba(0,0,0,0) 70%), {SURFACE};
       border: 1px solid {BORDER};}}
.hero h1 {{font-size: 1.85rem; font-weight: 800; margin: 0 0 .25rem 0; padding: 0;}}
.hero p {{color: {MUTED}; margin: 0; font-size: .98rem; line-height: 1.5;}}
.kpi {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: .9rem 1.05rem; height: 100%;}}
.kpi .lbl {{color: {MUTED}; font-size: .78rem; font-weight: 500; letter-spacing: .02em;}}
.kpi .val {{font-family: {HEAD_FONT}; font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 1.3;
           font-variant-numeric: tabular-nums;}}
.kpi .sub {{color: {GREEN}; font-size: .78rem;}}
.insight {{background: rgba(30,215,96,.06); border: 1px solid rgba(30,215,96,.22); border-radius: 12px;
          padding: .8rem 1rem; margin: .4rem 0 .9rem 0; line-height: 1.55;}}
.insight b {{color: {GREEN};}}
div[data-testid="stTabs"] button p {{font-size: .95rem; font-weight: 500;}}
section[data-testid="stSidebar"] {{border-right: 1px solid {BORDER};}}
</style>
"""


def page(title: str, icon: str, subtitle: str):
    st.set_page_config(page_title=f"{title} · Spotify Insight", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (lbl, val, sub) in zip(cols, items):
        c.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
                   f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, h=420, legend=True):
    fig.update_layout(template="spotify", height=h, margin=dict(l=10, r=10, t=56 if fig.layout.title.text else 16, b=10),
                      showlegend=legend)
    return fig


def _check():
    if not os.path.exists(os.path.join(ART, "analysis.joblib")):
        st.error("Folder `artifacts/` belum ada. Jalankan dulu: `python prepare.py` (sekitar 3-5 menit).")
        st.stop()


@st.cache_data(show_spinner="Memuat data lagu…")
def load_tracks() -> pd.DataFrame:
    _check()
    return pd.read_parquet(os.path.join(ART, "tracks.parquet"))


@st.cache_data(show_spinner="Memuat data per genre…")
def load_songs() -> pd.DataFrame:
    _check()
    return pd.read_parquet(os.path.join(ART, "songs_genre.parquet"))


@st.cache_resource(show_spinner="Memuat hasil analisis…")
def load_art() -> dict:
    _check()
    return joblib.load(os.path.join(ART, "analysis.joblib"))


@st.cache_resource
def load_hit_model():
    return joblib.load(os.path.join(ART, "hit_model.joblib"))


def sidebar_filters(tracks: pd.DataFrame, key: str = "f") -> pd.DataFrame:
    """Filter global di sidebar: genre, rentang popularity, explicit."""
    with st.sidebar:
        st.markdown("### 🎛️ Filter")
        genres = sorted(tracks.track_genre.unique())
        g = st.multiselect("Genre", genres, placeholder="Semua genre", key=f"{key}_g")
        p = st.slider("Popularity", 0, 100, (0, 100), key=f"{key}_p")
        e = st.radio("Explicit", ["Semua", "Ya", "Tidak"], horizontal=True, key=f"{key}_e")
    out = tracks[tracks.popularity.between(*p)]
    if g:
        out = out[out.track_genre.isin(g)]
    if e != "Semua":
        out = out[out.explicit == (e == "Ya")]
    st.sidebar.caption(f"{len(out):,} dari {len(tracks):,} lagu ditampilkan")
    return out


def radar_fig(rows: dict, title: str = "", feats=RADAR, h=430, colors=None):
    """rows: {label: Series/array nilai fitur}"""
    fig = go.Figure()
    palette = [GREEN] + PALETTE
    for i, (lbl, v) in enumerate(rows.items()):
        v = list(np.asarray(v, dtype=float))
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=feats + feats[:1], fill="toself", name=lbl,
                                      line_color=(colors or palette)[i % len(colors or palette)], opacity=.7))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#262A36"),
                                 angularaxis=dict(gridcolor="#262A36", tickfont=dict(size=11, color="#C9CED8")),
                                 bgcolor="rgba(0,0,0,0)", domain=dict(x=[.12, .88], y=[.12, .92])), title=title)
    style(fig, h)
    fig.update_layout(legend=dict(orientation="h", y=-0.02, x=.5, xanchor="center"))
    return fig


def spotify_embed(track_id: str, height: int = 80):
    url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
    if hasattr(st, "iframe"):
        st.iframe(url, height=height)
    else:
        st.components.v1.iframe(url, height=height)
