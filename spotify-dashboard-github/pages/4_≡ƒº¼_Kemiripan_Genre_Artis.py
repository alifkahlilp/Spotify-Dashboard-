import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from utils import page, load_art, insight, style, kpi_row, radar_fig, CF, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Kemiripan Genre & Artis", "🧬", "Mengelompokkan 114 genre dan 300 artis terpopuler berdasarkan rata-rata 9 fitur audio Spotify (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo).")
art = load_art()
gp = art["genre_profile"]; fam = art["genre_family"]; Zl = art["genre_linkage"]

tabs = st.tabs(["🌳 Pohon kemiripan genre", "🗂️ 8 kelompok genre", "🔍 Cari genre termirip", "🕸️ Jaringan artis"])

from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from utils import FAMILY, FAMILY_NAME, FAMILY_COLOR, RADAR, BORDER, MUTED
fam_name = fam.map(FAMILY_NAME)


def legend_chips():
    st.markdown(" ".join(f'<span style="display:inline-block;margin:0 14px 6px 0;color:#C9CED8">'
                         f'<span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:{c};margin-right:6px"></span>{n}</span>'
                         for n, c in FAMILY_COLOR.items()), unsafe_allow_html=True)


with tabs[0]:
    st.caption("Setiap genre diringkas jadi rata-rata 9 fitur audio. Genre yang cabangnya bertemu lebih dekat ke kanan punya profil lebih mirip. Contoh: latino dan reggaeton langsung bergabung karena nilai danceability, energy, dan valence-nya hampir sama.")
    kpi_row([("Genre", f"{len(gp)}", "diringkas per profil audio"), ("Kelompok genre", f"{fam.nunique()}", "hasil memotong pohon"),
             ("Cophenetic correlation", f"{art['genre_cophenet']:.2f}", "≥0,7 = sangat representatif")])
    st.write("")
    legend_chips()
    dn = dendrogram(Zl, labels=gp.index.tolist(), no_plot=True)
    leaves = dn["ivl"]; cut = Zl[-8, 2]
    fig = go.Figure()
    for xs, ys in zip(dn["icoord"], dn["dcoord"]):
        if max(ys) <= cut:
            leaf = leaves[int(round((min(xs) - 5) / 10))]
            col, w = FAMILY[fam[leaf]][1], 2
        else:
            col, w = "#4A5163", 1.4
        fig.add_trace(go.Scatter(x=ys, y=xs, mode="lines", line=dict(color=col, width=w), hoverinfo="skip", showlegend=False))
    ypos = [5 + 10 * i for i in range(len(leaves))]
    fig.add_trace(go.Scatter(x=[0] * len(leaves), y=ypos, mode="markers+text", text=leaves, textposition="middle right",
                             marker=dict(size=8, color=[FAMILY[fam[l]][1] for l in leaves]),
                             textfont=dict(size=11, color="#D5D9E2"), customdata=[fam_name[l] for l in leaves],
                             hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>", showlegend=False))
    fig.add_vline(cut, line_dash="dot", line_color=MUTED, annotation_text="dipotong di sini → 8 kelompok", annotation_font_color=MUTED)
    dmax = Zl[:, 2].max()
    fig.update_xaxes(range=[dmax * 1.03, -dmax * 0.2], title="jarak (semakin kecil = semakin mirip)", showgrid=False)
    fig.update_yaxes(visible=False, range=[-5, 10 * len(leaves) + 5])
    fig.update_layout(title="Pohon kekerabatan genre")
    st.plotly_chart(style(fig, 1850, False), width="stretch")

with tabs[1]:
    st.caption("Pohon di tab sebelumnya dipotong menjadi 8 kelompok. Genre dalam satu kelompok punya profil fitur audio yang mirip, misalnya kelompok Tenang & Instrumental berisi ambient, classical, piano, dan sleep. Nama kelompok kami beri berdasarkan ciri fitur audio yang paling menonjol.")
    legend_chips()
    rows = []
    for k, d in gp.groupby(fam):
        m = d[CF].mean()
        rows.append({"kelompok": FAMILY_NAME[k], "genre": len(d), "ciri menonjol": ", ".join(m.nlargest(2).index),
                     "ciri rendah": ", ".join(m.nsmallest(2).index), "popularity": art["genre_popularity"][d.index].mean(),
                     "anggota": ", ".join(d.index)})
    rows = pd.DataFrame(rows)
    c1, c2 = st.columns([1.5, 1])
    c1.dataframe(rows.round(1), hide_index=True, width="stretch", height=330,
                 column_config={"anggota": st.column_config.TextColumn(width="large"),
                                "popularity": st.column_config.ProgressColumn("popularity", min_value=0, max_value=60, format="%.1f")})
    raw = pd.DataFrame(art["scaler"].inverse_transform(gp[CF]), index=gp.index, columns=CF).groupby(fam_name).mean()
    pick = c2.multiselect("Bandingkan kelompok", list(FAMILY_COLOR), ["Tenang & Instrumental", "Dansa & Ceria", "Rock & Keras"])
    if pick:
        c2.plotly_chart(radar_fig({p_: raw.loc[p_, RADAR] for p_ in pick}, "Profil kelompok", h=360,
                                  colors=[FAMILY_COLOR[p_] for p_ in pick]), width="stretch")
    order = gp.assign(f=fam).sort_values(["f"]).index
    fig = make_subplots(rows=1, cols=2, column_widths=[.03, .97], horizontal_spacing=.005, shared_yaxes=True)
    fam_ids = sorted(FAMILY)
    dscale = []
    for i, k in enumerate(fam_ids):
        dscale += [[i / len(fam_ids), FAMILY[k][1]], [(i + 1) / len(fam_ids), FAMILY[k][1]]]
    fig.add_trace(go.Heatmap(z=[[fam[g_]] for g_ in order], y=list(order), x=["kelompok"], colorscale=dscale,
                             zmin=.5, zmax=len(fam_ids) + .5, showscale=False, customdata=[[fam_name[g_]] for g_ in order],
                             hovertemplate="%{y}<br>%{customdata}<extra></extra>"), 1, 1)
    fig.add_trace(go.Heatmap(z=gp.loc[order, CF].values, x=CF, y=list(order), colorscale=DIVERGING, zmid=0, zmin=-2.5, zmax=2.5,
                             colorbar=dict(title="z-score", len=.4), hovertemplate="%{y} · %{x}<br>z = %{z:.2f}<extra></extra>"), 1, 2)
    fig.update_yaxes(tickfont=dict(size=9), autorange="reversed")
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=11))
    fig.update_layout(title="Profil audio tiap genre")
    st.plotly_chart(style(fig, 1600, False), width="stretch")
    st.caption("Kuning = di atas rata-rata semua lagu, biru = di bawah rata-rata.")

with tabs[2]:
    st.caption("Pilih satu genre untuk melihat genre lain dengan profil 9 fitur audio paling dekat (jarak Euclid terkecil).")
    D = pd.DataFrame(squareform(pdist(gp[CF])), index=gp.index, columns=gp.index)
    c1, c2 = st.columns([1, 1.3])
    g = c1.selectbox("Pilih genre", gp.index.tolist(), index=gp.index.tolist().index("k-pop"))
    near = D[g].drop(g).nsmallest(8)
    fig = px.bar(x=near.values[::-1], y=near.index[::-1], orientation="h", title=f"Genre termirip {g}",
                 color_discrete_sequence=[GREEN])
    fig.update_layout(xaxis_title="jarak Euclid profil audio", yaxis_title=None)
    c1.plotly_chart(style(fig, 420, False), width="stretch")
    from utils import RADAR
    raw_prof = pd.DataFrame(art["scaler"].inverse_transform(gp[CF]), index=gp.index, columns=CF)
    c2.plotly_chart(radar_fig({g: raw_prof.loc[g, RADAR], near.index[0]: raw_prof.loc[near.index[0], RADAR],
                               D[g].idxmax(): raw_prof.loc[D[g].idxmax(), RADAR]}, "Termirip vs paling beda"),
                    width="stretch")
    pairs = D.where(np.triu(np.ones(D.shape, bool), 1)).stack().dropna().sort_values()
    insight("Pasangan genre dengan profil fitur audio paling mirip: " + ", ".join(f"<b>{a}–{b}</b>" for a, b in pairs.index[1:6]) +
            ". <b>songwriter</b> dan <b>singer-songwriter</b> berjarak 0 karena isinya lagu yang sama.")

with tabs[3]:
    nodes = art["net_nodes"].copy(); edges = art["net_edges"]
    lab = nodes.groupby("komunitas").genre.agg(lambda g: " · ".join(g.value_counts().head(2).index))
    nodes["komunitas"] = nodes.komunitas + " " + nodes.komunitas.map(lab)
    kpi_row([("Artis (node)", f"{len(nodes)}", "300 terpopuler, ≥5 lagu"), ("Koneksi (edge)", f"{len(edges):,}", "5 tetangga termirip"),
             ("Komunitas", f"{nodes.komunitas.nunique()}", "algoritma Louvain"), ("Modularity", f"{art['net_modularity']:.2f}", ">0,3 = komunitas jelas")])
    st.markdown("""**Cara membaca jaringan ini**
1. Tiap artis diringkas menjadi **rata-rata 9 fitur audio** dari semua lagunya (nilai distandardisasi), misalnya Eminem: speechiness tinggi, acousticness rendah.
2. Kemiripan dua artis diukur dengan **cosine similarity** antar vektor 9 fitur itu (1 = profil identik).
3. Tiap artis dihubungkan ke **5 artis paling mirip** (hanya jika similarity > 0,5). Ukuran titik = rata-rata popularity.
4. **Komunitas** (warna) dicari dengan algoritma Louvain: kelompok artis yang lebih banyak terhubung satu sama lain.

Artinya, dua artis bertetangga karena *angka fitur audionya* mirip, bukan karena satu genre, satu negara, atau sering kolaborasi.""")
    c1, c2 = st.columns([1, 3.2])
    kom = c1.multiselect("Tampilkan komunitas", sorted(nodes.komunitas.unique()), placeholder="Semua")
    fokus = c1.selectbox("Fokus artis", ["—"] + sorted(nodes.artis), index=0)
    show = set(nodes[nodes.komunitas.isin(kom)].artis) if kom else set(nodes.artis)
    pos = nodes.set_index("artis")[["x", "y"]]
    ex, ey = [], []
    for u, v, w in edges:
        if u in show and v in show:
            ex += [pos.at[u, "x"], pos.at[v, "x"], None]; ey += [pos.at[u, "y"], pos.at[v, "y"], None]
    nd = nodes[nodes.artis.isin(show)]
    fig = px.scatter(nd, x="x", y="y", color="komunitas", size="pop", size_max=22, hover_name="artis",
                     hover_data={"genre": True, "pop": ":.1f", "koneksi": True, "x": False, "y": False},
                     category_orders={"komunitas": sorted(nodes.komunitas.unique(), key=lambda k: int(k.split()[0][1:]))})
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=.6, color="rgba(220,220,220,.22)"), hoverinfo="skip", showlegend=False))
    fig.data = fig.data[-1:] + fig.data[:-1]
    fig.update_layout(legend=dict(title=None, font=dict(size=11)))
    lbl = nd.nlargest(25, "pop")
    fig.add_trace(go.Scatter(x=lbl.x, y=lbl.y, mode="text", text=lbl.artis, textfont=dict(size=10, color="white"),
                             textposition="top center", hoverinfo="skip", showlegend=False))
    if fokus != "—":
        nb = [v for u, v, w in edges if u == fokus] + [u for u, v, w in edges if v == fokus]
        f = nodes[nodes.artis.isin(nb + [fokus])]
        fig.add_trace(go.Scatter(x=f.x, y=f.y, mode="markers+text", text=f.artis, textposition="bottom center",
                                 marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="white", width=2)), name="fokus"))
        c1.markdown(f"**Tetangga {fokus}:**  \n" + ", ".join(nb))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(title="Jaringan artis")
    c2.plotly_chart(style(fig, 720), width="stretch")
    ctab = (nodes.groupby("komunitas").apply(lambda d: pd.Series({
        "artis": len(d), "genre dominan": ", ".join(d.genre.value_counts().head(3).index),
        "contoh": ", ".join(d.nlargest(5, "pop").artis),
        "ciri audio": ", ".join(art["artist_profile"].loc[d.artis].mean().nlargest(2).index)})))
    st.dataframe(ctab, width="stretch")
