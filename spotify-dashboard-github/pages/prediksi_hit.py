import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from utils import page, load_art, load_hit_model, load_songs, load_tracks, insight, style, kpi_row, CF, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Prediksi Hit & Klasifikasi Genre", "🔮", "Random Forest + SHAP: fitur apa yang mendorong lagu menjadi hit, dan seberapa akurat genre ditebak dari fitur audio.")
art = load_art(); hit = art["hit"]; model = load_hit_model(); tracks = load_tracks(); songs = load_songs()
FS = hit["features"]; Xs = hit["shap_X"]; SV = hit["shap_values"]

tabs = st.tabs(["📊 Performa model", "🧠 SHAP global", "🔍 Jelaskan satu lagu", "🎚️ Coba sendiri", "🏷️ Klasifikasi genre"])

with tabs[0]:
    kpi_row([("Definisi hit", "popularity ≥ 60", f"{hit['base_rate']:.1%} lagu (10% teratas)"),
             ("ROC-AUC audio saja", f"{hit['auc']:.3f}", "0,5 = tebak acak"),
             ("ROC-AUC + genre", f"{hit['auc_genre']:.3f}", f"naik {hit['auc_genre'] - hit['auc']:+.3f}")])
    fpr, tpr, _ = roc_curve(hit["y_test"], hit["p_test"])
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=GREEN, width=3), name=f"Random Forest (AUC {hit['auc']:.2f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="acak")
    fig.update_layout(title="Kurva ROC", xaxis_title="False positive rate", yaxis_title="True positive rate")
    c1, c2 = st.columns([1.2, 1])
    c1.plotly_chart(style(fig, 430), width="stretch")
    thr = c2.slider("Ambang probabilitas", .05, .95, .5, .05)
    pred = hit["p_test"] >= thr
    cm = confusion_matrix(hit["y_test"], pred)
    fig = px.imshow(cm, text_auto=True, x=["prediksi: bukan hit", "prediksi: hit"], y=["asli: bukan hit", "asli: hit"],
                    color_continuous_scale=SEQUENTIAL, title="Confusion matrix")
    fig.update_traces(texttemplate="%{z:,}")
    c2.plotly_chart(style(fig, 360, False), width="stretch")
    insight("Fitur audio hanya memprediksi hit secara moderat. Menambahkan genre menaikkan ROC-AUC cukup besar: "
            "<b>konteks genre/pasar lebih menentukan popularitas daripada fitur audio</b>.")

with tabs[1]:
    imp = pd.Series(np.abs(SV).mean(0), index=Xs.columns).sort_values()
    arah = pd.Series({c: np.corrcoef(Xs[c], SV[:, i])[0, 1] if Xs[c].std() > 0 else 0 for i, c in enumerate(Xs.columns)})
    c1, c2 = st.columns([1, 1.4])
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color=[GREEN if arah[c] > 0 else NEG for c in imp.index]))
    fig.update_layout(title="Fitur paling berpengaruh")
    fig.update_layout(yaxis_title=None, xaxis_title="mean |SHAP|")
    c1.plotly_chart(style(fig, 520, False), width="stretch")
    c1.caption("Hijau = nilai tinggi menaikkan peluang hit · Pink = nilai tinggi menurunkan.")
    long = []
    for i, c in enumerate(Xs.columns):
        v = Xs[c].values; r = (v - v.min()) / (np.ptp(v) or 1)
        long.append(pd.DataFrame({"fitur": c, "shap": SV[:, i], "nilai (relatif)": r}))
    long = pd.concat(long)
    fig = px.scatter(long, x="shap", y="fitur", color="nilai (relatif)", color_continuous_scale=DIVERGING, opacity=.55,
                     category_orders={"fitur": imp.index[::-1].tolist()}, title="Efek fitur per lagu")
    fig.update_traces(marker_size=4)
    c2.plotly_chart(style(fig, 520), width="stretch")
    f = st.selectbox("Dependence plot fitur", imp.index[::-1].tolist())
    i = list(Xs.columns).index(f)
    fig = px.scatter(x=Xs[f], y=SV[:, i], opacity=.5, color_discrete_sequence=[GREEN],
                     labels={"x": f, "y": "SHAP (kontribusi ke peluang hit)"}, title=f"Efek {f}")
    fig.add_hline(0, line_color="grey")
    st.plotly_chart(style(fig, 380, False), width="stretch")

with tabs[2]:
    meta = hit["shap_meta"].copy()
    meta["prob_hit"] = model.predict_proba(Xs)[:, 1]
    meta["label"] = meta.track_name + " — " + meta.artists.str.slice(0, 30) + " (pop " + meta.popularity.astype(str) + ")"
    urut = st.radio("Urutkan pilihan", ["prob hit tertinggi", "popularity tertinggi", "prob hit terendah"], horizontal=True)
    key = {"prob hit tertinggi": ("prob_hit", False), "popularity tertinggi": ("popularity", False), "prob hit terendah": ("prob_hit", True)}[urut]
    meta = meta.sort_values(key[0], ascending=key[1])
    pick = st.selectbox("Pilih lagu", meta.label.tolist()[:300])
    idx = meta.index[meta.label == pick][0]; row = list(Xs.index).index(idx)
    contrib = pd.Series(SV[row], index=Xs.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10][::-1]
    c1, c2 = st.columns([1.5, 1])
    fig = go.Figure(go.Waterfall(orientation="h", y=[f"{c} = {Xs.at[idx, c]:.2f}" for c in top.index], x=top.values,
                                 base=hit["shap_base"], increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=NEG)))
    fig.update_layout(title="Kenapa lagu ini diprediksi begitu", xaxis_title="probabilitas hit")
    c1.plotly_chart(style(fig, 460, False), width="stretch")
    m = meta.loc[idx]
    c2.metric("Probabilitas hit", f"{m.prob_hit:.0%}")
    c2.metric("Popularity asli", int(m.popularity), "hit" if m.popularity >= 60 else "bukan hit", delta_color="off")
    c2.markdown(f"**{m.track_name}**  \n{m.artists}  \n_genre: {m.track_genre}_")

with tabs[3]:
    st.markdown("Atur fitur audio dan lihat probabilitas lagu menjadi hit menurut model.")
    med = tracks[FS].astype(float).median()
    cols = st.columns(4); vals = {}
    rng = {"loudness": (-35.0, 2.0), "tempo": (50.0, 210.0), "duration_min": (1.0, 8.0)}
    for i, f in enumerate([c for c in FS if c not in ["key", "time_signature", "mode", "explicit"]]):
        lo, hi = rng.get(f, (0.0, 1.0))
        vals[f] = cols[i % 4].slider(f, lo, hi, float(np.clip(med[f], lo, hi)), key=f"s_{f}")
    c1, c2, c3 = st.columns(3)
    vals["explicit"] = float(c1.toggle("explicit"))
    vals["mode"] = float(c2.toggle("mode mayor", True))
    vals["key"] = float(med["key"]); vals["time_signature"] = 4.0
    p = model.predict_proba(pd.DataFrame([vals])[FS])[0, 1]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN},
                                        "steps": [{"range": [0, 50], "color": "#24262c"}, {"range": [50, 100], "color": "#1f3a29"}]},
                                 title={"text": "Probabilitas hit"}))
    c3.plotly_chart(style(fig, 260, False), width="stretch")
    c3.caption("Model dilatih dengan class_weight='balanced', jadi angka ini adalah skor relatif (lagu rata-rata ≈ 50–60%), "
               "bukan peluang sebenarnya. Bandingkan naik-turunnya saat slider digeser.")

with tabs[4]:
    FG = CF + ["duration_min", "explicit"]
    gsel = st.multiselect("Pilih genre yang akan ditebak (2–10)", sorted(songs.track_genre.unique()),
                          ["classical", "hip-hop", "edm", "country", "heavy-metal"], max_selections=10)
    if len(gsel) >= 2:
        @st.cache_data(show_spinner="Melatih Random Forest…")
        def train_genre(gs):
            d = songs[songs.track_genre.isin(gs)]
            a, b, c, e = train_test_split(d[FG].astype(float), d.track_genre, test_size=.25, stratify=d.track_genre, random_state=42)
            m = RandomForestClassifier(200, n_jobs=-1, random_state=42).fit(a, c)
            p = m.predict(b)
            return accuracy_score(e, p), confusion_matrix(e, p, labels=list(gs)), pd.Series(m.feature_importances_, index=FG)
        acc, cm, fi = train_genre(tuple(gsel))
        kpi_row([("Akurasi", f"{acc:.1%}", f"tebak acak = {1 / len(gsel):.0%}"), ("Genre", str(len(gsel)), "1.000 lagu per genre")])
        c1, c2 = st.columns([1.3, 1])
        fig = px.imshow(cm, x=gsel, y=gsel, text_auto=True, color_continuous_scale=SEQUENTIAL, labels=dict(x="prediksi", y="asli"),
                        title="Confusion matrix")
        fig.update_traces(texttemplate="%{z:,}")
        c1.plotly_chart(style(fig, 460, False), width="stretch")
        fi = fi.sort_values()
        fig = px.bar(x=fi.values, y=fi.index, orientation="h", title="Fitur terpenting", color_discrete_sequence=[GREEN])
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        c2.plotly_chart(style(fig, 460, False), width="stretch")
        insight("Genre yang fitur audionya sangat berbeda mudah ditebak (5 genre ≈ 87%). Untuk 114 genre sekaligus akurasi hanya ~26%, "
                "karena banyak genre adalah label budaya/bahasa dengan fitur audio yang mirip.")
