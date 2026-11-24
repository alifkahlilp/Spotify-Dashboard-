"""Hitung semua hasil analisis berat SEKALI, simpan ke folder artifacts/.
Jalankan:  python prepare.py      (sekitar 3-5 menit)
Setelah itu dashboard (streamlit run app.py) cukup memuat hasilnya sehingga cepat.
"""
import os, warnings, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)
RNG = 42
CF = ["danceability", "energy", "loudness", "speechiness", "acousticness",
      "instrumentalness", "liveness", "valence", "tempo"]
art = {}

def step(msg): print(f"[prepare] {msg}", flush=True)

# 1. Kualitas data & cleaning ---------------------------------------------------
step("load & kualitas data")
raw = pd.read_csv(DATA, index_col=0)
art["quality"] = pd.DataFrame({
    "Masalah": ["Nilai kosong (artists/album/track)", "Baris duplikat persis", "track_id muncul di >1 genre",
                "tempo = 0 (tidak valid)", "time_signature 0/1 (tidak valid)", "popularity = 0",
                "songwriter = singer-songwriter (lagu identik)"],
    "Jumlah": [int(raw.isna().any(axis=1).sum()), int(raw.duplicated().sum()), int(raw.track_id.duplicated().sum()),
               int((raw.tempo == 0).sum()), int(raw.time_signature.isin([0, 1]).sum()), int((raw.popularity == 0).sum()),
               int(raw[raw.track_genre.isin(["songwriter", "singer-songwriter"])].groupby("track_id").track_genre.nunique().eq(2).sum())],
})
art["quality"]["% data"] = (art["quality"].Jumlah / len(raw) * 100).round(2)
art["raw_shape"] = raw.shape
art["missing_raw"] = raw.isna().sum()

df = raw.dropna(subset=["artists", "track_name"]).drop_duplicates().copy()
df.loc[df.tempo == 0, "tempo"] = np.nan
df.loc[df.time_signature.isin([0, 1]), "time_signature"] = np.nan

step("uji imputasi tempo")
valid = df[df.tempo.notna()].sample(20000, random_state=RNG)
feats = CF
X = valid[feats].copy(); truth = X.tempo.copy()
mask = np.random.default_rng(RNG).random(len(X)) < 0.10
X.loc[mask, "tempo"] = np.nan
imp_rows, imp_preview = [], {}
for name, imp in [("Mean", SimpleImputer(strategy="mean")), ("Median", SimpleImputer(strategy="median")),
                  ("KNN (k=5)", KNNImputer(n_neighbors=5))]:
    sc = StandardScaler().fit(X)
    est = pd.Series(sc.inverse_transform(imp.fit_transform(sc.transform(X)))[:, feats.index("tempo")], index=X.index)[mask]
    imp_rows.append({"Metode": name, "MAE (BPM)": np.abs(est - truth[mask]).mean(), "RMSE (BPM)": np.sqrt(((est - truth[mask]) ** 2).mean())})
    imp_preview[name] = est.values
art["imputation"] = pd.DataFrame(imp_rows).round(2)
art["imputation_preview"] = {"asli": truth[mask].values, **imp_preview}

cols = CF + ["time_signature"]
sc = StandardScaler().fit(df[cols])
df[cols] = sc.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(sc.transform(df[cols])))
df["time_signature"] = df.time_signature.round().clip(3, 7)
df["duration_min"] = df.duration_ms / 60000

tracks = df.sort_values("popularity", ascending=False).drop_duplicates("track_id").reset_index(drop=True)
tracks["n_genre"] = tracks.track_id.map(df.track_id.value_counts())
tracks["mood"] = np.select([(tracks.energy >= .5) & (tracks.valence >= .5), tracks.energy >= .5, tracks.valence >= .5],
                           ["Senang & energik", "Tegang / marah", "Tenang & positif"], "Sedih / muram")

# 2. Clustering -----------------------------------------------------------------
step("clustering (elbow, silhouette, K-Means k=5)")
scaler = StandardScaler().fit(tracks[CF]); Xc = scaler.transform(tracks[CF])
samp = np.random.default_rng(RNG).choice(len(Xc), 10000, replace=False)
rows = []
for k in range(2, 11):
    km = KMeans(k, n_init=5, random_state=RNG).fit(Xc)
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xc[samp], km.labels_[samp])})
art["elbow"] = pd.DataFrame(rows)
km = KMeans(5, n_init=10, random_state=RNG).fit(Xc)
prof = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=CF)
# beri nama otomatis berdasarkan ciri pusat cluster
names = {}
names[int(prof.speechiness.idxmax())] = "Spoken word / Live"
rest = prof.drop(index=list(names))
names[int(rest.instrumentalness.idxmax())] = "Ambient / Sleep"
rest = rest.drop(index=list(set(names) & set(rest.index)))
names[int(rest.energy.idxmax())] = "Keras & Intens"
rest = rest.drop(index=list(set(names) & set(rest.index)))
names[int(rest.valence.idxmax())] = "Party & Latin"
rest = rest.drop(index=list(set(names) & set(rest.index)))
names[int(rest.index[0])] = "Akustik Mellow"
tracks["cluster"] = pd.Series(km.labels_).map(names)

step("K-Prototype & Fuzzy C-Means (sampel 5.000)")
from kmodes.kprototypes import KPrototypes
import skfuzzy as fuzz
s5 = tracks.sample(5000, random_state=RNG)
Xn = StandardScaler().fit_transform(s5[CF])
Xkp = np.column_stack([Xn, s5[["key", "mode", "explicit", "time_signature"]].astype(int).astype(str).values]).astype(object)
kp = KPrototypes(n_clusters=5, init="Cao", n_init=3, random_state=RNG).fit(Xkp, categorical=[9, 10, 11, 12])
_, u, *_, fpc = fuzz.cluster.cmeans(Xn.T, c=5, m=2, error=1e-5, maxiter=300, seed=RNG)
art["method_compare"] = {"ari_kproto_kmeans": adjusted_rand_score(s5.cluster, kp.labels_), "fpc": fpc,
                         "fcm_maxu": u.max(axis=0), "fcm_ambigu": float((u.max(axis=0) < 0.5).mean())}

# 3. UMAP -----------------------------------------------------------------------
step("UMAP (15.000 lagu)")
import umap
s15 = tracks.sample(15000, random_state=RNG).copy()
emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=RNG).fit_transform(scaler.transform(s15[CF]))
s15["umap_x"], s15["umap_y"] = emb[:, 0], emb[:, 1]
art["umap"] = s15[["track_id", "track_name", "artists", "track_genre", "popularity", "cluster", "mood", "umap_x", "umap_y"] + CF]

# 4. Peta genre (hierarchical) --------------------------------------------------
step("hierarchical clustering genre")
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
gprof = pd.DataFrame(scaler.transform(df[CF]), columns=CF, index=df.index).groupby(df.track_genre).mean()
Zl = linkage(gprof, "ward")
art["genre_profile"] = gprof
art["genre_linkage"] = Zl
art["genre_cophenet"] = cophenet(Zl, pdist(gprof))[0]
art["genre_family"] = pd.Series(fcluster(Zl, 8, "maxclust"), index=gprof.index)
art["genre_popularity"] = df.groupby("track_genre").popularity.mean()

# 5. Network artis --------------------------------------------------------------
step("network artis + Louvain")
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
a = tracks.assign(artist=tracks.artists.str.split(";")).explode("artist")
st_ = a.groupby("artist").agg(n=("track_id", "count"), pop=("popularity", "mean"), genre=("track_genre", lambda s: s.mode()[0]))
sel = st_[st_.n >= 5].nlargest(300, "pop")
aprof = pd.DataFrame(scaler.transform(a[CF]), index=a.index, columns=CF).groupby(a.artist).mean().loc[sel.index]
S = cosine_similarity(aprof); np.fill_diagonal(S, -1)
G = nx.Graph()
for n_ in sel.index: G.add_node(n_)
for i, n_ in enumerate(sel.index):
    for j in np.argsort(S[i])[::-1][:5]:
        if S[i, j] > .5: G.add_edge(n_, sel.index[j], weight=float(S[i, j]))
comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=RNG), key=len, reverse=True)
comm_of = {n_: k for k, c in enumerate(comms) for n_ in c}
pos = nx.spring_layout(G, weight="weight", seed=RNG, k=.25)
art["net_nodes"] = pd.DataFrame([{"artis": n_, "x": pos[n_][0], "y": pos[n_][1], "pop": sel.at[n_, "pop"], "lagu": sel.at[n_, "n"],
                                  "genre": sel.at[n_, "genre"], "komunitas": f"K{comm_of[n_]}", "koneksi": G.degree(n_)} for n_ in G])
art["net_edges"] = [(u_, v_, d["weight"]) for u_, v_, d in G.edges(data=True)]
art["net_modularity"] = nx.community.modularity(G, comms, weight="weight")
art["artist_profile"] = aprof

# 6. Prediksi hit + SHAP --------------------------------------------------------
step("model hit + SHAP")
import shap
FS = CF + ["duration_min", "explicit", "mode", "key", "time_signature"]
Xh = tracks[FS].astype(float); yh = (tracks.popularity >= 60).astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xh, yh, test_size=.25, stratify=yh, random_state=RNG)
rf = RandomForestClassifier(200, max_depth=14, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RNG).fit(Xtr, ytr)
p_te = rf.predict_proba(Xte)[:, 1]
Xg = pd.get_dummies(tracks.track_genre, prefix="g").astype(float)
rf2 = RandomForestClassifier(200, max_depth=14, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RNG).fit(pd.concat([Xh, Xg], axis=1).loc[Xtr.index], ytr)
auc_genre = roc_auc_score(yte, rf2.predict_proba(pd.concat([Xh, Xg], axis=1).loc[Xte.index])[:, 1])
Xs = Xte.sample(1500, random_state=RNG)
sv = shap.TreeExplainer(rf)(Xs)[:, :, 1]
art["hit"] = {"features": FS, "auc": roc_auc_score(yte, p_te), "auc_genre": auc_genre, "base_rate": yh.mean(),
              "y_test": yte.values, "p_test": p_te,
              "shap_values": sv.values, "shap_base": float(np.ravel(sv.base_values)[0]), "shap_X": Xs,
              "shap_meta": tracks.loc[Xs.index, ["track_name", "artists", "track_genre", "popularity"]]}
joblib.dump(rf, os.path.join(OUT, "hit_model.joblib"), compress=3)

# 7. Association rules ----------------------------------------------------------
step("apriori")
from mlxtend.frequent_patterns import apriori, association_rules
B = pd.DataFrame({
    "energi_tinggi": tracks.energy >= .7, "energi_rendah": tracks.energy < .3, "danceable": tracks.danceability >= .7,
    "bahagia": tracks.valence >= .6, "sedih": tracks.valence < .3, "akustik": tracks.acousticness >= .6,
    "instrumental": tracks.instrumentalness >= .5, "keras": tracks.loudness >= -6, "pelan": tracks.loudness < -12,
    "vokal_rap": tracks.speechiness >= .2, "explicit": tracks.explicit, "mayor": tracks["mode"] == 1,
    "cepat": tracks.tempo >= 140, "pendek": tracks.duration_ms < 180000,
    "populer": tracks.popularity >= 60, "tidak_populer": tracks.popularity < 20})
fi = apriori(B, min_support=0.003, use_colnames=True, max_len=4)
rules = association_rules(fi, metric="lift", min_threshold=1.0)
rules["antecedent"] = rules.antecedents.apply(lambda s: " + ".join(sorted(s)))
rules["consequent"] = rules.consequents.apply(lambda s: " + ".join(sorted(s)))
rules["n_ant"] = rules.antecedents.apply(len); rules["n_cons"] = rules.consequents.apply(len)
art["rules"] = rules[["antecedent", "consequent", "support", "confidence", "lift", "n_ant", "n_cons"]]
art["item_share"] = B.mean().sort_values(ascending=False)

# 8. Anomali per genre ----------------------------------------------------------
step("isolation forest per genre")
Zall = pd.DataFrame(scaler.transform(df[CF]), columns=CF, index=df.index)
df["anomali_skor"] = 0.0; df["anomali"] = False
for g, idx in df.groupby("track_genre").groups.items():
    iso = IsolationForest(n_estimators=200, contamination=.02, random_state=RNG).fit(Zall.loc[idx])
    df.loc[idx, "anomali_skor"] = -iso.score_samples(Zall.loc[idx]); df.loc[idx, "anomali"] = iso.predict(Zall.loc[idx]) == -1
dev = Zall - Zall.groupby(df.track_genre).transform("mean")
top = dev.abs().idxmax(axis=1)
df["paling_beda"] = [f"{c} {'↑' if dev.at[i, c] > 0 else '↓'}" for i, c in zip(df.index, top)]

# simpan ------------------------------------------------------------------------
art["scaler"] = scaler
art["CF"] = CF
joblib.dump(art, os.path.join(OUT, "analysis.joblib"), compress=3)
tracks.to_parquet(os.path.join(OUT, "tracks.parquet"))
df.reset_index(drop=True).to_parquet(os.path.join(OUT, "songs_genre.parquet"))
step(f"selesai → {OUT}")
