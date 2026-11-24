import plotly.express as px
import streamlit as st
from utils import page, load_art, insight, style, GREEN
from utils import DIVERGING, SEQUENTIAL, PALETTE, NEG, MOOD_WARNA, GREEN  # noqa

page("Pola Asosiasi (Apriori)", "🔗", "Kombinasi ciri lagu yang sering muncul bersama, misalnya 'danceable + explicit → populer'.")
art = load_art(); rules = art["rules"]

with st.expander("ℹ️ Definisi item", expanded=False):
    st.markdown("""| Item | Aturan | Item | Aturan |
|---|---|---|---|
| energi_tinggi | energy ≥ 0,7 | energi_rendah | energy < 0,3 |
| danceable | danceability ≥ 0,7 | bahagia / sedih | valence ≥ 0,6 / < 0,3 |
| akustik | acousticness ≥ 0,6 | instrumental | instrumentalness ≥ 0,5 |
| keras / pelan | loudness ≥ -6 / < -12 dB | vokal_rap | speechiness ≥ 0,2 |
| cepat | tempo ≥ 140 BPM | pendek | durasi < 3 menit |
| populer | popularity ≥ 60 | tidak_populer | popularity < 20 |""")

c1, c2, c3, c4 = st.columns(4)
items = ["(semua)"] + sorted(rules[rules.n_cons == 1].consequent.unique())
cons = c1.selectbox("Konsekuen (THEN)", items, index=items.index("populer"))
ms = c2.slider("Min support", .003, .10, .004, .001, format="%.3f")
mc = c3.slider("Min confidence", 0.0, 1.0, .1, .05)
ml = c4.slider("Min lift", 1.0, 5.0, 1.2, .1)
ant_has = st.multiselect("Antecedent harus mengandung", sorted({i for a in rules.antecedent for i in a.split(" + ")}))

r = rules[(rules.support >= ms) & (rules.confidence >= mc) & (rules.lift >= ml)]
if cons != "(semua)":
    r = r[r.consequent == cons]
for a in ant_has:
    r = r[r.antecedent.str.split(" + ").apply(lambda s: a in s)]
r = r.sort_values("lift", ascending=False)
st.caption(f"{len(r):,} aturan memenuhi syarat")

c1, c2 = st.columns([1.1, 1])
with c1:
    fig = px.scatter(r.head(400), x="support", y="confidence", size="lift", color="lift", hover_name="antecedent",
                     hover_data={"consequent": True, "lift": ":.2f"}, color_continuous_scale=SEQUENTIAL,
                     title="Peta aturan")
    st.plotly_chart(style(fig, 470), width="stretch")
with c2:
    top = r.head(12).iloc[::-1]
    if len(top):
        fig = px.bar(top, x="lift", y=top.antecedent + "  →  " + top.consequent, orientation="h",
                     text=top.confidence.map(lambda v: f"conf {v:.0%}"), color_discrete_sequence=[GREEN], title="Aturan terkuat")
        fig.add_vline(1, line_dash="dash", line_color="grey")
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(style(fig, 470, False), width="stretch")

st.dataframe(r.drop(columns=["n_ant", "n_cons"]).head(200).round(3), hide_index=True, width="stretch",
             column_config={"lift": st.column_config.ProgressColumn("lift", min_value=0, max_value=float(max(r.lift.max(), 1)) if len(r) else 5, format="%.2f")})

share = art["item_share"]
fig = px.bar(x=share.index, y=share.values * 100, title="Frekuensi tiap item", color_discrete_sequence=[PALETTE[0]])
fig.update_layout(xaxis_title=None, yaxis_title="% lagu")
st.plotly_chart(style(fig, 320, False), width="stretch")

insight("<b>danceable + explicit</b> membuat lagu ±1,9× lebih mungkin populer, sedangkan <b>instrumental + bahagia/danceable</b> "
        "±2,3× lebih mungkin tidak populer. Aturan terkuat secara umum bersifat fisik: <b>akustik + pelan + sedih → energi rendah</b>.")
