"""Tableau de bord AI Unit Economics : couche MESURÉE uniquement.

Lancer :  streamlit run src/dashboard.py
Lit data/telemetry.db en lecture seule. Aucune hypothèse, aucune valeur modélisée.
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.environ.get("AUE_DB", ROOT / "data" / "telemetry.db"))

SYSTEMS = ["naif", "naif_strict", "agent"]
LABELS = {"naif": "RAG original", "naif_strict": "RAG strict", "agent": "Agent"}
COLORS = {"RAG original": "#9aa5b1", "RAG strict": "#2a7f9e", "Agent": "#e07b39"}
STEP_ORDER = ["route", "rewrite", "embed", "grade", "generate", "say"]
KEY = "c.run_id = o.run_id AND c.system = o.system AND c.question_id = o.question_id AND c.trial = o.trial"

st.set_page_config(page_title="AI Unit Economics", layout="wide")


@st.cache_data(show_spinner=False)
def q(sql, params=(), _mtime=0.0):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, con, params=params)
    finally:
        con.close()


def query(sql, params=()):
    return q(sql, params, _mtime=DB.stat().st_mtime)


def label(df):
    df = df.copy()
    df["Système"] = df["system"].map(LABELS)
    return df


if not DB.exists():
    st.error(f"Base introuvable : {DB}")
    st.stop()

runs = query("SELECT run_id, COUNT(*) AS n FROM outcomes GROUP BY run_id ORDER BY MIN(ts)")
if runs.empty:
    st.warning("Aucun résultat dans la base.")
    st.stop()

with st.sidebar:
    run = st.selectbox("Lot de mesure", runs["run_id"], index=len(runs) - 1)
    show_judge = st.checkbox("Inclure le coût du juge dans les totaux", value=False)
    st.caption("Par défaut, le coût du juge est séparé : il n'existerait pas en production.")

st.title("Combien coûte une réponse réussie ?")
st.caption(
    f"Lot « {run} » · mesuré uniquement (tokens réels × prix datés) · "
    "aucune valeur métier ni ROI sur cette page."
)

# ---------- Données de base : un coût par passage (question × essai × système) ----------
passes = query(
    f"""
    SELECT o.system, o.trial, o.question_id, o.qtype, v.success,
           COALESCE(SUM(CASE WHEN c.step != 'judge' THEN c.cost_usd END), 0) AS prod_cost,
           COALESCE(SUM(CASE WHEN c.step  = 'judge' THEN c.cost_usd END), 0) AS judge_cost
    FROM outcomes o
    JOIN v_outcomes v ON v.run_id=o.run_id AND v.system=o.system
                     AND v.question_id=o.question_id AND v.trial=o.trial
    LEFT JOIN calls c ON {KEY}
    WHERE o.run_id = ?
    GROUP BY o.system, o.trial, o.question_id
    """,
    (run,),
)
passes["total_cost"] = passes["prod_cost"] + (passes["judge_cost"] if show_judge else 0)


def summarize(g):
    n, ok = len(g), int(g["success"].sum())
    spend = g["total_cost"].sum()
    return pd.Series(
        {
            "passages": n,
            "succès": ok,
            "taux": ok / n,
            "dépense": spend,
            "coût_par_succès": spend / ok if ok else None,
            "gaspillé": g.loc[g["success"] == 0, "total_cost"].sum() / spend if spend else 0,
        }
    )


by_sys = passes.groupby("system").apply(summarize, include_groups=False).reindex(SYSTEMS).dropna(how="all")

# ---------- 1. Chiffres clés ----------
st.subheader("1. Chiffres clés")
cols = st.columns(len(by_sys))
for col, (system, r) in zip(cols, by_sys.iterrows()):
    with col:
        st.markdown(f"**{LABELS[system]}**")
        st.metric("Coût par succès", f"{r['coût_par_succès'] * 1000:.3f} $ / 1 000 succès")
        st.metric("Taux de succès", f"{r['taux']:.0%}", help=f"{int(r['succès'])} succès sur {int(r['passages'])} passages")
        st.metric("Dépense gaspillée", f"{r['gaspillé']:.0%}", help="Part de la dépense consacrée à des passages en échec")

tot = passes["prod_cost"].sum()
jud = passes["judge_cost"].sum()
st.caption(
    f"Production : {tot:.4f} $ · juge : {jud:.4f} $ "
    f"(+{jud / tot:.0%} par rapport à la production) · "
    "coût par succès = dépense de tous les passages ÷ nombre de succès."
)

# ---------- 2. Variation par essai ----------
st.subheader("2. Le résultat tient-il d'un essai à l'autre ?")
by_trial = (
    passes.groupby(["system", "trial"]).apply(summarize, include_groups=False).reset_index()
)
by_trial = label(by_trial)
c1, c2 = st.columns(2)
with c1:
    fig = px.line(by_trial, x="trial", y="taux", color="Système", markers=True, color_discrete_map=COLORS)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Taux de succès", xaxis_title="Essai", xaxis_dtick=1, height=330)
    st.plotly_chart(fig, width="stretch")
with c2:
    by_trial["coût_1000"] = by_trial["coût_par_succès"] * 1000
    fig = px.line(by_trial, x="trial", y="coût_1000", color="Système", markers=True, color_discrete_map=COLORS)
    fig.update_layout(yaxis_title="$ pour 1 000 succès", xaxis_title="Essai", xaxis_dtick=1, height=330)
    st.plotly_chart(fig, width="stretch")
st.caption(
    "Lecture : si l'écart entre deux systèmes est plus petit que la variation d'un essai à l'autre "
    "pour un même système, on ne peut pas les départager sur ce critère."
)

# ---------- 3. Coût par étape ----------
st.subheader("3. Où va l'argent ?")
steps = query(
    f"""
    SELECT c.system, c.step, SUM(c.cost_usd) AS cost, COUNT(*) AS calls,
           AVG(c.input_tokens) AS avg_in
    FROM calls c JOIN outcomes o ON {KEY}
    WHERE c.run_id = ? AND c.step != 'judge'
    GROUP BY c.system, c.step
    """,
    (run,),
)
steps["part"] = steps["cost"] / steps.groupby("system")["cost"].transform("sum")
steps = label(steps)
fig = px.bar(
    steps, x="cost", y="Système", color="step", orientation="h", barmode="stack",
    category_orders={"step": STEP_ORDER, "Système": [LABELS[s] for s in SYSTEMS]},
    labels={"cost": "Dépense de production ($)", "step": "Étape"},
    color_discrete_sequence=["#6b8f71", "#c9a227", "#9aa5b1", "#e07b39", "#2a7f9e", "#b5838d"],
)
fig.update_layout(height=300)
st.plotly_chart(fig, width="stretch")
show = steps.sort_values(["system", "cost"], ascending=[True, False])[
    ["Système", "step", "calls", "avg_in", "cost", "part"]
].rename(columns={"step": "Étape", "calls": "Appels", "avg_in": "Tokens d'entrée moyens", "cost": "Coût ($)", "part": "Part"})
st.dataframe(
    show, hide_index=True, width="stretch",
    column_config={
        "Tokens d'entrée moyens": st.column_config.NumberColumn(format="%.0f"),
        "Coût ($)": st.column_config.NumberColumn(format="%.5f"),
        "Part": st.column_config.NumberColumn(format="percent"),
    },
)

# ---------- 4. Chemins ----------
st.subheader("4. Les chemins suivis")
path_sys = st.radio("Système", SYSTEMS, format_func=LABELS.get, horizontal=True, index=len(SYSTEMS) - 1)
paths = query(
    f"""
    WITH ordered AS (
        SELECT c.system, c.trial, c.question_id, c.step, c.cost_usd
        FROM calls c JOIN outcomes o ON {KEY}
        WHERE c.run_id = ? AND c.system = ? AND c.step != 'judge'
        ORDER BY c.id
    ), p AS (
        SELECT system, trial, question_id, group_concat(step, ' > ') AS path, SUM(cost_usd) AS cost
        FROM ordered GROUP BY system, trial, question_id
    )
    SELECT p.path, COUNT(*) AS passages, AVG(p.cost) AS avg_cost, SUM(p.cost) AS spend,
           AVG(v.success) AS taux
    FROM p JOIN v_outcomes v ON v.run_id = ? AND v.system = p.system
                            AND v.question_id = p.question_id AND v.trial = p.trial
    GROUP BY p.path ORDER BY spend DESC LIMIT 8
    """,
    (run, path_sys, run),
)
if paths.empty:
    st.info("Pas de chemins pour ce système.")
else:
    paths["part"] = paths["spend"] / paths["spend"].sum()
    st.dataframe(
        paths.rename(columns={"path": "Chemin", "passages": "Passages", "avg_cost": "Coût moyen ($)",
                              "spend": "Dépense ($)", "taux": "Taux de succès", "part": "Part des 8 chemins"}),
        hide_index=True, width="stretch",
        column_config={
            "Coût moyen ($)": st.column_config.NumberColumn(format="%.6f"),
            "Dépense ($)": st.column_config.NumberColumn(format="%.4f"),
            "Taux de succès": st.column_config.NumberColumn(format="percent"),
            "Part de la dépense": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption("Les 8 chemins les plus chers. Un chemin avec trois fois « rewrite > embed > grade » est une boucle de reformulation.")

# ---------- 5. Causes d'échec ----------
st.subheader("5. Pourquoi les échecs ?")
fails = query(
    """
    SELECT system,
      CASE
        WHEN error IS NOT NULL THEN 'Erreur d''exécution'
        WHEN expect_answer = 1 AND refused = 1 THEN 'Refus à tort'
        WHEN expect_answer = 1 AND unsupported > 0 THEN 'Réponse infidèle'
        WHEN expect_answer = 1 AND COALESCE(hit_used, 1) = 0 THEN 'Mauvais passage'
        WHEN expect_answer = 0 AND refused = 0 THEN 'Réponse à tort'
        ELSE 'Autre'
      END AS cause, COUNT(*) AS n
    FROM v_outcomes WHERE run_id = ? AND success = 0
    GROUP BY system, cause
    """,
    (run,),
)
if fails.empty:
    st.success("Aucun échec dans ce lot.")
else:
    fails = label(fails)
    fig = px.bar(
        fails, x="n", y="Système", color="cause", orientation="h", barmode="stack",
        category_orders={"Système": [LABELS[s] for s in SYSTEMS]},
        labels={"n": "Passages en échec", "cause": "Cause"},
        color_discrete_sequence=["#b5838d", "#e07b39", "#c9a227", "#2a7f9e", "#6b8f71", "#9aa5b1"],
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

# ---------- Limites ----------
with st.expander("Limites de cette mesure"):
    st.markdown(
        """
- Le juge est un modèle de la même famille que les générateurs (Haiku) : un biais d'auto-préférence est possible.
- Un seul énoncé non soutenu suffit à faire échouer un passage.
- La langue de la réponse est enregistrée mais ne compte pas dans le succès.
- Questions de type FAQ uniquement ; les essais répétés ne sont pas indépendants.
- Les tokens du cache ne sont pas valorisés. Seuls Haiku et Titan sont chiffrés, aux prix datés de `config/prices.json`.
- Les coûts Titan sont trop faibles pour être vérifiés sur la facture AWS.
"""
    )