"""Mesure coût ET qualité du RAG naïf, du RAG strict et de l'agent (projet 1).

Pour chaque passage (système, question, essai) :
  1. le système répond (ses appels sont enregistrés dans `calls`) ;
  2. le juge du projet 1 note la réponse (appels étiquetés `judge`) ;
  3. le verdict est enregistré dans `outcomes`, sans aucun texte.
Le code du projet 1 n'est pas modifié : on importe eval/run_eval.py et on
remplace les clients Anthropic et Bedrock des modules par des versions mesurées.

Exemples :
  python -X utf8 src/run_measure.py --set main --n 2 --run-id test_2q
  python -X utf8 src/run_measure.py --set all --trials 3 --run-id lot_A
Relancer la même commande avec le même --run-id reprend là où elle s'est arrêtée.
"""
import argparse
import collections
import datetime
import json
import os
import pathlib
import sys
import time

from telemetry.pricing import load_prices
from telemetry.store import connect, log_outcome, outcome_exists
from telemetry.wrap import install, set_context, stats, tag_step

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT1 = ROOT.parent / "Aws-Agentic-Rag"
QUESTION_FILES = {
    "main": "eval/questions.jsonl",
    "multi": "eval/multilingual.jsonl",
    "holdout": "eval/holdout.jsonl",
}
ALL_SYSTEMS = ("naif", "naif_strict", "agent")


def load_run_eval():
    """Importe eval/run_eval.py du projet 1 (après avoir vérifié qu'il n'a pas d'effet de bord)."""
    path = PROJECT1 / "eval" / "run_eval.py"
    source = path.read_text(encoding="utf-8")
    if "__name__ == " not in source:
        sys.exit("run_eval.py n'a pas de garde `if __name__ == \"__main__\"` : "
                 "l'importer lancerait toute l'évaluation. Arrêt.")
    os.chdir(PROJECT1)  # run_eval lit src/ et eval/ depuis la racine du projet 1
    sys.path.insert(0, str(PROJECT1 / "eval"))
    import run_eval
    return run_eval


def load_set(name, n):
    path = PROJECT1 / QUESTION_FILES[name]
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


def run_pass(conn, re, fn, run_id, system, qid, row, trial):
    set_context(run_id=run_id, system=system, question_id=qid, trial=trial)
    verdict, error = {}, None
    try:
        out = fn(row["question"])
        with tag_step("judge"):
            j = re.judge(row["question"], out["used"], re.strip_sources(out["answer"]))
        verdict = dict(
            hit5=re.found(row, out["retrieved"]),
            hit_used=re.found(row, out["used"]),
            refused=bool(j.get("refused")),
            unsupported=int(j.get("unsupported", 0)),
            same_language=bool(j.get("same_language")),
        )
    except Exception as exc:  # une erreur est un échec, on continue
        error = type(exc).__name__
    log_outcome(
        conn, run_id=run_id, system=system, question_id=qid, trial=trial,
        qtype=row.get("type"), lang=row.get("lang", "en"),
        expect_answer=int(row.get("type") == "answerable"), error=error, **verdict,
    )


def describe(conn, run_id, system, qid, trial):
    calls = conn.execute(
        "SELECT step, status, COALESCE(cost_usd, 0) FROM calls "
        "WHERE run_id=? AND system=? AND question_id=? AND trial=? ORDER BY id",
        (run_id, system, qid, trial),
    ).fetchall()
    prod = [c for c in calls if c[0] != "judge"]
    path = " > ".join(c[0] + ("!" if c[1] != "ok" else "") for c in prod) or "(aucun appel)"
    cost_prod = sum(c[2] for c in prod)
    cost_judge = sum(c[2] for c in calls if c[0] == "judge")
    success, error = conn.execute(
        "SELECT success, error FROM v_outcomes WHERE run_id=? AND system=? "
        "AND question_id=? AND trial=?", (run_id, system, qid, trial)).fetchone()
    label = "erreur " + error if error else ("succès" if success else "échec")
    return (f"{system:<11} {qid} essai {trial} : {path}  "
            f"[{cost_prod:.6f} $ + juge {cost_judge:.6f} $]  {label}")


def summary(conn, run_id):
    print("\nRésultats par système (coût de production = hors juge)")
    print(f"  {'système':<11} {'passages':>8} {'succès':>7} {'taux':>6} "
          f"{'coût prod.':>11} {'coût juge':>10} {'$/succès':>10} {'gaspillé':>9}")
    rows = conn.execute(
        """
        SELECT o.system, COUNT(*), SUM(o.success),
               SUM(COALESCE(c.prod_cost, 0)),
               SUM(CASE WHEN o.success = 0 THEN COALESCE(c.prod_cost, 0) ELSE 0 END),
               SUM(COALESCE(c.judge_cost, 0))
        FROM v_outcomes o
        LEFT JOIN (
            SELECT run_id, system, question_id, trial,
                   SUM(CASE WHEN step != 'judge' THEN COALESCE(cost_usd, 0) ELSE 0 END) AS prod_cost,
                   SUM(CASE WHEN step = 'judge' THEN COALESCE(cost_usd, 0) ELSE 0 END) AS judge_cost
            FROM calls GROUP BY run_id, system, question_id, trial
        ) c USING (run_id, system, question_id, trial)
        WHERE o.run_id = ? GROUP BY o.system ORDER BY o.system
        """, (run_id,)).fetchall()
    for system, n, ok, prod, wasted, judge in rows:
        per = f"{prod / ok:.6f}" if ok else "n/a"
        share = f"{wasted / prod:.0%}" if prod else "n/a"
        print(f"  {system:<11} {n:>8} {ok:>7} {ok / n:>6.0%} {prod:>11.6f} {judge:>10.6f} {per:>10} {share:>9}")

    print("\nAppels par système et par étape")
    for r in conn.execute(
        "SELECT system, step, COUNT(*), COALESCE(SUM(input_tokens),0), "
        "COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) "
        "FROM calls WHERE run_id=? GROUP BY system, step ORDER BY system, step", (run_id,)):
        print(f"  {r[0]:<11} {r[1]:<9} {r[2]:>5} appels {r[3]:>9} tok. entrée "
              f"{r[4]:>8} tok. sortie {r[5]:.6f} $")

    total, n, n_err, n_unknown, n_nocost = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0), COUNT(*), COALESCE(SUM(status!='ok'),0), "
        "COALESCE(SUM(step='unknown'),0), COALESCE(SUM(cost_usd IS NULL AND status='ok'),0) "
        "FROM calls WHERE run_id=?", (run_id,)).fetchone()
    n_failed = conn.execute(
        "SELECT COUNT(*) FROM outcomes WHERE run_id=? AND error IS NOT NULL", (run_id,)).fetchone()[0]
    print(f"\nTotal : {n} appels, {total:.6f} $ (juge compris)")
    print(f"Contrôles : {n_err} appel(s) en erreur, {n_unknown} étape(s) 'unknown', "
          f"{n_nocost} sans coût, {n_failed} passage(s) en erreur, "
          f"{stats['log_errors']} échec(s) d'enregistrement")


def main():
    ap = argparse.ArgumentParser(description="Mesure coût et qualité des systèmes du projet 1")
    ap.add_argument("--set", dest="sets", default="main",
                    help="main, multi, holdout ou all (plusieurs : main,multi)")
    ap.add_argument("--n", type=int, default=0, help="N premières questions de chaque jeu (0 = toutes)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    set_names = list(QUESTION_FILES) if args.sets == "all" else args.sets.split(",")
    systems = args.systems.split(",")
    if set(set_names) - set(QUESTION_FILES) or set(systems) - set(ALL_SYSTEMS):
        sys.exit(f"--set : {list(QUESTION_FILES)} ou all ; --systems : {list(ALL_SYSTEMS)}")
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    re = load_run_eval()
    conn = connect()
    prices = load_prices()
    for label, mod in (("rag", re.rag), ("rag_strict", re.rag_strict), ("agent", re.agent)):
        print(f"clients enveloppés dans {label} : {install(mod, conn, prices) or 'aucun !'}")
    print(f"clients enveloppés dans run_eval : {install(re, conn, prices, {'judge_client': 'anthropic'}) or 'aucun !'}")

    runners = {
        "naif": re.run_naive,
        "naif_strict": lambda q: re.run_naive(q, re.rag_strict),
        "agent": re.run_agent,
    }
    questions = {name: load_set(name, args.n) for name in set_names}
    for name, rows in questions.items():
        kinds = dict(collections.Counter(r.get("type") for r in rows))
        print(f"  jeu {name} : {len(rows)} questions, types {kinds}")
    total = len(systems) * sum(len(r) for r in questions.values()) * args.trials
    print(f"run_id : {run_id} | systèmes : {systems} | essais : {args.trials} | {total} passages\n")

    done, t0 = 0, time.time()
    for trial in range(1, args.trials + 1):
        for system in systems:
            for name, rows in questions.items():
                for i, row in enumerate(rows, start=1):
                    done += 1
                    qid = f"{name}-{i:03d}"
                    if outcome_exists(conn, run_id, system, qid, trial):
                        continue
                    run_pass(conn, re, runners[system], run_id, system, qid, row, trial)
                    minutes = (time.time() - t0) / 60
                    print(f"[{done}/{total} | {minutes:.1f} min] "
                          f"{describe(conn, run_id, system, qid, trial)}", flush=True)
    summary(conn, run_id)
    conn.close()


if __name__ == "__main__":
    main()