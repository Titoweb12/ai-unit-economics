"""Lance des questions sur le RAG naïf et l'agent du projet 1, avec télémétrie.

Le code du projet 1 n'est pas modifié : on importe ses modules et on remplace
leurs clients `claude` et `bedrock` par des versions qui enregistrent chaque
appel. Aucun texte de question ni de réponse n'est stocké : seulement des
identifiants (ex. main-003).

Exemple :  python -X utf8 src/run_measure.py --n 5
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import pathlib
import sys

from telemetry.pricing import load_prices
from telemetry.store import connect
from telemetry.wrap import install, set_context, stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT1 = ROOT.parent / "Aws-Agentic-Rag"
QUESTION_FILES = {"main": "eval/questions.jsonl", "multi": "eval/multilingual.jsonl"}


def load_questions(set_name, n):
    path = PROJECT1 / QUESTION_FILES[set_name]
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


def describe(conn, run_id, system, qid, trial):
    """Une ligne par question : le chemin parcouru, le nombre d'appels, le coût."""
    rows = conn.execute(
        "SELECT step, status, cost_usd FROM calls "
        "WHERE run_id=? AND system=? AND question_id=? AND trial=? ORDER BY id",
        (run_id, system, qid, trial),
    ).fetchall()
    path = " > ".join(r[0] + ("!" if r[1] != "ok" else "") for r in rows) or "(aucun appel)"
    cost = sum(r[2] or 0 for r in rows)
    return f"  {system:<5} {qid} essai {trial} : {path}  [{len(rows)} appels, {cost:.6f} $]"


def summary(conn, run_id):
    print("\nRésumé par système et par étape")
    for r in conn.execute(
        "SELECT system, step, COUNT(*), COALESCE(SUM(input_tokens),0), "
        "COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) "
        "FROM calls WHERE run_id=? GROUP BY system, step ORDER BY system, step",
        (run_id,),
    ):
        print(f"  {r[0]:<5} {r[1]:<9} {r[2]:>4} appels  {r[3]:>8} tok. entrée  "
              f"{r[4]:>7} tok. sortie  {r[5]:.6f} $")
    total, n, n_err, n_unknown, n_nocost = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0), COUNT(*), "
        "COALESCE(SUM(status!='ok'),0), COALESCE(SUM(step='unknown'),0), "
        "COALESCE(SUM(cost_usd IS NULL AND status='ok'),0) "
        "FROM calls WHERE run_id=?", (run_id,),
    ).fetchone()
    print(f"\nTotal : {n} appels, {total:.6f} $")
    print(f"Contrôles : {n_err} en erreur, {n_unknown} étape(s) 'unknown', "
          f"{n_nocost} sans coût, {stats['log_errors']} échec(s) d'enregistrement")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5, help="N premières questions (0 = toutes)")
    ap.add_argument("--trials", type=int, default=1, help="nombre d'essais")
    ap.add_argument("--systems", default="naif,agent", help="naif, agent ou naif,agent")
    ap.add_argument("--set", dest="set_name", choices=QUESTION_FILES, default="main")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    systems = args.systems.split(",")
    if set(systems) - {"naif", "agent"}:
        sys.exit("--systems accepte seulement : naif, agent")
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    # Le projet 1 lit ses fichiers (data/, .env) depuis sa racine.
    os.chdir(PROJECT1)
    sys.path.insert(0, str(PROJECT1 / "src"))
    import agent  # noqa: E402
    import rag  # noqa: E402

    conn = connect()
    prices = load_prices()
    install(rag, conn, prices)
    install(agent, conn, prices)
    runners = {
        "naif": lambda q: rag.answer(q),
        "agent": lambda q: agent.app.invoke({"question": q}),
    }

    questions = load_questions(args.set_name, args.n)
    print(f"run_id : {run_id} | {len(questions)} questions | systèmes : {systems} | essais : {args.trials}")
    for trial in range(1, args.trials + 1):
        for system in systems:
            for i, row in enumerate(questions, start=1):
                qid = f"{args.set_name}-{i:03d}"
                set_context(run_id=run_id, system=system, question_id=qid, trial=trial)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):  # le projet 1 imprime beaucoup
                        runners[system](row["question"])
                except Exception as exc:
                    print(f"  ! {type(exc).__name__} sur {qid}", file=sys.stderr)
                print(describe(conn, run_id, system, qid, trial))
    summary(conn, run_id)
    conn.close()


if __name__ == "__main__":
    main()