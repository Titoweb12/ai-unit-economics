"""Rapport SQL sur un lot mesuré : variation entre essais, causes d'échec, chemins, réconciliation.

Usage :  python -X utf8 src/report.py lot_A [--day 2026-11-25]
Lecture seule : ne modifie pas la base.
"""
import argparse
import datetime

from telemetry.store import connect

PER_TRIAL = """
SELECT o.system, o.trial, COUNT(*), SUM(o.success),
       SUM(COALESCE(c.prod_cost, 0))
FROM v_outcomes o
LEFT JOIN (
    SELECT run_id, system, question_id, trial,
           SUM(CASE WHEN step != 'judge' THEN COALESCE(cost_usd, 0) ELSE 0 END) AS prod_cost
    FROM calls GROUP BY run_id, system, question_id, trial
) c USING (run_id, system, question_id, trial)
WHERE o.run_id = ? GROUP BY o.system, o.trial ORDER BY o.system, o.trial
"""

CAUSES = """
SELECT system,
  COUNT(*),
  COALESCE(SUM(error IS NOT NULL), 0),
  COALESCE(SUM(error IS NULL AND expect_answer = 1 AND refused = 1), 0),
  COALESCE(SUM(error IS NULL AND expect_answer = 1 AND refused = 0 AND unsupported > 0), 0),
  COALESCE(SUM(error IS NULL AND expect_answer = 1 AND refused = 0 AND unsupported = 0
               AND COALESCE(hit_used, 1) = 0), 0),
  COALESCE(SUM(error IS NULL AND expect_answer = 0 AND refused = 0), 0)
FROM v_outcomes WHERE run_id = ? AND success = 0 GROUP BY system ORDER BY system
"""

PATHS = """
SELECT system, path, COUNT(*), AVG(cost) FROM (
  SELECT system, question_id, trial, group_concat(step, ' > ') AS path,
         SUM(COALESCE(cost_usd, 0)) AS cost
  FROM (SELECT * FROM calls WHERE run_id = ? AND step != 'judge' ORDER BY id)
  GROUP BY system, question_id, trial
) GROUP BY system, path ORDER BY system, COUNT(*) DESC
"""

STEP_SHARE = """
SELECT system, step, COALESCE(SUM(cost_usd), 0) AS cost
FROM calls WHERE run_id = ? AND step != 'judge' GROUP BY system, step ORDER BY system, cost DESC
"""


def section(title):
    print(f"\n=== {title} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--day", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                    help="jour UTC pour la réconciliation (défaut : aujourd'hui)")
    args = ap.parse_args()
    conn = connect()
    run_id = args.run_id

    section("Succès et coût par essai (variation entre essais)")
    print(f"  {'système':<11} {'essai':>5} {'passages':>8} {'succès':>7} {'taux':>6} {'$/succès':>10}")
    for system, trial, n, ok, cost in conn.execute(PER_TRIAL, (run_id,)):
        per = f"{cost / ok:.6f}" if ok else "n/a"
        print(f"  {system:<11} {trial:>5} {n:>8} {ok:>7} {ok / n:>6.1%} {per:>10}")

    section("Causes d'échec par système (une cause par échec, dans cet ordre)")
    print(f"  {'système':<11} {'échecs':>6} {'erreur':>7} {'refus à tort':>13} {'infidèle':>9} "
          f"{'mauvais passage':>16} {'réponse à tort':>15}")
    for r in conn.execute(CAUSES, (run_id,)):
        print(f"  {r[0]:<11} {r[1]:>6} {r[2]:>7} {r[3]:>13} {r[4]:>9} {r[5]:>16} {r[6]:>15}")

    section("Types de questions")
    for qtype, expect, n in conn.execute(
            "SELECT qtype, expect_answer, COUNT(*) FROM outcomes WHERE run_id = ? "
            "GROUP BY qtype, expect_answer", (run_id,)):
        print(f"  {str(qtype):<14} réponse attendue : {'oui' if expect else 'non (refus)'} | {n} passages")

    section("Appels en erreur")
    rows = conn.execute(
        "SELECT system, question_id, trial, step, error_type FROM calls "
        "WHERE run_id = ? AND status != 'ok'", (run_id,)).fetchall()
    for r in rows:
        print("  ", r)
    for r in conn.execute(
            "SELECT system, question_id, trial, error FROM outcomes WHERE run_id = ? AND error IS NOT NULL",
            (run_id,)):
        print("   passage en erreur :", r)
    if not rows:
        print("  aucun")

    section("Chemins les plus fréquents (production, hors juge)")
    shown = {}
    for system, path, n, cost in conn.execute(PATHS, (run_id,)):
        shown[system] = shown.get(system, 0) + 1
        if shown[system] <= 6:
            print(f"  {system:<11} {n:>4} fois  {cost:.6f} $ en moyenne  {path}")

    section("Répartition du coût de production par étape")
    totals = {}
    rows = conn.execute(STEP_SHARE, (run_id,)).fetchall()
    for system, _, cost in rows:
        totals[system] = totals.get(system, 0) + cost
    for system, step, cost in rows:
        print(f"  {system:<11} {step:<9} {cost:.6f} $  {cost / totals[system]:>5.0%}")

    section(f"Réconciliation : jour {args.day} (UTC), tous lots confondus")
    for provider in ("anthropic", "bedrock"):
        n, tin, tout, cost = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            "COALESCE(SUM(cost_usd),0) FROM calls WHERE provider = ? AND substr(ts, 1, 10) = ?",
            (provider, args.day)).fetchone()
        print(f"  {provider:<10} {n:>5} appels, {tin} tok. entrée + {tout} tok. sortie "
              f"= {tin + tout} tokens, {cost:.6f} $")
    conn.close()


if __name__ == "__main__":
    main()