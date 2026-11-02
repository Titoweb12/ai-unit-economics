import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from telemetry.pricing import cost_usd, load_prices
from telemetry.store import connect, log_call

prices = load_prices()
c = cost_usd(prices, "claude-haiku-5-5", 1000, 200)
print("Coût Haiku (1000 en entrée, 200 en sortie) :",
      f"{c:.6f} $" if c is not None else "prix non renseigné")

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    conn = connect(pathlib.Path(d) / "test.db")
    for step, tin, tout in [("route", 500, 40), ("rewrite", 300, 60), ("generate", 2000, 350)]:
        log_call(
            conn, run_id="test", system="agent", question_id="q1", trial=1, step=step,
            provider="anthropic", model="claude-haiku-5-5",
            input_tokens=tin, output_tokens=tout, latency_ms=100,
            cost_usd=cost_usd(prices, "claude-haiku-5-5", tin, tout),
        )
    n, tin, tout = conn.execute(
        "SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens) FROM calls"
    ).fetchone()
    print(f"{n} appels enregistrés : {tin} tokens en entrée, {tout} en sortie")
    conn.close()