import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from telemetry.store import connect
from telemetry.wrap import install, set_context, stats, tag_step

PRICES = {"models": {
    "claude-haiku-5-5": {"input_per_million_tokens": 0.10, "output_per_million_tokens": 0.50},
    "amazon.titan-embed-text-v2:0": {"input_per_million_tokens": 0.20, "output_per_million_tokens": 0},
}}

# Faux module du projet 1 : mêmes noms de fonctions, aucun appel réseau.
FAKE = '''
import io, json

class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0

class _Msg:
    def __init__(self, i, o):
        self.usage = _Usage(i, o)
        self.content = ["texte"]

class _Messages:
    def create(self, **kw):
        if kw.get("boom"):
            raise RuntimeError("panne")
        return _Msg(100, 20)

class _Claude:
    messages = _Messages()

class _Bedrock:
    def invoke_model(self, **kw):
        return {"ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-input-token-count": "7"}},
                "body": io.BytesIO(b'{"embedding": [0.1]}')}

claude = _Claude()
bedrock = _Bedrock()

def ask(prompt):
    return claude.messages.create(model="claude-haiku-5-5", max_tokens=10)

def route_node(): return ask("x")
def grade_node(): return ask("x")
def say(): return ask("x")

def embed():
    resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body="{}")
    return json.loads(resp["body"].read())

def retrieve_node(): return embed()
def failing(): return claude.messages.create(model="claude-haiku-5-5", boom=True)
'''

mod = types.ModuleType("fake_rag")
exec(FAKE, mod.__dict__)

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    conn = connect(pathlib.Path(d) / "test.db")
    install(mod, conn, PRICES)
    install(mod, conn, PRICES)  # une 2e installation ne doit pas empiler d'enveloppes
    set_context(run_id="t", system="agent", question_id="q1", trial=1)

    assert mod.route_node().content == ["texte"]          # réponse rendue intacte
    mod.grade_node()
    assert mod.retrieve_node() == {"embedding": [0.1]}     # corps Titan encore lisible
    mod.say()
    with tag_step("judge"):
        mod.ask("x")
    try:
        mod.failing()
    except RuntimeError:
        pass

    rows = conn.execute(
        "SELECT step, status, error_type, input_tokens, output_tokens, cost_usd, "
        "run_id, system, question_id, trial FROM calls ORDER BY id").fetchall()
    steps = [r[0] for r in rows]
    assert steps == ["route", "grade", "embed", "say", "judge", "unknown"], steps
    assert rows[5][1:3] == ("error", "RuntimeError"), rows[5]
    assert rows[0][6:] == ("t", "agent", "q1", 1), rows[0]
    assert abs(rows[0][5] - 0.00002) < 1e-12, rows[0]       # (100*0.10 + 20*0.50)/1e6
    assert abs(rows[2][5] - 0.0000014) < 1e-12, rows[2]     # 7*0.20/1e6
    tin, tout = conn.execute("SELECT SUM(input_tokens), SUM(output_tokens) FROM calls").fetchone()
    assert (tin, tout) == (407, 80), (tin, tout)
    assert stats["log_errors"] == 0
    print(f"OK : {len(rows)} appels enregistrés, étapes {steps}")
    print(f"     {tin} tokens en entrée, {tout} en sortie, coût d'un appel route = {rows[0][5]:.7f} $")
    conn.close()