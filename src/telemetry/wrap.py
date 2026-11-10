"""Enveloppe les clients Anthropic et Bedrock d'un module, sans modifier son code.

Le module cible (rag.py ou agent.py du projet 1) crée, au niveau du module,
deux objets : `claude` et `bedrock`. Après l'import, on les remplace par des
versions qui enregistrent chaque appel (tokens, durée, coût) puis rendent la
réponse intacte. Aucun texte de question, de prompt ou de réponse n'est lu
ni stocké : seulement les compteurs.
"""
import contextlib
import sys
import time

from .pricing import cost_usd
from .store import log_call

# Fonction du projet 1 qui a déclenché l'appel -> nom de l'étape.
STEP_BY_FUNCTION = {
    "route_node": "route",
    "rewrite_node": "rewrite",
    "rewrite": "rewrite",
    "grade_node": "grade",
    "generate_node": "generate",
    "answer": "generate",
    "say": "say",
    "unknown_node": "say",
    "refuse_node": "say",
    "embed": "embed",
    "judge": "judge",
}

_ctx = {"run_id": None, "system": None, "question_id": None, "trial": None}
_forced_steps = []
stats = {"log_errors": 0}


def set_context(**kwargs):
    """Renseigne run_id, system, question_id, trial pour les appels suivants."""
    unknown = set(kwargs) - set(_ctx)
    if unknown:
        raise ValueError(f"champs inconnus : {sorted(unknown)}")
    _ctx.update(kwargs)


@contextlib.contextmanager
def tag_step(name):
    """Force le nom d'étape pour les appels faits dans le bloc (ex. le juge)."""
    _forced_steps.append(name)
    try:
        yield
    finally:
        _forced_steps.pop()


def detect_step():
    if _forced_steps:
        return _forced_steps[-1]
    frame = sys._getframe(1)
    while frame is not None:
        step = STEP_BY_FUNCTION.get(frame.f_code.co_name)
        if step:
            return step
        frame = frame.f_back
    return "unknown"


def _ms(t0):
    return int((time.perf_counter() - t0) * 1000)


class _Recorder:
    def __init__(self, conn, prices):
        self.conn = conn
        self.prices = prices

    def log(self, **fields):
        try:
            log_call(self.conn, **_ctx, **fields)
        except Exception as exc:  # ne jamais casser l'application mesurée
            stats["log_errors"] += 1
            print(f"[telemetry] enregistrement impossible : {exc!r}", file=sys.stderr)


class _WrappedMessages:
    def __init__(self, real, rec):
        self._real = real
        self._rec = rec

    def create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        step = detect_step()
        t0 = time.perf_counter()
        try:
            msg = self._real.create(*args, **kwargs)
        except Exception as exc:
            self._rec.log(step=step, provider="anthropic", model=model,
                          latency_ms=_ms(t0), status="error",
                          error_type=type(exc).__name__)
            raise
        usage = msg.usage
        tin, tout = usage.input_tokens, usage.output_tokens
        self._rec.log(
            step=step, provider="anthropic", model=model,
            input_tokens=tin, output_tokens=tout,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            latency_ms=_ms(t0),
            cost_usd=cost_usd(self._rec.prices, model, tin, tout),
        )
        return msg


class WrappedAnthropic:
    def __init__(self, real, rec):
        self._real = real
        self.messages = _WrappedMessages(real.messages, rec)

    def __getattr__(self, name):
        return getattr(self._real, name)


class WrappedBedrock:
    def __init__(self, real, rec):
        self._real = real
        self._rec = rec

    def invoke_model(self, **kwargs):
        model = kwargs.get("modelId", "unknown")
        step = detect_step()
        t0 = time.perf_counter()
        try:
            resp = self._real.invoke_model(**kwargs)
        except Exception as exc:
            self._rec.log(step=step, provider="bedrock", model=model,
                          latency_ms=_ms(t0), status="error",
                          error_type=type(exc).__name__)
            raise
        # Les tokens sont lus dans l'en-tête : le corps (lisible une seule fois)
        # reste intact pour le code du projet 1.
        headers = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        raw = headers.get("x-amzn-bedrock-input-token-count")
        tin = int(raw) if raw is not None else None
        self._rec.log(
            step=step, provider="bedrock", model=model,
            input_tokens=tin, output_tokens=0, latency_ms=_ms(t0),
            cost_usd=cost_usd(self._rec.prices, model, tin, 0) if tin is not None else None,
        )
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


def install(module, conn, prices):
    """Remplace les clients `claude` et `bedrock` d'un module déjà importé."""
    rec = _Recorder(conn, prices)
    if not isinstance(module.claude, WrappedAnthropic):
        module.claude = WrappedAnthropic(module.claude, rec)
    if not isinstance(module.bedrock, WrappedBedrock):
        module.bedrock = WrappedBedrock(module.bedrock, rec)