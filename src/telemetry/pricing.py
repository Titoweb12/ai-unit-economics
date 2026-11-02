import json
import pathlib

PRICES_PATH = pathlib.Path(__file__).resolve().parents[2] / "config" / "prices.json"


def load_prices(path=PRICES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cost_usd(prices, model, input_tokens, output_tokens):
    """Coût en USD, ou None si le prix du modèle n'est pas renseigné."""
    p = prices["models"].get(model)
    if not p:
        return None
    price_in = p.get("input_per_million_tokens")
    price_out = p.get("output_per_million_tokens")
    if price_in is None or price_out is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000