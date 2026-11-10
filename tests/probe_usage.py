import json
import pathlib

import anthropic
import boto3
from dotenv import load_dotenv

# Réutilise le .env du projet 1 (on ne copie pas les clés).
load_dotenv(pathlib.Path(__file__).resolve().parents[2] / "Aws-Agentic-Rag" / ".env")

claude = anthropic.Anthropic()
msg = claude.messages.create(
    model="claude-haiku-5-5",
    max_tokens=500,
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print("ANTHROPIC usage :", msg.usage)

bedrock = boto3.client("bedrock-runtime")
body = json.dumps({"inputText": "hello world", "dimensions": 1024, "normalize": True})
resp = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
headers = resp["ResponseMetadata"]["HTTPHeaders"]
print("BEDROCK en-têtes avec 'token' :", {k: v for k, v in headers.items() if "token" in k.lower()})
data = json.loads(resp["body"].read())
print("BEDROCK champs du corps :", list(data.keys()))
print("BEDROCK inputTextTokenCount :", data.get("inputTextTokenCount"))