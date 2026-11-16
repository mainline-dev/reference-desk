"""The answering model: its settings, and one chat call through the OpenAI client.

Settings come from .env at the repo root (git ignores it; see .env.example) or
from the environment, which wins:

  REFDESK_API_KEY   required
  REFDESK_BASE_URL  default https://openrouter.ai/api/v1
  REFDESK_MODEL     default deepseek/deepseek-chat-v3-0324: DeepSeek V3, a chat
                    model, not a reasoning one
  REFDESK_ZDR       default true: OpenRouter may route the request only to a
                    zero data retention provider

The base URL and model can point anywhere that speaks the OpenAI API, such as
Ollama at http://localhost:11434/v1.
"""
import os
import re
import sys

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
TEMPERATURE = 0  # the same passages and question give the same answer, as near as the model allows
TRUE, FALSE = {"true", "1", "yes", "on"}, {"false", "0", "no", "off"}
NO_ZDR_PROVIDER = re.compile(r"data policy|zero data retention|\bzdr\b|no endpoints", re.I)


def settings():
    load_dotenv()  # finds .env from the script's folder upwards; does not override the environment
    key = os.environ.get("REFDESK_API_KEY", "").strip()
    if not key:
        sys.exit("REFDESK_API_KEY is not set. Copy .env.example to .env and add your key.")
    zdr = os.environ.get("REFDESK_ZDR", "true").strip().lower()
    if zdr not in TRUE | FALSE:
        sys.exit(f"REFDESK_ZDR is {zdr!r}: use true or false.")
    return {
        "base_url": os.environ.get("REFDESK_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        "model": os.environ.get("REFDESK_MODEL", "").strip() or DEFAULT_MODEL,
        "api_key": key,
        "zdr": zdr in TRUE,
    }


def chat(config, system, user):
    """(the model's reply, the provider that served it, its reasoning tokens) for one
    exchange. The provider and reasoning tokens are None where the API does not say.

    With zdr on, every request carries provider: {"zdr": true}. If the API turns
    it down, for instance because no zero retention provider serves the model,
    this stops. It never sends the request again without zdr.
    """
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    try:
        response = client.chat.completions.create(
            model=config["model"],
            temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"provider": {"zdr": True}} if config["zdr"] else None,
        )
    except APIStatusError as error:
        body = error.body if isinstance(error.body, dict) else {}
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        detail = inner.get("message") or error.message or str(error)
        if config["zdr"] and (error.status_code == 404 or NO_ZDR_PROVIDER.search(detail)):
            sys.exit(f"No zero data retention provider will serve {config['model']} for this request "
                     f"({error.status_code}: {detail}).\nStopped. The request was not sent again "
                     "without zero data retention. Choose a model with a ZDR provider in REFDESK_MODEL.")
        sys.exit(f"The model request failed ({error.status_code}): {detail}")
    provider = (response.model_extra or {}).get("provider")
    # The model is meant to answer without reasoning; this records whether it did.
    details = getattr(response.usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    return response.choices[0].message.content or "", provider, reasoning
