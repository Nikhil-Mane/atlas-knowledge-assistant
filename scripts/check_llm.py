"""Smoke-test the LLM deployments with the smallest possible calls.

Default: one embedding of the word "ping" (~1 token). Confirms the endpoint,
key, API version and embedding deployment work, and that the deployment
accepts `dimensions=EMBEDDING_DIM`.
  --vision  also sends one 32x32 image with a 3-word prompt (low detail).
  --chat    also sends one 5-token prompt to the chat and fast deployments.

Usage:  .venv/Scripts/python scripts/check_llm.py [--vision] [--chat]
"""
import argparse
import base64
import io
import sys
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from PIL import Image

from rag.config import get_settings
from rag.models import chat_llm, model_id, raw_dense_embeddings, vision_llm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    s = get_settings()

    where = urlparse(s.azure_openai_endpoint or "").netloc if s.llm_provider == "azure" else "api.openai.com"
    print(f"provider: {s.llm_provider}   endpoint: {where}   api_version: {s.azure_openai_api_version}")
    ok = True

    try:
        vec = raw_dense_embeddings(s).embed_query("ping")
        good = len(vec) == s.embedding_dim
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} embedding  {model_id(s, 'embedding')}: {len(vec)} dims "
              f"(expected {s.embedding_dim})")
    except Exception as e:
        ok = False
        print(f"FAIL embedding  {model_id(s, 'embedding')}: {type(e).__name__}: {str(e)[:300]}")

    if args.vision:
        img = Image.new("RGB", (32, 32), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        msg = HumanMessage(content=[{"type": "text", "text": "Reply with OK."},
                                    {"type": "image_url", "image_url": {"url": url, "detail": "low"}}])
        try:
            reply = vision_llm(s).invoke([msg])
            print(f"OK   vision     {model_id(s, 'vision')}: replied {str(reply.content)[:40]!r}")
        except Exception as e:
            ok = False
            print(f"FAIL vision     {model_id(s, 'vision')}: {type(e).__name__}: {str(e)[:300]}")

    if args.chat:
        for fast in (False, True):
            kind = "fast" if fast else "chat"
            try:
                reply = chat_llm(s, fast=fast).invoke("Reply with OK.")
                print(f"OK   {kind:10} {model_id(s, kind)}: replied {str(reply.content)[:40]!r}")
            except Exception as e:
                ok = False
                print(f"FAIL {kind:10} {model_id(s, kind)}: {type(e).__name__}: {str(e)[:300]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
