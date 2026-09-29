"""Local stand-ins for the paid models (LLM_PROVIDER=fake).

They let the whole application run end to end (ingestion, OCR path, query
graph, API streaming, load tests) with no API key and zero cost. Results are
deterministic and roughly sensible, not intelligent:

* `HashingEmbeddings`: bag-of-words hashed into a vector, so texts sharing
  words are similar. Good enough for retrieval tests; not semantic.
* `FakeChatModel`: answers by quoting the first source with a citation,
  grades everything relevant, says every answer is grounded, and "OCRs"
  images with a placeholder. Supports streaming and structured output.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda

_WORD = re.compile(r"[A-Za-z0-9_.$]+")


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return " ".join(p.get("text", "") for p in message.content if isinstance(p, dict))


def _has_image(messages: list[BaseMessage]) -> bool:
    return any(isinstance(m.content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in m.content) for m in messages)


class FakeChatModel(BaseChatModel):
    relevant: bool = True       # what the grader answers; tests flip it to exercise retries
    chit_chat: bool = False     # rewrite reports 'not a document question'
    null_lists: bool = False    # mimic real models that send null for empty lists
    content_filter: bool = False  # raise like Azure's content filter does
    echo_system: bool = False   # misbehave: repeat the system prompt (prompt-leak test)
    extra: str = ""             # appended to answers (e.g. a secret, for output-guard tests)
    grounded: bool = True

    @property
    def _llm_type(self) -> str:
        return "fake-rag"

    def _reply(self, messages: list[BaseMessage]) -> str:
        if self.content_filter:
            raise ValueError("Error code: 400 - {'error': {'code': 'content_filter', 'message': "
                             "'The response was filtered due to the prompt triggering Azure "
                             "OpenAI's content management policy.'}}")
        if self.echo_system and messages and messages[0].type == "system":
            return f"My instructions are: {_text(messages[0])}"
        if _has_image(messages):
            return "[fake OCR] Transcribed page text would appear here."
        prompt = "\n".join(_text(m) for m in messages)
        sources = re.findall(r"^\[(\d+)\][^\n]*\n(.+?)(?=^\[\d+\]|\Z)", prompt, flags=re.M | re.S)
        if not sources:
            return "I don't know based on the indexed documents."
        n, body = sources[0]
        body = re.sub(r"</?source>", " ", body)
        sentence = re.split(r"(?<=[.!?])\s", " ".join(body.split()), maxsplit=1)[0][:300]
        return f"According to the documents: {sentence} [{n}]{self.extra}"

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply(messages)))])

    def _stream(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for i, word in enumerate(self._reply(messages).split(" ")):
            token = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def with_structured_output(self, schema, **kwargs):
        def respond(messages) -> Any:
            msgs = messages.to_messages() if hasattr(messages, "to_messages") else messages
            prompt = "\n".join(_text(m) for m in msgs)
            name = schema.__name__
            if name == "RewriteResult":
                question = re.search(r"QUESTION:\s*(.+)", prompt)
                return schema(needs_search=not self.chit_chat,
                              search_query=(question.group(1) if question else prompt)[:300].strip(),
                              file_types=None if self.null_lists else [])
            if name == "GradeResult":
                ids = [int(i) for i in re.findall(r"^\[(\d+)\]", prompt, flags=re.M)]
                return schema(relevant_ids=ids if self.relevant else [])
            if name == "GroundedResult":
                return schema(grounded=self.grounded,
                              unsupported=None if self.null_lists else ([] if self.grounded else ["claim"]))
            raise ValueError(f"FakeChatModel has no fake for {name}")
        return RunnableLambda(respond)
