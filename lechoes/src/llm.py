"""
Provider-agnostic access to the generative models the pipeline and its evaluation
use, so that a model can be swapped without touching the classifier.

Two response shapes are needed, and both are *structurally* constrained rather
than requested in prose — the point of a multi-model comparison is to measure the
models, not how tightly each one was prompted:

``SCHEMA_CATEGORY``
    exactly one member of a fixed set of strings. Gemini constrains this with
    ``response_mime_type="text/x.enum"``; Mistral has no enum mime type, so the
    equivalent is ``chat.parse`` against a one-field pydantic model whose field
    is that enum. Both refuse to emit anything outside the set.

``SCHEMA_INT_LIST``
    an array of integers — the 1-based candidate positions the type selector
    returns, or ``[0]`` to abstain. Gemini takes a JSON schema directly; Mistral
    takes a pydantic model, so the list arrives wrapped in one field and is
    unwrapped here.

``SCHEMA_INT``
    a single integer — the entity linker picks exactly one candidate, or 0 for
    "none suitable", so a list would admit a shape it cannot act on.

``SCHEMA_METADATA``
    one object per requested field, for metadata extraction. The *field names* are
    the caller's, passed as a ``MetadataFields`` spec rather than written here:
    which fields a collection records is domain configuration, and hard-coding one
    museum's schema into the provider layer would make the next collection edit
    this module. ``quoted=True`` turns every list into an array of
    ``{value, quote}``, which is what makes an evidence check possible.

The wrapping is why ``raw_text`` and ``parsed`` are reported separately: the raw
text differs in shape between providers (``Merchandise`` against
``{"category": "Merchandise"}``) while the parsed value does not, and only the
parsed value is scored.

Model specs are ``provider:model`` — ``mistral:ministral-14b-2512``. A bare name
means Google, so existing callers that pass ``gemini-2.5-flash`` keep working.
"""
from __future__ import annotations

import enum
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA_CATEGORY = "category"
SCHEMA_INT_LIST = "int_list"
SCHEMA_INT = "int"
SCHEMA_METADATA = "metadata"

PROVIDER_GOOGLE = "google"
PROVIDER_MISTRAL = "mistral"
DEFAULT_PROVIDER = PROVIDER_GOOGLE

# Environment variable consulted for each provider when no key is passed.
API_KEY_ENV = {PROVIDER_GOOGLE: "GEMINI_API_KEY",
               PROVIDER_MISTRAL: "MISTRAL_API_KEY"}

# A burst of requests is what gets a key rate-limited mid-sweep; the linker
# applies the same courtesy to WDQS for the same reason. Per provider, since the
# budgets are separate.
MIN_INTERVAL = {PROVIDER_GOOGLE: 0.0, PROVIDER_MISTRAL: 0.35}

_throttle_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _throttle(provider: str) -> None:
    interval = MIN_INTERVAL.get(provider, 0.0)
    if not interval:
        return
    with _throttle_lock:
        elapsed = time.monotonic() - _last_call.get(provider, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_call[provider] = time.monotonic()


@dataclass
class MetadataFields:
    """
    The shape one ``SCHEMA_METADATA`` call should return, supplied by the caller.

    `list_fields` are multi-valued; `string_fields` and `int_fields` are scalars.
    Under `quoted`, each list item becomes ``{value, quote}`` so the caller can check
    the quote against its input and drop what is not there.
    """
    list_fields: Sequence[str]
    string_fields: Sequence[str] = ()
    int_fields: Sequence[str] = ()
    quoted: bool = False

    def names(self) -> list:
        return list(self.list_fields) + list(self.string_fields) + list(self.int_fields)


@dataclass
class LLMResponse:
    """One completion, normalised across providers."""
    text: str | None          # exactly what the provider returned
    parsed: Any               # str for SCHEMA_CATEGORY, list[int] for SCHEMA_INT_LIST
    model: str
    provider: str
    cached: bool = False


def parse_model_spec(spec: str) -> tuple[str, str]:
    """
    Splits ``provider:model``. A bare name is Google, which keeps the pipeline's
    existing ``generator_model='gemini-2.5-flash'`` callers working unchanged.
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
        provider = provider.strip().lower()
        if provider not in API_KEY_ENV:
            raise ValueError(f"unknown provider {provider!r}; "
                             f"expected one of {sorted(API_KEY_ENV)}")
        return provider, model.strip()
    return DEFAULT_PROVIDER, spec.strip()


class LLMClient:
    """Common surface: one call, one normalised response."""

    provider: str = ""

    def __init__(self, model: str, api_key: str = "", temperature: float = 0.0,
                 timeout_s: float | None = None):
        self.model = model
        self.temperature = temperature
        # Neither SDK imposes a deadline by default, and a stalled request
        # otherwise blocks forever — a bulk linking run was once found wedged
        # overnight on a single call, consuming no CPU.
        self.timeout_s = timeout_s
        self.api_key = api_key or os.environ.get(API_KEY_ENV[self.provider], "")
        if not self.api_key:
            raise ValueError(
                f"no API key for provider {self.provider!r}: pass one or set "
                f"{API_KEY_ENV[self.provider]}")
        self._client = self._build_client()

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    def _build_client(self):                       # pragma: no cover - trivial
        raise NotImplementedError

    def generate(self, prompt: str, schema: str, *,
                 system_instruction: str | None = None,
                 choices: Sequence[str] | None = None,
                 fields: MetadataFields | None = None) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(LLMClient):
    provider = PROVIDER_GOOGLE

    def _build_client(self):
        from google import genai
        from google.genai import types as genai_types
        options = (genai_types.HttpOptions(timeout=int(self.timeout_s * 1000))
                   if self.timeout_s else None)      # the SDK expects millis
        return genai.Client(api_key=self.api_key, http_options=options)

    def generate(self, prompt, schema, *, system_instruction=None, choices=None,
                 fields=None):
        config: dict[str, Any] = {"temperature": self.temperature}
        if system_instruction:
            config["system_instruction"] = system_instruction

        if schema == SCHEMA_METADATA:
            if not fields:
                raise ValueError("SCHEMA_METADATA needs a MetadataFields spec")
            config["response_mime_type"] = "application/json"
            config["response_schema"] = _metadata_json_schema(fields)
            _throttle(self.provider)
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config)
            text = response.text
            parsed = response.parsed
            if not isinstance(parsed, dict):
                parsed = _dict_from_text(text)
            return LLMResponse(text=text, parsed=parsed, model=self.model,
                               provider=self.provider)

        if schema == SCHEMA_CATEGORY:
            if not choices:
                raise ValueError("SCHEMA_CATEGORY needs the allowed choices")
            config["response_mime_type"] = "text/x.enum"
            config["response_schema"] = enum.Enum(
                "CategoryChoice", {c: c for c in choices})
        elif schema == SCHEMA_INT_LIST:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = {"type": "array",
                                         "items": {"type": "integer"}}
        elif schema == SCHEMA_INT:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = {"type": "integer"}
        else:
            raise ValueError(f"unknown schema {schema!r}")

        _throttle(self.provider)
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=config)

        text = response.text
        if schema == SCHEMA_CATEGORY:
            parsed = (text or "").strip()
        elif schema == SCHEMA_INT:
            parsed = response.parsed
            if not isinstance(parsed, int):
                parsed = _int_from_text(text)
        else:
            parsed = response.parsed
            if not isinstance(parsed, list):
                parsed = _ints_from_text(text)
        return LLMResponse(text=text, parsed=parsed, model=self.model,
                           provider=self.provider)


class MistralClient(LLMClient):
    provider = PROVIDER_MISTRAL

    def _build_client(self):
        # 2.9 moved the SDK under mistralai.client; 1.x exported it at the top
        # level. Accept either so the module does not pin a single SDK layout.
        try:
            from mistralai.client import Mistral
        except ImportError:                        # pragma: no cover
            from mistralai import Mistral
        return Mistral(api_key=self.api_key)

    def generate(self, prompt, schema, *, system_instruction=None, choices=None,
                 fields=None):
        from pydantic import BaseModel, create_model

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        if schema == SCHEMA_METADATA:
            if not fields:
                raise ValueError("SCHEMA_METADATA needs a MetadataFields spec")
            model_cls = _metadata_pydantic_model(fields, BaseModel, create_model)
        elif schema == SCHEMA_CATEGORY:
            if not choices:
                raise ValueError("SCHEMA_CATEGORY needs the allowed choices")
            choice_enum = enum.Enum("CategoryChoice",
                                    {f"C{i}": c for i, c in enumerate(choices)},
                                    type=str)
            model_cls = create_model("CategoryAnswer",
                                     category=(choice_enum, ...),
                                     __base__=BaseModel)
        elif schema == SCHEMA_INT_LIST:
            model_cls = create_model("SelectionAnswer",
                                     selected=(list[int], ...),
                                     __base__=BaseModel)
        elif schema == SCHEMA_INT:
            model_cls = create_model("ChoiceAnswer",
                                     choice=(int, ...),
                                     __base__=BaseModel)
        else:
            raise ValueError(f"unknown schema {schema!r}")

        _throttle(self.provider)
        extra = {"timeout_ms": int(self.timeout_s * 1000)} if self.timeout_s else {}
        response = self._client.chat.parse(
            model=self.model, messages=messages, temperature=self.temperature,
            response_format=model_cls, **extra)
        message = response.choices[0].message
        text = message.content
        parsed_obj = message.parsed

        if schema == SCHEMA_METADATA:
            parsed = (parsed_obj.model_dump() if parsed_obj is not None
                      else _dict_from_text(text))
        elif schema == SCHEMA_CATEGORY:
            # Unwrap {"category": "..."} to the bare string Gemini returns, so
            # everything downstream sees one shape.
            value = getattr(parsed_obj, "category", None)
            parsed = getattr(value, "value", value)
            parsed = (parsed or "").strip() if isinstance(parsed, str) else parsed
        elif schema == SCHEMA_INT:
            parsed = getattr(parsed_obj, "choice", None)
            if not isinstance(parsed, int):
                parsed = _int_from_text(text)
        else:
            parsed = getattr(parsed_obj, "selected", None)
            if not isinstance(parsed, list):
                parsed = _ints_from_text(text)
        return LLMResponse(text=text, parsed=parsed, model=self.model,
                           provider=self.provider)


def _metadata_json_schema(fields: MetadataFields) -> dict:
    """
    The Gemini response schema for one metadata extraction.

    `propertyOrdering` is set because Gemini honours it and a stable key order makes
    the cached raw text diffable between runs.
    """
    quoted_item = {"type": "object",
                   "properties": {"value": {"type": "string"},
                                  "quote": {"type": "string"}},
                   "required": ["value", "quote"],
                   "propertyOrdering": ["value", "quote"]}
    properties: dict[str, Any] = {}
    for name in fields.list_fields:
        properties[name] = {"type": "array",
                            "items": dict(quoted_item) if fields.quoted
                            else {"type": "string"}}
    for name in fields.string_fields:
        properties[name] = {"type": "string"}
    for name in fields.int_fields:
        properties[name] = {"type": "integer"}
    names = fields.names()
    return {"type": "object", "properties": properties,
            "required": names, "propertyOrdering": names}


def _metadata_pydantic_model(fields: MetadataFields, base, create_model):
    """
    The Mistral equivalent: `chat.parse` takes a pydantic model, not a JSON schema.

    Every field is required with a default, so a model that omits one yields an empty
    cell rather than a validation error — the caller must be able to tell "asserted
    nothing" from "the call failed", and losing the whole object to one missing key
    would collapse the two.
    """
    if fields.quoted:
        item = create_model("QuotedValue", value=(str, ...), quote=(str, ...),
                            __base__=base)
        list_type = list[item]
    else:
        list_type = list[str]
    spec: dict[str, Any] = {name: (list_type, []) for name in fields.list_fields}
    spec.update({name: (str, "") for name in fields.string_fields})
    spec.update({name: (int, 0) for name in fields.int_fields})
    return create_model("MetadataAnswer", __base__=base, **spec)


def _dict_from_text(text: str | None) -> dict:
    """
    Last resort when a provider returns unparsed content: the JSON object out of the
    body. Returns {} rather than raising, so the caller records a parse failure for
    one object instead of losing the run.
    """
    if not text:
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _ints_from_text(text: str | None) -> list:
    """
    Last resort when a provider returns unparsed content: pull a JSON array of
    integers out of the text. Returns [] rather than raising, so the caller can
    record `error_malformed` instead of losing the cell.
    """
    if not text:
        return []
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value
    if isinstance(value, dict):
        for candidate in value.values():
            if isinstance(candidate, list) and all(isinstance(v, int) for v in candidate):
                return candidate
    return []


def _int_from_text(text: str | None):
    """Single integer out of a response body, or None if there isn't one."""
    if text is None:
        return None
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        stripped = (text or "").strip()
        return int(stripped) if stripped.lstrip("-").isdigit() else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for candidate in value.values():
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
    return None


CLIENTS = {PROVIDER_GOOGLE: GeminiClient, PROVIDER_MISTRAL: MistralClient}


def cache_identity(spec: str) -> str:
    """
    The name a model contributes to a cache key.

    The bare model for the default provider, the full ``provider:model``
    otherwise. Caches built before this module existed keyed on the bare name,
    and rendering the same model differently would invalidate every one of them
    — an expensive way to change nothing.
    """
    provider, model = parse_model_spec(spec)
    return model if provider == DEFAULT_PROVIDER else f"{provider}:{model}"


def build_client(model_spec: str, api_key: str = "", temperature: float = 0.0,
                 timeout_s: float | None = None) -> LLMClient:
    """Constructs the client for ``provider:model`` (bare name means Google)."""
    provider, model = parse_model_spec(model_spec)
    return CLIENTS[provider](model=model, api_key=api_key,
                             temperature=temperature, timeout_s=timeout_s)


@dataclass
class CachingLLMClient:
    """
    Disk-backed cache in front of any `LLMClient`.

    Keyed by ``sha1(provider, model, schema, system_instruction, prompt,
    run_index)``. The model is part of the key, so one cache file can hold a
    whole multi-model sweep without collisions, and an interrupted run resumes
    without re-billing the cells it already paid for.
    """
    client: LLMClient
    path: Path
    run_index: int = 1
    hits: int = 0
    misses: int = 0
    _cache: dict = field(default_factory=dict)
    _unsaved: int = 0

    def __post_init__(self):
        self.path = Path(self.path)
        if self.path.exists():
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def spec(self) -> str:
        return self.client.spec

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def provider(self) -> str:
        return self.client.provider

    def _key(self, prompt: str, schema: str, system_instruction: str | None,
             fields: MetadataFields | None = None) -> str:
        # The response *shape* is part of the key: the same prompt asked for quoted
        # and unquoted values comes back differently, and serving one from the other
        # would be a silent corruption rather than a cache hit.
        shape = ("|".join(fields.names()) + f"|quoted={fields.quoted}"
                 if fields else "")
        payload = "\x00".join([self.client.provider, self.client.model, schema,
                               system_instruction or "", str(self.run_index), shape,
                               prompt])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def generate(self, prompt, schema, *, system_instruction=None, choices=None,
                 fields=None):
        key = self._key(prompt, schema, system_instruction, fields)
        if key in self._cache:
            self.hits += 1
            entry = self._cache[key]
            return LLMResponse(text=entry["text"], parsed=entry["parsed"],
                               model=self.client.model,
                               provider=self.client.provider, cached=True)
        response = self.client.generate(
            prompt, schema, system_instruction=system_instruction, choices=choices,
            fields=fields)
        self.misses += 1
        self._cache[key] = {"text": response.text, "parsed": response.parsed}
        self._unsaved += 1
        if self._unsaved >= 10:          # survive an interrupted sweep
            self.save()
        return response

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")
        self._unsaved = 0
