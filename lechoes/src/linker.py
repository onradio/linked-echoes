from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import requests
from pathlib import Path

from jinja2 import Template

try:  # package import (lechoes.src.linker) and bare import (notebooks) both work
    from .entichecks import get_eck
    from .llm import SCHEMA_INT, build_client
except ImportError:  # pragma: no cover
    from entichecks import get_eck
    from llm import SCHEMA_INT, build_client

# FIXME: Eventually move these to a config file and another for all prompts
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL_ENDPOINT = "http://dbpedia.org/sparql"

USER_AGENT = "BMEEntityLinker/1.0 (radio@liv.ac.uk)"

# WDQS throttles bursts aggressively: a handful of back-to-back queries starts
# returning 429/503, and an unthrottled validation loop degrades into a stream
# of failures that look like "every candidate is invalid". Serialise calls with
# a minimum gap and retry on the transient status codes.
SPARQL_MIN_INTERVAL = 1.0
SPARQL_MAX_RETRIES = 3

# Maximum candidates folded into a single batched validation request.
# Cost grows super-linearly in the number of branches WDQS has to evaluate at
# once: the colour check (whose branches include an unbounded wdt:P279*) runs in
# 1.3s at 2 candidates, 7.8s at 8, and exceeds the 60s server limit at 13.
# Eight is comfortable for every type in entichecks; raise it only with the
# `--batch-equivalence` test as evidence.
BATCH_VALIDATION_CHUNK = 8

# The batched query is an optimisation with a per-candidate ASK fallback, so it
# should give up quickly rather than retry an expensive query three times.
BATCH_VALIDATION_RETRIES = 2

# The MediaWiki API is a different service from WDQS with its own budget, so it
# gets its own throttle rather than sharing the SPARQL one.
API_MIN_INTERVAL = 1.0
API_MAX_RETRIES = 3

_throttle_lock = threading.Lock()
_throttle_last_call: dict = {}


def _throttle(service: str, min_interval: float) -> None:
    """Space calls to one service at least `min_interval` seconds apart."""
    with _throttle_lock:
        elapsed = time.monotonic() - _throttle_last_call.get(service, 0.0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _throttle_last_call[service] = time.monotonic()


def _retry_after(response, attempt: int) -> float:
    """Seconds to wait before retrying, honouring Retry-After when sane."""
    header = response.headers.get("Retry-After")
    try:
        wait = float(header) if header else 5.0 * (attempt + 1)
    except ValueError:
        wait = 5.0 * (attempt + 1)
    return min(wait, 60.0)


def _throttle_sparql(min_interval: float) -> None:
    _throttle("sparql", min_interval)

LINKER_PROMPT_TEMPLATE = """
You are an expert entity linker. Your task is to identify the single most
relevant entity from the provided list based on a search term, expected entity
type (e.g. music artist) and any additional context to orient the selection.

**Search Term:** "{search_term}"
**Expected Type:** "{entity_type}"
**Context:** "{context_query}"

Analyse the following candidate entities with their labels and descriptions.
The context is crucial for disambiguation. For example, if the context mentions
"John Williams guitar collection", you should pick the correct entity for John 
Williams (the guitarist).

Based on all the information, which entity is the best match? Respond only with
the number corresponding to your choice, or "0" if none are suitable.

**Candidate Entities:**
{candidates_text}
"""


def run_sparql_query(query: str,
                     endpoint: str,
                     timeout: int = 60,
                     retries: int = SPARQL_MAX_RETRIES,
                     min_interval: float = SPARQL_MIN_INTERVAL,
                     method: str = "GET"):
    """
    Runs a SPARQL query against a given endpoint and returns the results as a
    Python dictionary. Returns None on error.

    Calls are throttled and retried with backoff on the transient status codes
    WDQS uses when it is being hammered (429 / 500 / 502 / 503). Without this a
    validation loop silently degrades: failed queries return None, which the
    caller cannot distinguish from "this entity is the wrong type".

    Pass ``method="POST"`` for long queries: a GET puts the query in the URL and
    WDQS answers 414 once it grows past a few kilobytes, which batched
    validation does easily.
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(retries):
        _throttle_sparql(min_interval)
        try:
            if method.upper() == "POST":
                response = requests.post(
                    endpoint,
                    data={"query": query, "format": "json"},
                    headers=headers,
                    timeout=timeout,
                )
            else:
                response = requests.get(
                    endpoint,
                    params={"query": query, "format": "json"},
                    headers=headers,
                    timeout=timeout,
                )
            if response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(_retry_after(response, attempt))
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(3.0 * (attempt + 1))
                continue
            print(f"Error running query: {e}")
            return None
    return None


def wikidata_get_labels(entity_ids: list,
                        min_interval: float = API_MIN_INTERVAL) -> dict:
    """
    Batch-fetch English label and description for a list of Q-numbers.

    Returns ``{qid: (label, description)}``; entities that do not exist are
    omitted. Uses the MediaWiki API rather than SPARQL (no rate-limit pressure
    on WDQS, and 50 entities per request).

    Returns None if a batch could not be fetched, so a caller checking whether
    entities exist cannot mistake a failed request for a missing entity.
    """
    out = {}
    ids = [q for q in entity_ids if q]
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels|descriptions",
            "languages": "en",
            "format": "json",
        }
        data = None
        for attempt in range(API_MAX_RETRIES):
            _throttle("wikidata_api", min_interval)
            try:
                response = requests.get(
                    WIKIDATA_API_ENDPOINT,
                    params=params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=60,
                )
                if (response.status_code in (429, 500, 502, 503)
                        and attempt < API_MAX_RETRIES - 1):
                    time.sleep(_retry_after(response, attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                print(f"Error fetching labels: {e}")
                return None
        if data is None:
            return None
        for qid, entity in (data.get("entities") or {}).items():
            if "missing" in entity:
                continue
            out[qid] = (
                entity.get("labels", {}).get("en", {}).get("value", ""),
                entity.get("descriptions", {}).get("en", {}).get("value", ""),
            )
    return out

def wikidata_get_claims(entity_ids: list,
                        properties: tuple = ("P31",),
                        min_interval: float = API_MIN_INTERVAL) -> dict:
    """
    Batch-fetch statement values for a list of Q-numbers, 50 ids per request.

    Returns ``{qid: {prop: [{"id": ..., "rank": ...}, ...]}}``.

    Both the truthy and the non-truthy statements are returned, because they are
    not the same set and the difference matters: the ECKs match on ``wdt:``,
    which exposes only best-rank non-deprecated statements, so a class that
    appears here may be one the validator cannot see. Use ``truthy_values`` to
    get the validator's view.

    Returns None if any batch could not be fetched, so callers do not mistake a
    failed request for an entity with no statements.
    """
    out: dict = {}
    ids = [q for q in entity_ids if q]
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "claims",
            "format": "json",
        }
        data = None
        for attempt in range(API_MAX_RETRIES):
            _throttle("wikidata_api", min_interval)
            try:
                response = requests.get(
                    WIKIDATA_API_ENDPOINT, params=params,
                    headers={"User-Agent": USER_AGENT}, timeout=60,
                )
                if (response.status_code in (429, 500, 502, 503)
                        and attempt < API_MAX_RETRIES - 1):
                    time.sleep(_retry_after(response, attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                print(f"Error fetching claims: {e}")
                return None
        if data is None:
            return None
        for qid, entity in (data.get("entities") or {}).items():
            record = {p: [] for p in properties}
            if "missing" in entity:
                out[qid] = record
                continue
            for prop in properties:
                for statement in entity.get("claims", {}).get(prop, []):
                    value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
                    if isinstance(value, dict) and "id" in value:
                        record[prop].append({
                            "id": value["id"],
                            "rank": statement.get("rank", "normal"),
                        })
            out[qid] = record
    return out


def truthy_values(statements: list) -> list:
    """
    Reduce statements to the set ``wdt:`` would expose: drop deprecated, then
    keep only preferred if any preferred exists, otherwise the normal ones.
    """
    live = [s for s in statements if s.get("rank") != "deprecated"]
    preferred = [s for s in live if s.get("rank") == "preferred"]
    return [s["id"] for s in (preferred or live)]


def wikidata_entity_lookup(search_term,
                           num_results=10,
                           retries: int = API_MAX_RETRIES,
                           min_interval: float = API_MIN_INTERVAL):
    """
    Entity lookup via Wikidata retrieves Q numbers, labels, and descriptions.

    Throttled and retried on the transient status codes, like
    ``run_sparql_query``. Returns a list of candidates (possibly empty, which is
    a genuine "Wikidata knows nothing by this name"), or **None** if the request
    could not be completed. Callers that persist results must keep those two
    apart: freezing a failed request as an empty pool silently turns an outage
    into a retrieval miss.
    """
    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "en",
        "format": "json",
        "type": "item",
        "limit": num_results,
    }
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries):
        _throttle("wikidata_api", min_interval)
        try:
            response = requests.get(
                WIKIDATA_API_ENDPOINT,
                params=params,
                headers=headers,
                timeout=60,
            )
            if response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(_retry_after(response, attempt))
                continue
            response.raise_for_status()
            data = response.json()
            results_list = []
            for result in data.get("search", []):
                entity_display = result.get("display", {})
                entity_label = entity_display.get("label", {}).get("value")
                entity_desc = entity_display.get("description", {}).get("value")
                results_list.append({
                    "wikidata_id": result["id"],
                    "label": entity_label,
                    "description": entity_desc,
                })
            return results_list
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < retries - 1:
                time.sleep(3.0 * (attempt + 1))
                continue
            print(f"Error during Wikidata request: {e}")
            return None
    return None


def validate_entity(entity_id, validation_template, source="wd"):
    """
    Validate a tentative entity ID against a given validation SPARQL ask query.

    Returns True/False, or None if the query could not be run — callers must
    treat None as "unknown", not as "invalid".
    """
    template = Template(validation_template[source])
    valid_q = template.render(entity_id=entity_id)
    endpoint = WIKIDATA_SPARQL_ENDPOINT if source == "wd" \
        else DBPEDIA_SPARQL_ENDPOINT
    result = run_sparql_query(valid_q, endpoint=endpoint)
    return result["boolean"] if result else None


def validate_entities_batch(entity_ids: list,
                            validation_template: dict,
                            chunk: int = None) -> dict:
    """
    Validate several Wikidata entities with a single SPARQL request.

    The per-entity ASK query is rewritten into one SELECT whose branches each
    keep a *constant* subject::

        SELECT ?e WHERE {
          { <ask body for Q1> BIND(wd:Q1 AS ?e) }
          UNION
          { <ask body for Q2> BIND(wd:Q2 AS ?e) }
          ...
        }

    Binding the subject with ``VALUES ?e { ... }`` instead would be shorter but
    is much slower: WDQS only plans a transitive path efficiently when both of
    its endpoints are constants, so the subject has to stay inlined per branch.

    Returns ``{qid: bool}``, or None on failure so the caller can fall back to
    per-entity ASKs.
    """
    if not entity_ids:
        return {}

    # Chunked to bound how much work one request asks WDQS for; the request
    # itself goes out as POST, so query length is not the constraint. The best
    # size is type-dependent — on real candidate pools the music-artist check
    # (whose wdt:P106/wdt:P279* branch is costly on humans with many
    # occupations) is measurably *worse* per entity at 8 than at 4 — so callers
    # doing bulk work should pass their own.
    size = chunk or BATCH_VALIDATION_CHUNK
    if len(entity_ids) > size:
        merged = {}
        for start in range(0, len(entity_ids), size):
            part = validate_entities_batch(
                entity_ids[start:start + size], validation_template, chunk=size)
            if part is None:
                return None
            merged.update(part)
        return merged

    template = Template(validation_template["wd"])
    branches = []
    for qid in entity_ids:
        ask_query = template.render(entity_id=qid).strip()
        start, end = ask_query.find("{"), ask_query.rfind("}")
        if start == -1 or end == -1:
            return None  # not an ASK we can rewrite; caller falls back
        body = ask_query[start + 1:end]
        # Drop the "# label" comments the ASK bodies carry: they would otherwise
        # comment out whatever follows on the same line once reflowed, and they
        # are pure bulk in a query repeated once per candidate.
        body = " ".join(
            line.split("#", 1)[0].strip() for line in body.splitlines()
        ).strip()
        branches.append(f"  {{ {{ {body} }} BIND(wd:{qid} AS ?e) }}")

    query = "SELECT DISTINCT ?e WHERE {\n" + "\n  UNION\n".join(branches) + "\n}"
    result = run_sparql_query(query, endpoint=WIKIDATA_SPARQL_ENDPOINT,
                              method="POST", retries=BATCH_VALIDATION_RETRIES)
    if not result:
        return None
    try:
        matched = {
            binding["e"]["value"].rsplit("/", 1)[-1]
            for binding in result["results"]["bindings"]
        }
    except (KeyError, TypeError):
        return None
    return {qid: qid in matched for qid in entity_ids}

# ##############################################################################
# RAG Entity Linker
# ##############################################################################

class RAGEntityLinker:
    def __init__(self,
                 generator_model: str = 'gemini-2.5-flash',
                 generator_api_key: str = '',
                 cache_path: Path | None = None,
                 temperature: float = 0.0,
                 llm_timeout_s: float = 90.0,
                 llm_retries: int = 3,
                 generator: object = None):
        """
        Initialises the RAGEntityLinker.

        Args:
            generator_model: The selection model as ``provider:model`` — e.g.
                        ``mistral:ministral-14b-2512``. A bare name means Google,
                        so existing callers are unaffected.
            generator: A ready-made ``llm.LLMClient`` (or caching wrapper) used
                        in preference to building one, so an evaluation can share
                        a cache across linkers.
            cache_path: Optional path to a JSON file for caching linker decisions.
                        Keyed on (search_term, entity_type). On a cache hit the
                        LLM call and SPARQL validation are skipped entirely.
            temperature: Sampling temperature for the selection call. Defaults
                        to 0.0 — picking one candidate from a numbered list has
                        no reason to be creative, and leaving it at the provider
                        default made repeated runs vary for the wrong reason.
            llm_timeout_s: Per-request deadline for the selection call. The SDK
                        does not impose one by default, and a request that
                        stalls will otherwise block forever — a bulk run left
                        overnight was found wedged on a single call with no
                        CPU time consumed.
            llm_retries: Attempts per selection call before reporting an error.
        """
        self.temperature = temperature
        self.llm_timeout_s = llm_timeout_s
        self.llm_retries = llm_retries
        # The provider layer resolves the key for whichever provider the spec
        # names and raises if it is absent, so the linker no longer hardcodes
        # GEMINI_API_KEY.
        self.generator = generator or build_client(
            generator_model, api_key=generator_api_key,
            temperature=temperature, timeout_s=llm_timeout_s)
        self.generator_model = getattr(self.generator, "spec", generator_model)
        print(f"LLM client initialised: {self.generator_model}")

        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict = {}
        self._cache_hits = 0
        self._cache_misses = 0
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
                print(f"Linker cache loaded: {len(self._cache)} entries from {self._cache_path}")
            except Exception as e:
                print(f"Warning: could not load linker cache: {e}")

    def _cache_key(self, search_term: str, entity_type: str) -> str:
        raw = f"{search_term.strip().lower()}|{entity_type.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def cache_stats(self) -> dict:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_cached": len(self._cache),
        }

    def _search_candidates(self, search_term: str, top_k: int = 10, verbose: bool = False) -> list:
        """
        Searches Wikidata for candidate entities matching the search term.
        """
        candidates = wikidata_entity_lookup(search_term, num_results=top_k)
        if not candidates and verbose:
            print("Could not retrieve any candidates from Wikidata.")
            return []
        return candidates

    def _validate_candidates(self, candidates: list, entity_type_eck: dict, verbose: bool = False) -> list:
        """
        Validates candidates using SPARQL ASK queries based on the provided ECK.
        """
        validated_candidates = []
        for i, candidate in enumerate(candidates):
            is_valid = validate_entity(candidate['wikidata_id'], entity_type_eck)
            if verbose:  # print validation result after each candidate
                status = "Valid" if is_valid else "Invalid"
                print(f"  {i+1}. {candidate['label']} ({candidate['wikidata_id']}) -> {status}")
            if is_valid:  # only keep valid candidates
                validated_candidates.append(candidate)

        return validated_candidates

    def generate_prompt(self, search_term: str,
                        context_query: str,
                        entity_type: str,
                        candidates: list) -> str:
        candidates_text = ""
        for i, candidate in enumerate(candidates):
            # A candidate may not have a description
            desc = candidate.get('description', 'No description available.')
            candidates_text += (
                f"({i+1}) Label: {candidate['label']} |"
                f" Description: {desc} |"
                f" ID: {candidate['wikidata_id']}\n"
            )

        return LINKER_PROMPT_TEMPLATE.format(
            search_term=search_term,
            entity_type=entity_type,
            context_query=context_query,
            candidates_text=candidates_text
        )

    def link(self,
             search_term: str,
             context_query: str = "",
             entity_type: str = "",
             entity_type_eck: dict = None,
             top_k: int = 10,
             validate: bool = True,
             verbose: bool = True) -> dict | None:
        """
        Performs the full RAG entity linking process.
        """
        # Normalise curly/smart quotes to straight apostrophe
        for ch in "\u2018\u2019\u02bc\u0060\u00b4":
            search_term = search_term.replace(ch, "'")

        # Step 0: Some input validation
        if validate and not entity_type_eck:
            raise ValueError("ECK must be provided if validate is True.")
        if search_term.strip() == "":
            raise ValueError("Search term must be a non-empty string.")

        # Cache check — skip retrieval + Gemini if already resolved
        cache_key = self._cache_key(search_term, entity_type)
        if cache_key in self._cache:
            self._cache_hits += 1
            if verbose:
                print(f"Cache hit for '{search_term}' ({entity_type})")
            return self._cache[cache_key]
        self._cache_misses += 1

        # Step 1: Search for initial candidates and validate them if needed
        candidates = self._search_candidates(search_term, top_k, verbose)
        if not candidates:
            return None
        if validate:  # Validate candidates via SPARQL ASK queries, if required
            candidates = self._validate_candidates(
                candidates, entity_type_eck, verbose)
        # Check for no candidates or single candidate from retrieval/validation
        if not candidates:
            if verbose:
                print("\nNo suitable candidates found after filtering.")
            return None
        if len(candidates) == 1:
            if verbose:
                print("\nOnly one suitable candidate found.")
        # Step 2: Generate a linking prompt for the LLM and fire it
        choice, _raw, error = self._select_with_llm(
            search_term, context_query, entity_type, candidates, verbose)
        if error or choice is None:
            return None
        # The LLM should respond with "0" if no entity is suitable
        if choice == 0:
            if verbose:
                print("\nLLM indicated no suitable match.")
            return None

        # Convert 1-based choice to 0-based index and check bounds
        choice_index = choice - 1
        if 0 <= choice_index < len(candidates):
            chosen_entity = candidates[choice_index]
            self._cache[cache_key] = chosen_entity
            self._save_cache()
            return chosen_entity
        if verbose:  # out of bound index exceeding options
            print(f"\nLLM returned an out-of-bounds index: {choice_index}")
        return None

    def _select_with_llm(self,
                         search_term: str,
                         context_query: str,
                         entity_type: str,
                         candidates: list,
                         verbose: bool = False):
        """
        Ask the LLM to pick one candidate.

        Returns ``(choice, raw_response, error)`` where ``choice`` is the
        1-based index the model returned (0 meaning "none suitable"), and
        ``error`` is None, ``"parse"`` or ``"api"``. Shared by ``link`` and
        ``link_instrumented`` so the two cannot drift apart — they must issue
        an identical prompt for the evaluation's S2/S3 comparison to be sound.
        """
        prompt = self.generate_prompt(
            search_term, context_query, entity_type, candidates)
        # Temperature is pinned on the client rather than left to the provider
        # default: selection should be as close to deterministic as the API
        # allows, and an evaluation reporting variance over repeated runs has to
        # know what temperature that variance was measured at.
        if verbose:
            print("\nGenerated Prompt:\n", prompt)
            print("\nInvoking LLM for final selection...")
        response = None
        for attempt in range(max(1, self.llm_retries)):
            try:
                response = self.generator.generate(prompt, SCHEMA_INT)
                return response.parsed, response.text, None
            except (ValueError, IndexError):
                # A malformed body is deterministic — retrying will not help.
                if verbose:
                    print(f"\nCould not parse LLM response: '{response}'")
                return None, getattr(response, "text", None), "parse"
            except Exception as e:
                if attempt < self.llm_retries - 1:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                print(f"An error occurred during LLM generation: {e}")
                return None, None, "api"
        return None, None, "api"

    def link_instrumented(self,
                          search_term: str,
                          context_query: str = "",
                          entity_type: str = "",
                          candidates: list = None,
                          validity: dict = None,
                          entity_type_eck: dict = None,
                          top_k: int = 10,
                          validate: bool = True,
                          verbose: bool = False) -> dict:
        """
        Link a mention and report *why*, in one pass.

        ``link`` collapses "nothing retrieved", "everything filtered out", "the
        model abstained" and "the model erred" into a single ``None``, which
        makes error attribution impossible. This returns the distinguishing
        counts alongside the prediction.

        Both inputs can be injected so an evaluation replays frozen artefacts
        instead of re-querying:

        * ``candidates`` — a ranked candidate list; sliced to ``top_k``. When
          omitted, retrieval runs as usual.
        * ``validity`` — ``{wikidata_id: True | False | None}``. When omitted,
          the ECK is evaluated live. ``None`` means the check could not be
          run, and such candidates are **kept**, never silently dropped.

        ``outcome`` is one of ``linked``, ``abstain_no_candidates``,
        ``abstain_all_filtered``, ``abstain_llm``, ``error_out_of_bounds``,
        ``error_api``.
        """
        for ch in "‘’ʼ`´":
            search_term = search_term.replace(ch, "'")

        started = time.monotonic()
        record = {
            "predicted_qid": None, "predicted_label": None, "outcome": None,
            "n_retrieved": 0, "n_validated": 0, "n_unevaluable": 0,
            "llm_raw_response": None, "latency_ms": 0,
        }

        def done(outcome):
            record["outcome"] = outcome
            record["latency_ms"] = int((time.monotonic() - started) * 1000)
            return record

        pool = (self._search_candidates(search_term, top_k, verbose)
                if candidates is None else list(candidates))
        pool = (pool or [])[:top_k]
        record["n_retrieved"] = len(pool)
        if not pool:
            return done("abstain_no_candidates")

        if validate:
            eck = entity_type_eck
            if validity is None:
                if eck is None:
                    eck = get_eck(entity_type)
                if eck is None:
                    validity = {}
                else:
                    validity = validate_entities_batch(
                        [c["wikidata_id"] for c in pool], eck) or {}
            kept = [c for c in pool if validity.get(c["wikidata_id"]) is not False]
            record["n_unevaluable"] = sum(
                1 for c in pool if validity.get(c["wikidata_id"]) is None)
        else:
            kept = pool
        record["n_validated"] = len(kept)

        if not kept:
            return done("abstain_all_filtered")

        choice, raw, error = self._select_with_llm(
            search_term, context_query, entity_type, kept, verbose)
        record["llm_raw_response"] = raw
        if error == "api":
            return done("error_api")
        if choice is None:
            return done("error_api" if error else "abstain_llm")
        if choice == 0:
            return done("abstain_llm")
        index = choice - 1
        if not (0 <= index < len(kept)):
            return done("error_out_of_bounds")
        record["predicted_qid"] = kept[index]["wikidata_id"]
        record["predicted_label"] = kept[index]["label"]
        return done("linked")


# ##############################################################################
# Typed RAG Entity Linker (v2)
# ##############################################################################

class TypedRAGEntityLinker(RAGEntityLinker):
    """
    Entity linker that looks its own type check up from ``entity_type``.

    Differences from :class:`RAGEntityLinker`:

    * **No ECK plumbing.** Callers pass ``entity_type="music_artist"`` and the
      linker resolves the matching ECK through ``entichecks.get_eck``; the
      caller never has to import or pass ``entity_type_eck``. Common spellings
      ("artist", "music genre", "location", "album", ...) are accepted.
    * **Validation on by default**, because the checks it uses are the ones
      covered by ``test_linker.py``.
    * **One SPARQL request per mention** instead of one per candidate, which is
      what makes validation affordable against a rate-limited public WDQS. Falls
      back to per-candidate ASKs if the batch request fails.
    * **Query failures do not silently discard candidates.** A candidate whose
      check could not be run is kept (and flagged) rather than treated as the
      wrong type — the old behaviour turned a WDQS outage into "nothing links".

    Unknown entity types fall back to no validation with a warning, so a new
    field in the extractor cannot take the pipeline down. Pass
    ``strict_types=True`` to raise instead.

    Usage::

        linker = TypedRAGEntityLinker(generator_api_key=API_KEY)
        linker.link(
            search_term="Bill Evans",
            context_query="Bill Evans' saxophone",
            entity_type="music_artist",   # resolves MUSIC_ARTIST_ECK itself
        )

    ``entity_type`` accepts the pipeline's field names (``music_artist``,
    ``music_work``, ``music_album``, ``genre``, ``event``, ``place``,
    ``country``, ``venue``, ``colour``, ``material``) and common variants such
    as "Music Artist", "artist", "location", "album" or "color".
    """

    def __init__(self,
                 generator_model: str = 'gemini-2.5-flash',
                 generator_api_key: str = '',
                 cache_path: Path | None = None,
                 strict_types: bool = False,
                 batch_validate: bool = True,
                 temperature: float = 0.0,
                 llm_timeout_s: float = 90.0,
                 llm_retries: int = 3,
                 generator: object = None):
        super().__init__(generator_model=generator_model,
                         generator_api_key=generator_api_key,
                         cache_path=cache_path,
                         temperature=temperature,
                         llm_timeout_s=llm_timeout_s,
                         llm_retries=llm_retries,
                         generator=generator)
        self.strict_types = strict_types
        self.batch_validate = batch_validate
        self._unknown_types_seen = set()

    def resolve_eck(self, entity_type: str) -> dict | None:
        """Return the ECK for ``entity_type``, or None if there is no check."""
        eck = get_eck(entity_type)
        if eck is None and entity_type:
            if self.strict_types:
                raise ValueError(
                    f"No entity check defined for type '{entity_type}'. "
                    f"Add it to entichecks.WD_TYPE_SPECS or pass "
                    f"strict_types=False to skip validation."
                )
            if entity_type not in self._unknown_types_seen:
                self._unknown_types_seen.add(entity_type)
                print(f"Warning: no entity check for type '{entity_type}' — "
                      f"candidates will not be type-validated.")
        return eck

    def _validate_candidates(self,
                             candidates: list,
                             entity_type_eck: dict,
                             verbose: bool = False) -> list:
        """
        Filter candidates by type, batching the checks into one SPARQL request.

        Candidates whose check could not be evaluated are kept, so a transient
        WDQS failure costs precision rather than dropping the link entirely.
        """
        if not candidates or not entity_type_eck:
            return candidates

        ids = [c["wikidata_id"] for c in candidates]
        verdicts = None
        if self.batch_validate:
            verdicts = validate_entities_batch(ids, entity_type_eck)
            if verdicts is None and verbose:
                print("  Batch validation failed; falling back to per-candidate ASKs.")
        if verdicts is None:
            verdicts = {
                qid: validate_entity(qid, entity_type_eck, source="wd")
                for qid in ids
            }

        kept = []
        for i, candidate in enumerate(candidates):
            is_valid = verdicts.get(candidate["wikidata_id"])
            if is_valid is None:  # query failed — keep rather than guess
                status = "Unknown (check failed, kept)"
                kept.append(candidate)
            elif is_valid:
                status = "Valid"
                kept.append(candidate)
            else:
                status = "Invalid"
            if verbose:
                print(f"  {i+1}. {candidate['label']} "
                      f"({candidate['wikidata_id']}) -> {status}")
        return kept

    def link(self,
             search_term: str,
             context_query: str = "",
             entity_type: str = "",
             entity_type_eck: dict = None,
             top_k: int = 10,
             validate: bool = True,
             verbose: bool = True) -> dict | None:
        """
        Link ``search_term`` to a Wikidata entity of type ``entity_type``.

        ``entity_type_eck`` is optional and only needed to override the check
        resolved from ``entity_type``; the signature is kept compatible with
        :meth:`RAGEntityLinker.link` so this class is a drop-in replacement.
        """
        if validate and entity_type_eck is None:
            entity_type_eck = self.resolve_eck(entity_type)
            validate = entity_type_eck is not None

        return super().link(
            search_term=search_term,
            context_query=context_query,
            entity_type=entity_type,
            entity_type_eck=entity_type_eck,
            top_k=top_k,
            validate=validate,
            verbose=verbose,
        )
