"""Test-session setup for the API suite (`lechoes/test_api.py`).

Two problems have to be solved before `import lechoes` can succeed in an analysis
environment, and both are solved here rather than in the tests:

1. **`lechoes/__init__.py` pulls in the serving stack.** `SPARQLWrapper` and `sqlmodel` are
   absent from the `lechoes` env — that absence is a documented property of it, and it is
   why the pipeline stubs `lechoes` out of `sys.modules` and the evaluation packs import
   bare. The same trick is applied here: a minimal stand-in is installed in `sys.modules`
   for whichever of the two is missing, so the API's own modules import unmodified. If the
   real package *is* installed, it is used and nothing is stubbed.

2. **The queries need a store to run against.** Rather than mock the SPARQL layer away —
   which would test the mocks and not the queries — the endpoints are pointed at the
   released BME KG (`knowledge_graph.ttl` + `core.ttl` from a checkout of
   `onradio/bme-kg`, see `BME_KG_DIR` below) loaded into an in-memory rdflib graph. The templates in `sparql_queries.py` are executed verbatim, so a query
   that is malformed, or that asks for a predicate the KG no longer has, fails the test.

This file lives at the repository root deliberately. pytest loads the rootdir conftest
before it collects anything, which is the only point at which the stubs can be installed
ahead of `lechoes/__init__.py` being imported as part of collecting the `lechoes` package.
"""

from __future__ import annotations

import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = REPO_ROOT / "lechoes"
# The graph is the BME instance's published data, so it is read from a bme-kg checkout
# rather than vendored here: BME_KG_DIR if set, else a sibling `bme-kg/` next to this
# repository. Without one the KG-backed tests skip. Every count asserted in
# `test_api.py` was confirmed against bme-kg's kg-v1.1.0 release.
KG_BUNDLE_DIR = Path(os.environ.get("BME_KG_DIR", REPO_ROOT.parent / "bme-kg")).expanduser()
KG_BUNDLE = KG_BUNDLE_DIR / "knowledge_graph.ttl"
CORE_TTL = KG_BUNDLE_DIR / "core.ttl"

sys.path.insert(0, str(REPO_ROOT))

# Note: the working directory is deliberately left alone. `lechoes/src/schema.py` opens
# "./mappings.json" by a relative path at import time, but it is reached only through
# `lechoes.main`, which these tests do not import — every other module resolves its paths
# from `__file__`. Changing directory here would break the sibling suites that a run from
# the repository root also picks up.


def _install_config_stub() -> None:
    """Stand in for `lechoes.config` when the deployment's `.env` is not present.

    `Settings` takes `db_uri`, `db_user`, `db_password` and `api_keys` from `lechoes/.env`,
    which is gitignored and absent from a checkout. They are declared nowhere in the
    class, so pydantic-settings will not pick them up from the environment either — the
    module raises `AttributeError` on import without that file.

    Seeding `sys.modules` is used rather than writing a `.env`: that path holds real
    deployment credentials on a machine that has one, and a test run must not create or
    overwrite it. Where a `.env` *does* exist the real module is imported untouched; the
    endpoints are redirected at the rdflib graph either way, so no connection is made to
    whatever `db_uri` names.
    """
    import json
    from types import SimpleNamespace

    module = ModuleType("lechoes.config")
    with open(PACKAGE_DIR / "scopes.json") as handle:
        scopes = json.load(handle)

    module.settings = SimpleNamespace(
        app_name="BME API",
        base_uri="http://radio.liv.ac.uk/",
        min_password_len=6,
        uri_strategy="short_hash",
        scopes=scopes,
        db_uri="http://localhost:7200/repositories/bme/statements",
        db_query_uri="http://localhost:7200/repositories/bme",
        db_user="test-user",
        db_password="test-password",
        api_keys='["test-api-key"]',
    )
    module.load_scopes = lambda: scopes
    module.in_dev = False
    sys.modules["lechoes.config"] = module


if not (PACKAGE_DIR / ".env").exists():
    _install_config_stub()


# --------------------------------------------------------------------------------------
# Stand-ins for the serving dependencies, installed only when genuinely absent
# --------------------------------------------------------------------------------------

def _install_sparqlwrapper_stub() -> None:
    module = ModuleType("SPARQLWrapper")

    class SPARQLWrapper:  # noqa: N801 - mirrors the real class name
        def __init__(self, endpoint: str = "", *_args, **_kwargs):
            self.endpoint = endpoint
            self.queryString = ""

        def setQuery(self, query):  # noqa: N802
            self.queryString = query

        def setReturnFormat(self, *_a, **_k):  # noqa: N802
            pass

        def setHTTPAuth(self, *_a, **_k):  # noqa: N802
            pass

        def setCredentials(self, *_a, **_k):  # noqa: N802
            pass

        def setRequestMethod(self, *_a, **_k):  # noqa: N802
            pass

        def query(self):
            raise RuntimeError("No SPARQL endpoint bound; use the `kg` fixture.")

        def queryAndConvert(self):  # noqa: N802
            raise RuntimeError("No SPARQL endpoint bound; use the `kg` fixture.")

    module.SPARQLWrapper = SPARQLWrapper
    module.JSON = "json"
    module.POST = "POST"
    module.GET = "GET"
    sys.modules["SPARQLWrapper"] = module


def _install_sqlmodel_stub() -> None:
    """A stand-in covering only what the API's import chain touches.

    `models.ItemIdentifiers` needs a declarative base that tolerates `table=True`;
    `utils.hash_exists` needs `select(...).where(...)`; `queries.items.delete` needs
    `delete(...).where(...)`. Nothing here talks to a database — the tests that exercise
    the SQLite side pass a recording session object instead (see `FakeSession`).
    """
    module = ModuleType("sqlmodel")

    class _Statement:
        def __init__(self, target=None):
            self.target = target
            self.clauses = []

        def where(self, *clauses):
            self.clauses.extend(clauses)
            return self

    class _Metadata:
        def create_all(self, *_a, **_k):
            pass

    class SQLModel:  # noqa: N801
        metadata = _Metadata()

        def __init_subclass__(cls, **_kwargs):
            # Swallows `table=True`.
            super().__init_subclass__()

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Session:  # noqa: N801
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def Field(*_a, **kwargs):  # noqa: N802
        return kwargs.get("default")

    def create_engine(*_a, **_k):
        return object()

    module.SQLModel = SQLModel
    module.Session = Session
    module.Field = Field
    module.create_engine = create_engine
    module.select = lambda target=None, *a, **k: _Statement(target)
    module.delete = lambda target=None, *a, **k: _Statement(target)
    sys.modules["sqlmodel"] = module


if find_spec("SPARQLWrapper") is None:
    _install_sparqlwrapper_stub()
if find_spec("sqlmodel") is None:
    _install_sqlmodel_stub()


# --------------------------------------------------------------------------------------
# An rdflib-backed stand-in for a SPARQL endpoint
# --------------------------------------------------------------------------------------

_UPDATE_FORMS = {"INSERT", "DELETE", "WITH", "LOAD", "CLEAR", "DROP", "CREATE", "ADD", "MOVE", "COPY"}
_COMMENT_RE = re.compile(r"#[^\n]*")
_PROLOGUE_RE = re.compile(r"(?im)^\s*(?:PREFIX\s+\S*\s*<[^>]*>|BASE\s*<[^>]*>)\s*")
_FIRST_WORD_RE = re.compile(r"[A-Za-z]+")


def is_update_query(query: str) -> bool:
    """Decide whether to run a query or an update, from the *first keyword*.

    Searching the whole string for INSERT/DELETE is not good enough: it misfires on any
    SELECT whose filter value happens to contain the word — a lookup for the identifier
    `test-insert-01` was routed to `graph.update()` and failed to parse. Only the form of
    the request itself decides, so comments and the PREFIX/BASE prologue are stripped and
    the first remaining keyword is what is read.
    """
    body = _PROLOGUE_RE.sub("", _COMMENT_RE.sub("", query))
    match = _FIRST_WORD_RE.search(body)
    return bool(match) and match.group(0).upper() in _UPDATE_FORMS


class RDFLibEndpoint:
    """Speaks the slice of the SPARQLWrapper protocol the API uses, over an rdflib Graph.

    Results are converted into the SPARQL 1.1 JSON results shape, including each
    binding's ``type`` — the API reads that to tell a URI-valued colour (the superseded
    KG) from a literal one (the current KG), so flattening it to plain strings would hide
    exactly the defect these tests exist to catch.
    """

    def __init__(self, graph):
        self.graph = graph
        self.queryString = ""
        self.executed: list[str] = []

    # -- SPARQLWrapper protocol ---------------------------------------------------------
    def setQuery(self, query):  # noqa: N802
        self.queryString = query

    def setReturnFormat(self, *_a, **_k):  # noqa: N802
        pass

    def setHTTPAuth(self, *_a, **_k):  # noqa: N802
        pass

    def setCredentials(self, *_a, **_k):  # noqa: N802
        pass

    def setRequestMethod(self, *_a, **_k):  # noqa: N802
        pass

    def query(self):
        return self._run()

    def queryAndConvert(self):  # noqa: N802
        return self._run()

    # -- execution ----------------------------------------------------------------------
    def _run(self):
        from rdflib.plugins.sparql.results.jsonresults import JSONResultSerializer  # noqa: F401

        self.executed.append(self.queryString)
        if is_update_query(self.queryString):
            self.graph.update(self.queryString)
            return _UpdateResult()
        return self._to_json(self.graph.query(self.queryString))

    @staticmethod
    def _to_json(result):
        from rdflib import BNode, Literal, URIRef

        variables = [str(v) for v in (result.vars or [])]
        bindings = []
        for row in result:
            binding = {}
            for var in variables:
                value = row[var]
                if value is None:
                    continue
                if isinstance(value, URIRef):
                    entry = {"type": "uri", "value": str(value)}
                elif isinstance(value, BNode):
                    entry = {"type": "bnode", "value": str(value)}
                else:
                    entry = {"type": "literal", "value": str(value)}
                    if isinstance(value, Literal):
                        if value.language:
                            entry["xml:lang"] = value.language
                        if value.datatype:
                            entry["datatype"] = str(value.datatype)
                binding[var] = entry
            bindings.append(binding)
        return {"head": {"vars": variables}, "results": {"bindings": bindings}}


class _UpdateResult:
    """SPARQLWrapper returns a response object whose `.convert()` the API calls."""

    def convert(self):
        return {}


class FakeSession:
    """Records what the SQLite side is asked to do, without a database.

    `hash_exists` is driven by `existing_hashes`, so both the happy path and the
    duplicate-insert rejection can be exercised.
    """

    def __init__(self, existing_hashes=()):
        self.existing_hashes = set(existing_hashes)
        self.added = []
        self.executed = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def exec(self, statement):
        self.executed.append(statement)
        return _FakeResult(self)


class _FakeResult:
    def __init__(self, session):
        self._session = session

    def first(self):
        return next(iter(self._session.existing_hashes), None) if self._session.existing_hashes else None


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def kg_source():
    """The released base layer plus core.ttl's curated Exhibition/Era nodes, parsed once."""
    from rdflib import Graph

    if not KG_BUNDLE.exists():  # pragma: no cover - guards a mis-checkout
        pytest.skip(f"BME KG not found at {KG_BUNDLE}; set BME_KG_DIR to a bme-kg checkout")
    graph = Graph()
    graph.parse(str(KG_BUNDLE), format="turtle")
    if CORE_TTL.exists():
        graph.parse(str(CORE_TTL), format="turtle")
    return graph


@pytest.fixture
def kg(kg_source, monkeypatch):
    """Bind the API's query and update endpoints to a private copy of the KG.

    A copy per test, so the tests that insert and delete cannot influence the ones that
    count items. The module-level `sparql_query` / `sparql_update` names are rebound in
    every module that imported them by value — patching `lechoes.sparql_query` alone would
    leave those stale bindings pointing at the real endpoint.
    """
    from rdflib import Graph

    graph = Graph()
    for triple in kg_source:
        graph.add(triple)

    endpoint = RDFLibEndpoint(graph)

    import lechoes
    import lechoes.src.queries.categories as categories_module
    import lechoes.src.queries.entities as entities_module
    import lechoes.src.queries.exhibitions as exhibitions_module
    import lechoes.src.queries.items as items_module

    monkeypatch.setattr(lechoes, "sparql_query", endpoint, raising=False)
    monkeypatch.setattr(lechoes, "sparql_update", endpoint, raising=False)
    for module in (items_module, categories_module, exhibitions_module, entities_module):
        if hasattr(module, "sparql_query"):
            monkeypatch.setattr(module, "sparql_query", endpoint)
        if hasattr(module, "sparql_update"):
            monkeypatch.setattr(module, "sparql_update", endpoint)

    return endpoint


@pytest.fixture
def session():
    return FakeSession()
