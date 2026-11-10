"""
Local-only dev runner: serves the real `lechoes` FastAPI app against the released
`knowledge_graph.ttl` plus `core.ttl`'s curated Exhibition/Era nodes, both loaded straight
into memory with rdflib, instead of a real GraphDB. Both are read from a checkout of
`onradio/bme-kg`: `BME_KG_DIR` if set, else a sibling `bme-kg/` next to this repository
(the same resolution `conftest.py` uses).

Not part of the deployed app or the test suite — just a fast way to click around the
dashboard against real data without bringing up GraphDB. Reuses the same rdflib-backed
SPARQL-endpoint stand-in the test suite already trusts (`conftest.RDFLibEndpoint`).

Usage: python run_local.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
KG_BUNDLE_DIR = Path(os.environ.get("BME_KG_DIR", REPO_ROOT.parent / "bme-kg")).expanduser()
KG_BUNDLE = KG_BUNDLE_DIR / "knowledge_graph.ttl"
CORE_TTL = KG_BUNDLE_DIR / "core.ttl"

sys.path.insert(0, str(REPO_ROOT))

from conftest import RDFLibEndpoint  # noqa: E402

import lechoes  # noqa: E402
import lechoes.src.queries.categories as categories_module  # noqa: E402
import lechoes.src.queries.entities as entities_module  # noqa: E402
import lechoes.src.queries.exhibitions as exhibitions_module  # noqa: E402
import lechoes.src.queries.items as items_module  # noqa: E402

from rdflib import Graph  # noqa: E402

graph = Graph()
graph.parse(str(KG_BUNDLE), format="turtle")
graph.parse(str(CORE_TTL), format="turtle")
endpoint = RDFLibEndpoint(graph)

lechoes.sparql_query = endpoint
lechoes.sparql_update = endpoint
for module in (items_module, categories_module, exhibitions_module, entities_module):
    if hasattr(module, "sparql_query"):
        module.sparql_query = endpoint
    if hasattr(module, "sparql_update"):
        module.sparql_update = endpoint


# `schema.py` opens "./mappings.json" by a relative path at import time (only reached via
# `lechoes.main`), matching the working directory the deployed API image runs from.
os.chdir(REPO_ROOT / "lechoes")

from lechoes.main import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    print(f"Loaded {len(graph)} triples from {KG_BUNDLE} + {CORE_TTL}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
