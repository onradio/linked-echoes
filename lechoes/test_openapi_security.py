"""The lock in Swagger must match what the API actually enforces.

    pytest lechoes/test_openapi_security.py

`verify_api_key` is applied app-wide so a new route is protected by default, but it
exempts GET and OPTIONS at runtime. That branch is invisible to the schema generator, so
every endpoint — including the public reads — was marked as requiring a key. The API is
published as open linked data, and a lock on a read endpoint tells an external consumer
they need a credential they have no way to obtain.

Both halves are asserted here. Testing the schema alone would pass if the dependency were
removed outright, which would fix the docs by unprotecting the writes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

WRITE_METHODS = ("post", "put", "patch", "delete")


@pytest.fixture(scope="module")
def app():
    """`src/schema.py` reads "./mappings.json" at import time, so importing the app at all
    depends on the working directory. Import it from the one directory that works."""
    cwd = Path.cwd()
    os.chdir(Path(__file__).parent)
    try:
        from lechoes.main import app as fastapi_app

        return fastapi_app
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def schema(app):
    app.openapi_schema = None
    return app.openapi()


def test_reads_carry_no_key_requirement(schema):
    locked = [
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method in ("get", "options")
        if method in operations and operations[method].get("security")
    ]
    assert not locked, f"public reads marked as requiring a key: {locked}"


def test_writes_still_carry_the_key_requirement(schema):
    unlocked = [
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method in WRITE_METHODS
        if method in operations and not operations[method].get("security")
    ]
    assert not unlocked, f"writes not marked as requiring a key: {unlocked}"


def test_the_dependency_is_still_applied_app_wide(app):
    """The schema fix must not become a reason to move enforcement onto each route."""
    names = [
        getattr(dependency.dependency, "__name__", "")
        for dependency in app.router.dependencies
    ]
    assert "verify_api_key" in names
