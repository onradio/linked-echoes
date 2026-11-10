from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from lechoes import create_db_and_tables, load_mappings
from lechoes.src.auth_utils import verify_api_key
from lechoes.src.routes.config import config_router
from lechoes.src.routes.items import items_router
from lechoes.src.routes.wikidata import wikidata_router
from lechoes.src.routes.exhibitions import exhibitions_router
from lechoes.src.routes.categories import categories_router
from lechoes.src.routes.entities import entities_router

app = FastAPI(root_path="/bme-api", dependencies=[Depends(verify_api_key)])

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(items_router)
app.include_router(config_router)
app.include_router(wikidata_router)
app.include_router(exhibitions_router)
app.include_router(categories_router)
app.include_router(entities_router)


PUBLIC_METHODS = ("get", "options")


def custom_openapi():
    """Drop the key requirement from GET and OPTIONS in the schema.

    `verify_api_key` is an app-wide dependency so that a new route is protected by
    default, but it exempts reads at runtime — a branch the schema generator cannot see.
    Without this, Swagger draws a lock on every public endpoint and tells an external
    consumer they need a credential to read open data.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    for path in schema.get("paths", {}).values():
        for method in PUBLIC_METHODS:
            if method in path:
                path[method].pop("security", None)

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


app.add_event_handler("startup", create_db_and_tables)
app.add_event_handler("startup", load_mappings)


@app.exception_handler(ValidationError)
async def bad_request_exception_handler(_request: Request, _exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Oops! that is a bad request!"},
    )


@app.get("/")
async def root():
    return {"message": "Hello World"}
