import logging
from typing import Annotated

from fastapi import Depends
from SPARQLWrapper import JSON, SPARQLWrapper
from sqlmodel import Session, SQLModel, create_engine

from lechoes.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


logger = logging.getLogger("lechoes")

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def load_mappings():
    from lechoes.src import init_descriptors

    # inject mappings into the descriptor classes
    init_descriptors("./mappings.json")


def create_db_and_tables():
    from lechoes.src.models import ItemIdentifiers  # pylint: disable=unused-import

    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

sparql_query = SPARQLWrapper(settings.db_query_uri)
sparql_update = SPARQLWrapper(settings.db_uri)

sparql_update.setHTTPAuth("BASIC")
sparql_query.setHTTPAuth("BASIC")
sparql_update.setCredentials(settings.db_user, settings.db_password)
sparql_query.setCredentials(settings.db_user, settings.db_password)


sparql_update.setReturnFormat(JSON)
sparql_query.setReturnFormat(JSON)
