import json
import os
import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

in_dev = "dev" in sys.argv


@lru_cache
def load_scopes() -> dict[str, list[str]]:
    with open(os.path.join(os.path.dirname(__file__), "scopes.json")) as f:
        return json.loads(f.read())


class Settings(BaseSettings):
    app_name: str = "BME API"
    base_uri: str = "http://radio.liv.ac.uk/"
    min_password_len: int = 6
    uri_strategy: str = "short_hash"
    scopes: dict = load_scopes()
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"), extra="allow"
    )


settings = Settings()
settings.db_uri = settings.db_uri_dev if in_dev else settings.db_uri
settings.db_query_uri = settings.db_uri.replace("/statements", "")
