import os
import ast
import json
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from lechoes.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def parse_keys_list(raw_val: str) -> list[str]:
    raw_val = raw_val.strip()
    if not raw_val:
        return []
    
    # Handle list brackets
    if raw_val.startswith("[") and raw_val.endswith("]"):
        # Try JSON parser
        try:
            parsed = json.loads(raw_val)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except Exception:
            pass
            
        # Try Python literal parser
        try:
            parsed = ast.literal_eval(raw_val)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except Exception:
            pass
            
        # Fallback to simple comma-split inside brackets (e.g. [key1, key2])
        inner = raw_val[1:-1]
        return [k.strip().strip("'\"") for k in inner.split(",") if k.strip()]
        
    # Standard comma-separated string fallback
    return [k.strip().strip("'\"") for k in raw_val.split(",") if k.strip()]

async def verify_api_key(request: Request, api_key: str | None = Depends(api_key_header)):
    # Allow GET and OPTIONS methods without API Key
    if request.method in ("GET", "OPTIONS"):
        return None

    # Check Authorization header as fallback (e.g. Bearer token)
    if not api_key:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing",
        )

    # Collect all configured valid keys
    valid_keys = set()

    # Load api_keys setting from configuration or environment
    api_keys_raw = (
        getattr(settings, "api_keys", None)
        or os.environ.get("API_KEYS")
        or os.environ.get("api_keys")
    )
    
    if api_keys_raw:
        parsed_keys = parse_keys_list(str(api_keys_raw))
        valid_keys.update(parsed_keys)

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    
    return api_key
