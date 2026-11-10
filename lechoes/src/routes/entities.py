from fastapi import APIRouter, HTTPException, status

from lechoes import logger
from lechoes.src.queries.entities import get_entity_detail

entities_router = APIRouter(prefix="/entities", dependencies=[])


@entities_router.get("/{entity_type}/{entity_id}", status_code=status.HTTP_200_OK)
async def get_entity_detail_route(entity_type: str, entity_id: str):
    """
    Get a minted entity's label and the items that reference it.

    Backs the dashboard's `/entity/:type/:id` page — the landing spot for links that used
    to point straight at the KG's own node URI (`bme:Zone/5`, `bme:Person/...`) and 404'd
    because that path is also where the dashboard SPA is served from. See
    `queries/entities.py` for why every minted entity's detail is just a label plus its
    referencing items.
    """
    try:
        logger.info(f"Fetching entity detail: {entity_type}/{entity_id}")
        return get_entity_detail(entity_type, entity_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching entity '{entity_type}/{entity_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch entity '{entity_type}/{entity_id}': {e}")
