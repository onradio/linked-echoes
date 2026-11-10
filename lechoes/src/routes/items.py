import typing
from typing import Annotated
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import ValidationError

from lechoes import SessionDep, logger, settings
from lechoes.src.queries.items import (
    delete,
    get_items_by_category,
    get_search_triples,
    insert,
    update,
    get_item_by_identifier,
    search_items,
    get_items_by_zone,
)
from lechoes.src.queries.exhibitions import get_exhibition_by_zone

from lechoes.src.metamodel import Item
# from lechoes.src.uri_factory import create_uri

items_router = APIRouter(prefix="/items", dependencies=[])


# Load mappings once at module level
mappings_path = Path(__file__).parent.parent.parent / "mappings_min.json"
with open(mappings_path, 'r') as f:
    mappings = json.load(f)

# Build category reverse mapping from the mappings structure
category_reverse_mapping = {}
category_config = mappings.get('category', {})
for cat_name, cat_data in category_config.items():
    if cat_name.startswith('_'):  # Skip metadata fields like _relation_uri
        continue
    # Get the type URIs for this category and use the description as the label
    type_uris = cat_data.get('_type_uri', [])
    description = cat_data.get('_description', cat_name)  # fallback to cat_name if no description
    for uri in type_uris:
        category_reverse_mapping[uri] = description

logger.info(f"Loaded category reverse mapping: {category_reverse_mapping}")

# Create reverse mapping from relation URIs to field names
predicate_to_field_mapping = {}
for field_name, field_config in mappings.items():
    if field_name.startswith('_'):
        continue
    relation_uris = field_config.get('_relation_uri', [])
    for uri in relation_uris:
        predicate = uri.split('/')[-1] if '/' in uri else uri.split('#')[-1]
        if predicate not in predicate_to_field_mapping:
            predicate_to_field_mapping[predicate] = []
        predicate_to_field_mapping[predicate].append(field_name)

# Special field mappings that don't follow the standard pattern
special_field_mappings = {
    "itemLocation": "case_number",
    "startTime": "lend_start", 
    "endTime": "lend_end",
    "lender": "lender",
    "isPartOf": "exhibition"
}


@items_router.post("/", status_code=status.HTTP_201_CREATED)
async def add_item(
    item: Item,
    session: SessionDep,
):
    """
    NOTE: This requires /statements to be added to the end of the endpoint.
    Adds an item to the KG only supports authenticated requests.

    Args:
        item (Item): Items description (refer schema.py)
        session (SessionDep): The database session dependency.

    Returns:
        dict: Confirming the addition of the Item

    Raises:
        HTTPException: If the insert fails.

    """
    logger.info("Inserted item")
    
    try:
        insert(item, session)
        return {"detail": "Inserted Item!"}
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insert failed: {e}")


@items_router.delete("/item/{identifier}", status_code=status.HTTP_200_OK)
async def delete_items_route(
    identifier: str,
    session: SessionDep,
):
    delete(identifier, session)
    return {"detail": "Deleted item"}


@items_router.patch("/item/{identifier}", status_code=status.HTTP_200_OK)
async def update_item_route(
    identifier: str,
    item: Item,
    session: SessionDep,
):
    """
    Update an existing item by its identifier.
    
    This endpoint performs a complete replacement of the item (delete + insert).
    
    Args:
        identifier (str): The identifier of the item to update.
        item (Item): The updated item data.
        session (SessionDep): The database session dependency.
        
    Returns:
        dict: Confirmation message with the updated item details.
        
    Raises:
        HTTPException: If the update fails or the item is not found.
    """
    try:
        logger.info(f"Updating item with identifier '{identifier}'")
        
        # Convert Item to dict for the update function
        item_dict = item.model_dump(exclude_unset=True)
        
        # Call the update function
        update(identifier, item_dict, session)
        
        return {
            "detail": "Updated item successfully",
            "identifier": identifier,
            "updated_by": "API"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValidationError as e:
        logger.error(f"Validation error updating item '{identifier}': {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except Exception as e:
        import traceback
        logger.error(f"Failed to update item '{identifier}': {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update item: {e}")


@items_router.get("/search", status_code=status.HTTP_200_OK)
async def search_items_route(
    q: Annotated[
        str, Query(description="Search term for item names, descriptions, etc.")
    ]
):
    """
    Search for items across multiple fields using the `SEARCH_OBJECTS_QUERY`.

    This endpoint searches the search term in:
    - Item names
    - Item descriptions
    - Item captions
    - Labels of linked entities (e.g., artists, bands)

    Args:
        q: The search term (case-insensitive).

    Returns:
        A list of complete `Item` objects matching the search criteria.
    """
    try:
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="Search term cannot be empty")

        return search_items(q.strip())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for term '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@items_router.get("/category/{category}", status_code=status.HTTP_200_OK)
async def get_category_items_route(
    category: str,
    limit: int = 100,
):
    """
    Get all the items belonging to a category.

    Args:
        category (str): Selected category.
        limit (int): Max number of items to return.

    Returns:
        dict: Sparql's JSON response.
    """
    try:
        logger.info(f"Fetching items in category: {category} with limit {limit}")   
        return get_items_by_category(category)
    except Exception as e:
        logger.error(f"Error fetching items in category '{category}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch items for category '{category}': {e}")


@items_router.get("/zone/{zone}", status_code=status.HTTP_200_OK)
async def get_zone_items_route(
    zone: str,
):
    """
    Get all items belonging to a specific exhibition zone.

    Args:
        zone (str): The zone identifier (e.g., zone URI or zone number).

    Returns:
        List[Item]: List of Item objects belonging to the specified zone.
    """
    try:
        logger.info(f"Fetching items in zone: {zone}")
        return get_items_by_zone(zone)
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error fetching items in zone '{zone}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch items for zone '{zone}': {e}")


@items_router.get("/item/{identifier}", status_code=status.HTTP_200_OK)
async def get_item_triple_route(identifier: str):
    """
    Get an item by its identifier using the comprehensive RETRIEVE_OBJECT_QUERY.
    This provides a complete, structured Item response with all related data.

    Args:
        hash (str): Item's identifier.

    Returns:
        Item: Complete item object with all fields populated.
    """
    try:
        logger.info(f"Fetching item with identifier: {identifier}")
        return get_item_by_identifier(identifier)
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        import traceback
        logger.error(f"Failed to fetch item with identifier '{identifier}': {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch item: {e}")
