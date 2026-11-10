from fastapi import APIRouter, HTTPException, status

from lechoes import logger
from lechoes.src.queries.categories import get_categories

categories_router = APIRouter(prefix="/categories", dependencies=[])


@categories_router.get("/", status_code=status.HTTP_200_OK)
async def get_categories_list():
    """
    Fetch categories from the knowledge graph with their Wikidata Q numbers.
    Categories are defined as scho:DefinedTerm and skos:Concept with rdfs:label.
    """
    try:
        results = get_categories()["results"]["bindings"]
        categories = []
        
        for result in results:
            category_uri = result["category"]["value"]
            category_label = result["categoryLabel"]["value"]
            categories.append({
                "uri": category_uri,
                "label": category_label
            })
        
        return categories
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch categories: {e}"
        )


@categories_router.post("/", status_code=status.HTTP_200_OK)
async def list_categories():
    """
    Fetch categories from the knowledge graph with their Wikidata Q numbers.
    Categories are defined as scho:DefinedTerm and skos:Concept with rdfs:label.
    """
    try:
        results = get_categories()["results"]["bindings"]
        categories = []
        
        for result in results:
            category_uri = result["category"]["value"]
            category_label = result["categoryLabel"]["value"]
            categories.append({
                "uri": category_uri,
                "label": category_label
            })
        
        return categories
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch categories: {e}"
        )
