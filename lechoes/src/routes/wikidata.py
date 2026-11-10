import requests
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any

wikidata_router = APIRouter(prefix="/wikidata", dependencies=[])


@wikidata_router.get("/search", status_code=status.HTTP_200_OK)
async def search_entities(q: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search Wikidata entities and return results with custom labels.
    
    Args:
        q (str): Search query string
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        Dict containing search results with custom_label field in format "label-id-description"
    
    Raises:
        HTTPException: If the request to Wikidata fails
    """
    try:
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": q,
                "language": "en",
                "format": "json",
                "type": "item",
                "limit": limit,
                "origin": "*"
            },
            headers={
                "User-Agent": "bme-kg/1.0"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Transform the results to include custom_label
        if "search" in data:
            for item in data["search"]:
                label = item.get("label", "")
                item_id = item.get("id", "")
                description = item.get("description", "")

                # Create custom label in format "label | description"
                item["custom_label"] = f"{label} | {description}"
        
        return data
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch data from Wikidata: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}"
        )


@wikidata_router.get("/labels", status_code=status.HTTP_200_OK)
async def get_entity_label(q_number: str) -> str:
    """
    Get the English label for a specific Wikidata entity.
    
    Args:
        q_number (str): Wikidata Q number (e.g., "Q8341")
    
    Returns:
        str: English label of the entity
    
    Raises:
        HTTPException: If the Q number is invalid or request fails
    """
    try:
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": q_number,
                "props": "labels",
                "format": "json",
            },
            headers={
                "User-Agent": "bme-kg/1.0"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return data["entities"][q_number]["labels"]["en"]["value"]
        
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid Q Number!")
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch data from Wikidata: {str(e)}"
        )
