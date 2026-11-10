import re

from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any

from lechoes.src.queries.exhibitions import get_exhibitions, get_exhibition_by_zone

exhibitions_router = APIRouter(prefix="/exhibitions", dependencies=[])


async def _get_exhibitions_data(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Shared function to get exhibitions data from the knowledge graph.
    
    Args:
        limit (int): Maximum number of exhibitions to return (default: 100)
    
    Returns:
        List[Dict[str, Any]]: List of exhibition objects with metadata
    """
    try:
        results = get_exhibitions(limit)["results"]["bindings"]
        
        # Group results by exhibition URI
        exhibitions_data = {}
        
        for result in results:
            exhibition_uri = result["exhibition"]["value"]
            # Split on both separators: rdfs:label ends `.../rdf-schema#label`, so
            # splitting on "/" alone yields "rdf-schema#label" and matches no field.
            pred = re.split(r"[/#]", result["pred"]["value"])[-1]
            obj = result["obj"]["value"]
            
            if exhibition_uri not in exhibitions_data:
                exhibitions_data[exhibition_uri] = {
                    "uri": exhibition_uri,
                    "name": "",
                    "start_date": "",
                    "end_date": "",
                    "description": "",
                    "image": ""
                }
            
            # Map predicates to exhibition fields. Real ExhibitionEvent nodes carry both
            # scho:name ("Punk & New Romantics") and rdfs:label ("Exhibition: Punk & New
            # Romantics") — name wins since it is the display value; label is only a
            # fallback for whichever node type has nothing else (the old zone stand-in).
            if pred == "name":
                exhibitions_data[exhibition_uri]["name"] = obj
            elif pred == "label" and not exhibitions_data[exhibition_uri]["name"]:
                exhibitions_data[exhibition_uri]["name"] = obj
            elif pred == "startDate":
                exhibitions_data[exhibition_uri]["start_date"] = obj
            elif pred == "endDate":
                exhibitions_data[exhibition_uri]["end_date"] = obj
            elif pred == "description":
                exhibitions_data[exhibition_uri]["description"] = obj
            elif pred == "image":
                exhibitions_data[exhibition_uri]["image"] = obj
        
        return list(exhibitions_data.values())
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch exhibitions: {e}"
        )


@exhibitions_router.get("/", status_code=status.HTTP_200_OK)
async def list_exhibitions(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all exhibitions from the knowledge graph.

    Args:
        limit (int): Maximum number of exhibitions to return (default: 100)

    Returns:
        List[Dict[str, Any]]: List of exhibition objects with metadata
    """
    return await _get_exhibitions_data(limit)


@exhibitions_router.post("/", status_code=status.HTTP_200_OK)
async def list_exhibitions_post(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all exhibitions from the knowledge graph (POST endpoint).

    Args:
        limit (int): Maximum number of exhibitions to return (default: 100)

    Returns:
        List[Dict[str, Any]]: List of exhibition objects with metadata
    """
    return await _get_exhibitions_data(limit)