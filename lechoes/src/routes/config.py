from lechoes.src.schema import DeleteItem, NewItem, UpdateItem
from fastapi import APIRouter, status

config_router = APIRouter(prefix="/config", dependencies=[])


@config_router.post("/new-item", status_code=status.HTTP_201_CREATED)
async def add_item(
    new_item: NewItem,
):
    """
    Args:
        new_item (NewItem): Items description (refer schema.py)
        session (SessionDep): The database session dependency.

    Returns:
        dict: Confirming the addition of the Item

    Raises:
        HTTPException: If the new insert fails.
    """
    new_item.insert()
    return {"detail": "Inserted Item!"}


@config_router.delete("/delete-item", status_code=status.HTTP_200_OK)
async def delete_item(
    delete_item: DeleteItem,
):
    """
    Args:
        new_item (NewItem): Items description (refer schema.py)
        session (SessionDep): The database session dependency.

    Returns:
        dict: Confirming the addition of the Item

    Raises:
        HTTPException: If the new insert fails.
    """
    delete_item.delete()
    return {"detail": "Deleted Item!"}


@config_router.patch("/update-item", status_code=status.HTTP_200_OK)
async def update_item(
    name: str,
    update_item: UpdateItem,
):
    print(name, update_item)
    return {"detail": "Updated Item!"}