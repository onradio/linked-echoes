import typing

from sqlmodel import select

from lechoes.src.models import ItemIdentifiers

if typing.TYPE_CHECKING:
    from lechoes import SessionDep


def to_camel_case(snake_str):
    """
    Converts a snake_case string to CamelCase.
    """
    snake_str = snake_str.replace(" ", "_")
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_snake_case(camel_str, sep="_"):
    """
    Converts a CamelCase string to snake_case.
    """
    return "".join([sep + i.lower() if i.isupper() else i for i in camel_str]).lstrip(
        sep
    )


def hash_exists(hash: str, session: "SessionDep") -> bool:
    return bool(
        session.exec(
            select(ItemIdentifiers).where(ItemIdentifiers.hash == hash)
        ).first()
    )
