from sqlmodel import Field, SQLModel


class ItemIdentifiers(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    case_number: int
    hash: str = Field(max_length=8)
