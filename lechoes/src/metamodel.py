import hashlib
from datetime import datetime
from typing import List, Optional, Tuple, Union, Any, Dict
from pydantic import BaseModel, model_validator, Field, field_validator


# This makes the structure explicit: {"uri": "...", "label": "..."}
class LinkedEntity(BaseModel):
    uri: Optional[str] = None
    label: str
    # How many items link to this entity, so a client can rank connections by weight.
    # Optional: literal-valued links have no entity to count, and the lookup may fail.
    count: Optional[int] = None

class Item(BaseModel):
    """
    Pydantic model for the API. This model includes:
    - Standard data fields for describing a museum object.
    - Custom getter methods to retrieve either full data or just display labels.
    - A 'finalisation' validator to ensure data integrity by checking for
      the presence of URIs when an object is considered complete.
    - A method to easily identify all labels that are missing URIs.
    - Validators to ensure consistent date formatting.

    """
    # Core identifying information
    identifier: str
    name: str

    # Categorization and description
    category: LinkedEntity
    type: Optional[LinkedEntity] = None
    description: Optional[str] = None
    caption: Optional[str] = None

    # Location and context
    case_number: Optional[str] = None
    case_name: Optional[str] = None
    item_number: Optional[str] = None
    location: Optional[List[LinkedEntity]] = None
    exhibition: Optional[List[LinkedEntity]] = None

    # Associated media
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None

    # Music-related fields (can have multiple values)
    genre: Optional[List[LinkedEntity]] = None
    music_work: Optional[List[LinkedEntity]] = None
    music_album: Optional[List[LinkedEntity]] = None
    music_artist: Optional[List[LinkedEntity]] = None

    # Event and location data (can have multiple values)
    event: Optional[List[LinkedEntity]] = None
    country: Optional[List[LinkedEntity]] = None
    place: Optional[List[LinkedEntity]] = None
    venue: Optional[List[LinkedEntity]] = None

    # Physical characteristics
    colour: Optional[List[LinkedEntity]] = None
    material: Optional[List[LinkedEntity]] = None

    # Temporal coverage (plain year strings, not linked entities)
    year: Optional[List[str]] = None

    # Lending information
    lender: Optional[str] = None
    lend_start: Optional[Union[datetime, str]] = None
    lend_end: Optional[Union[datetime, str]] = None
    
    @field_validator("lend_start", "lend_end", mode='before')
    @classmethod
    def format_dates(cls, value: Union[datetime, str, None]):
        """Formats date values into a consistent DD-MM-YYYY string."""
        if not value:
            return value

        if isinstance(value, datetime):
            return value.strftime("%d-%m-%Y")

        if isinstance(value, str):
            parts = value.split("-")
            # Check for YYYY-MM-DD format and convert it
            if len(parts) == 3 and len(parts[0]) == 4:
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
        
        return value

    def get_labels_without_uri(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary of fields and their corresponding labels that are missing a URI.
        """
        missing_uri_labels: Dict[str, List[str]] = {}
        fields_to_check = [
            ('category', self.category),
            ('type', self.type),
            ('location', self.location),
            ('exhibition', self.exhibition),
            ('genre', self.genre),
            ('music_work', self.music_work),
            ('music_album', self.music_album),
            ('music_artist', self.music_artist),
            ('event', self.event), ('country', self.country),
            ('place', self.place), ('venue', self.venue),
            ('colour', self.colour), ('material', self.material)
        ]

        for field_name, value in fields_to_check:
            if value is None:
                continue

            labels_without_uri = []
            entities = value if isinstance(value, list) else [value]
            
            # UPDATED: Use attribute access (.uri, .label) instead of tuple unpacking
            for entity in entities:
                if not entity.uri:
                    labels_without_uri.append(entity.label)
            
            if labels_without_uri:
                missing_uri_labels[field_name] = labels_without_uri
        
        return missing_uri_labels

    def is_final(self) -> Tuple[bool, List[str]]:
        """
        Checks if the instance is "final" by calling get_labels_without_uri.
        """
        missing_labels_dict = self.get_labels_without_uri()
        if not missing_labels_dict:
            return True, []
        error_messages = []
        for field_name, labels in missing_labels_dict.items():
            for label in labels:
                error_messages.append(f"{field_name} ('{label}')")
        
        return False, error_messages

    def _extract_labels(self, data: Optional[Union[LinkedEntity, List[LinkedEntity]]]) -> Optional[Union[str, List[str]]]:
        """Helper to extract labels from single or list of LinkedEntity objects."""
        if data is None:
            return None
        if isinstance(data, list):
            return [entity.label for entity in data]
        return data.label
    
    def generate_identifier(self) -> str:
        """Generates a unique identifier based on the content of the item."""
        if self.identifier and self.identifier.strip():
            raise ValueError("Identifier already set for this item.")
        item_str = self.model_dump_json()
        return hashlib.sha256(item_str.encode('utf-8')).hexdigest()[:10]

    # --- Getter Methods (No signature changes needed) ---
    def get_category(self, labels_only: bool = False) -> Optional[Union[str, LinkedEntity]]:
        return self._extract_labels(self.category) if labels_only else self.category
    
    def get_type(self, labels_only: bool = False) -> Optional[Union[str, LinkedEntity]]:
        return self._extract_labels(self.type) if labels_only else self.type

    def get_exhibition(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.exhibition) if labels_only else self.exhibition

    def get_genre(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.genre) if labels_only else self.genre

    def get_music_work(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.music_work) if labels_only else self.music_work

    def get_music_album(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.music_album) if labels_only else self.music_album

    def get_music_artist(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.music_artist) if labels_only else self.music_artist
    
    def get_event(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.event) if labels_only else self.event

    def get_country(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.country) if labels_only else self.country

    def get_place(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.place) if labels_only else self.place

    def get_venue(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.venue) if labels_only else self.venue

    def get_colour(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.colour) if labels_only else self.colour

    def get_material(self, labels_only: bool = False) -> Optional[Union[List[str], List[LinkedEntity]]]:
        return self._extract_labels(self.material) if labels_only else self.material

    # --- Standard Dunder Methods ---
    def __str__(self):
        return f"Item(name='{self.name}', identifier='{self.identifier}')"

    def __repr__(self):
        return self.__str__()



# --- Example Usage (UPDATED to use dictionaries) ---
if __name__ == '__main__':
    # 1. Example of an item that is NOT final (missing URIs)
    draft_item_data = {
        "identifier": "OBJ-001",
        "name": "Vintage Fender Stratocaster",
        "category": {"uri": "http://www.wikidata.org/entity/Q34379",
                     "label": "Musical Instrument"},
        "type": {"label": "Electric Guitar"}, # This now has a missing URI
        "genre": [
            {"uri": "http://www.wikidata.org/entity/Q8341",
             "label": "Rock"},
            {"label": "Blues"}
        ],
        "material": [
            {"uri": "http://www.wikidata.org/entity/Q287",
             "label": "Wood"},
            {"label": "Plastic"},
            {"uri": "http://www.wikidata.org/entity/Q11427",
             "label": "Metal"}
        ],
        "lend_start": datetime(2023, 1, 15),
        "lend_end": "2024-12-25"
    }

    draft_item = Item.model_validate(draft_item_data)
    
    print("--- Draft Item ---")
    print(draft_item)
    print(f"\nDraft Item as dict:\n{draft_item.model_dump_json(indent=2)}")
    
    is_final, missing_fields = draft_item.is_final()
    print(f"\nIs the draft item final? {is_final}")
    if not is_final:
        print(f"Reason: Missing URIs in fields: {missing_fields}")
        
    labels_to_mint = draft_item.get_labels_without_uri()
    print(f"\nLabels needing URIs: {labels_to_mint}")

    print(f"\nFormatted lend_start: {draft_item.lend_start}")
    print(f"Formatted lend_end: {draft_item.lend_end}")
    
    print("\n" + "="*40 + "\n")
    
    # 2. Example of a FINAL item (all URIs present)
    final_item_data = {
        "identifier": "OBJ-002",
        "name": "Stage-worn Jacket",
        "category": {"uri": "http://www.wikidata.org/entity/Q11460",
                     "label": "Clothing"},
        "type": {"uri": "http://www.wikidata.org/entity/Q11460",
                 "label": "Clothing"},
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q12345",
                          "label": "Famous Artist"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q1088",
                    "label": "Red"}]
    }

    final_item = Item.model_validate(final_item_data)

    print("--- Final Item ---")
    print(final_item)

    is_final, missing_fields = final_item.is_final()
    print(f"\nIs the final item final? {is_final}")
    if not is_final:
        print(f"Reason: Missing URIs in fields: {missing_fields}")
    else:
        print("This item is final and ready for storage.")
        
    print(f"\nLabels needing URIs: {final_item.get_labels_without_uri()}")

