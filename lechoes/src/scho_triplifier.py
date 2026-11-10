import json
import logging
import re
import unicodedata
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal


from rdflib import Graph, Literal as RdfLiteral, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD

# Assuming item_model.py is in the same directory or accessible in the path
from .metamodel import Item, LinkedEntity
from .namespaces import SCHO, bind_namespaces

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



def slugify(value: str) -> str:
    """
    Normalizes a string by converting it to lowercase, removing non-alpha
    characters, and converting spaces to hyphens. Essential for minting clean URIs.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    return value

def to_iso_date(date_str: Optional[str]) -> Optional[str]:
    """
    Robustly converts a date string to ISO 8601 format (YYYY-MM-DD).
    Handles YYYY-MM-DD and DD-MM-YYYY formats. Returns None if input is None.
    """
    if not date_str:
        return None
    # Check if already in the correct ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            # Catches invalid dates like '2023-13-40'
            pass
    # Check for and convert DD-MM-YYYY format
    if re.match(r"^\d{2}-\d{2}-\d{4}$", date_str):
        try:
            return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    # If all else fails, log a warning and return the original string
    logger.warning(
        f"Date '{date_str}' is not in a recognized format (YYYY-MM-DD or DD-MM-YYYY). "
        "Proceeding without conversion, which may cause serialization errors."
    )
    return date_str

class Triplifier:
    """
    Handles the conversion of an Item Pydantic model to an RDF graph based on
    a declarative mappings file and a specified URI minting strategy.
    """
    def __init__(self, mappings_path: str, base_uri: str, uri_strategy: Literal['slug', 'uuid'] = 'slug',
                 location_lookup: Optional[Dict[str, Any]] = None):
        """
        Initializes the Triplifier.

        Args:
            mappings_path: Path to the JSON file containing the mappings.
            base_uri: The base URI for minting new resources (e.g., "http://radio.liv.ac.uk/bme/").
            uri_strategy: The strategy for creating new URIs from labels ('slug' or 'uuid').
            location_lookup: Optional dict mapping case_number (str) to {zone, case_name}.
        """
        self.mappings = self._load_json(mappings_path)
        self.base_uri = Namespace(base_uri)
        self.uri_strategy = uri_strategy
        self.location_lookup = location_lookup or {}
        self.uri = None
        self.g = Graph()
        bind_namespaces(self.g)

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Loads a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Mappings file not found at: {path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from: {path}")
            raise

    def _get_class_name_from_uri(self, uri: str) -> str:
        """Helper to extract the class name from a full URI."""
        return re.split('/|#', uri)[-1]

    def _mint_uri(self, entity_label: str, entity_class_name: str) -> URIRef:
        """
        Mints a new URI based on the configured strategy and the RDF class name.
        """
        if self.uri_strategy == 'slug':
            slug = slugify(entity_label)
            return self.base_uri[f"{entity_class_name}/{slug}"]
        elif self.uri_strategy == 'uuid':
            return self.base_uri[f"{entity_class_name}/{uuid.uuid4()}"]
        elif self.uri_strategy == 'hash':
            return self.base_uri[f"{entity_class_name}/{hashlib.sha256(entity_label.encode('utf-8')).hexdigest()[:10]}"]
        
        raise NotImplementedError(f"URI strategy '{self.uri_strategy}' not implemented.")

    def triplify(self, item: Item, output_path: Optional[str] = None) -> Graph:
        """
        Generates a set of RDF triples for a given Item object.

        Args:
            item: The Pydantic Item object to triplify.
            output_path: If provided, saves the RDF graph to this file path in Turtle format.

        Returns:
            An rdflib.Graph object containing the generated triples.
        """
        self.g = Graph()
        self.g = bind_namespaces(self.g)

        # 1. Create the main subject URI for the item (using the shorthand)
        main_class_name = self._get_class_name_from_uri(self.mappings['_type_shorthand'])
        item_subject = self.base_uri[f"{main_class_name}/{item.identifier}"]
        self.uri = self.base_uri[f"{main_class_name}/{item.identifier}"]

        # 2. Add the main type(s) for the item from the mapping root
        for type_uri in self.mappings.get('_type_uri', []):
            self.g.add((item_subject, RDF.type, URIRef(type_uri)))

        # 3a. Year — typed as xsd:gYear literals on schema:temporalCoverage
        # Only assert 4-digit numeric years (xsd:gYear requires integer)
        if item.year:
            for y in item.year:
                y_str = str(y).strip() if y else ""
                if re.match(r"^\d{4}$", y_str):
                    self.g.add((item_subject, SCHO.temporalCoverage, RdfLiteral(y_str, datatype=XSD.gYear)))

        # 3. Process each field in the Pydantic model
        for field_name, field_value in item.model_dump().items():
            if field_value is None or field_name in ["lender", "lend_start", "lend_end", "case_number", "item_number", "year"]:
                continue # Lending info is handled separately
            
            field_mapping = self.mappings.get(field_name)
            if not field_mapping:
                logger.warning(f"No mapping found for field '{field_name}'. Skipping.")
                continue

            relation_uris = field_mapping.get('_relation_uri', [])
            for predicate_uri in relation_uris:
                predicate = URIRef(predicate_uri)
                # Handle lists of linked entities
                if isinstance(field_value, list): 
                    for entity_dict in field_value:
                        entity = LinkedEntity.model_validate(entity_dict)
                        self._add_linked_entity_triple(item_subject, predicate, entity, field_name, field_mapping)
                # Handle single linked entities
                elif isinstance(field_value, dict):
                    entity = LinkedEntity.model_validate(field_value)
                    self._add_linked_entity_triple(item_subject, predicate, entity, field_name, field_mapping)
                # Handle simple literal values
                else:
                    # Check if the value is a URL and should be a URIRef
                    if field_name in ['image', 'audio', 'video', 'url'] and isinstance(field_value, str) and field_value.startswith('http'):
                        self.g.add((item_subject, predicate, URIRef(field_value)))
                    else:
                        self.g.add((item_subject, predicate, RdfLiteral(field_value)))

        # 4. Handle location hierarchy (case_number + item_number)
        self._add_location_info(item_subject, item)

        # 5. Handle the special case for lending information
        self._add_lending_info(item_subject, item)

        # 5. Save to file if requested
        if output_path:
            try:
                self.g.serialize(destination=output_path, format="turtle")
                logger.info(f"Graph successfully saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save graph to {output_path}: {e}")

        return self.g

    def _add_linked_entity_triple(self, subject: URIRef, predicate: URIRef, entity: LinkedEntity, field_name: str, mapping: Dict):
        """
        Adds triples for a LinkedEntity, minting a URI if necessary.
        If the mapping type is xsd:string, the label is emitted as a plain literal.
        """
        type_uris = mapping.get('_type_uri', [])

        # Emit a plain literal when the mapping declares xsd:string as the type
        if type_uris and type_uris[0] == str(XSD.string):
            label = entity.label.lower() if entity.label else ""
            if label:
                self.g.add((subject, predicate, RdfLiteral(label)))
            return

        object_uri = None
        if entity.uri:
            object_uri = URIRef(entity.uri)
        else:
            # Mint a new URI and log a warning
            logger.warning(f"Missing URI for '{field_name}' with label '{entity.label}'. Minting new URI.")
            if not type_uris:
                logger.error(f"Cannot mint URI for '{entity.label}'. No '_type_uri' in mapping for field '{field_name}'.")
                return

            # Use the first type URI to determine the class name for the path
            entity_class_name = self._get_class_name_from_uri(type_uris[0])
            object_uri = self._mint_uri(entity.label, entity_class_name)

        # Add the primary triple (e.g., item -> hasGenre -> genre)
        self.g.add((subject, predicate, object_uri))

        # Add supporting triples for the object (e.g., genre -> is a -> MusicGenre, genre -> hasLabel -> "Rock")
        for type_uri in type_uris:
            self.g.add((object_uri, RDF.type, URIRef(type_uri)))
        self.g.add((object_uri, RDFS.label, RdfLiteral(entity.label, lang='en')))

    def _add_location_info(self, subject: URIRef, item: Item):
        """
        Builds a Place hierarchy for schema:contentLocation using case_number
        and item_number. Mints deterministic slug URIs so the same case/zone
        always produces the same URI across runs.

        Structure:
            item -> schema:contentLocation -> bme:Case/{n}/Item/{m}  (or .../default)
            bme:Case/{n}/Item/{m} -> schema:containedInPlace -> bme:Case/{n}
            bme:Case/{n} -> schema:containedInPlace -> bme:Zone/{z}
        """
        if not item.case_number:
            return

        case_key = str(item.case_number).strip()
        case_info = self.location_lookup.get(case_key, {})
        zone = str(case_info.get("zone", "unknown"))
        case_name = case_info.get("case_name", f"Case {case_key}")

        # Zone node
        zone_uri = self.base_uri[f"Zone/{zone}"]
        self.g.add((zone_uri, RDF.type, URIRef("http://schema.org/Place")))
        self.g.add((zone_uri, RDFS.label, RdfLiteral(f"Zone {zone}", lang="en")))

        # Case node
        case_uri = self.base_uri[f"Case/{case_key}"]
        self.g.add((case_uri, RDF.type, URIRef("http://schema.org/Place")))
        self.g.add((case_uri, RDFS.label, RdfLiteral(case_name, lang="en")))
        self.g.add((case_uri, URIRef("http://schema.org/containedInPlace"), zone_uri))

        self.g.add((subject, URIRef("http://schema.org/location"), case_uri))

    def _add_lending_info(self, subject: URIRef, item: Item):
        """Creates a blank node for lending information if present."""
        if not any([item.lender, item.lend_start, item.lend_end]):
            return

        lend_mapping = self.mappings.get('lend', {})
        relation_uri = lend_mapping.get('_relation_uri', [SCHO.hasPart])[0]
        type_uri = lend_mapping.get('_type_uri', [SCHO.BorrowAction])[0]
        
        bnode = BNode()
        self.g.add((subject, URIRef(relation_uri), bnode))
        self.g.add((bnode, RDF.type, URIRef(type_uri)))

        if item.lender:
            pred = URIRef(self.mappings['lender']['_relation_uri'][0])
            self.g.add((bnode, pred, RdfLiteral(item.lender)))
        if item.lend_start:
            pred = URIRef(self.mappings['lend_start']['_relation_uri'][0])
            iso_date = to_iso_date(item.lend_start)
            if iso_date:
                self.g.add((bnode, pred, RdfLiteral(iso_date, datatype=XSD.date)))
        if item.lend_end:
            pred = URIRef(self.mappings['lend_end']['_relation_uri'][0])
            iso_date = to_iso_date(item.lend_end)
            if iso_date:
                self.g.add((bnode, pred, RdfLiteral(iso_date, datatype=XSD.date)))


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create an Item object (same data as in item_model.py for consistency)
    draft_item_data = {
        "identifier": "a4d8c1fe",
        "name": "Bryan May's Red Special",
        "description": "The iconic homemade electric guitar built by Queen's guitarist Brian May and his father.",
        "caption": "Famously crafted from an 18th-century fireplace mantel, it was instrumental in creating Queen's unique sound. The guitar that powered the sound of Queen, displayed in the Hard Rock exhibit.",
        "category": {"uri": "http://www.wikidata.org/entity/Q34379", "label": "Musical Instrument"},
        "type": {"uri": "http://www.mimo-db.eu/InstrumentsKeywords/3236", "label": "Electric Guitar"},
        "case_number": "1",
        "image": "https://media.musicarts.com/is/image/MMGS7/L85963000001000-09-2000x2000.jpg",
        "genre": [
            {"uri": "http://www.wikidata.org/entity/Q83270", "label": "Hard Rock"},
            {"label": "Glam Rock"} # Missing URI, will be minted
        ],
        "material": [
            {"uri": "http://www.wikidata.org/entity/Q2075708", "label": "Oak"},
            {"uri": "http://www.wikidata.org/entity/Q239788", "label": "Mahogany"},
        ],
        "lender": "Brian May",
        "lend_start": "2018-01-01",
        "lend_end": "2028-12-31"
    }
    item_to_process = Item.model_validate(draft_item_data)

    # 2. Initialize and run the Triplifier with the 'slug' strategy
    print("--- Running with 'slug' URI strategy ---")
    slug_triplifier = Triplifier(
        mappings_path=str(Path(__file__).parent.parent / "mappings_min.json"),
        base_uri="http://radio.liv.ac.uk/bme/",
        uri_strategy='slug'
    )
    slug_output_file = "output_graph_slug.ttl"
    slug_rdf_graph = slug_triplifier.triplify(item_to_process, output_path=slug_output_file)
    print(f"Triplification complete. Output saved to '{slug_output_file}'.")

    print("\n" + "="*50 + "\n")

    # 3. Initialize and run the Triplifier with the 'uuid' strategy
    print("--- Running with 'uuid' URI strategy ---")
    uuid_triplifier = Triplifier(
        mappings_path=str(Path(__file__).parent.parent / "mappings_min.json"),
        base_uri="http://radio.liv.ac.uk/bme/",
        uri_strategy='uuid'
    )
    uuid_output_file = "output_graph_uuid.ttl"
    uuid_rdf_graph = uuid_triplifier.triplify(item_to_process, output_path=uuid_output_file)
    print(f"Triplification complete. Output saved to '{uuid_output_file}'.")

    print("\n" + "="*50 + "\n")
    print("Running tests on the 7 example items...")
    from test_triplifier import TEST_DATA
    out_dir = Path(__file__).parent / "examples/test-triplifier"
    for id in TEST_DATA.keys():
        print(f"Processing test item '{id}'...")
        item_data = TEST_DATA.get(id)
        item = Item.model_validate(item_data)
        generated_graph = uuid_triplifier.triplify(item, f"{out_dir}/{id}_test.ttl")
