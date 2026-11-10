import pytest
from pathlib import Path
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.compare import isomorphic, graph_diff

from .scho_triplifier import Triplifier
from .namespaces import BME
from .metamodel import Item

# Define paths relative to the test file
MAPPINGS_PATH = str(Path(__file__).parent / "mappings_min.json")
GROUND_TRUTH_PATH = str(Path(__file__).parent / "testset_proto_3.ttl")

# --- Test Data Payloads ---
TEST_DATA = {
    "a4d8c1fe": {
        "identifier": "a4d8c1fe",
        "name": "Bryan May's Red Special",
        "description": "The iconic homemade electric guitar built by Queen's guitarist Brian May and his father.",
        "caption": "Famously crafted from an 18th-century fireplace mantel, it was instrumental in creating Queen's unique sound. The guitar that powered the sound of Queen, displayed in the Hard Rock exhibit.",
        "category": {"uri": "http://www.wikidata.org/entity/Q34379", "label": "Musical Instrument"},
        "type": {"uri": "http://www.mimo-db.eu/InstrumentsKeywords/3236", "label": "Electric Guitar"},
        "case_number": "1",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone6", "label": "Zone 6"}],
        "image": "https://media.musicarts.com/is/image/MMGS7/L85963000001000-09-2000x2000.jpg",
        "audio": "http://museum.example.org/audio/a4d8c1fe_guitarsolo.mp3",
        "video": "https://www.youtube.com/watch?v=IIVbUJnoduQ",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q15873", "label": "Brian May"}, {"uri": "http://www.wikidata.org/entity/Q15862", "label": "Queen"}],
        "venue": [{"uri": "http://www.wikidata.org/entity/Q43279", "label": "Wembley Stadium"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "place": [{"uri": "http://www.wikidata.org/entity/Q84", "label": "London"}],
        "event": [{"uri": "http://www.wikidata.org/entity/Q193740", "label": "Live Aid 1985"}],
        "music_work": [{"uri": "http://www.wikidata.org/entity/Q187745", "label": "Bohemian Rhapsody"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q83270", "label": "Hard Rock"}, {"uri": "http://www.wikidata.org/entity/Q76092", "label": "Glam Rock"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q2075708", "label": "Oak"}, {"uri": "http://www.wikidata.org/entity/Q239788", "label": "Mahogany"}, {"uri": "http://www.wikidata.org/entity/Q215865", "label": "Mother of pearl"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q3142", "label": "Red"}, {"uri": "http://www.wikidata.org/entity/Q23445", "label": "Black"}],
        "lender": "Brian May",
        "lend_start": "2018-01-01",
        "lend_end": "2028-12-31"
    },
    "b9e1f3a0": {
        "identifier": "b9e1f3a0",
        "name": "Sex Pistols 'Anarchy' Muslin Shirt",
        "description": "An original muslin shirt designed by Vivienne Westwood and Malcolm McLaren, worn by Johnny Rotten.",
        "caption": "A wearable manifesto of the punk movement. The deliberately offensive imagery and slogans defined the punk aesthetic. Worn by Johnny Rotten during the infamous Today show interview with Bill Grundy in 1976.",
        "category": {"uri": "http://www.wikidata.org/entity/Q11460", "label": "Clothing"},
        "type": {"uri": "http://thesaurus.europeanafashion.eu/thesaurus/10025", "label": "Shirt"},
        "case_number": "2",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone5", "label": "Zone 5"}],
        "image": "http://museum.example.org/images/b9e1f3a0.jpg",
        "audio": "https://open.spotify.com/track/2n26ggxLD72OSLjlHHdAZq?si=aa770611b8804751",
        "video": "https://www.youtube.com/watch?v=boQGEyAcZwo",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q82545", "label": "Sex Pistols"}, {"uri": "http://www.wikidata.org/entity/Q106662", "label": "John Lydon"}],
        "place": [{"uri": "http://www.wikidata.org/entity/Q84", "label": "London"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "event": [{"uri": "http://radio.liv.ac.uk/bme/Event/e9e1f3a0", "label": "Today (Bill Grundy show) interview"}],
        "music_album": [{"uri": "http://www.wikidata.org/entity/Q457275", "label": "Never Mind the Bollocks, Here's the Sex Pistols"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q3071", "label": "Punk Rock"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q34320", "label": "Cotton"}, {"uri": "http://www.wikidata.org/entity/Q11463", "label": "Ink"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q23444", "label": "White"}, {"uri": "http://www.wikidata.org/entity/Q23445", "label": "Black"}],
        "lender": "Victoria and Albert Museum",
        "lend_start": "2020-03-10",
    },
    "c5f0b2d3": {
        "identifier": "c5f0b2d3",
        "name": "Master Tape for 'Sgt. Pepper's Lonely Hearts Club Band'",
        "description": "The original 1-inch 4-track analogue master tape for The Beatles' seminal 1967 album, recorded at EMI Studios (now Abbey Road Studios).",
        "caption": "The original recording of a pop masterpiece. The Beatles' groundbreaking album that redefined popular music and culture. This master tape captures the innovative techniques and creativity that made 'Sgt. Pepper's' a timeless classic.",
        "category": {"uri": "http://www.wikidata.org/entity/Q758901", "label": "Audiovisual Media"},
        "type": {"uri": "http://pbcore.org/pbcore-controlled-vocabularies/titletype-vocabulary#1InchVideotape", "label": "1 inch videotape"},
        "case_number": "3",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone3", "label": "Zone 3"}],
        "image": "https://goauctionomega.blob.core.windows.net/stock/25311-2-medium.jpg?v=63798651902413",
        "audio": "https://open.spotify.com/track/6pcO0mAQbpKbqKeiFHuTRR?si=b73e631024924ab9",
        "video": "https://www.youtube.com/watch?v=opecMCX-SQQ",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q1299", "label": "The Beatles"}],
        "venue": [{"uri": "http://www.wikidata.org/entity/Q209651", "label": "Abbey Road Studios"}],
        "place": [{"uri": "http://www.wikidata.org/entity/Q84", "label": "London"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "music_album": [{"uri": "http://www.wikidata.org/entity/Q169226", "label": "Sgt. Pepper's Lonely Hearts Club Band"}],
        "music_work": [{"uri": "http://www.wikidata.org/entity/Q300385", "label": "A Day in the Life"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q206159", "label": "Psychedelic Rock"}, {"uri": "http://www.wikidata.org/entity/Q76092", "label": "Glam Rock"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q193663", "label": "Magnetic tape"}],
        "lender": "Apple Corps",
        "lend_start": "2017-06-01",
    },
    "d1e4c8a9": {
        "identifier": "d1e4c8a9",
        "name": "Handwritten Lyrics for 'Wonderwall'",
        "description": "Noel Gallagher's original handwritten lyrics for 'Wonderwall', written on a sheet of hotel stationery in 1995. Includes crossed-out lines and annotations.",
        "caption": "The draft of a generation-defining anthem. These handwritten lyrics by Noel Gallagher capture the raw emotion and introspection that made 'Wonderwall' an enduring classic of the Britpop era. Written during a stay at the Midland Hotel in Manchester, they offer a glimpse into the creative process behind one of Oasis's most beloved songs.",
        "category": {"uri": "http://www.wikidata.org/entity/Q49848", "label": "Textual and Graphic Works"},
        "type": {"uri": "http://www.wikidata.org/entity/Q169123", "label": "Lyrics"},
        "case_number": "4",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone7", "label": "Zone 7"}],
        "image": "https://news.justcollecting.com/wp-content/uploads/2020/02/58267_0.jpg",
        "audio": "https://open.spotify.com/track/1qPbGZqppFwLwcBC1JQ6Vr?si=e867c6046c164016",
        "video": "https://www.youtube.com/shorts/mAcHpVbdtKI",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q382890", "label": "Oasis"}, {"uri": "http://www.wikidata.org/entity/Q193459", "label": "Noel Gallagher"}],
        "event": [{"uri": "http://radio.liv.ac.uk/bme/Event/e1e4c8a9", "label": "Oasis live at Knebworth 1996"}],
        "venue": [{"uri": "http://www.wikidata.org/entity/Q19841", "label": "Knebworth House"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "music_album": [{"uri": "http://www.wikidata.org/entity/Q8360", "label": "(What's the Story) Morning Glory?"}],
        "music_work": [{"uri": "http://www.wikidata.org/entity/Q1045517", "label": "Wonderwall"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q189045", "label": "Britpop"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q11472", "label": "Paper"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q23442", "label": "White"}, {"uri": "http://www.wikidata.org/entity/Q1088", "label": "Blue"}],
        "lender": "Noel Gallagher",
        "lend_start": "2019-08-11",
    },
    "e6f7b9c2": {
        "identifier": "e6f7b9c2",
        "name": "Spice Girls 'Spice World' Lunchbox",
        "description": "A mass-produced metal lunchbox with matching plastic thermos, featuring a promotional image of the Spice Girls for their 1997 film 'Spice World'.",
        "caption": "A symbol of 'Girl Power' fan mania which could be found in school playgrounds across the UK. A quintessential piece of 90s pop merchandise.",
        "category": {"uri": "http://www.wikidata.org/entity/Q28877", "label": "Merchandise"},
        "type": {"uri": "http://vocab.getty.edu/aat/300199751", "label": "Lunchbox"},
        "case_number": "5",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone7", "label": "Zone 7"}],
        "image": "https://metro.co.uk/wp-content/uploads/2015/01/spice-girls.jpg?quality=90&strip=all&w=570",
        "audio": "https://open.spotify.com/album/3x2jF7blR6bFHtk4MccsyJ?si=BrygJ0cRQiWpu9H4cMRslA",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q55641", "label": "Spice Girls"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "music_album": [{"uri": "http://www.wikidata.org/entity/Q818781", "label": "Spice"}],
        "place": [{"uri": "http://www.wikidata.org/entity/Q84", "label": "London"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q37073", "label": "Pop music"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q1096", "label": "Tin"}, {"uri": "http://www.wikidata.org/entity/Q11474", "label": "Plastic"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q429220", "label": "Pink"}, {"uri": "http://www.wikidata.org/entity/Q317802", "label": "Silver"}],
    },
    "f8a0d7e4": {
        "identifier": "f8a0d7e4",
        "name": "John Lennon's Round 'Granny' Glasses",
        "description": "A pair of round, nickel-rimmed glasses famously worn by John Lennon during the mid-to-late 1960s. This style became an inseparable part of his public image during the psychedelic era and his subsequent solo career.",
        "caption": "The iconic eyewear of a music and cultural legend. These glasses were a key element of John Lennon's distinctive look, symbolizing his transformation from Beatle to countercultural icon. Worn during the height of The Beatles' fame and throughout his solo career, they remain a powerful emblem of his enduring influence on music and popular culture.",
        "category": {"uri": "http://www.wikidata.org/entity/Q115866017", "label": "Personal Effects"},
        "type": {"uri": "http://vocab.getty.edu/aat/300236409", "label": "Spectacles"},
        "case_number": "6",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone3", "label": "Zone 3"}],
        "image": "https://c.files.bbci.co.uk/c24f/live/b4199d60-4e81-11ef-8b97-e7e1e12cdeb2.jpg",
        "video": "https://www.youtube.com/watch?v=r8IlMfx3ZeA",
        "url": "https://www.beatlesstory.com/blog/john-lennons-spectacles/",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q1203", "label": "John Lennon"}, {"uri": "http://www.wikidata.org/entity/Q1299", "label": "The Beatles"}],
        "place": [{"uri": "http://www.wikidata.org/entity/Q24826", "label": "Liverpool"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "music_album": [{"uri": "http://www.wikidata.org/entity/Q169226", "label": "Sgt. Pepper's Lonely Hearts Club Band"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q206159", "label": "Psychedelic Rock"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q744", "label": "Nickel"}, {"uri": "http://www.wikidata.org/entity/Q11469", "label": "Glass"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q317802", "label": "Silver"}],
        "lender": "The Ono Lennon Estate",
        "lend_start": "2021-10-09",
    },
    "g2c5h4b7": {
        "identifier": "g2c5h4b7",
        "name": "BRIT Award for British Album of the Year (2012)",
        "description": "The BRIT Award trophy presented to Adele for her album '21' at the 2012 ceremony. The statue for that year was designed by Sir Peter Blake.",
        "caption": "A physical recognition of a record-breaking album. Adele's '21' became the best-selling album of the 21st century in the UK, and this BRIT Award trophy symbolizes her monumental achievement in British music history.",
        "category": {"uri": "http://www.wikidata.org/entity/Q190573", "label": "Award and Recognition"},
        "type": {"uri": "http://vocab.getty.edu/aat/300233975", "label": "Trophies"},
        "case_number": "7",
        "exhibition": [{"uri": "http://radio.liv.ac.uk/bme/Exhibition/Zone8", "label": "Zone 8"}],
        "image": "https://i.pinimg.com/736x/66/06/59/6606594978ed4227b3ad7dec06eea626.jpg",
        "audio": "https://open.spotify.com/album/0Lg1uZvI312TPqxNWShFXL?si=hvKnl5LYQ_uLUfXSfPWSqg",
        "video": "https://www.youtube.com/watch?v=I4RS4YodJU8",
        "music_artist": [{"uri": "http://www.wikidata.org/entity/Q23215", "label": "Adele"}],
        "place": [{"uri": "http://www.wikidata.org/entity/Q84", "label": "London"}],
        "country": [{"uri": "http://www.wikidata.org/entity/Q145", "label": "United Kingdom"}],
        "venue": [{"uri": "http://www.wikidata.org/entity/Q5364202", "label": "The O2 Arena"}],
        "music_album": [{"uri": "http://www.wikidata.org/entity/Q155169", "label": "21"}],
        "event": [{"uri": "http://www.wikidata.org/entity/Q2982137", "label": "2012 BRIT Awards"}],
        "genre": [{"uri": "http://www.wikidata.org/entity/Q131272", "label": "Soul music"}],
        "material": [{"uri": "http://www.wikidata.org/entity/Q663", "label": "Aluminium"}],
        "colour": [{"uri": "http://www.wikidata.org/entity/Q183127", "label": "Multicolour"}],
        "lender": "BRIT Awards Ltd.",
        "lend_start": "2013-02-20",
    }
}

def run_single_item_test(identifier: str):
    """
    Helper function to run the triplification and comparison for a single item.
    """
    # 1. Load the ground truth graph
    gt_graph = Graph().parse(GROUND_TRUTH_PATH, format="turtle")
    item_subject = BME[f"HumanMadeObject/{identifier}"]
    
    # 2. Create a filtered ground truth graph containing only triples for this item
    filtered_gt = Graph()
    # Add all direct triples for the item
    for p, o in gt_graph.predicate_objects(subject=item_subject):
        filtered_gt.add((item_subject, p, o))
        # If the object is not a literal, add its description from the main graph
        if isinstance(o, URIRef):
             for p2, o2 in gt_graph.predicate_objects(subject=o):
                 filtered_gt.add((o, p2, o2))
        # Handle blank nodes
        elif isinstance(o, BNode):
            for p_b, o_b in gt_graph.predicate_objects(subject=o):
                filtered_gt.add((o, p_b, o_b))

    # 3. Get the corresponding test data payload
    item_data = TEST_DATA.get(identifier)
    if not item_data:
        pytest.fail(f"No test data found for identifier: {identifier}")

    # 4. Initialize Triplifier and generate the graph
    triplifier = Triplifier(
        mappings_path=MAPPINGS_PATH,
        base_uri=str(BME),
        uri_strategy='slug'
    )
    item = Item.model_validate(item_data)
    generated_graph = triplifier.triplify(item)

    # 5. Compare the graphs
    if not isomorphic(generated_graph, filtered_gt):
        # If they are not identical, graph_diff helps to see what's different
        in_generated_only, in_gt_only, _ = graph_diff(generated_graph, filtered_gt)
        
        # Use a more detailed format for easier debugging
        diff_report = []
        if any(in_generated_only):
            diff_report.append("--- Triples in Generated Graph ONLY ---")
            for s, p, o in in_generated_only:
                diff_report.append(f"{s.n3()} {p.n3()} {o.n3()}.")
        
        if any(in_gt_only):
            diff_report.append("\n--- Triples in Ground Truth ONLY ---")
            for s, p, o in in_gt_only:
                diff_report.append(f"{s.n3()} {p.n3()} {o.n3()}.")

        # Fail the test with a useful message
        pytest.fail(
            f"Graph for '{identifier}' is not isomorphic to the ground truth.\n\n" + "\n".join(diff_report),
            pytrace=False
        )

# --- Pytest Test Functions ---

def test_musical_instrument():
    """Tests Category 1: Musical Instruments"""
    run_single_item_test("a4d8c1fe")

def test_clothing_fashion():
    """Tests Category 2: Clothing & Fashion"""
    run_single_item_test("b9e1f3a0")

def test_audiovisual_media():
    """Tests Category 3: Audio-Visual Media"""
    run_single_item_test("c5f0b2d3")

def test_textual_graphic_works():
    """Tests Category 4: Textual & Graphic Works"""
    run_single_item_test("d1e4c8a9")

def test_merchandise():
    """Tests Category 5: Merchandise"""
    run_single_item_test("e6f7b9c2")

def test_personal_effects():
    """Tests Category 6: Personal Effects"""
    run_single_item_test("f8a0d7e4")

def test_award_recognition():
    """Tests Category 7: Award and Recognition"""
    run_single_item_test("g2c5h4b7")
