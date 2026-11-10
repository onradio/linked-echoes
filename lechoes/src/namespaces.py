from rdflib import Namespace
from rdflib.namespace import RDF, RDFS, SKOS, XSD, FOAF

# --- Namespaces ---
BME = Namespace("http://radio.liv.ac.uk/bme/")
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
SCHO = Namespace("http://schema.org/")
WD = Namespace("http://www.wikidata.org/entity/")
DBP = Namespace("http://dbpedia.org/resource/")
AAT = Namespace("http://vocab.getty.edu/aat/")
MIMO = Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")
EUF = Namespace("http://thesaurus.europeanafashion.eu/thesaurus/")
PBCORE = Namespace("http://pbcore.org/pbcore-controlled-vocabularies/titletype-vocabulary/#")


def bind_namespaces(graph):
    """
    Binds commonly used namespaces to a given RDF graph for easier reference.
    """
    graph.bind("crm", CRM)
    graph.bind("rdfs", RDFS)
    graph.bind("skos", SKOS)
    graph.bind("scho", SCHO)
    graph.bind("xsd", XSD)
    graph.bind("wd", WD)
    graph.bind("foaf", FOAF)
    graph.bind("bme", BME)
    graph.bind("dbp", DBP)
    graph.bind("aat", AAT)
    graph.bind("mimo", MIMO)
    graph.bind("euf", EUF)
    graph.bind("pbcore", PBCORE)
    return graph


