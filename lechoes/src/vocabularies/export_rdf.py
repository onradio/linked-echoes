#!/usr/bin/env python3
"""
Release the harmonised controlled vocabularies as a single SKOS dataset in Turtle.

    python lechoes/src/vocabularies/export_rdf.py              # -> harmonised_vocabularies.ttl
    python lechoes/src/vocabularies/export_rdf.py --out PATH

Input is `all_concepts.csv`, the harmonised table from which the type-classification indices
are built (one row per term: label, definition, URI, source vocabulary). Every row becomes a
`skos:Concept` under the URI of its source vocabulary, with its label as `skos:prefLabel`@en and
its definition as `skos:definition`@en, in one `skos:ConceptScheme` per source vocabulary
(Getty AAT, MIMO, Europeana Fashion, PBCore). Where the source is itself distributed as RDF in
this folder (MIMO, Europeana Fashion), the labels it gives in other languages, and its
alternative labels, are added, so that the release keeps the multilingual labels where they
exist. A VoID/DCAT description of the dataset closes the file.

The table has 50,301 rows; they yield 50,298 concepts. Two Europeana Fashion terms are listed
twice and are merged, and the MIMO row whose URI is MIMO's concept scheme itself ("Musical
instruments") is recorded as the source of the MIMO scheme rather than as a concept, since SKOS
keeps concept schemes and concepts disjoint.

The concepts are the source vocabularies' own: their labels and definitions remain subject to
the terms of Getty AAT, MIMO, PBCore and Europeana Fashion.
"""
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCAT, DCTERMS, PROV, RDF, RDFS, SKOS, VOID, XSD

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "all_concepts.csv"
DEFAULT_OUT = HERE / "harmonised_vocabularies.ttl"

# Namespace for the resources this release adds (the dataset and its concept schemes); the
# concepts themselves keep the URIs of their source vocabularies.
NS = Namespace("http://radio.liv.ac.uk/vocabularies/")
RELEASE_DATE = "2026-11-16"
VERSION = "1.0"
REPOSITORY = URIRef("https://github.com/onradio/linked-echoes")

# source_vocabulary value in the table -> (scheme slug, title, homepage, scheme URI at the source)
SOURCES = {
    "AAT": ("aat", "Getty Art & Architecture Thesaurus (AAT)",
            "https://www.getty.edu/research/tools/vocabularies/aat/", "http://vocab.getty.edu/aat/"),
    "MIMO": ("mimo", "MIMO (Musical Instrument Museums Online) instrument keywords",
             "https://vocabulary.mimo-international.com", "http://www.mimo-db.eu/InstrumentsKeywords#"),
    "Europeana": ("europeana-fashion", "Europeana Fashion thesaurus",
                  "http://thesaurus.europeanafashion.eu", None),
    "PBCore-AssetType": ("pbcore", "PBCore controlled vocabularies", "https://pbcore.org", None),
    "PBCore-Instantiation": ("pbcore", "PBCore controlled vocabularies", "https://pbcore.org", None),
}
# PBCore contributes two of its vocabularies; each is kept as a collection within the scheme.
PBCORE_COLLECTIONS = {
    "PBCore-AssetType": ("pbcore/asset-types", "PBCore asset types"),
    "PBCore-Instantiation": ("pbcore/instantiation-formats", "PBCore physical instantiation formats"),
}
# Source RDF in this folder from which other-language and alternative labels are taken.
LABEL_SOURCES = {
    "MIMO": [(HERE / "mimo" / "mimo_doremus.ttl", "turtle")],
    "Europeana": [(HERE / "fashion" / "fashion.rdf", "xml")],
}


def _norm(text: str) -> str:
    return " ".join(text.split()).rstrip(".").casefold()


def read_table(path: Path):
    """Rows grouped by URI, in table order: {uri: {"source", "labels", "definitions"}}."""
    concepts = OrderedDict()
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            uri = row["uri"].strip()
            entry = concepts.setdefault(uri, {"source": row["source_vocabulary"].strip(),
                                              "labels": [], "definitions": []})
            if entry["source"] != row["source_vocabulary"].strip():
                raise SystemExit(f"{uri} is listed under two source vocabularies")
            # Whitespace is collapsed: a few definitions carry line breaks from their sources.
            label, definition = " ".join(row["term"].split()), " ".join(row["definition"].split())
            if label and label not in entry["labels"]:
                entry["labels"].append(label)
            if definition and _norm(definition) not in {_norm(d) for d in entry["definitions"]}:
                entry["definitions"].append(definition)
    return concepts


def source_labels(source: str, uris: set):
    """{uri: [(property, Literal)]} from the source RDF, restricted to `uris`."""
    found = {}
    for path, fmt in LABEL_SOURCES.get(source, []):
        g = Graph()
        g.parse(path, format=fmt)
        for prop in (SKOS.prefLabel, SKOS.altLabel):
            for s, o in g.subject_objects(prop):
                if str(s) in uris and isinstance(o, Literal) and o.language and str(o).strip():
                    found.setdefault(str(s), []).append((prop, Literal(" ".join(str(o).split()), lang=o.language)))
    return found


def build(table_path: Path = CSV_PATH) -> Graph:
    g = Graph()
    for prefix, ns in [("skos", SKOS), ("dct", DCTERMS), ("void", VOID), ("dcat", DCAT),
                       ("prov", PROV), ("rdfs", RDFS), ("xsd", XSD), ("lev", NS),
                       ("aat", Namespace("http://vocab.getty.edu/aat/")),
                       ("mimo", Namespace("http://www.mimo-db.eu/InstrumentsKeywords/")),
                       ("euf", Namespace("http://thesaurus.europeanafashion.eu/thesaurus/")),
                       ("pbcore", Namespace("http://pbcore.org/pbcore-controlled-vocabularies/titletype-vocabulary/#"))]:
        g.bind(prefix, ns)

    concepts = read_table(table_path)
    scheme_uris = {SOURCES[s][3] for s in SOURCES if SOURCES[s][3]}
    skipped = [u for u in concepts if u in scheme_uris]          # MIMO's scheme listed as a term
    for uri in skipped:
        del concepts[uri]

    extra = {}
    for source in LABEL_SOURCES:
        uris = {u for u, c in concepts.items() if c["source"] == source}
        extra.update(source_labels(source, uris))

    counts = {}
    for uri, c in concepts.items():
        slug = SOURCES[c["source"]][0]
        counts[slug] = counts.get(slug, 0) + 1
        node = URIRef(uri)
        g.add((node, RDF.type, SKOS.Concept))
        g.add((node, SKOS.inScheme, NS[slug]))
        pref = {"en": c["labels"][0]}
        alts = {("en", label) for label in c["labels"][1:]}
        for prop, lit in sorted(extra.get(uri, []), key=lambda x: (x[1].language, str(x[0]), str(x[1]))):
            lang, text = lit.language, str(lit)
            if prop == SKOS.prefLabel and lang not in pref:
                pref[lang] = text
            else:
                alts.add((lang, text))
        for lang, text in pref.items():
            g.add((node, SKOS.prefLabel, Literal(text, lang=lang)))
        for lang, text in sorted(alts):
            if pref.get(lang) != text:                           # prefLabel and altLabel disjoint
                g.add((node, SKOS.altLabel, Literal(text, lang=lang)))
        if c["definitions"]:
            g.add((node, SKOS.definition, Literal(max(c["definitions"], key=len), lang="en")))
        if c["source"] in PBCORE_COLLECTIONS:
            g.add((NS[PBCORE_COLLECTIONS[c["source"]][0]], SKOS.member, node))

    for slug, title, homepage, source_scheme in {v[:4] for v in SOURCES.values()}:
        scheme = NS[slug]
        g.add((scheme, RDF.type, SKOS.ConceptScheme))
        g.add((scheme, DCTERMS.title, Literal(title, lang="en")))
        g.add((scheme, DCTERMS.source, URIRef(homepage)))
        if source_scheme:
            g.add((scheme, DCTERMS.source, URIRef(source_scheme)))
        g.add((scheme, RDFS.comment, Literal(f"{counts[slug]:,} concepts of the {title}, as harmonised "
                                             "for the Linked Echoes workflow.", lang="en")))
        g.add((NS[""], DCTERMS.hasPart, scheme))
    for key, (path, title) in PBCORE_COLLECTIONS.items():
        g.add((NS[path], RDF.type, SKOS.Collection))
        g.add((NS[path], SKOS.prefLabel, Literal(title, lang="en")))

    dataset = NS[""]
    g.add((dataset, RDF.type, VOID.Dataset))
    g.add((dataset, RDF.type, DCAT.Dataset))
    g.add((dataset, DCTERMS.title, Literal("Linked Echoes harmonised controlled vocabularies", lang="en")))
    g.add((dataset, DCTERMS.description, Literal(
        "Terms of Getty AAT, MIMO, Europeana Fashion and PBCore, harmonised into a single dataset "
        "for the retrieval-augmented type classification of the Linked Echoes workflow. Concepts "
        "keep the URIs of their source vocabularies; labels in other languages and alternative "
        "labels are included where the source provides them.", lang="en")))
    g.add((dataset, DCTERMS.rights, Literal(
        "The concepts, labels and definitions remain subject to the terms of their source "
        "vocabularies (Getty AAT, MIMO, PBCore, Europeana Fashion), which should be credited "
        "when the data are reused.", lang="en")))
    g.add((dataset, DCTERMS.created, Literal(RELEASE_DATE, datatype=XSD.date)))
    g.add((dataset, DCTERMS.hasVersion, Literal(VERSION)))
    g.add((dataset, PROV.wasDerivedFrom,
           URIRef(f"{REPOSITORY}/blob/main/lechoes/src/vocabularies/all_concepts.csv")))
    g.add((dataset, RDFS.seeAlso, REPOSITORY))
    g.add((dataset, VOID.vocabulary, URIRef(str(SKOS))))
    g.add((dataset, VOID.entities, Literal(len(concepts), datatype=XSD.integer)))
    g.add((dataset, VOID.triples, Literal(len(g) + 1, datatype=XSD.integer)))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--table", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    g = build(args.table)
    args.out.write_text(g.serialize(format="turtle").rstrip("\n") + "\n", encoding="utf-8")
    n = len(set(g.subjects(RDF.type, SKOS.Concept)))
    print(f"wrote {args.out} ({len(g):,} triples, {n:,} concepts)")


if __name__ == "__main__":
    main()
