# KG Expansion Pipeline — Team Overview

## What this is

The BME Knowledge Graph currently holds structured data about music entities — artists, bands, albums, events, venues — extracted from exhibition descriptions and item metadata. Each entity in the KG has a link to its Wikidata counterpart, but that link was previously unused. This pipeline follows those links and pulls structured, factual data from Wikidata to enrich each entity automatically.

The result is a significantly richer KG — with birth dates, band members, founding locations, genres, awards, record labels, and more — without any manual data entry.

---

## The problem it solves

The KG was built from exhibition text, which is rich in context but sparse in facts. An item might reference "David Bowie" as a subject, but the KG entry for Bowie would only have a label and a Wikidata link. The pipeline fills that gap — pulling structured attributes for every entity the KG knows about.

---

## How it works

### 1. The frontier

The pipeline starts by scanning the KG for every entity that has a Wikidata link. These are collected into a **frontier** — a queue of entities waiting to be enriched. Each entity in the frontier tracks whether it has been enriched, what round it was discovered in, and what new entities it led to.

### 2. Type resolution

Every entity in Wikidata has a type declared via a property called `P31` (instance of). For example:
- David Bowie → `P31 = Q5` (human)
- The Rolling Stones → `P31 = Q215380` (rock band)
- Glastonbury Festival → `P31 = Q868557` (music festival)

We maintain a **type mapping table** (`type_mapping.json`) with 138 entries that translates these Wikidata type codes into our schema.org-aligned KG types:

| Wikidata P31 | KG Type |
|---|---|
| Q5 (human) | schema:Person |
| Q215380 (rock band) | schema:MusicGroup |
| Q482994 (studio album) | schema:MusicAlbum |
| Q483110 (venue) | schema:Venue |
| Q868557 (music festival) | schema:Event |

For entities with unusual or conflicting types, we maintain a separate **overrides table** (`entity_type_overrides.csv`) with 11 hand-curated entries for edge cases like the Grammy Awards, Buckingham Palace, and the British Invasion.

### 3. Expansion journeys

Once we know what type an entity is, we know exactly which Wikidata properties to fetch. We call these **journeys** — one per entity type. Each journey defines:

- Which attributes to pull (e.g. for a Person: birth date, nationality, occupation, genre, band memberships)
- Which attributes are purely informational vs. which point to other entities worth expanding
- Where to stop (terminal nodes — countries, cities, genres — that don't need further enrichment)

There are 7 journeys covering: Person, MusicGroup, MusicAlbum, MusicRecording, ConcertTour, Event/Concert, and Venue.

### 4. Loop protection

The graph is naturally cyclic — a band has members, each member belongs to a band, which has members, and so on. The pipeline prevents infinite loops by maintaining a **seen set**: every entity QID is processed exactly once. Once an entity is enriched, it is never revisited. New entities discovered during enrichment are queued but only if they haven't been seen before.

### 5. Chunked execution

The pipeline runs in controlled chunks of configurable size (currently 500 entities per run). This keeps each run predictable and interruptible. The frontier is saved to disk after every single entity, so runs can be stopped and resumed at any point without losing progress.

### 6. Merge back

After each chunk, the new triples are merged back into the KG. The original KG file is never overwritten — a new expanded file is written each time.

---

## What the output looks like

For each enriched entity, a small TTL file is written with the new triples. For example, for David Bowie:

```turtle
<http://www.wikidata.org/entity/Q5383>
    a schema:Person ;
    rdfs:label "David Bowie"@en ;
    schema:birthDate "1947-01-08"^^xsd:date ;
    schema:deathDate "2016-01-10"^^xsd:date ;
    schema:birthPlace <http://www.wikidata.org/entity/Q42462> ;
    schema:nationality <http://www.wikidata.org/entity/Q145> ;
    schema:genre <http://www.wikidata.org/entity/Q11399> ;
    schema:memberOf <http://www.wikidata.org/entity/Q128245> ;
    schema:sameAs "69178282-44dd-4931-b9bb-93e21f5e3c3c" .
```

---

## Current state

| Stage | Status |
|---|---|
| Type mapping (138 P31 entries) | Complete |
| Entity overrides (11 edge cases) | Complete |
| 7 expansion journeys specified | Complete |
| Loop protection | Complete |
| Chunked run with resume | Complete |
| Round 1 — seed entities (Person + MusicGroup) | Complete — 378 enriched, +4,884 triples |
| Round 2+ — discovered stubs (all types) | In progress |

### Numbers so far
- Seed KG: **9,415 triples**
- After round 1: **14,299 triples**
- Frontier: **2,600+ entities** queued (seed + discovered stubs)

---

## Open questions for the team

1. **MusicComposition** — the KG has 151 `MusicComposition` entities (songs as works, not recordings). Do we treat these the same as `MusicRecording` (performed instance), or do we want to enrich them separately with composer, lyricist, year written? This is the one schema decision still open.

2. **Review-needed entities** — after each run, some entities can't be typed because their Wikidata P31 code isn't in our mapping table. Currently ~29 per run. These are logged in the report CSV. Do we want to review these as a batch and add them to the mapping, or leave them unresolved?

3. **Round 2 scope** — the 1,500+ newly discovered stubs include band members, album artists, venues, and events. Do we enrich all of them, or gate by type (e.g. only enrich newly discovered Persons and MusicGroups, skip the rest)?

4. **Salience signals** — we're pulling award nominations, notable works, and event participations as `salience-candidate` attributes. These are intended as signals for competency questions like "most famous" or "most missed". Is the team happy with this approach, or should salience scoring be a separate step?

---

## Files

```
kg_expansion/
├── README.md                      — step-by-step run guide
├── EXPANSION_JOURNEYS.md          — full journey specs with diagrams
├── PIPELINE_OVERVIEW.md           — this document
├── run_expansion.sh               — single script to run everything
├── build_frontier.py              — scans KG, builds entity queue
├── kg_expander.py                 — fetches from Wikidata, writes TTL, reports
├── aggregate_expansion.py         — merges TTL files back into KG
├── enrich_hmos.py                 — post-merge HMO enrichment (genre, instrument, relatedTo) ← RUN AFTER EVERY AGGREGATION
├── type_mapping.json              — 138 Wikidata P31 → schema.org type mappings
├── entity_type_overrides.csv      — 11 hand-curated edge case assignments
├── frontier.json                  — live queue state (updated every run)
├── ttl_expanded/                  — per-entity enrichment output
└── reports/                       — per-run CSV report
```

---

## ⚠️ Post-expansion step — run every time

After `aggregate_expansion.py` produces `knowledge_graph_full.ttl` and/or `knowledge_graph_full_2hop.ttl`, always run:

```bash
python kg_expansion/enrich_hmos.py \
    --kg ../../assets/kg_triple_store_gt/knowledge_graph_full_2hop.ttl \
    --kg ../../assets/kg_triple_store_gt/knowledge_graph_full_2hop.ttl
```

This propagates artist genre and instrument data onto museum objects, and adds
same-case sibling links. Without this step, genre-based and instrument-based
routing in Lumi/Sonora will have gaps.

Use `--patch-dir kg_expansion/patches` to write separate patch TTLs instead of
modifying the KG in place (useful for auditing what changed).
