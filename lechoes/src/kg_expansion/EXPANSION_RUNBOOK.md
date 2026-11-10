# KG Expansion Runbook

> **Retired (12 Oct 2026):** `knowledge_graph_full.ttl` (1-hop) is no longer produced or distributed. Use `knowledge_graph_full_2hop.ttl`.

> **Moved (Nov 2026):** this runbook records the BME run. The graph, frontier and expansion files it names (`assets/kg_triple_store_gt/`, `frontier_gt*.json`, `expansion_gt*.ttl`) are BME data and now live in `onradio/bme-kg` (currently private), under `build/`, not in this repository. The commands are unchanged, but pass the files' new locations explicitly.


Wikidata enrichment pipeline for `assets/kg_triple_store_gt/knowledge_graph.ttl`.

Run all commands from `lechoes/src/`.

---

## Overview

```
GT KG (knowledge_graph.ttl)
    ↓  build_frontier.py
frontier_gt.json  [seed: 651 entities from scho:about]
    ↓  kg_expander.py  (round 1)
ttl_gt/  [per-entity TTL files]  +  frontier grows with new stubs
    ↓  kg_expander.py  (round 2+)
ttl_gt/  [round 2 stubs enriched]
    ↓  aggregate_expansion.py
expansion_gt.ttl  [merged enrichment layer]
    ↓  merge
knowledge_graph_full.ttl  [GT KG + Wikidata expansion]
```

---

## Step 1 — Build Frontier

Scans the GT KG for all `wd:` URIs referenced via `scho:about`. Outputs a seed frontier.

```bash
python3 kg_expansion/build_frontier.py \
  --kg ../../assets/kg_triple_store_gt/knowledge_graph.ttl \
  --output kg_expansion/frontier_gt.json
```

**Output:** `frontier_gt.json` — `{qid: {uri, label, types, enriched: false}}`

**Seed counts (current GT KG):**
| KG Type | Count |
|---|---|
| Person | 246 |
| MusicComposition | 151 |
| MusicAlbum | 68 |
| Place | 63 |
| DefinedTerm | 54 |
| Event | 34 |
| EventVenue | 31 |
| Q214609 (material) | 19 |
| Country | 10 |
| **Total** | **651** |

**Note:** `MusicComposition` → aliased to `MusicRecording` internally. `EventVenue` → aliased to `Venue`.

---

## Step 2 — Run Expansion

Iterative enrichment. Each run picks up all unenriched frontier entries, fetches from Wikidata, classifies, writes TTL, and adds new stubs to the frontier. Run the same command for each round — it resumes automatically.

```bash
python3 kg_expansion/kg_expander.py \
  --frontier kg_expansion/frontier_gt.json \
  --output-dir kg_expansion/ttl_gt \
  --report-dir kg_expansion/reports \
  --max-rounds 3 \
  --delay 0.5
```

**Options:**
| Flag | Default | Notes |
|---|---|---|
| `--frontier` | `frontier.json` | Frontier file to use |
| `--output-dir` | `ttl_expanded/` | Per-entity TTL output |
| `--report-dir` | `reports/` | CSV report per run |
| `--max-rounds` | 2 | Max frontier depth; round 1 = seeds, round 2 = stubs from seeds |
| `--delay` | 1.0 | Seconds between WD API fetches |
| `--max-entities` | None | Limit for test runs |
| `--type-filter` | None | e.g. `Person,MusicGroup` |
| `--dry-run` | False | Fetch + classify but don't write TTL |

**Run results (2026-07-07, GT KG):**

| Round | Processed | Enriched | Terminal | Unknown/Other | New Stubs Added |
|---|---|---|---|---|---|
| 1 | 355 | 342 (96%) | 11 | 2 | 1,649 |
| 2 | 1,649 | 822 (50%) | 823 | 550 Unknown | ~1,510 |

Round 2's 550 "Unknown" = stubs whose P31 didn't resolve in `type_mapping.json` (genre nodes, abstract concepts, some places). These get written as minimal terminal TTL files.

**Frontier state after round 2:**
| Status | Count |
|---|---|
| enriched | 1,164 |
| terminal_reference | 834 |
| terminal (pre-skipped) | 296 |
| none (pending round 3) | 1,510 |
| fetch_failed | 4 |
| review_needed | 2 |
| **Total** | **3,810** |

**TTL files produced:** 1,998 files in `ttl_gt/`

---

## Step 3 — Aggregate

Merges all per-entity TTL files into a single enrichment layer.

```bash
python3 kg_expansion/aggregate_expansion.py \
  --ttl-dir kg_expansion/ttl_gt \
  --output kg_expansion/expansion_gt.ttl
```

---

## Step 4 — Merge with GT KG

Combine the museum object graph with the Wikidata enrichment layer.

```bash
# Simple merge via rdflib or cat (TTL files are self-contained)
python3 -c "
from rdflib import ConjunctiveGraph
g = ConjunctiveGraph()
g.parse('../../assets/kg_triple_store_gt/knowledge_graph.ttl')
g.parse('kg_expansion/expansion_gt.ttl')
g.serialize('../../assets/kg_triple_store_gt/knowledge_graph_full.ttl', format='turtle')
print(f'Total triples: {len(g)}')
"
```

---

## Architecture Notes

### Classification (`wd_classifier.py`)
- Lookup order: entity override → exact P31 match in `type_mapping.json` → BFS anchor walk (depth 4) → Tier 2 terminal
- Tier 1 = full PROPERTY_MAP enrichment; Tier 2 = label + description + sameAs only
- `golden_set.csv` — 51 hand-curated test cases; run with `--golden`

### PROPERTY_MAP decisions
Full rationale in `bme_kg_property_strategy.xlsx`. Key decisions:
- **P1303 instrument** → `schema:instrument` via MIMO two-hop (WD P1303 → instrument QID → P3763 → MIMO URI). Fallback to WD URI if no MIMO ID.
- **P737 influenced by** → `schema:isRelatedTo` (no clean schema.org fit; flagged for future review)
- **P800 notable work** → **not in PROPERTY_MAP**. Separate `catalogue_expansion` pass planned: pull works as stubs, link via `schema:byArtist` (not direct artist→work triple)
- **P710 participant** → dropped (P175 `schema:performer` covers core CQ)
- **P131 located in** → dropped everywhere (inconsistent WD granularity)
- **Concert** → collapsed into Event; **NightClub/Theatre** → collapsed into Venue

### File separation
| File | Contents |
|---|---|
| `knowledge_graph.ttl` | Museum object graph — HumanMadeObjects, Cases, Zones, Exhibition structure |
| `expansion_gt.ttl` | Wikidata enrichment layer — properties per entity type |
| `knowledge_graph_full.ttl` | Merged final KG |
| `catalogue_expansion.ttl` | (planned) Notable works pulled via P800 for bot QA coverage |
| `bme_knowledge_graph.ttl` | Separate museum file — keep as-is, do not merge |

### KG Type Aliases
The GT KG uses different type labels than the expander's normalized types:
| GT KG Type | Expander Type |
|---|---|
| `MusicComposition` | `MusicRecording` |
| `EventVenue` | `Venue` |

### Known issues
- **550 "Unknown" in round 2** — stubs whose P31 not in `type_mapping.json`. Written as minimal terminal TTL. Run `wd_type_audit.py` to identify patterns and extend mappings.
- **4 fetch_failed** — transient WD API errors. Re-run will retry automatically.
- **1,510 pending (round 3)** — further stubs added by round 2; diminishing returns expected.
- **ConcertTour dates** — WD stores P580/P582 as qualifiers on P175 statement, not top-level. Current expander fetches top-level only → tour dates often absent.

---

## Useful Commands

```bash
# Check frontier state
python3 -c "
import json
f = json.loads(open('kg_expansion/frontier_gt.json').read())
from collections import Counter
statuses = Counter(v.get('status','none') for v in f.values())
for k,v in statuses.most_common(): print(f'  {k}: {v}')
print(f'Total: {len(f)}')
"

# Dry run — first 20 entities
python3 kg_expansion/kg_expander.py \
  --frontier kg_expansion/frontier_gt.json \
  --output-dir kg_expansion/ttl_gt \
  --max-entities 20 --dry-run

# Type-filtered run (Person + MusicGroup only)
python3 kg_expansion/kg_expander.py \
  --frontier kg_expansion/frontier_gt.json \
  --output-dir kg_expansion/ttl_gt \
  --type-filter Person,MusicGroup

# Run golden test
python3 kg_expansion/wd_classifier.py --golden

# Run regression test
python3 kg_expansion/wd_classifier.py --regression
```
