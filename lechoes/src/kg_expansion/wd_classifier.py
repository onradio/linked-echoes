"""
wd_classifier.py

Tier-1 / Tier-2 hierarchical type classifier for Wikidata entities.

Tier 1 — full enrichment (Person, MusicGroup, MusicAlbum, MusicRecording,
          Event, ConcertTour, Venue): resolved via SPARQL ASK subclass walk
          against anchor QIDs, with blocker vetoes, max depth 4.

Tier 2 — everything else: fetch label + description + P31 label only,
          write as terminal_reference. Never queued into frontier.

Promotion rule: P31 classes appearing ≥10 times in Tier 2 are flagged
for human review after each run.

Usage (standalone regression test):
    python kg_expansion/wd_classifier.py \
        --type-mapping kg_expansion/type_mapping.json \
        --overrides kg_expansion/entity_type_overrides.csv \
        --regression
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Wikidata Entity Data API — used for P279 walks (avoids WDQS rate limits)
# ---------------------------------------------------------------------------
WD_API = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WD_HEADERS = {"User-Agent": "BME-KG-Classifier/1.0 (https://github.com/bme-kg; research project; Python/urllib)"}

# ---------------------------------------------------------------------------
# Tier 1 configuration
# ---------------------------------------------------------------------------
# Each type has:
#   anchors  — QIDs that are the root classes for this type
#   blockers — QIDs that veto a match even if the anchor walk succeeds
#   priority — lower = higher priority when multiple types match
# ---------------------------------------------------------------------------
TIER1 = {
    "Person": {
        "priority": 1,
        "anchors": [
            "Q5",          # human
            "Q101352",     # person (generic)
            "Q2338524",    # musical artist
            "Q22808320",   # singer-songwriter
            "Q65115256",   # stage persona
            "Q89050960",   # alter ego
        ],
        "blockers": [],
    },
    "MusicGroup": {
        "priority": 2,
        "anchors": [
            "Q215380",    # rock band
            "Q5741069",   # music group
            "Q2088357",   # musical ensemble
            "Q54982412",  # songwriting team
            "Q84310035",  # record production team
        ],
        "blockers": [
            "Q5",       # don't match individuals as groups
        ],
    },
    "MusicAlbum": {
        "priority": 3,
        "anchors": [
            "Q482994",    # studio album
            "Q134556",    # single
            "Q169930",    # extended play
            "Q208569",    # compilation album
            "Q209939",    # live album
        ],
        "blockers": [],
    },
    "MusicRecording": {
        "priority": 4,
        "anchors": [
            "Q7366",       # song
            "Q55850593",   # audio track
            "Q55850643",   # music track without lyrics
        ],
        "blockers": [
            "Q134556",    # single is a release (MusicAlbum), not a recording
            "Q482994",    # studio album
        ],
    },
    "ConcertTour": {
        "priority": 5,
        "anchors": [
            "Q1573906",   # concert tour
        ],
        "blockers": [],
    },
    "Event": {
        "priority": 6,
        "anchors": [
            "Q182832",    # concert
            "Q868557",    # music festival
            "Q1955280",   # music competition
            "Q3267702",   # concert (alt)
            "Q1573906",   # concert tour (catches tours in event context)
            "Q17343829",  # concert series
            "Q17633526",  # charity event
        ],
        "blockers": [
            "Q215380",    # band — don't route music groups as events
            "Q5",         # person
            "Q11424",     # film
            "Q58483083",  # musical film
        ],
    },
    "Venue": {
        "priority": 7,
        "anchors": [
            "Q483110",    # venue
            "Q8719053",   # music venue
            "Q1060829",   # concert hall
            "Q27951514",  # indoor arena
            "Q641226",    # arena
            "Q24354",     # theatre
            "Q12819564",  # nightclub
            "Q1763828",   # multi-purpose hall
            "Q746369",    # recording studio
            "Q212198",    # public house
            "Q25550691",  # guildhall
            "Q96083918",  # former movie theater
        ],
        "blockers": [
            "Q3918",      # university
            "Q5341295",   # educational organisation
            "Q4830453",   # business (generic — too broad)
            "Q14092",     # gym
            "Q44782",     # sports facility / airport
            "Q1076486",   # sports venue (non-music)
            "Q55488",     # railway station
            "Q928830",    # metro station / underground station
            "Q11707",     # restaurant
            "Q1048525",   # hotel
        ],
    },
}

# Priority order for conflict resolution (lower = wins)
TIER1_PRIORITY = {t: cfg["priority"] for t, cfg in TIER1.items()}

MAX_DEPTH = 4
API_DELAY = 0.5   # seconds between entity data API calls


# ---------------------------------------------------------------------------
# Entity Data API helpers — used for P279 walks (avoids WDQS rate limits)
# ---------------------------------------------------------------------------

_entity_cache: dict[str, dict | None] = {}


def _fetch_entity_data(qid: str, retries: int = 3) -> dict | None:
    """Fetch entity JSON from Wikidata Entity Data API with caching."""
    if qid in _entity_cache:
        return _entity_cache[qid]
    url = WD_API.format(qid=qid)
    req = urllib.request.Request(url, headers=WD_HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            entity = data["entities"].get(qid)
            _entity_cache[qid] = entity
            time.sleep(API_DELAY)
            return entity
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = [10, 30, 60][attempt]
                print(f"    [api] 429 for {qid} — wait {wait}s")
                time.sleep(wait)
            else:
                time.sleep(5)
        except Exception as e:
            time.sleep(5)
    _entity_cache[qid] = None
    return None


def _get_p279_qids(entity: dict) -> list[str]:
    """Get direct P279 (subclass of) values from an entity."""
    qids = []
    for claim in entity.get("claims", {}).get("P279", []):
        try:
            qids.append(claim["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            pass
    return qids


# ---------------------------------------------------------------------------
# P279 subclass walk — BFS up to max_depth hops
# Checks whether start_qid is a subclass of target_qid within depth limit.
# ---------------------------------------------------------------------------

_subclass_cache: dict[tuple, bool] = {}


def is_subclass_of(child_qid: str, parent_qid: str, max_depth: int = MAX_DEPTH) -> bool:
    """
    BFS walk up P279 chain from child_qid.
    Returns True if parent_qid is reachable within max_depth hops.
    Uses entity data API — no SPARQL needed.
    """
    cache_key = (child_qid, parent_qid, max_depth)
    if cache_key in _subclass_cache:
        return _subclass_cache[cache_key]

    visited: set[str] = set()
    frontier = [child_qid]
    depth = 0

    while frontier and depth <= max_depth:
        next_frontier = []
        for qid in frontier:
            if qid in visited:
                continue
            visited.add(qid)
            if qid == parent_qid:
                _subclass_cache[cache_key] = True
                return True
            entity = _fetch_entity_data(qid)
            if entity:
                next_frontier.extend(_get_p279_qids(entity))
        frontier = next_frontier
        depth += 1

    _subclass_cache[cache_key] = False
    return False


def _cached_subclass(child: str, parent: str) -> bool:
    if child == parent:
        return True
    return is_subclass_of(child, parent)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class ClassificationResult:
    def __init__(self, tier: int, normalized_type: str, role: str,
                 matched_anchor: str = "", match_source: str = "",
                 p31_qids: list = None):
        self.tier = tier
        self.normalized_type = normalized_type
        self.role = role
        self.matched_anchor = matched_anchor
        self.match_source = match_source
        self.p31_qids = p31_qids or []

    def __repr__(self):
        return (f"ClassificationResult(tier={self.tier}, type={self.normalized_type}, "
                f"role={self.role}, anchor={self.matched_anchor}, source={self.match_source})")


def classify(entity_qid: str, p31_qids: list[str], overrides: dict,
             type_mapping: dict | None = None) -> ClassificationResult:
    """
    Classify an entity into Tier 1 or Tier 2.

    Resolution order:
      1. Entity-level override (entity_type_overrides.csv)
      2. Exact P31 match in type_mapping.json (hand-reviewed, authoritative)
      3. Tier 1 anchor walk via P279 BFS (with blocker check)
      4. Tier 2 — terminal_reference

    Returns ClassificationResult.
    """
    # 1. Override
    if entity_qid in overrides:
        norm_type = overrides[entity_qid]
        tier = 1 if norm_type in TIER1 else 2
        return ClassificationResult(
            tier=tier,
            normalized_type=norm_type,
            role="enrichment_target" if tier == 1 else "terminal_reference",
            matched_anchor="override",
            match_source="entity_override",
            p31_qids=p31_qids,
        )

    if not p31_qids:
        return ClassificationResult(
            tier=2,
            normalized_type="Unknown",
            role="terminal_reference",
            match_source="no_p31",
            p31_qids=[],
        )

    # 2. Exact match in type_mapping.json — hand-reviewed, authoritative
    if type_mapping:
        for p31_qid in p31_qids:
            if p31_qid in type_mapping:
                entry = type_mapping[p31_qid]
                norm_type = entry["normalized_type"]
                role = entry.get("role", "terminal_reference")
                tier = 1 if norm_type in TIER1 else 2
                return ClassificationResult(
                    tier=tier,
                    normalized_type=norm_type,
                    role=role if tier == 1 else "terminal_reference",
                    matched_anchor=p31_qid,
                    match_source="type_mapping_exact",
                    p31_qids=p31_qids,
                )

    # 3. Tier 1 walk — collect all matches
    matches = []  # list of (type_name, anchor_qid, p31_qid, priority)

    for p31_qid in p31_qids:
        for type_name, cfg in TIER1.items():
            # Check each anchor
            for anchor in cfg["anchors"]:
                hit = _cached_subclass(p31_qid, anchor)
                if not hit:
                    continue

                # Blocker check
                blocked = False
                for blocker in cfg["blockers"]:
                    if _cached_subclass(p31_qid, blocker):
                        blocked = True
                        break

                if not blocked:
                    matches.append((type_name, anchor, p31_qid, cfg["priority"]))
                    break  # first anchor match per type is enough

    if not matches:
        # 4. Tier 2
        return ClassificationResult(
            tier=2,
            normalized_type="Unknown",
            role="terminal_reference",
            match_source="no_tier1_match",
            p31_qids=p31_qids,
        )

    # Resolve conflicts — lowest priority number wins
    matches.sort(key=lambda m: m[3])
    best_type, best_anchor, best_p31, _ = matches[0]

    # Check for genuine ambiguity (two different types at same priority — shouldn't happen
    # given the priority numbering, but guard it)
    if len(matches) > 1 and matches[0][3] == matches[1][3] and matches[0][0] != matches[1][0]:
        return ClassificationResult(
            tier=1,
            normalized_type="Ambiguous",
            role="review_needed",
            matched_anchor=f"{matches[0][1]}+{matches[1][1]}",
            match_source="priority_conflict",
            p31_qids=p31_qids,
        )

    return ClassificationResult(
        tier=1,
        normalized_type=best_type,
        role="enrichment_target",
        matched_anchor=best_anchor,
        match_source=f"p31:{best_p31}→anchor:{best_anchor}",
        p31_qids=p31_qids,
    )


# ---------------------------------------------------------------------------
# Tier 2 — minimal TTL writer (label + description + P31 label + sameAs)
# ---------------------------------------------------------------------------

def tier2_triples(entity_qid: str, entity: dict, p31_qids: list[str]) -> list[tuple]:
    """
    Return a minimal list of (subject, predicate, object_str, obj_type) tuples
    for Tier 2 entities. The caller is responsible for writing to RDF.
    """
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import RDF, RDFS, OWL, XSD

    SCHO = Namespace("http://schema.org/")
    WD = Namespace("http://www.wikidata.org/entity/")

    uri = URIRef(f"http://www.wikidata.org/entity/{entity_qid}")
    g = Graph()
    g.bind("schema", SCHO)
    g.bind("wd", WD)
    g.bind("rdfs", RDFS)

    label = entity.get("labels", {}).get("en", {}).get("value", "")
    desc = entity.get("descriptions", {}).get("en", {}).get("value", "")

    if label:
        g.add((uri, RDFS.label, Literal(label, lang="en")))
    if desc:
        g.add((uri, SCHO.description, Literal(desc, lang="en")))

    # sameAs back to WD
    g.add((uri, OWL.sameAs, URIRef(f"http://www.wikidata.org/entity/{entity_qid}")))

    # P31 QIDs as rdfs:seeAlso (informational only)
    from rdflib.namespace import RDFS
    for p31_qid in p31_qids:
        g.add((uri, RDFS.seeAlso, URIRef(f"http://www.wikidata.org/entity/{p31_qid}")))

    return g


# ---------------------------------------------------------------------------
# Promotion tracker — P31 counter for Tier 2 entities
# ---------------------------------------------------------------------------

class Tier2Tracker:
    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self._p31_counter: Counter = Counter()

    def record(self, p31_qids: list[str]):
        for q in p31_qids:
            self._p31_counter[q] += 1

    def promotion_candidates(self) -> list[tuple[str, int]]:
        return [(qid, count) for qid, count in self._p31_counter.most_common()
                if count >= self.threshold]

    def report(self) -> str:
        candidates = self.promotion_candidates()
        if not candidates:
            return "No Tier 2 P31 classes crossed the promotion threshold."
        lines = [f"Tier 2 P31 classes ≥{self.threshold} occurrences — review for promotion:"]
        for qid, count in candidates:
            lines.append(f"  {qid:<15} {count:4d}  https://www.wikidata.org/wiki/{qid}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------

def run_regression(type_mapping_path: Path, overrides_path: Path):
    """
    Run the new classifier against every row in type_mapping.csv.
    Report disagreements between the classifier output and the hand-reviewed decision.
    """
    import sys

    type_mapping = json.loads(type_mapping_path.read_text())
    overrides = {}
    if overrides_path.exists():
        with open(overrides_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row.get("Entity QID", "").strip()
                ntype = row.get("Assigned KG type", "").strip()
                if qid and ntype:
                    overrides[qid] = ntype

    rows = list(type_mapping.items())
    total = len(rows)
    agreements = 0
    disagreements = []
    all_rows = []

    print(f"[regression] Testing {total} type_mapping.json entries against Tier1/Tier2 classifier...")
    print(f"[regression] This will make SPARQL calls — expect ~{total * 0.5:.0f}s minimum\n")

    for i, (p31_qid, tm_entry) in enumerate(rows, 1):
        expected = tm_entry["normalized_type"]
        expected_role = tm_entry["role"]

        print(f"  [{i}/{total}] {p31_qid} ({tm_entry['p31_label']}) ...", end=" ", flush=True)

        # Classify treating this P31 QID as the entity's P31
        # Pass type_mapping so exact-match step fires (self-referential: each entry should match itself)
        result = classify(entity_qid="__regression__", p31_qids=[p31_qid], overrides={},
                          type_mapping=type_mapping)

        # Agreement: type matches OR both are terminal
        tier2_types = {"terminal_context", "terminal_reference"}
        expected_is_terminal = expected_role in tier2_types or expected in {
            "Country", "City", "DefinedTerm", "Organisation", "Award",
            "Film", "TVShow", "Material", "MusicGenre", "Island",
        }
        result_is_terminal = result.tier == 2

        agreed = result.normalized_type == expected or (expected_is_terminal and result_is_terminal)
        row = {
            "p31_qid": p31_qid,
            "p31_label": tm_entry["p31_label"],
            "expected": expected,
            "expected_role": expected_role,
            "classifier_type": result.normalized_type,
            "classifier_role": result.role,
            "match_source": result.match_source,
            "result": "pass" if agreed else "fail",
        }
        if agreed:
            agreements += 1
            label = expected if result.normalized_type == expected else f"both terminal ({expected}/{result.normalized_type})"
            print(f"✓ {label}")
        else:
            disagreements.append(row)
            print(f"✗ expected={expected} got={result.normalized_type} ({result.match_source})")
        all_rows.append(row)

        time.sleep(API_DELAY)

    # Always write full report
    report_path = Path("kg_expansion/regression_report.csv")
    fieldnames = ["p31_qid", "p31_label", "expected", "expected_role",
                  "classifier_type", "classifier_role", "match_source", "result"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"  REGRESSION RESULTS")
    print(f"{'='*60}")
    print(f"  Total entries : {total}")
    print(f"  Agreements    : {agreements} ({100*agreements/total:.1f}%)")
    print(f"  Disagreements : {len(disagreements)}")
    print(f"  Report        : {report_path}")
    if disagreements:
        print(f"\n  Disagreements to resolve:")
        for d in disagreements:
            print(f"    {d['p31_qid']:<15} expected={d['expected']:<20} got={d['classifier_type']}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Golden set acceptance test — independent of type_mapping.json
# ---------------------------------------------------------------------------

def run_golden(golden_path: Path, type_mapping_path: Path, overrides: dict):
    """
    Test classifier against golden_set.csv — hand-curated ground truth
    that is independent of type_mapping.json.

    expected_type conventions:
      - A KG type name (Person, Venue, MusicAlbum, etc.) → classifier must return that type
      - NOT_VENUE  → classifier must NOT return Venue
      - NOT_EVENT  → classifier must NOT return Event
      - Tier 2 rows (expected_tier=2) → classifier must return tier=2

    This catches errors that the self-referential regression cannot:
    e.g. type_mapping.json saying gym→Venue when gym should be blocked.
    """
    type_mapping = json.loads(type_mapping_path.read_text())

    with open(golden_path, newline="", encoding="utf-8") as f:
        golden_rows = list(csv.DictReader(f))

    total = len(golden_rows)
    passes = 0
    failures = []
    all_rows = []

    print(f"[golden] Testing {total} hand-curated entries from {golden_path} ...")
    print(f"[golden] Fetching from Wikidata — will take ~{total * 0.5:.0f}s minimum\n")

    for i, row in enumerate(golden_rows, 1):
        qid = row["qid"].strip()
        label = row["label"].strip()
        expected = row["expected_type"].strip()
        expected_tier = int(row["expected_tier"].strip())
        notes = row.get("notes", "")

        print(f"  [{i}/{total}] {qid} ({label}) ...", end=" ", flush=True)

        test_mode = row.get("test_mode", "entity").strip()

        if test_mode == "p31_direct":
            # Treat the QID itself as a P31 value — no Wikidata fetch needed
            p31_qids = [qid]
            result = classify(entity_qid="__golden__", p31_qids=p31_qids, overrides=overrides,
                              type_mapping=type_mapping)
        else:
            # Fetch real entity from Wikidata and use its P31 claims
            entity = _fetch_entity_data(qid)
            if entity is None:
                print("SKIP — fetch failed")
                all_rows.append({**row, "classifier_type": "fetch_failed",
                                  "classifier_tier": "", "result": "skip"})
                continue

            p31_qids = []
            for claim in entity.get("claims", {}).get("P31", []):
                try:
                    p31_qids.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                except (KeyError, TypeError):
                    pass

            result = classify(entity_qid=qid, p31_qids=p31_qids, overrides=overrides,
                              type_mapping=type_mapping)

        # Evaluate
        if expected.startswith("NOT_"):
            forbidden = expected[4:]  # e.g. "VENUE" → "Venue"
            # Capitalise first letter only
            forbidden_type = forbidden[0] + forbidden[1:].lower()
            passed = result.normalized_type != forbidden_type
            verdict = "pass" if passed else f"FAIL — got {result.normalized_type}, must not be {forbidden_type}"
        elif expected_tier == 2:
            passed = result.tier == 2
            verdict = "pass" if passed else f"FAIL — expected Tier 2 got Tier {result.tier} ({result.normalized_type})"
        else:
            passed = result.normalized_type == expected
            verdict = "pass" if passed else f"FAIL — expected {expected} got {result.normalized_type}"

        if passed:
            passes += 1
            print(f"✓ {verdict if verdict == 'pass' else ''}{result.normalized_type}")
        else:
            failures.append({**row, "classifier_type": result.normalized_type,
                              "classifier_tier": result.tier,
                              "match_source": result.match_source,
                              "result": "fail", "verdict": verdict})
            print(f"✗ {verdict}")

        all_rows.append({**row, "classifier_type": result.normalized_type,
                         "classifier_tier": result.tier,
                         "match_source": result.match_source,
                         "result": "pass" if passed else "fail"})

        time.sleep(API_DELAY)

    # Write full report
    report_path = Path("kg_expansion/golden_report.csv")
    fieldnames = list(golden_rows[0].keys()) + ["classifier_type", "classifier_tier",
                                                  "match_source", "result"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"  GOLDEN SET RESULTS")
    print(f"{'='*60}")
    print(f"  Total   : {total}")
    print(f"  Pass    : {passes}")
    print(f"  Fail    : {len(failures)}")
    print(f"  Report  : {report_path}")
    if failures:
        print(f"\n  Failures:")
        for f_ in failures:
            print(f"    {f_['qid']:<12} {f_['label']:<35} {f_['verdict']}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WD Tier-1/2 Classifier")
    parser.add_argument("--type-mapping", type=Path,
                        default=Path("kg_expansion/type_mapping.json"))
    parser.add_argument("--overrides", type=Path,
                        default=Path("kg_expansion/entity_type_overrides.csv"))
    parser.add_argument("--regression", action="store_true",
                        help="Run regression test against type_mapping.json")
    parser.add_argument("--golden", type=Path, default=None,
                        help="Run acceptance test against golden_set.csv (independent ground truth)")
    parser.add_argument("--qid", default=None,
                        help="Classify a single QID (for testing)")
    parser.add_argument("--p31", default=None,
                        help="Comma-separated P31 QIDs to classify against")
    args = parser.parse_args()

    overrides = {}
    if args.overrides.exists():
        with open(args.overrides, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row.get("Entity QID", "").strip()
                ntype = row.get("Assigned KG type", "").strip()
                if qid and ntype:
                    overrides[qid] = ntype

    if args.golden:
        run_golden(args.golden, args.type_mapping, overrides)

    elif args.regression:
        run_regression(args.type_mapping, args.overrides)

    elif args.qid and args.p31:
        p31_qids = [q.strip() for q in args.p31.split(",")]
        print(f"Classifying {args.qid} with P31={p31_qids} ...")
        result = classify(args.qid, p31_qids, overrides)
        print(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
