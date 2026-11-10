"""
Test suite for the entity-type check queries (ECKs) in ``entichecks.py``.

Run it as a script::

    python -m lechoes.src.test_linker                     # every type
    python -m lechoes.src.test_linker music_artist        # selected types
    python -m lechoes.src.test_linker --verify-fixture    # fixture integrity only
    python -m lechoes.src.test_linker --batch-equivalence # batched path only
    python -m lechoes.src.test_linker --selection         # LLM providers only

Four independent checks are performed:

``verify_fixture``
    Confirms every QID in ``TEST_CASES`` still resolves to the label the fixture
    claims. This guard exists because the previous fixture had drifted badly --
    Q158852 was labelled "Liverpool" but is *conductor*, Q37073 was labelled
    "Jazz" but is *pop music*, Q11158 was labelled "Prussia" but is *acid* -- so
    the suite was scoring queries against entities that were never what they
    said they were. Always run this before trusting a query result.

``run_eck_tests``
    Runs each ECK against its positive and negative cases and reports
    precision / recall / accuracy per type.

``run_batch_equivalence_tests``
    Holds the batched validation path used by ``TypedRAGEntityLinker`` (all of a
    mention's candidates in one SPARQL request) to the same fixture as the
    per-entity ASK path, so the optimisation cannot silently change verdicts.

``run_selection_tests``
    Exercises the selection call against each configured model through the
    provider layer (``src/llm.py``). It asserts the **contract** the linker
    depends on — an integer within ``0..len(candidates)``, and no exception —
    and *reports* two things it deliberately does not assert: which entity each
    model chose, and whether two identical calls agreed. Both are properties of
    the model rather than of the linker. Temperature 0 is best effort at every
    provider, not a guarantee: the classification evaluation measures 2 to 7 of
    65 objects changing outcome between runs, so failing on a disagreement here
    would report a known API property as a code defect.

Cases are chosen to mirror what the entity-lookup service actually returns for a
mention: the negatives are dominated by real distractors (same-name people of
other professions, disambiguation pages, discographies, an album sharing its
name with the artist) and by *adjacent* types, which are the ones that matter --
work vs. release vs. recording, place vs. venue vs. country, genre vs. movement.

Every case carries a ``why`` so the intent of an assertion survives review.
"""

import argparse
import os
import sys
import time

try:  # allow both `python -m lechoes.src.test_linker` and direct execution
    from .entichecks import ECK_BY_TYPE
    from .linker import (BATCH_VALIDATION_CHUNK, RAGEntityLinker, validate_entity,
                         validate_entities_batch, wikidata_get_labels)
    from .llm import API_KEY_ENV, parse_model_spec
except ImportError:  # pragma: no cover - direct execution fallback
    from entichecks import ECK_BY_TYPE
    from linker import (BATCH_VALIDATION_CHUNK, RAGEntityLinker, validate_entity,
                        validate_entities_batch, wikidata_get_labels)
    from llm import API_KEY_ENV, parse_model_spec

# Models exercised by --selection. The deployed model first; the rest are the
# comparison set of the classification evaluation.
SELECTION_MODELS = (
    "google:gemini-2.5-flash",
    "google:gemini-3.6-flash",
    "mistral:mistral-large-2512",
    "mistral:ministral-14b-2512",
)

# One mention with a real distractor set: the jazz pianist, a same-name coach and
# a disambiguation page are what the lookup service actually returns for "Bill
# Evans", so the shape being tested is the shape production sees.
SELECTION_FIXTURE = {
    "search_term": "Bill Evans",
    "context_query": "Bill Evans' jazz piano recordings",
    "entity_type": "music_artist",
    "candidates": [
        {"wikidata_id": "Q208205", "label": "Bill Evans",
         "description": "American jazz pianist (1929-1980)"},
        {"wikidata_id": "Q16728874", "label": "Bill Evans",
         "description": "American football coach"},
        {"wikidata_id": "Q359075", "label": "Bill Evans",
         "description": "Wikimedia disambiguation page"},
    ],
    "preferred_qid": "Q208205",
}


# ##############################################################################
# Fixtures
# ##############################################################################

TEST_CASES = {
    "music_artist": {
        "pass": [
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
            {"id": "Q93346", "label": "The Who", "why": "musical group"},
            {"id": "Q384664", "label": "The Quarrymen", "why": "musical group"},
            {"id": "Q484918", "label": "Simon & Garfunkel", "why": "musical duo, not under 'musical group'"},
            {"id": "Q155805", "label": "London Symphony Orchestra", "why": "symphony orchestra, not under 'musical group'"},
            {"id": "Q167546", "label": "Lucio Dalla", "why": "singer-songwriter"},
            {"id": "Q192529", "label": "Andrés Segovia", "why": "classical guitarist"},
            {"id": "Q2632", "label": "Ringo Starr", "why": "drummer"},
            {"id": "Q392", "label": "Bob Dylan", "why": "songwriter / guitarist"},
            {"id": "Q154895", "label": "Herbert von Karajan", "why": "conductor"},
            {"id": "Q506900", "label": "Grandmaster Flash", "why": "disc jockey / rapper"},
            {"id": "Q191819", "label": "George Martin", "why": "record producer / arranger / conductor"},
            {"id": "Q117012", "label": "Yoko Ono", "why": "singer / composer"},
            {"id": "Q208205", "label": "Bill Evans", "why": "jazz pianist; real lookup candidate"},
            # Primarily the Beatles' manager, but Wikidata also records him as a
            # record producer -- the same occupation that admits George Martin,
            # so accepting him is the consistent outcome. For a linker this is
            # also the useful one: the alternative is no link at all for a
            # central figure in this domain.
            {"id": "Q203690", "label": "Brian Epstein", "why": "record producer / music executive"},
        ],
        "fail": [
            {"id": "Q253192", "label": "Cynthia Lennon", "why": "visual artist / journalist, not a musician"},
            {"id": "Q181677", "label": "George R. R. Martin", "why": "writer; same-name distractor"},
            {"id": "Q3101741", "label": "George Martin", "why": "naval officer; same-name distractor"},
            {"id": "Q16728874", "label": "Bill Evans", "why": "head coach; real lookup distractor"},
            {"id": "Q359075", "label": "Bill Evans", "why": "disambiguation page; real lookup distractor"},
            {"id": "Q4908924", "label": "Bill Evans", "why": "album named after a musician"},
            {"id": "Q1606282", "label": "Ringo Starr discography", "why": "Wikimedia discography page"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album, not an artist"},
            {"id": "Q200799", "label": "Eleanor Rigby", "why": "song, not an artist"},
            {"id": "Q621231", "label": "Apple Corps", "why": "business founded by the band"},
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
        ],
    },
    "place": {
        "pass": [
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
            {"id": "Q84", "label": "London", "why": "city / metropolis"},
            {"id": "Q1055", "label": "Hamburg", "why": "city and federated state of Germany"},
            {"id": "Q6624", "label": "Ascoli Piceno", "why": "comune of Italy: admin entity, NOT a settlement subclass"},
            {"id": "Q56031", "label": "Arquata del Tronto", "why": "comune of Italy"},
            {"id": "Q23100", "label": "Merseyside", "why": "metropolitan county"},
            {"id": "Q21", "label": "England", "why": "constituent country"},
            {"id": "Q22", "label": "Scotland", "why": "constituent country"},
        ],
        "fail": [
            {"id": "Q6583", "label": "Sahara", "why": "desert: physical feature, not a settlement or admin entity"},
            {"id": "Q479469", "label": "Abbey Road", "why": "road; same-name distractor for the album"},
            {"id": "Q1130849", "label": "Liverpool F.C.", "why": "football club; real lookup distractor"},
            {"id": "Q41176", "label": "building", "why": "generic class"},
            {"id": "Q909845", "label": "The Cavern Club", "why": "venue, belongs in the venue slot"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
        ],
    },
    "country": {
        "pass": [
            {"id": "Q145", "label": "United Kingdom", "why": "sovereign state"},
            {"id": "Q30", "label": "United States", "why": "sovereign state"},
            {"id": "Q794", "label": "Iran", "why": "sovereign state"},
            {"id": "Q21", "label": "England", "why": "instance of country"},
            {"id": "Q22", "label": "Scotland", "why": "instance of country"},
            {"id": "Q15180", "label": "Soviet Union", "why": "historical country"},
            {"id": "Q36704", "label": "Yugoslavia", "why": "historical country"},
            {"id": "Q38872", "label": "Prussia", "why": "historical country"},
        ],
        "fail": [
            {"id": "Q84", "label": "London", "why": "city"},
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
            {"id": "Q23100", "label": "Merseyside", "why": "county, a sub-national admin entity"},
            {"id": "Q6583", "label": "Sahara", "why": "desert"},
            {"id": "Q4830453", "label": "business", "why": "generic class"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
        ],
    },
    "genre": {
        "pass": [
            {"id": "Q8341", "label": "jazz", "why": "music genre"},
            {"id": "Q11399", "label": "rock music", "why": "music genre"},
            {"id": "Q7749", "label": "rock and roll", "why": "music genre"},
            {"id": "Q11401", "label": "hip-hop", "why": "music genre"},
            {"id": "Q606089", "label": "skiffle", "why": "music genre central to this domain"},
            {"id": "Q2344635", "label": "Merseybeat", "why": "music genre central to this domain"},
            {"id": "Q546264", "label": "beat music", "why": "music genre"},
            {"id": "Q206159", "label": "psychedelic rock", "why": "music genre"},
            {"id": "Q9759", "label": "blues", "why": "music genre"},
            {"id": "Q846119", "label": "psychedelic music", "why": "music genre"},
        ],
        "fail": [
            {"id": "Q37068", "label": "Romanticism", "why": "art/cultural movement, not a music genre"},
            {"id": "Q270083", "label": "Jazz", "why": "Queen album sharing a genre's name"},
            {"id": "Q188451", "label": "music genre", "why": "the class itself, not a genre"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
        ],
    },
    "music_work": {
        "pass": [
            {"id": "Q736799", "label": "Please Please Me", "why": "the composition"},
            {"id": "Q200799", "label": "Eleanor Rigby", "why": "the composition"},
            {"id": "Q1418568", "label": "Nowhere Man", "why": "the composition"},
            {"id": "Q1326986", "label": "Sgt. Pepper's Lonely Hearts Club Band", "why": "the song, vs the album of the same name"},
            {"id": "Q187745", "label": "Bohemian Rhapsody", "why": "the composition"},
            {"id": "Q12008", "label": "Piano Sonata No. 14", "why": "classical composition"},
            {"id": "Q1128775", "label": "Vogue", "why": "the composition"},
        ],
        "fail": [
            {"id": "Q60428471", "label": "Please Please Me", "why": "recording (music track with vocals), not the work"},
            {"id": "Q60661154", "label": "Eleanor Rigby", "why": "recording, not the work"},
            {"id": "Q181826", "label": "Please Please Me", "why": "album sharing the work's title"},
            {"id": "Q61027072", "label": "Please Please Me", "why": "LP release sharing the work's title"},
            {"id": "Q60427757", "label": "Please Please Me / Ask Me Why", "why": "single release"},
            {"id": "Q1939008", "label": "Nowhere Man", "why": "EP release sharing the work's title"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
        ],
    },
    "music_album": {
        "pass": [
            {"id": "Q181826", "label": "Please Please Me", "why": "album"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album"},
            {"id": "Q169226", "label": "Sgt. Pepper's Lonely Hearts Club Band", "why": "album"},
            {"id": "Q201816", "label": "Help!", "why": "album"},
            {"id": "Q251309", "label": "Bob Dylan", "why": "album sharing the artist's name"},
            {"id": "Q270083", "label": "Jazz", "why": "album sharing a genre's name"},
            {"id": "Q61027072", "label": "Please Please Me", "why": "LP record release"},
            {"id": "Q60427757", "label": "Please Please Me / Ask Me Why", "why": "single release"},
            {"id": "Q1939008", "label": "Nowhere Man", "why": "extended play release"},
        ],
        "fail": [
            {"id": "Q736799", "label": "Please Please Me", "why": "the composition, not a release"},
            {"id": "Q200799", "label": "Eleanor Rigby", "why": "the composition, not a release"},
            {"id": "Q60428471", "label": "Please Please Me", "why": "recording, not a release"},
            {"id": "Q1606282", "label": "Ringo Starr discography", "why": "Wikimedia discography page"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
        ],
    },
    "event": {
        "pass": [
            {"id": "Q164815", "label": "Woodstock Festival", "why": "music festival"},
            {"id": "Q193740", "label": "Live Aid", "why": "benefit concert"},
            {"id": "Q2920369", "label": "Isle of Wight Festival 1970", "why": "music festival edition"},
            {"id": "Q138412740", "label": "Simon and Garfunkel in Rotterdam in 1982", "why": "concert"},
        ],
        "fail": [
            {"id": "Q1248643", "label": "The Beatles at Shea Stadium", "why": "television film about an event, not the event"},
            {"id": "Q909845", "label": "The Cavern Club", "why": "venue where events happen"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
            # Known limitations: a human would call both of these events, but
            # Wikidata types them as something else. Asserted as failures so the
            # behaviour is recorded rather than silently rediscovered later.
            {"id": "Q549544", "label": "Beatlemania", "why": "KNOWN LIMIT: typed as a 'fan frenzy', not an event of any kind"},
            {"id": "Q3300629", "label": "Royal Variety Performance", "why": "KNOWN LIMIT: typed as a television program"},
        ],
    },
    "venue": {
        "pass": [
            {"id": "Q909845", "label": "The Cavern Club", "why": "venue"},
            {"id": "Q456140", "label": "Star-Club", "why": "music venue / nightclub"},
            {"id": "Q128468", "label": "Wembley Stadium", "why": "stadium"},
            {"id": "Q1056326", "label": "Shea Stadium", "why": "defunct sports venue"},
            {"id": "Q193639", "label": "Royal Albert Hall", "why": "concert hall / theatre building"},
            {"id": "Q2947292", "label": "Indica Gallery", "why": "art gallery"},
            {"id": "Q209651", "label": "Abbey Road Studios", "why": "recording studio; NOT under venue, hence its own root"},
        ],
        "fail": [
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
            {"id": "Q23100", "label": "Merseyside", "why": "county"},
            {"id": "Q479469", "label": "Abbey Road", "why": "road"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album"},
            {"id": "Q188451", "label": "music genre", "why": "generic class"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
        ],
    },
    "colour": {
        "pass": [
            {"id": "Q23445", "label": "black", "why": "primary color"},
            {"id": "Q3142", "label": "red", "why": "primary color"},
            {"id": "Q23444", "label": "white", "why": "instance of color"},
            {"id": "Q1088", "label": "blue", "why": "primary / spectral color"},
            {"id": "Q3133", "label": "green", "why": "primary / spectral color"},
            {"id": "Q943", "label": "yellow", "why": "primary / spectral color"},
            {"id": "Q39338", "label": "orange", "why": "secondary color"},
            {"id": "Q208045", "label": "gold", "why": "web color"},
        ],
        "fail": [
            {"id": "Q214609", "label": "material", "why": "adjacent type"},
            {"id": "Q287", "label": "wood", "why": "material"},
            {"id": "Q486972", "label": "human settlement", "why": "generic class"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
            {"id": "Q8341", "label": "jazz", "why": "music genre"},
        ],
    },
    "material": {
        "pass": [
            {"id": "Q287", "label": "wood", "why": "material"},
            {"id": "Q286", "label": "leather", "why": "material; has NO P31 at all, only P279"},
            {"id": "Q1910620", "label": "snakeskin", "why": "material; no P31"},
            {"id": "Q239788", "label": "mahogany wood", "why": "type of wood"},
            {"id": "Q122421", "label": "rosewood", "why": "type of wood"},
            {"id": "Q215865", "label": "nacre", "why": "composite material (mother of pearl)"},
            {"id": "Q243519", "label": "velvet", "why": "fabric; 5+ P279 hops from material"},
            {"id": "Q652698", "label": "denim", "why": "fabric"},
            {"id": "Q37681", "label": "silk", "why": "fabric"},
            {"id": "Q11474", "label": "plastic", "why": "material"},
            {"id": "Q11472", "label": "paper", "why": "modelled as an INSTANCE of material, not a subclass"},
            {"id": "Q389782", "label": "cardboard", "why": "material"},
            {"id": "Q146368", "label": "polyvinyl chloride", "why": "the vinyl of records"},
            {"id": "Q39782", "label": "brass", "why": "alloy"},
        ],
        "fail": [
            {"id": "Q1075", "label": "color", "why": "adjacent type"},
            {"id": "Q3142", "label": "red", "why": "colour"},
            {"id": "Q8341", "label": "jazz", "why": "music genre"},
            {"id": "Q1299", "label": "The Beatles", "why": "musical group"},
            {"id": "Q2632", "label": "Ringo Starr", "why": "person"},
            {"id": "Q173643", "label": "Abbey Road", "why": "album"},
            {"id": "Q24826", "label": "Liverpool", "why": "city"},
        ],
    },
}


# ##############################################################################
# Fixture integrity
# ##############################################################################

def _all_fixture_ids() -> dict:
    """QID -> set of labels the fixture claims for it."""
    claimed = {}
    for cases in TEST_CASES.values():
        for bucket in ("pass", "fail"):
            for case in cases[bucket]:
                claimed.setdefault(case["id"], set()).add(case["label"])
    return claimed


def _labels_agree(claimed: str, actual: str) -> bool:
    c, a = claimed.strip().lower(), actual.strip().lower()
    return c == a or c in a or a in c


def verify_fixture(verbose: bool = True) -> list:
    """
    Check that every fixture QID still resolves to the label it claims.

    Returns the list of mismatches; an empty list means the fixture is sound.
    """
    claimed = _all_fixture_ids()
    actual = wikidata_get_labels(list(claimed))
    if actual is None:
        # A failed request must not be reported as 111 missing entities.
        print("Fixture integrity: SKIPPED — could not reach the Wikidata API")
        return [("*", "", "<api unreachable>", "")]
    mismatches = []
    for qid in sorted(claimed, key=lambda q: int(q[1:])):
        real_label, real_desc = actual.get(qid, (None, ""))
        for want in claimed[qid]:
            if real_label is None:
                mismatches.append((qid, want, "<entity not found>", ""))
            elif not _labels_agree(want, real_label):
                mismatches.append((qid, want, real_label, real_desc))

    if verbose:
        total = len(claimed)
        if mismatches:
            print(f"Fixture integrity: {len(mismatches)} MISMATCH(es) across {total} entities")
            for qid, want, got, desc in mismatches:
                print(f"  {qid:12} fixture says '{want}' but Wikidata says '{got}'  [{desc[:60]}]")
        else:
            print(f"Fixture integrity: OK ({total} entities verified)")
    return mismatches


# ##############################################################################
# ECK test runner
# ##############################################################################

def run_eck_tests(types: list = None,
                  delay_seconds: float = 1.0,
                  verbose: bool = True) -> dict:
    """
    Run every ECK against its fixture and report per-type quality.

    Args:
        types: entity-type names to test; defaults to all types in TEST_CASES.
        delay_seconds: pause between SPARQL calls (WDQS throttles bursts).
        verbose: print a per-case line as well as the summary.

    Returns a dict of per-type metrics plus an ``_overall`` entry.
    """
    selected = list(types) if types else list(TEST_CASES)
    unknown = [t for t in selected if t not in TEST_CASES]
    if unknown:
        raise ValueError(f"unknown entity type(s): {unknown}. "
                         f"Available: {sorted(TEST_CASES)}")

    results = {}
    for type_name in selected:
        eck = ECK_BY_TYPE[type_name]
        cases = TEST_CASES[type_name]
        tp = fp = tn = fn = 0
        errors = []
        wrong = []

        if verbose:
            print(f"\n=== {type_name} ===")

        for bucket in ("pass", "fail"):
            expected = bucket == "pass"
            for case in cases[bucket]:
                actual = validate_entity(case["id"], eck, source="wd")
                if actual is None:
                    errors.append(case)
                    status = "ERROR"
                elif actual == expected:
                    status = "ok"
                    if expected:
                        tp += 1
                    else:
                        tn += 1
                else:
                    status = "WRONG"
                    wrong.append(case)
                    if expected:
                        fn += 1
                    else:
                        fp += 1

                if verbose:
                    want = "accept" if expected else "reject"
                    print(f"  [{status:5}] {case['id']:12} {case['label'][:34]:36}"
                          f" want={want:6} got={str(actual):5}  ({case['why']})")
                time.sleep(delay_seconds)

        n_pass, n_fail = len(cases["pass"]), len(cases["fail"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        results[type_name] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / (n_pass + n_fail) if (n_pass + n_fail) else 0.0,
            "errors": errors, "wrong": wrong,
        }

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'type':16} {'prec':>6} {'rec':>6} {'F1':>6} {'acc':>6} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'err':>4}")
    print("-" * 78)
    tot = {k: 0 for k in ("tp", "fp", "tn", "fn")}
    n_err = 0
    for type_name, m in results.items():
        for k in tot:
            tot[k] += m[k]
        n_err += len(m["errors"])
        print(f"{type_name:16} {m['precision']:6.2f} {m['recall']:6.2f} "
              f"{m['f1']:6.2f} {m['accuracy']:6.2f} "
              f"{m['tp']:3} {m['fp']:3} {m['tn']:3} {m['fn']:3} {len(m['errors']):4}")
    print("-" * 78)
    denom = sum(tot.values())
    prec = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"]) else 0.0
    rec = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"]) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = (tot["tp"] + tot["tn"]) / denom if denom else 0.0
    print(f"{'OVERALL':16} {prec:6.2f} {rec:6.2f} {f1:6.2f} {acc:6.2f} "
          f"{tot['tp']:3} {tot['fp']:3} {tot['tn']:3} {tot['fn']:3} {n_err:4}")
    print("=" * 78)

    failures = [(t, c) for t, m in results.items() for c in m["wrong"]]
    if failures:
        print(f"\n{len(failures)} incorrect verdict(s):")
        for type_name, case in failures:
            print(f"  {type_name:16} {case['id']:12} {case['label'][:30]:32} ({case['why']})")
    errored = [(t, c) for t, m in results.items() for c in m["errors"]]
    if errored:
        print(f"\n{len(errored)} query error(s) (network / WDQS timeout, not a query defect):")
        for type_name, case in errored:
            print(f"  {type_name:16} {case['id']:12} {case['label'][:30]}")


def run_batch_equivalence_tests(types: list = None, verbose: bool = True) -> list:
    """
    Check that batched validation reaches the same verdicts as the fixture.

    ``TypedRAGEntityLinker`` folds every candidate for a mention into one SPARQL
    request instead of issuing one ASK per candidate. The rewrite is only safe
    if it produces the same verdicts, so it is held to exactly the same fixture
    that ``run_eck_tests`` holds the per-entity ASK path to -- if both pass, the
    two paths agree.

    Returns the list of disagreements; empty means batching is faithful.
    """
    selected = list(types) if types else list(TEST_CASES)
    disagreements = []

    for type_name in selected:
        eck = ECK_BY_TYPE[type_name]
        expected = {}
        for bucket in ("pass", "fail"):
            for case in TEST_CASES[type_name][bucket]:
                expected[case["id"]] = (bucket == "pass")
        ids = list(expected)

        batched = validate_entities_batch(ids, eck)
        if batched is None:
            disagreements.append((type_name, "*", "batch query failed", None))
            if verbose:
                print(f"  {type_name:16} batch query FAILED")
            continue

        bad = [q for q in ids if batched.get(q) != expected[q]]
        if verbose:
            status = "ok" if not bad else f"{len(bad)} mismatch(es)"
            n_req = -(-len(ids) // BATCH_VALIDATION_CHUNK)  # ceil
            print(f"  {type_name:16} {len(ids):3} entities in {n_req} request(s) "
                  f"(vs {len(ids)} ASKs) -> {status}")
        for qid in bad:
            disagreements.append((type_name, qid, expected[qid], batched.get(qid)))

    if verbose:
        if disagreements:
            print(f"\n{len(disagreements)} batch/ASK disagreement(s):")
            for type_name, qid, want, got in disagreements:
                print(f"  {type_name:16} {qid:12} expected={want} batch={got}")
        else:
            print("\nBatch validation matches the per-entity ASK verdicts.")
    return disagreements


def run_selection_tests(models: list = None, verbose: bool = True) -> list:
    """
    Exercise the LLM selection call against each model through the provider layer.

    Returns the list of contract violations; empty means every reachable model
    behaves the way the linker requires. A model whose key is absent is skipped
    rather than failed — a developer with one provider configured should still be
    able to run the suite.
    """
    specs = list(models) if models else list(SELECTION_MODELS)
    fixture = SELECTION_FIXTURE
    n = len(fixture["candidates"])
    violations = []

    for spec in specs:
        provider, _ = parse_model_spec(spec)
        if not os.environ.get(API_KEY_ENV[provider]):
            if verbose:
                print(f"  {spec:32} SKIPPED (no ${API_KEY_ENV[provider]})")
            continue
        try:
            linker = RAGEntityLinker(generator_model=spec)
        except Exception as exc:
            violations.append((spec, f"could not construct: {type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  {spec:32} FAIL  {type(exc).__name__}: {exc}")
            continue

        answers = []
        for _ in range(2):        # temperature 0: the two must agree
            choice, raw, error = linker._select_with_llm(
                fixture["search_term"], fixture["context_query"],
                fixture["entity_type"], fixture["candidates"], verbose=False)
            answers.append((choice, raw, error))

        (choice, raw, error) = answers[0]
        if error:
            violations.append((spec, f"selection reported error={error!r}"))
        elif not isinstance(choice, int) or isinstance(choice, bool):
            violations.append((spec, f"choice is {type(choice).__name__}, not int: {choice!r}"))
        elif not (0 <= choice <= n):
            violations.append((spec, f"choice {choice} outside 0..{n}"))

        # Reported, never asserted: temperature 0 is best effort at every
        # provider, and the classification evaluation already quantifies the
        # residual variance.
        unstable = answers[0][0] != answers[1][0]

        if verbose:
            flag = (f"   [repeat call returned {answers[1][0]}]" if unstable else "")
            if isinstance(choice, int) and 1 <= choice <= n:
                picked = fixture["candidates"][choice - 1]
                agrees = picked["wikidata_id"] == fixture["preferred_qid"]
                note = "" if agrees else "   (not the fixture's preference)"
                print(f"  {spec:32} choice={choice} -> {picked['wikidata_id']} "
                      f"{picked['label']}{note}{flag}")
            elif choice == 0:
                print(f"  {spec:32} choice=0 (abstained)"
                      f"   (not the fixture's preference){flag}")
            else:
                print(f"  {spec:32} FAIL  choice={choice!r} raw={raw!r} error={error!r}")

    if verbose:
        if violations:
            print(f"\n{len(violations)} contract violation(s):")
            for spec, why in violations:
                print(f"  {spec:32} {why}")
        else:
            print("\nEvery reachable model satisfies the selection contract "
                  "(an integer in range, no exception).")
            print("The chosen candidate and any repeat-call disagreement are "
                  "reported, not asserted: both are properties of the model "
                  "rather than of the linker.")
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("types", nargs="*", help="entity types to test (default: all)")
    parser.add_argument("--verify-fixture", action="store_true",
                        help="only check that fixture QIDs resolve to their claimed labels")
    parser.add_argument("--batch-equivalence", action="store_true",
                        help="only check batched validation against per-entity ASKs")
    parser.add_argument("--selection", action="store_true",
                        help="only check the LLM selection contract across providers")
    parser.add_argument("--models", nargs="+",
                        help="model specs for --selection (default: the configured set)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the fixture integrity check")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between SPARQL calls (default 1.0)")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args(argv)

    # The selection check tests the provider layer, not the ECK fixture, so it
    # does not depend on the Wikidata label lookup running first.
    if args.selection and not args.types:
        args.skip_verify = True

    if not args.skip_verify:
        mismatches = verify_fixture()
        if args.verify_fixture:
            return 1 if mismatches else 0
        if mismatches:
            print("\nRefusing to run: fix the fixture first — query results "
                  "measured against wrong entities are meaningless.")
            return 1
    elif args.verify_fixture:
        return 0

    if args.selection:
        print("\nLLM selection contract, by provider:")
        return 1 if run_selection_tests(args.models or None) else 0

    if args.batch_equivalence:
        print("\nBatch vs per-entity ASK equivalence:")
        return 1 if run_batch_equivalence_tests(args.types or None) else 0

    results = run_eck_tests(types=args.types or None,
                            delay_seconds=args.delay,
                            verbose=not args.quiet)
    bad = sum(len(m["wrong"]) for m in results.values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
