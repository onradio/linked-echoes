# Evaluating the neuro-symbolic entity linker

Companion narrative to `entity_linking_evaluation_spec.md`, written for §4.4 of the
Linked Echoes article. The spec says what to build; this says what was done, what came
out, and what the numbers will and will not support.

Figures and tables live in `data/results/latex/` and are regenerated with
`python score.py && python report.py`. Numbers quoted below are current as of the run of
2026-08-10 (`eck_version = v1`); the authoritative copies are `data/results/*.csv` and
`latex/macros.tex`, and prose should cite the macros rather than repeat literals.

---

## 1. What the evaluation is for

The linker is a retrieve–validate–select pipeline: a lexical search over Wikidata returns
candidates, a SPARQL `ASK` query per entity type discards those of the wrong type, and an
LLM picks one of the survivors. The prompt *already* names the expected type. The question
is therefore not whether type information helps — both configurations receive it — but
whether **enforcing** the type symbolically beats **stating** it and trusting the model to
respect it.

Three claims follow, and the design exists to test exactly these:

- **C1** Hard type filtering beats soft type conditioning.
- **C2** Validation makes the linker robust as retrieval widens.
- **C3** The benefit concentrates where surface strings are ambiguous.

## 2. Benchmark

**BME-EL** derives from 65 catalogued objects of the *Beyond the Music* collection. Each
object carries metadata fields for ten entity types; splitting and normalising those fields
yields **115 distinct mentions over 194 instances**, where a mention is a distinct
(normalised label, entity type) pair.

Normalisation is deliberately shallow — Unicode, quotes and whitespace only. Spelling and
casing are left as found, so *Freddy Mercury* and *Edge Road* (for Edgware Road) survive
into the benchmark. Repairing them would remove precisely the noise a heritage linker has
to cope with.

The collection supplies its own hard cases without any synthetic construction:
cross-domain homonyms (*Cream*, *Suede*, *Queen*, *Band Aid*), titles that read as ordinary
language (*Blue Monday*, *Foundations*, *White Room*, *Apache*, *Freight Train*), person
versus eponymous ensemble (*Ken Colyer* / *Ken Colyer's Jazzmen*), alias drift
(*The Detours* → *The High Numbers*), and generic venue names (*Railway Hotel*,
*Torquay Town Hall*).

### Gold standard

A single annotator resolved every mention using Wikidata search and the entity pages
directly, **not** the systems' candidate lists. This matters: because the gold was
established independently of what retrieval returned, retrieval misses are measurable
rather than invisible. Where more than one item is defensible, alternates are recorded and
scoring accepts any of them.

Mentions with no correct Wikidata item are labelled NIL under one of four codes —
`no_entity`, `out_of_scope`, `too_vague`, `data_error`. The first three are scored:
**correct abstention is a result, not a nuisance.** `data_error` marks legacy column shifts
that are not entities at all (a year in the colour field); those three mentions are
excluded, leaving **112 scored**.

One annotation policy deserves stating plainly in the paper, because it shapes a headline
number. Wikidata frequently holds the *single* for a song but not the composition. Linking
a single into a `music_work` slot writes a factually wrong statement into the knowledge
graph, so such mentions are annotated NIL. The decision is about schema integrity and was
taken before any system was run — but it does favour systems able to abstain, and its
effect is concentrated in the music-work type. Where an entity is typed as *both* a song
and a single, it is linked.

### Frozen candidate pools

Wikidata's search ranking drifts, so the top 50 candidates per mention were retrieved once
and pinned as an immutable artefact, together with each candidate's `P31` classes. Every
depth `k` reported is a slice of that list, never a re-query, which makes the whole
evaluation replayable offline apart from the LLM call.

Measured pool depth is shallower than the design assumed: mean 23.4, median 17, and only a
third of mentions return the full 50. Six mentions return nothing at all and are a
guaranteed miss for every system.

## 3. Type checks

Ten SPARQL `ASK` queries, one per entity type. Two properties of their provenance matter
for the paper's credibility.

First, they were **rewritten before the evaluation existed**. The previous revision
referenced ten QIDs that did not mean what their comments claimed — `Q133998`, labelled
"single", is a Belgian village; `Q730221`, labelled "EP", is an airport — and two checks
were structurally incapable of passing, because they required a `P31` that class-like
entities such as *leather* and *velvet* simply do not have.

Second, the rewrite was driven by an **independent fixture** of 158 assertions over 111
Wikidata entities, not by any mention, pool or gold QID in this benchmark. The checks were
then frozen as `v1` before the pools were retrieved. This is the "freeze first" option of
the spec's revision protocol: `v1` is a pre-registered validator, not one fitted to the
test set, and the paper may say so.

A query that cannot be evaluated is recorded as `error_api` and **never** as a rejection —
otherwise a transient network failure would be scored as the validator doing its job. The
final validation pass contains no such records.

## 4. Systems and protocol

All four systems consume the same frozen pool truncated to `k`, and both LLM systems
receive an identical prompt including the expected type.

| Id | Pipeline | Cost |
| --- | --- | --- |
| `S0` | candidate at rank 1; never abstains | free, offline |
| `S1` | highest-ranked ECK-valid candidate; abstain if none | free, offline |
| `S2` | LLM selects from the unfiltered top-`k`; may abstain | LLM |
| `S3` | ECK filter, then LLM selects; may abstain | LLM |

`S2` and `S3` are built from the same class with validation toggled, so they differ in the
filter and nothing else. `S0` is the baseline that is routinely underestimated; `S1`
isolates how much the generative step adds over the filter alone.

`k ∈ {5, 10, 25, 50}`, headline `k = 10`. Three runs per cell for the LLM systems at
temperature 0; `S0` and `S1` are deterministic and run once. Significance is McNemar's
exact test on paired decisions; 95% intervals are a cluster bootstrap over objects
(10,000 resamples), since mentions on the same object share disambiguating context.

## 5. Results

### C1 — symbolic validation beats soft type conditioning

At `k = 10`, accuracy rises from **0.827** (`S2`) to **0.935** (`S3`), a gain of **+0.107**
with 95% CI **[0.052, 0.164]**. McNemar's exact test gives **p = 0.0018** on discordant
counts b = 1 / c = 13 — thirteen mentions that only the validated system gets right, one
that only the unvalidated one does. Link precision, the metric that matters when a wrong
link writes a false statement into the graph, rises from **0.831** to **0.946**.

The mechanism is visible in the error attribution (`figure_attribution.pdf`): spurious
links — asserting an entity where the gold is NIL — fall from 13 to 4, and wrong selections
from 4 to 1. Validation converts confident wrong answers into abstentions, which a curator
can act on; a wrong link is silent.

### C2 — robustness to distractor pressure

As retrieval widens from `k = 5` to `k = 50`, the unvalidated system degrades
(0.863 → 0.824) while the validated one stays essentially flat (0.943 → 0.929) and leads at
every depth. Mean pruning at `k = 10` is 6.8 → 1.5 candidates.

The practical reading is that validation lets an institution widen retrieval — raising the
chance the right entity is in the pool at all — without paying for it in precision.

**Caveat to state when discussing the figure.** The sweep is diluted by pool saturation:
36% of mentions return fewer than five candidates, so their pool is identical at every `k`
and they contribute a constant to every curve. The measured degradation of `S2` is
therefore a lower bound on what deeper pools would show, not an estimate of it.

### C3 — where the benefit lands

Gains concentrate exactly where surface strings are ambiguous: music work **+0.348**,
country **+0.200**, genre **+0.111**, album **+0.100**. Types whose names are near-unique —
colour, material, event — are already saturated at 1.000 for both systems and gain nothing,
which is the expected shape rather than a null result.

The music-work figure is the largest and needs the honest gloss from §2: much of it is
`S3` correctly abstaining where Wikidata offers only the single, and `S2` confidently
linking it.

**C3 cannot be claimed evenly across types.** Event has 2 distinct mentions and venue 8, of
which three retrieve nothing. No difference measured on that many mentions is reportable
with or without a CI. Works (23) and albums (10) are the defensible half of the claim;
venues and events should be reported as indicative and left to Table 2 as descriptive rows.

### The symbolic step costs nothing measurable

The validator false-rejection rate — gold entities retrieved but rejected by their own
check — is **0.00 overall and in every individual type**. Retrieval recall at `k = 10` is
**0.99**, with a single gold entity absent from its pool. The filter is not discarding
correct answers; it is discarding wrong ones. This is the number a sceptical reader will
look for, and it should be reported prominently rather than buried.

### The filter carries most of the weight

`S1` — the checks plus "take the highest-ranked survivor", with no language model at all —
reaches **0.893**, above `S2` at every depth and at zero inference cost. The generative step
adds precision on top (0.889 → 0.946) rather than doing the bulk of the disambiguation.

This is worth stating rather than hiding. It reframes the contribution from "an LLM helped
by types" to "a symbolic filter that an LLM refines", which is both more defensible and
more useful to an institution weighing whether to pay for inference at all.

## 6. Limitations

- **Small per-type samples.** See C3 above.
- **Pool saturation.** See C2 above.
- **Gold treats a release as not a work.** A schema-integrity decision, taken before any
  system ran, that nonetheless favours systems able to abstain.
- **Wikidata is a moving target.** The pools are frozen and timestamped, so the benchmark
  replays exactly; the `ASK` queries, however, run against live WDQS, so validator results
  reproduce only to within Wikidata's drift. The retrieval timestamp is recorded in the
  release.

## 7. Artefacts

| Path | Contents |
| --- | --- |
| `data/gold_objects.csv` | 65 source objects |
| `data/annotation_sheet.csv` | 115 mentions with gold QIDs, alternates, NIL reasons |
| `data/pools_raw.jsonl` | frozen top-50 pools with `P31`, timestamped — **immutable** |
| `data/pools_valid_v1.jsonl` | 2,694 verdicts, one per (mention, candidate) |
| `data/predictions/` | 29 cells: system × k × run |
| `data/results/` | scored metrics as CSV/JSON |
| `data/results/latex/` | figures (PDF + pgfplots) and tables for the article |

Reproduce end to end with `freeze_pools.py` → `validate_pools.py` → `run_systems.py` →
`score.py` → `report.py`. Only the middle three touch the network.
