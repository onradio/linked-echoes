# Evaluation of the category and type classifiers

*Draft material for Section 4.3 of the Linked Echoes article (Knowledge-Based Systems).
Every figure quoted here is produced by `score.py` from the prediction files in
`data/predictions/` and formatted by `report.py`; nothing is transcribed by hand. The
companion `report.md` holds the same tables without the discussion.*

**Artefacts for the article.** The tables below are reproduced as markdown for reading; the
typeset versions are generated and should be `\input` rather than retyped.

| Deliverable | Source | Placement |
| --- | --- | --- |
| Table 1 — category classification | `latex/table1_category.tex` (`tab:category`) | §3, main text |
| Table 2 — type classification | `latex/table2_type.tex` (`tab:type`) | §5, main text |
| Figure 1 — ceiling vs. achieved | `figures/retrieval_and_selection.pdf` | §5, main text |
| Table 3 — full retrieval grid | `latex/table3_retrieval_grid.tex` (`tab:retrieval_grid`) | §4, appendix |
| Figure 2 — retrieval grid | `figures/retrieval_grid.pdf` | §4, appendix |
| hit@10 by query formulation | `latex/table_query_formulations.tex` (`tab:queries`) | §4, appendix |
| Figure captions | `latex/figure_captions.tex` | — |

Figure 2 has a **truncated vertical axis** (begins at 40%), which its generated caption states;
Table 3 sits beside it carrying every value exactly.

---

## 1. What is evaluated, and why in two stages

The construction pipeline assigns each object two things: a **broad category**, one of seven,
and a **fine-grained type**, a term drawn from one of four harmonised controlled vocabularies.
The two are not independent — the category selects which vocabulary the type is drawn from —
so a failure in the first propagates into the second. They are evaluated separately, and the
type classifier is evaluated **with the gold category supplied**, which isolates it from
category errors and makes its figures an upper bound on end-to-end behaviour.

Type assignment is itself two mechanisms in sequence: a retriever proposes candidate terms
from a vocabulary index, and an LLM selects among them or declines. Reporting only the
end-to-end number would leave the two indistinguishable, so retrieval is evaluated on its own
first (Experiment B), establishing the ceiling that selection (Experiment C) works against.
This matters more than it might appear: the two mechanisms turn out to respond to the same
parameter — the number of candidates `k` — in *opposite* directions.

| Experiment | Task | Systems | LLM calls |
| --- | --- | --- | --- |
| A | category | `C0` majority, `C1` embedding, `C2` LLM | 195 (+195 ablation) |
| B | type retrieval | 6 configurations × 6 depths | none |
| C | type selection | `T0` retrieval top-1, `T1` RAG | 1,630 |

---

## 2. Evaluation design

### 2.1 Gold standard

Sixty-five objects, drawn as three random objects per display case across 21 cases (one case
contributes five). The sampling is by case rather than uniform because objects within a case
share thematic context and often near-identical text; drawing by case makes the clustering
explicit rather than hidden, and every interval reported below is bootstrapped over cases for
the same reason. The skew the sample exhibits — 39 of 65 objects are Textual & Graphic Work —
is the skew of the collection, not an artefact of selection.

A single annotator produced the gold standard, so no inter-annotator agreement is available.
The full sheet is released so the labels can be audited. All 102 term URIs it records were
verified to resolve in the indexed vocabulary with labels matching, which catches the failure
mode where a gold term is a plausible string that no retriever can ever return.

### 2.2 Three tiers of type gold

Annotation surfaced a distinction the original design did not anticipate, and it is
load-bearing for how the type results are read:

| Tier | *n* | Meaning |
| --- | --- | --- |
| **strong** | 50 | a precise term exists in the searched vocabularies |
| **weak** | 14 | no exact term exists; a defensible approximation was recorded |
| **NIL** | 1 | no usable term at all |

The weak tier is not annotator hesitancy — it is a measurement of a documented vocabulary gap.
Nine of the fourteen are song lyrics, for which Getty AAT has no term (`song sheets` was
recorded as the nearest); five are tour programmes (`souvenir programs`). On a weak row,
**two behaviours are correct**: returning the recorded approximation, and declining to assert
anything. Only a third term is an error. Scoring these rows as though a single answer were
required would have penalised the system for exactly the judgement the annotator made.

This tier structure also explains an absence. The original design expected some gold terms to
be unreachable — lyrics reused from Wikidata, tour programmes minted locally, as the pipeline
in fact does — and would have scored abstention on them as correct. In the completed
annotation there are **no unreachable golds**: every such case was resolved to an AAT
approximation instead. The consequence is that the strict-NIL population is a single object,
so abstention behaviour is reported over the weak and NIL rows together (*n* = 15) and is
indicative rather than conclusive.

### 2.3 Metrics

For the category task: **accuracy** (crediting a match against either the primary or the
secondary gold category), **strict accuracy** (primary only), and macro-averaged
**precision / recall / F1** over the attested classes. Micro-averages are not reported —
every object receives exactly one prediction, so micro-P, micro-R and accuracy are the same
number.

For the type task, each decision falls into one of six outcomes:

| | reachable gold | no reachable gold |
| --- | --- | --- |
| **asserted the accepted term** | `TP` | — |
| **asserted another term** | `WT` | `SA` |
| **abstained** | `FA` (strong) / `TA_weak` (weak) | `TA` |

from which: **accuracy** = (TP + TA + TA_weak)/N; **term precision** = TP/(TP + WT + SA), the
metric the argument rests on, since a wrong type writes a false `scho:additionalType` into the
graph whereas an abstention leaves a gap a curator can fill; **term recall** =
TP/(TP + WT + FA), which excludes `TA_weak` because a correct abstention is not a miss; and
**abstention accuracy** = (TA + TA_weak)/(TA + TA_weak + SA + WT_weak). Because precision and
recall are computed over slightly different populations, their harmonic mean is reported as a
summary rather than as an independently interpretable quantity, and **strong-row P/R/F1** are
given beside them, restricted to the 50 rows where exactly one behaviour is correct and the
three have their textbook meaning.

### 2.4 Statistical treatment

Three runs at temperature 0 for every LLM cell; results are averaged for reporting and
majority-voted per object for paired testing. Comparisons use **exact McNemar** on the
discordant pairs, reported with the discordant counts, which on a sample this size carry more
information than the *p*-value. Confidence intervals are **percentile bootstrap, 10,000
resamples, clustered by display case**.

---

## 3. Experiment A — category classification

*LaTeX: `latex/table1_category.tex` (`tab:category`).*

| System | Accuracy | 95% CI | Strict | Macro-P | Macro-R | Macro-F1 | McNemar *p* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `C0` majority class | 60.0 | [47.8, 73.0] | 60.0 | 15.0 | 25.0 | 18.8 | <0.001 |
| `C1` nearest definition | 26.2 | [16.9, 36.2] | 24.6 | 59.9 | 37.0 | 37.2 | <0.001 |
| **`C2` LLM (ours)** | **98.0** | [94.2, 100.0] | **97.4** | **100.0** | **98.4** | **99.2** | — |

Macro averages are over the four attested classes. *p* is against `C2`.

**The LLM categoriser is at ceiling.** It misclassifies between one and two of 65 objects
depending on the run, and its per-class precision is 1.00 on every attested class: nothing it
places in a category belongs elsewhere. What it loses is a little recall on Clothing (0.97)
and Textual & Graphic Work (0.97). The bootstrap interval reaches 100%, and the sample cannot
distinguish the system from a perfect classifier — a limitation of the evaluation, not a claim
about the system.

**The embedding baseline fails, and instructively.** `C1` — nearest of the seven curated
scope notes by sentence embedding — reaches 26.2%, less than half the majority baseline. Its
per-class behaviour shows why: it predicts **Merchandise 28 times on a sample containing no
Merchandise objects**, and recovers only 2 of 39 Textual & Graphic Work objects (recall
0.051). Thirty-eight of its 65 predictions land in categories with no gold instances at all.
The failure is not marginal discrimination but systematic drift: embedding similarity between
an object description and an abstract category definition is dominated by surface vocabulary,
and the Merchandise scope note ("commercially produced items… featuring an artist's name,
image, or logo") is lexically close to a great many music-museum descriptions. This is worth
reporting precisely because a non-generative baseline is the obvious thing a reviewer will ask
for; it is available, it is free, and on this task it does not work.

**A richer prompt did not help.** A variant carrying the seven scope notes and eight worked
examples (`C2-scoped`) was evaluated over the same three runs: 100.0% accuracy against `C2`'s
98.0%, **one discordant object, McNemar *p* = 1.0**. Since the additional prompt material buys
no measurable accuracy, the simpler instruction is the one deployed.

The comparison is more interesting than the aggregate suggests, and the article should say so
in a sentence rather than leave it at "no difference". The two prompts differ in *which*
errors survive. `C2-scoped` is never outright wrong on this sample: both its strict errors are
secondary-gold matches on composite objects — a plectrum and a toy guitar, which its scope
notes explicitly assign to Personal Effects and Merchandise while the curator made Musical
Instrument primary. `C2` instead makes two genuine misreadings, taking membership cards for
Personal Effects and a poster for Merchandise. **Both of those confusions fall in categories
this sample does not attest**, and which the scope notes were written to disambiguate. The
parsimony argument is therefore sound on the evidence available, and should be stated that
way; confirming it for the rare categories requires a sample that contains them.

A second observation belongs in the limitations rather than the results. The gap between
accuracy and strict accuracy is entirely the four composite objects, and for `C2-scoped` it is
entirely a *documented disagreement between the deployed scope notes and the curator's
revision* — not model error. That is evidence for admitting a secondary category rather than a
caveat about it.

---

## 4. Experiment B — type retrieval

*LaTeX: `latex/table3_retrieval_grid.tex` (`tab:retrieval_grid`),
`latex/table_query_formulations.tex` (`tab:queries`). Figure:
`figures/retrieval_grid.pdf`.*

Retrieval is swept over two index strategies (the term label alone, or label + scope note),
three query formulations, and six depths. Every configuration is retrieved **once at
`k` = 50** and every smaller `k` is a slice of that same pool, never a re-query, so the depth
axis is exact rather than an approximation over independent retrievals.

| Index | Query | @5 | @10 | @20 | @30 | @40 | @50 | MRR@50 | Selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| term | description | 54.7 | 57.8 | 67.2 | 70.3 | 71.9 | 71.9 | 0.388 | — |
| term | name + description | 57.8 | 62.5 | 68.8 | 75.0 | 78.1 | 78.1 | 0.478 | — |
| **term** | **hint + description** | **68.8** | **71.9** | **76.6** | 76.6 | 78.1 | 78.1 | **0.602** | yes |
| combine | description | 45.3 | 54.7 | 62.5 | 68.8 | 75.0 | 76.6 | 0.345 | — |
| **combine** | **name + description** | 53.1 | 62.5 | 71.9 | **78.1** | 78.1 | **82.8** | 0.370 | yes |
| combine | hint + description | 62.5 | 70.3 | 76.6 | 78.1 | 78.1 | 79.7 | 0.481 | — |

hit@*k* over the 64 objects with a reachable gold, AAT fallback pool included.

**Similarity is not an experimental dimension after all.** The design intended to compare
L2-on-unnormalised embeddings (the deployed behaviour) against cosine. They are the same
ranking: `all-MiniLM-L6-v2` terminates in a normalisation layer, so its output is already
unit-norm — `normalize_embeddings=True` changes it by 0.0 — and for unit vectors
‖a−b‖² = 2 − 2⟨a,b⟩, a strictly decreasing function. Converting the recorded inner products
reproduces the L2 distances to six decimal places across the entire grid. The article's claim
of cosine similarity is therefore **correct as written**, even though the code indexes with
`IndexFlatL2`. Both builds were run and retained as the evidence; only one is reported.

The residual disagreements — 36 gold ranks out of 1,560 — are floating-point tie-breaks
between *duplicate vocabulary entries*, which leads to a finding about the vocabulary itself:
4,766 of the 50,301 harmonised terms share a label with another term (2,192 distinct labels),
and AAT alone contains three terms labelled `magazines` (an ammunition store, a firearm
component, and a periodical) and two labelled `leaflets`, one of them botanical. Retrieval
routinely returns several of these together, and it is the scope note in the selection prompt
that distinguishes them.

**The type hint is worth more than the index strategy, until it isn't.** Prefixing the
retrieval query with the type predicted by metadata enrichment is worth 14 points of hit@5 on
the term index (68.8 vs 54.7) and has by far the best mean reciprocal rank (0.602 against
0.478 and 0.388) — it places the gold term high, not merely in the pool. But its advantage
decays: by `k` = 40 the three query formulations on the term index converge on 78.1%, and
`name + description` overtakes it on both indices at depth.

**Term-only indexing dominates at shallow depth; label + scope note wins at depth.** The
`combine` index is worse everywhere up to `k` = 20 and produces the best single figure in the
grid, 82.8% at `k` = 50. The two are not the same curve at different heights: the term index
*saturates* — its three formulations are flat from `k` = 40 — while the combine index is still
climbing at `k` = 50. Embedding a term with its definition adds discriminative signal that
only becomes usable once the pool is deep enough to contain the competition.

The two configurations carried into Experiment C follow from this table rather than from a
search over it: **term + hint** because it is the deployed pipeline and the best or joint-best
term-index configuration at every depth, and **combine + name** because it is the only
configuration reaching 82.8% and the only one still improving at `k` = 50. The remaining four
are dominated — each is beaten at every depth by one of the two selected. Reporting the full
grid means the choice is transparent rather than tuned on the evaluation set.

---

## 5. Experiment C — type selection

*LaTeX: `latex/table2_type.tex` (`tab:type`). Figure:
`figures/retrieval_and_selection.pdf`.*

| Retrieval | Sys. | *k* | Acc. | Term P | Term R | F1 | Abst. | hit@*k* | *p* vs T0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| term + hint | `T0` | — | 44.6 | 44.6 | 45.3 | 45.0 | — | — | — |
| term + hint | `T1` | 5 | **75.4** | **78.4** | 71.4 | **74.8** | 60.0 | 68.8 | <0.001 |
| term + hint | `T1` | 10 | 74.4 | 74.2 | 70.8 | 72.5 | 53.3 | 71.9 | <0.001 |
| term + hint | `T1` | 20 | 70.8 | 71.5 | 68.3 | 69.9 | 35.7 | 76.6 | <0.001 |
| term + hint | `T1` | 50 | 67.2 | 68.8 | 66.7 | 67.7 | 12.8 | 78.1 | <0.001 |
| combine + name | `T0` | — | 26.2 | 26.2 | 26.6 | 26.4 | — | — | — |
| combine + name | `T1` | 5 | 58.5 | 72.1 | 53.4 | 61.4 | 70.0 | 53.1 | <0.001 |
| combine + name | `T1` | 10 | 64.1 | 76.4 | 60.5 | 67.5 | 66.7 | 62.5 | <0.001 |
| combine + name | `T1` | 20 | 64.1 | 74.8 | 61.1 | 67.3 | 62.5 | 71.9 | <0.001 |
| combine + name | `T1` | 50 | 71.8 | 75.6 | 70.3 | 72.8 | 47.6 | 82.8 | <0.001 |

### 5.1 The selector earns its place

`T1` beats retrieval top-1 by 31 points in the deployed configuration (75.4% against 44.6%)
and by 46 points in the other (71.8% against 26.2%). The comparisons are paired per object and
significant at every depth in both configurations (*p* < 0.001; 16 to 31 discordant objects,
essentially all favouring `T1`). This is the clearest result in the section and it is not a
close call: a retriever alone, however well tuned, assigns the wrong type to more than half
the collection.

The mechanism is visible in the selector's behaviour rather than only in its score. At
`k` = 10 the term it asserts is retrieval's top-ranked candidate in **59.5%** of assertions
under `term + hint` but only **35.0%** under `combine + name`, where the median chosen rank is
3. The selector overrides the ranking exactly where the ranking is worse — which is the same
ordering quality that MRR captures in §4, and it is why a system that merely ratified rank 1
would score like `T0`.

### 5.2 Depth is not free

The two configurations respond to `k` in opposite directions, which is what makes running both
worthwhile:

- **`term + hint` degrades with depth**: 75.4% → 67.2%, term precision 78.4% → 68.8%, while
  its retrieval ceiling *rises* from 68.8% to 78.1%. It is handed 9 more points of reachable
  gold and gives back 8 points of accuracy.
- **`combine + name` improves**: 58.5% → 71.8%, converting 13 of the 30 points of ceiling it
  gains.

The paired tests qualify this carefully. `combine + name`'s improvement from `k` = 5 to
`k` = 50 is significant (17 discordant, 13 to 4, *p* = 0.049). **`term + hint`'s degradation is
not** (11 discordant, 8 to 3, *p* = 0.227): the direction is consistent across every
intermediate depth and across all three runs, but 65 objects clustered in 21 cases cannot
establish it. It should be reported as a consistent trend that the sample does not confirm,
which is a weaker and more accurate claim than the raw accuracies invite. Likewise
`term + hint` significantly outperforms `combine + name` at `k` = 5 (*p* = 0.027) and at no
other depth.

### 5.3 Where the degradation actually happens

Decomposing `term + hint` by gold tier locates it exactly:

| Tier | *n* | | `k` = 5 | `k` = 50 |
| --- | --- | --- | --- | --- |
| strong | 50 | TP | 40.0 | 40.0 |
| | | WT | 5.0 | 7.7 |
| | | FA | 5.0 | 2.3 |
| weak | 14 | TP | 0.0 | 2.0 |
| | | WT | 6.0 | 11.0 |
| | | `TA_weak` | 8.0 | 1.0 |

**On the 50 strong rows the selector is exactly as accurate at `k` = 50 as at `k` = 5** — 40
true positives in both. Its strong-row F1 barely moves across the whole sweep (84.2 → 81.9).
The entire headline decline is the weak tier, where correct abstentions collapse from 8 to 1
and wrong terms rise from 6 to 11. Presented with fifty candidates instead of five, the model
stops declining to answer: abstention accuracy falls from 60.0% to 12.8%.

This is a precise and, we think, generalisable statement about RAG over controlled
vocabularies. **Deepening the candidate pool does not degrade the model's ability to identify
a correct term; it degrades its willingness to say that no correct term exists.** The
STRICT RULE in the prompt — the instruction to return 0 rather than the closest available
match — is the mechanism credited with leaving objects untyped in the deployed graph rather
than mis-typed, and it is the first thing to fail under distractor pressure.

`combine + name` shows the complementary pattern: its gain is concentrated in the strong tier
(TP 26.0 → 35.3, FA 15.0 → 4.3) because at `k` = 5 its retrieval simply had not found the gold
term. Depth helps where retrieval is failing and hurts where it is not.

Run-to-run instability points the same way: at temperature 0, `term + hint` produces a
different outcome on 2 of 65 objects between runs at `k` = 5, rising to 7 of 65 at `k` = 50.
Deeper pools make the selector measurably less deterministic.

### 5.4 Error attribution

One cause per object, in precedence order (mean per run, `term + hint`):

| *k* | correct | retrieval miss | selection error | false abstention |
| --- | --- | --- | --- | --- |
| 5 | 49.0 | 12.0 | 4.0 | 0.0 |
| 10 | 48.3 | 11.0 | 5.7 | 0.0 |
| 20 | 46.0 | 11.0 | 6.3 | 1.7 |
| 50 | 43.7 | 13.0 | 7.0 | 1.0 |

Retrieval misses are the largest single error class at every depth and are essentially flat —
the objects whose gold term the retriever cannot surface are largely the same objects
regardless of `k`. What grows is selection error. For `combine + name` the same table shows
retrieval misses falling from 24.0 to 8.7 while selection errors rise from 1.0 to 7.3: deeper
retrieval converts misses into opportunities, and the selector converts most but not all of
them.

### 5.5 Vocabulary size dominates difficulty

Grouping the best cell (`term + hint`, `k` = 5) by the vocabulary the accepted answer was
drawn from — which is the routed vocabulary except where the AAT fallback fired, on three
objects:

| Vocabulary | Terms | Objects answered | Accuracy |
| --- | --- | --- | --- |
| MIMO | 2,324 | 6 | 100.0 |
| PBCore | 93 | 7 | 100.0 |
| Europeana Fashion | 1,481 | 10 | 70.0 |
| Getty AAT | 46,403 | 42 | 69.0 |

The two small, domain-focused vocabularies are solved outright. The difficulty is concentrated
in AAT, which is twenty times larger and is the destination for four of the seven categories
as well as for every fallback. Counts are small enough that these are indicative, but the
direction is unambiguous and it suggests the productive engineering lever is narrowing the AAT
search space — by facet, or by category-specific subtree — rather than improving the selector.

---

## 6. What the evaluation changed in the system

The evaluation was not only measurement; it found defects, and they are worth reporting
because they are the kind that produce plausible numbers rather than crashes.

1. **FAISS pads short result sets with `-1`.** Passed directly to a positional lookup, this
   silently returned the *last row of the vocabulary* as a candidate. PBCore holds 93 terms,
   so any sweep near that value was injecting phantom candidates into the prompt.
2. **The index and the query must be encoded identically.** Normalising one and not the other
   yields a ranking that is neither cosine nor L2 and looks entirely normal. The similarity is
   now recorded in a manifest written beside each index and recovered at load time, and a
   declared value contradicting the index raises.
3. **Missing scope notes reached the prompt as the string `nan`.**
4. **The prompt conflated two different hints.** Where no predicted type existed, the *category*
   was interpolated into a field labelled `PREDICTED TYPE`, so the model was shown a category
   name as a proposed object type. They are now separate fields and which was used is recorded.
5. **The AAT fallback fired on any empty result**, including API failures — laundering a
   transient outage into an AAT type assertion. It now fires only on a genuine abstention.

The evaluation also changed a deployed component: the category prompt was reduced to the seven
names, on the parsimony argument in §3. Note for the record that the released BME KG (v1.1)
was built with the richer prompt, since it predates this measurement.

---

## 7. Limitations

1. **Ceiling effects on the category task.** `C2` sits at 98.0% with an interval reaching 100%.
   The sample can establish that the LLM categoriser vastly outperforms both baselines; it
   cannot rank it against a plausible alternative LLM configuration.
2. **Three of seven categories are unattested as primary labels.** Award & Recognition is
   absent entirely (2 objects, 0.4% of the collection); Merchandise and Personal Effects occur
   only as secondary labels. Macro-averages are over the four attested classes. This is
   expected from a random-by-case sample of a skewed collection, but it means the categories
   where the scope notes would most plausibly bind are precisely the ones not tested.
3. **A fifth of the type gold standard is approximate** (§2.2), concentrated in two object
   types and clustered by display case.
4. **Type figures assume correct categories.** They are an upper bound on end-to-end
   behaviour; the bound is nearly tight here only because category accuracy is at ceiling.
5. **Small per-cell counts.** Several breakdowns rest on 6–14 objects and are indicative.
6. **A single annotator**, hence no inter-annotator agreement.
7. **One collection, one language, one embedding model.** Whether the depth/abstention
   interaction of §5.3 generalises is untested.

---

## 8. Reproducibility

| Stage | Script | Output |
| --- | --- | --- |
| gold standard | `gold.py --verify` | tiers, reachability, URI audit |
| indices | `build_indices.py` | 16 tagged FAISS builds |
| Experiment A | `run_category.py` | `data/predictions/category_*.jsonl` |
| Experiment B | `run_retrieval.py` | `data/predictions/retrieval_*.jsonl` |
| Experiment C | `run_selection.py` | `data/predictions/selection_*.jsonl` |
| metrics | `score.py` | `data/results/*.csv`, `stats.json` |
| deliverables | `report.py` | `latex/*.tex`, `figures/*.pdf`, `report.md` |

All LLM responses are cached on disk keyed by `sha1(model, prompt, run_index)`, so the
experiments replay without re-billing and the reported numbers are reproducible from the
prediction files alone. Retrieval and scoring require no API access at all.
