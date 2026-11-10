# Evaluation of metadata extraction

*Draft material for Section 4.2 of the Linked Echoes article. Every figure quoted here is produced by `score.py` from the prediction files in `data/predictions/` and formatted by `report.py`; nothing is transcribed by hand. The prose numbers are also emitted as `latex/macros.tex`, so a sentence in the article cannot quote a stale figure.*

| Deliverable | Source | Placement |
| --- | --- | --- |
| Figure 1 — the operating plane | `figures/operating_plane.pdf` | §4.2, main text |
| Table 1 — systems | `latex/table1_systems.tex` | §4.2, main text |
| Table 2 — per field | `latex/table2_per_field.tex` | §4.2, main text |
| Prose numbers | `latex/macros.tex` | inline |

---

## 1. What the gold supports

209 gold values over 62 objects in 21 display cases; 173 occur literally in the case evidence and 36 can only be reached by inference. 496 of the 682 (object, field) cells are empty, so precision is mostly lost on cells where there is nothing to find and recall can only be won on the 186 that carry a value.

**139 sibling distractors over 49 of the 62 objects** — values visible in an object's display-case evidence that belong to a neighbour. That is what exhaustive extraction runs into, and it is imposed by the corpus rather than by any prompt. **Recall is a ceiling here — for extractive systems; precision is a reference point, not a bound** (§2.4):

| Scope | Evidence | Asserts | Exhaustive P | Recall ceiling | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | description only | 103 | 0.942 | 0.464 | 0.622 |
| E2 | + own caption | 231 | 0.723 | 0.799 | 0.759 |
| E3 | + siblings | 392 | 0.441 | 0.828 | 0.576 |

## 2. Systems

| System | Scope | Policy | Values | P | R | F1 | R_sup | Yield | Spurious | Exh. P | Adj. P | McNemar p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0a spaCy | E2 | n/a | 385.0 | 0.314 | 0.579 | 0.407 | 0.671 | 0.139 | 0.089 | 0.723 | 0.314 | 0.931 |
| M0b GLiNER | E2 | n/a | 297.0 | 0.380 | 0.541 | 0.447 | 0.618 | 0.167 | 0.163 | 0.723 | 0.380 | 0.271 |
| M1 | E1 | free | 170.3 | 0.667 | 0.544 | 0.599 | 0.620 | 0.176 | 0.076 | 0.942 | 0.778 | 0.000 |
| M2 | E1 | quote | 129.3 | 0.714 | 0.442 | 0.546 | 0.522 | 0.056 | 0.051 | 0.942 | 0.714 | 0.000 |
| M3 (ref) | E2 | free | 395.7 | 0.414 | 0.783 | 0.541 | 0.904 | 0.204 | 0.200 | 0.723 | 0.420 | — |
| M4 | E2 | quote | 399.0 | 0.404 | 0.772 | 0.531 | 0.902 | 0.148 | 0.200 | 0.723 | 0.404 | 0.265 |
| M5 | E3 | free | 316.0 | 0.506 | 0.766 | 0.610 | 0.869 | 0.269 | 0.167 | 0.441 | 0.527 | 0.001 |
| M6 | E3 | quote | 318.0 | 0.481 | 0.732 | 0.581 | 0.850 | 0.167 | 0.164 | 0.441 | 0.481 | 0.005 |

### What the table says

**Attribution is close to free.** Moving from `E2` to `E3` — the same model, the same policy, with the attribution instruction and the siblings named — raises precision by 0.093 (0.414 → 0.506) at a cost of 0.018 recall (0.783 → 0.766), so F1 rises from 0.541 to 0.610. The mechanism is visible in the error attribution: the misattribution share of false positives falls from 0.138 to 0.060.

**M5 and M6 exceed the exhaustive precision of their scope** (0.506 and 0.481 against 0.441). That is not a bound being broken: exhaustive precision is what you get by copying every gold-vocabulary value visible in the evidence, and any well-chosen subset of those assertions is more precise. Exceeding it is the definition of doing attribution rather than copying, and no other cell does it. The *recall* column of §2.4 is the one that is a genuine ceiling, and only for extractive systems: every quote-policy cell respects its scope's, while M1 exceeds one — by inferring values that occur in no evidence at all, which is what `inference_yield` measures.

**The evidence check is not the lever the attribution instruction is.** The quote policy raises precision at `E1` (0.667 → 0.714) but costs recall (0.544 → 0.442), and at `E2` and `E3` it moves both slightly down. What it does do is eliminate hallucination outright: every quote-policy cell has a hallucination share of exactly 0.000, because a value whose quote must occur in the input cannot fall outside it.

**Both NER baselines are floors, and they are not close.** `M0a` reaches six of the eleven fields — the `EVENT` label never fires on this corpus — and its recall is bounded above by 0.761 before any error of its own. `M0b`, which takes its label set at inference time, reaches all eleven and still scores 0.380 precision.

## 3. Per field

`†` marks a field with fewer than 10 gold values, whose figures are indicative only. Columns are M3 (ref) and M5.

| Field | Gold | Sup. | Inf. | M3 (ref) P | M3 (ref) R | M5 P | M5 R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `music_genres` | 24 | 9 | 15 | 0.226 | 0.333 | 0.261 | 0.403 |
| `music_works` | 26 | 26 | 0 | 0.375 | 0.962 | 0.629 | 0.936 |
| `music_albums` | 14 | 11 | 3 | 0.241 | 0.500 | 0.429 | 0.500 |
| `music_artists` | 73 | 71 | 2 | 0.585 | 0.909 | 0.680 | 0.904 |
| `events`† | 1 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `countries`† | 9 | 2 | 7 | 0.296 | 0.593 | 0.295 | 0.667 |
| `places`† | 6 | 6 | 0 | 0.286 | 1.000 | 0.353 | 1.000 |
| `venues`† | 9 | 8 | 1 | 0.727 | 0.889 | 0.727 | 0.889 |
| `years` | 35 | 35 | 0 | 0.540 | 0.971 | 0.630 | 0.829 |
| `colours`† | 3 | 3 | 0 | 0.500 | 1.000 | 0.600 | 1.000 |
| `materials`† | 9 | 1 | 8 | 0.214 | 0.111 | 0.214 | 0.111 |

## 4. In prose

- **Zero-shot ablation (§3.8), and what adjudication does to it.** Removing the few-shot block from the reference cell costs 0.043 of *strict* precision (0.414 → 0.371) at unchanged recall, and raises the hallucination share more than fivefold (0.024 → 0.129). **The adjudication reverses that reading.** 19 of the zero-shot cell's hallucinated values per run were judged correct but unrecorded, against 0 for few-shot, so adjudicated precision is 0.427 against 0.420 and the two brackets overlap entirely. The examples do teach restraint — but what the restraint suppresses is largely *true*, and absent from the catalogue rather than wrong. Which of the two a museum wants is the question §0.3 says the institution answers, not the evaluation.
- **Gold incompleteness (§4.5), adjudicated.** The largest hallucination share is 0.359 (M1) against a pre-registered threshold of 0.15, so the conditional fired. The pool — 91 distinct proposals — was **judged whole**, not sampled, so every hallucinated value carries a verdict and nothing is extrapolated: 41 correct, 19 incorrect, 31 unverifiable. Adjudicated precision is reported as a bracket, because the annotators marked a value `unverifiable` whenever the description and caption could not settle it *even if outside knowledge suggests it is true* — an exclusion that selects precisely for the values the evidence cannot reach, so extrapolating across it is the one assumption worth refusing to make silently.
- **What the correction does and does not move.** Strict precision is a lower bound on M3 by 0.006 and on `M1` by 0.111 — the largest movement in the table, and the reason is structural: `M1` reads the description alone under the free policy, so most of what it adds is knowledge the caption never states. **Every quote-policy cell is unmoved**, because a value whose quote must occur in the input cannot be a hallucination: for `M2`, `M4` and `M6`, strict precision is not a lower bound but the number itself. The ranking is unchanged under every reading of the bracket.
- **Determinism.** At temperature 0, cells whose outcome differs across the three runs: M1 24, M2 7, M3 10, M3-zeroshot 36, M4 15, M5 23, M6 23 of 682.
- **Failures.** 0 API or parse failures across the whole sweep; no object was excluded from any cell.
- **Annotation protocol.** The gold is a consensus product of three annotators working to a single agreed set of values, so inter-annotator agreement is undefined by construction rather than missing.

---

*Regenerate with `python score.py && python report.py`. Do not edit this file.*
