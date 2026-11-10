# Ingest Pipeline

End-to-end pipeline: raw item descriptions → knowledge graph (TTL).

Run all commands from the `lechoes/src/` directory.

## Inputs

The pipeline reads an institution's own data; none of it lives in this repository.

- **The collection CSV** (`csv_path`), one row per object.
- **The layout files** `zone_definitions.csv` and `zone_case_mappings.csv`, which give `core.ttl` its zones and cases and give each object its location. They are read from `--layout-dir`, which defaults to the collection CSV's own directory, so keeping the three files together needs no flag. If they are missing the run continues, but it logs a warning and the graph has no locations.

The British Music Experience's inputs live in [`onradio/bme-kg`](https://github.com/onradio/bme-kg).

---

## Basic usage

```bash
python ingest_pipeline.py path/to/collection.csv \
    --api-key YOUR_KEY \
    --output-dir path/to/output
```

The input CSV must contain a `description` or `item_description` column, and optionally a `caption` column.

---

## Common commands

### Process all rows
```bash
python ingest_pipeline.py path/to/collection.csv \
    --api-key YOUR_KEY \
    --output-dir path/to/output
```

### Process first N rows (for testing)
```bash
python ingest_pipeline.py path/to/collection.csv \
    --api-key YOUR_KEY \
    --output-dir path/to/output \
    --n-rows 5
```

### Debug mode (first 5 rows, no `--n-rows` needed)
```bash
python ingest_pipeline.py path/to/collection.csv \
    --api-key YOUR_KEY \
    --output-dir path/to/output \
    --debug
```

### Resume after interruption
Picks up from where the last run left off using the checkpoint file.
```bash
python ingest_pipeline.py path/to/collection.csv \
    --api-key YOUR_KEY \
    --output-dir path/to/output \
    --resume
```

### Retry failed rows from a previous report
Reads a report CSV, reprocesses only rows where `status=failed` and `retryable=true`,
and writes a new `ingest_report_retry_{timestamp}.csv` — the original report is not modified.
```bash
python ingest_pipeline.py path/to/collection.csv \
    --api-key YOUR_KEY \
    --output-dir path/to/output \
    --retry-report path/to/output/reports/ingest_report_YYYYMMDD_HHMMSS.csv
```

### Aggregate existing TTL files without re-running ingestion
Reads all `.ttl` files from `{output-dir}/ttl/`, merges them, and writes
`{output-dir}/knowledge_graph.ttl`. Also runs KG validation and stats.
```bash
python ingest_pipeline.py \
    --api-key YOUR_KEY \
    --output-dir path/to/output \
    --aggregate-ttl-only
```

With source CSV for coverage validation (source vs TTL check):
```bash
python ingest_pipeline.py \
    --api-key YOUR_KEY \
    --output-dir path/to/output \
    --aggregate-ttl-only \
    --source-csv path/to/collection.csv
```

---

## All options

| Flag | Default | Description |
|---|---|---|
| `csv_path` | *(required)* | Input CSV file |
| `--api-key` | *(required)* | Gemini API key |
| `--output-dir` | `assets/triples` (gitignored) | Directory for TTL files and reports |
| `--aggregate-file` | `knowledge_graph.ttl` | Filename for the aggregated graph |
| `--vocab-path` | `vocabularies/all_concepts.csv` | Path to vocabulary CSV |
| `--base-uri` | `http://radio.liv.ac.uk/bme/` | Base URI for new resources |
| `--uri-strategy` | `uuid` | URI minting strategy (`uuid` or `slug`) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Embedding model |
| `--generator-model` | `gemini-2.5-flash` | Gemini model for classification/linking |
| `--classifier-top-types` | `20` | Top-k types for classification |
| `--linker-top-entities` | `10` | Top-k entities for linking |
| `--n-rows` | *(all)* | Limit to first N rows |
| `--debug` | `false` | Process first 5 rows only |
| `--resume` | `false` | Skip rows already processed per checkpoint |
| `--retry-report` | — | Report CSV to retry failed rows from |
| `--aggregate-ttl-only` | `false` | Aggregate TTLs only, skip ingestion |
| `--source-csv` | — | Source CSV for coverage validation (with `--aggregate-ttl-only`) |
| `--layout-dir` | the CSV's directory | Directory holding `zone_definitions.csv` and `zone_case_mappings.csv` |

---

## Output files

```
{output-dir}/
├── core.ttl                                  # generated at pipeline startup
├── knowledge_graph.ttl                       # aggregated graph
├── ttl/
│   └── {timestamp}_{row:03d}_{identifier}.ttl   # per-item TTL files
├── checkpoints/
│   └── ingest_checkpoint.json               # resume state
├── extractions/
│   └── {timestamp}_{row:03d}.json           # raw extraction output per row
├── reports/
│   ├── ingest_report_{timestamp}.csv        # row-level status report
│   ├── kg_validation_{timestamp}.txt        # KG validation report
│   └── kg_stats_{timestamp}.txt             # KG stats report
└── logs/
    ├── run_manifest_{timestamp}.json
    └── run_report_{timestamp}.txt
```

### Report CSV columns

| Column | Description |
|---|---|
| `source_file` | Input CSV path |
| `row_index` | Row number in source file |
| `record_id` | `case_number` if present |
| `status` | `success` or `failed` |
| `last_successful_stage` | Last stage that completed |
| `failed_stage` | Stage that raised the exception |
| `failure_type` | Exception class name |
| `failure_message` | Error message |
| `failure_trace` | Full traceback |
| `output_file` | Output file path |
| `ttl_file` | TTL filename |
| `processed_at` | UTC timestamp |
| `retryable` | `true` or `false` |

---

## Resilience features

- **Circuit breaker**: halts the pipeline after 3 consecutive or 5 total Gemini failures
- **Checkpointing**: every row (success or failure) is checkpointed; use `--resume` to continue
- **Retry**: pass any report CSV to `--retry-report` to reprocess only failed retryable rows
- **Report CSV**: flushed after every row — intact even if the pipeline crashes mid-run
