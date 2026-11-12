# Quantum Physics Paper Corpus

This project prepares a collection of quantum physics paper titles and abstracts
for information retrieval. Data comes from
[konsman/physics-corpus](https://huggingface.co/datasets/konsman/physics-corpus),
using its `quantum-physics` configuration and `train` split.

## Project structure

```text
scripts/
  ingest.py                           # Download, filter, clean and save papers
data/
  physics-quant-ph.jsonl               # Saved papers: one JSON object per line
  physics-quant-ph.progress.json       # Resume progress and processing counts
requirements.txt                      # Python dependencies
```

The `data` directory is created automatically and excluded from Git.

## Setup and ingestion

Use Python 3.10 or newer. Run these commands from the project folder:

```powershell
python -m pip install -r requirements.txt
python scripts/ingest.py --category quant-ph --limit 1000
```

`quant-ph` is the default category. `--limit 1000` scans the next 1,000 source
records; it does **not** guarantee 1,000 saved papers, because category filtering,
missing abstracts and duplicates can reduce the saved count.

Run the command again to continue from the saved progress. To process all
remaining source records:

```powershell
python scripts/ingest.py --category quant-ph --limit 0
```

Internet access to Hugging Face is required. Default output paths are relative to
the project folder, regardless of where the script is launched. A custom
`--output` path is relative to your current terminal directory.

## Category filtering

`primary_category` is a paper's main subject. `arxiv_categories` contains its
subject tags, including cross-listed categories.

The processor keeps a paper if `quant-ph` appears in `arxiv_categories`, or its
`primary_category` is `quant-ph`. For example, a paper with primary category
`hep-th` and tags `["hep-th", "quant-ph"]` is included.

This filter selects quantum physics broadly, including topics such as quantum
computing and entanglement. It does not classify papers into individual topics.
Category definitions are available in the
[arXiv taxonomy](https://arxiv.org/category_taxonomy).

To select another category, use `--category hep-th` or `--category gr-qc`.
Use `--category all` to disable filtering. Each category uses a separate default
output filename; `all` uses `data/physics.jsonl`.

## Processing steps

1. Stream only the needed source columns instead of downloading the full corpus.
2. Select papers matching the requested category.
3. Normalize Unicode to NFC and collapse repeated whitespace in titles and abstracts.
4. Skip papers with empty or missing abstracts.
5. Remove duplicates using paper IDs or SHA-256 hashes of cleaned title-plus-abstract text.
6. Append accepted papers to UTF-8 JSONL and save progress every 100 scanned records.

Cleaning preserves case, punctuation, equations and scientific symbols. It does
not check factual quality or detect near-duplicate papers. Full text, sections,
chunking, tokenization and embeddings are not included yet.

## Saved fields

| Field | Meaning |
| --- | --- |
| `paper_id` | Source paper ID; falls back to arXiv ID or text hash if missing |
| `title` | Cleaned title |
| `abstract` | Cleaned abstract |
| `text` | Title and abstract joined by a blank line, ready for retrieval |
| `text_hash` | SHA-256 hash of `text`, used for deduplication |
| `primary_category` | Main arXiv subject category |
| `arxiv_categories` | All available arXiv category tags |
| `submission_date` | Submission date provided by the source |

JSONL stores one complete JSON object per line, rather than a single JSON array.
You can open it in a text editor or read one record at a time:

```python
import json

with open("data/physics-quant-ph.jsonl", encoding="utf-8") as papers:
    for line in papers:
        paper = json.loads(line)
        print(paper["title"])
```

## Resume progress and processing counts

The progress file stores the source identity, source offset, committed output size,
document count and processing counts. It is updated every 100 scanned records
and when a run finishes. On restart, the script discards any incomplete batch after
the saved progress, reloads saved IDs and hashes, and resumes processing. There is no
SQLite dependency. ID and hash sets grow in memory as the corpus grows; paper
text is processed one record at a time. Resuming replays earlier source records,
so restarting later in the corpus can take longer.

The source identity includes the exact Hugging Face commit and category. If the
upstream `main` revision changes, use `--revision COMMIT` with the commit from
your progress file's `source` value, or select a new output path. Do not edit the
JSONL or progress file manually, or run two ingestion processes against the same
output at once.

The same progress file contains scanned, inserted and skipped counts for the
current or most recent run, alongside the cumulative source offset and saved
document count. Its `status` is `in_progress` during ingestion and `completed`
when a run finishes successfully. There is no separate checkpoint or report file.

For a local JSONL source with the same input fields:

```powershell
python scripts/ingest.py --input-jsonl input/papers.jsonl --output data/local-quant-ph.jsonl --limit 1000
```

Keep the local input unchanged when resuming. Keep source attribution when
sharing processed data; consult the dataset card for licensing details.
