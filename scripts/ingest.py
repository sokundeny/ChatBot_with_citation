"""Stream, clean and deduplicate physics papers into resumable JSONL files."""
import argparse
import hashlib
import json
import re
import os
import unicodedata
from itertools import islice
from pathlib import Path

DATASET = "konsman/physics-corpus"
CONFIG = "quantum-physics"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["paper_id", "arxiv_id", "title", "abstract", "primary_category", "arxiv_categories", "submission_date", "full_text"]


def clean(value):
    # Preserve case, punctuation, symbols and equations for scientific retrieval.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value or "")).strip()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def process(rows, output, identity, limit=1000, batch_size=100, category=None):
    if category:
        identity = f"{identity}:category={category}"
    output = Path(output)
    if output.suffix.lower() != ".jsonl":
        raise ValueError("Use a .jsonl output filename (one JSON paper per line).")
    if limit < 0 or batch_size < 1:
        raise ValueError("limit must be >= 0 and batch_size must be >= 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(".progress.json")
    state = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    if state and state["source"] != identity:
        raise ValueError("Output belongs to another source/configuration; use another --output.")
    if state is None:
        if output.exists() and output.stat().st_size:
            raise ValueError("Existing JSONL has no progress file; use a new output path.")
        state = dict(source=identity, source_offset=0, output_bytes=0, total_documents=0)
    if state["output_bytes"] and not output.exists():
        raise ValueError("Progress file exists but its data file is missing.")
    seen_ids, seen_hashes = set(), set()
    offset = state["source_offset"]
    scanned = inserted = empty = duplicates = filtered = 0
    with output.open("r+b" if output.exists() else "w+b") as handle:
        handle.seek(0, 2)
        if handle.tell() < state["output_bytes"]:
            raise ValueError("Data file is shorter than its saved progress.")
        # Discard an incomplete batch after an interrupted write.
        handle.truncate(state["output_bytes"])
        handle.seek(0)
        count = 0
        for line in handle:
            document = json.loads(line)
            seen_ids.add(document["paper_id"])
            seen_hashes.add(document["text_hash"])
            count += 1
        if count != state["total_documents"]:
            raise ValueError("Saved document count does not match the saved progress.")
        # Only IDs and hashes are retained in RAM; paper text remains on disk.
        def commit(completed=False):
            handle.flush()
            os.fsync(handle.fileno())
            state.update(source_offset=offset + scanned, output_bytes=handle.tell(),
                         total_documents=count + inserted,
                         scanned_this_run=scanned, inserted_this_run=inserted,
                         skipped_empty_abstract=empty, skipped_duplicate=duplicates,
                         skipped_category=filtered, category=category,
                         status="completed" if completed else "in_progress")
            write_json(progress_path, state)

        commit()
        # Resume replays source rows up to the saved progress without saving them again.
        for row in islice(rows, offset, None if limit == 0 else offset + limit):
            title, abstract = clean(row.get("title")), clean(row.get("abstract"))
            text = "\n\n".join(x for x in (title, abstract) if x)
            categories = row.get("arxiv_categories") or []
            if category and category != row.get("primary_category") and category not in categories:
                filtered += 1
            elif not abstract:
                empty += 1
            else:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                paper_id = str(row.get("paper_id") or row.get("arxiv_id") or digest)
                if paper_id in seen_ids or digest in seen_hashes:
                    duplicates += 1
                else:
                    document = dict(paper_id=paper_id, title=title, abstract=abstract,
                                    text=text, text_hash=digest,
                                    primary_category=row.get("primary_category"),
                                    arxiv_categories=categories,
                                    submission_date=row.get("submission_date"),
                                    full_text=clean(row.get("full_text")))
                    handle.write((json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8"))
                    seen_ids.add(paper_id)
                    seen_hashes.add(digest)
                    inserted += 1
            scanned += 1
            if scanned % batch_size == 0:
                commit()
                print(f"Scanned {scanned}; saved {inserted}; source offset {offset + scanned}", flush=True)
        commit(completed=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Defaults to data/physics-CATEGORY.jsonl")
    parser.add_argument("--category", default="quant-ph", help="arXiv tag to include (primary or cross-listed); all disables filtering")
    parser.add_argument("--limit", type=int, default=1000, help="Source rows per run; 0 processes all remaining rows")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--input-jsonl", type=Path, help="Use local JSONL instead of Hugging Face")
    parser.add_argument("--revision", default="main", help="Pin a Hub commit for reproducible processing")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("limit must be >= 0 and batch-size must be >= 1")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", args.category):
        parser.error("category must be an arXiv tag or 'all'")
    category = None if args.category == "all" else args.category
    output = args.output or PROJECT_ROOT / "data" / (f"physics-{category}.jsonl" if category else "physics.jsonl")
    if args.input_jsonl:
        identity = f"local:{args.input_jsonl.resolve()}"
        with args.input_jsonl.open(encoding="utf-8") as handle:
            result = process((json.loads(line) for line in handle if line.strip()),
                             output, identity, args.limit, args.batch_size, category)
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        # Resolve main once; persist the exact commit so resume cannot silently mix revisions.
        revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
        identity = f"{DATASET}:{CONFIG}:train:{revision}:title-abstract-fulltext-v1"
        rows = load_dataset(DATASET, CONFIG, split="train", streaming=True,
                            columns=COLUMNS, revision=revision)
        result = process(iter(rows), output, identity, args.limit, args.batch_size, category)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
