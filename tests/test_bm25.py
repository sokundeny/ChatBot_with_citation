import json
import math
from pathlib import Path
import random
import sys
import tempfile

# Ensure repository root is on sys.path so 'scripts.index_bm25' is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.index_bm25 import BM25Index

# Same three-document fixture as the TF-IDF test
FIXTURE_CHUNKS = [
    {
        "chunk_id": "D1",
        "paper_id": "paper_001",
        "chunk_index": 0,
        "title": "Quantum Physics",
        "text": "Quantum quantum physics",
        "tokens": ["quantum", "quantum", "physics"]
    },
    {
        "chunk_id": "D2",
        "paper_id": "paper_002",
        "chunk_index": 0,
        "title": "Quantum Computing",
        "text": "Quantum computing",
        "tokens": ["quantum", "computing"]
    },
    {
        "chunk_id": "D3",
        "paper_id": "paper_003",
        "chunk_index": 0,
        "title": "Physics Experiment",
        "text": "Physics experiment",
        "tokens": ["physics", "experiment"]
    }
]


def close(actual, expected):
    return math.isclose(
        actual,
        expected,
        rel_tol=1e-5,
        abs_tol=1e-6
    )


def raises(error_type, function):
    """True if calling function() raises error_type (works with or without pytest)."""
    try:
        function()
    except error_type:
        return True
    return False


def brute_force_scores(chunks, query_tokens, k1=1.5, b=0.75):
    """Score every chunk straight from the BM25 formula, with no inverted index."""
    n = len(chunks)
    avgdl = sum(len(c["tokens"]) for c in chunks) / n
    scores = {}
    for chunk in chunks:
        total = 0.0
        for term in query_tokens:
            df = sum(1 for c in chunks if term in c["tokens"])
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = chunk["tokens"].count(term)
            if tf == 0:
                continue
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(chunk["tokens"]) / avgdl))
        if total > 0:
            scores[chunk["chunk_id"]] = total
    return scores


def test_bm25_index():
    # Build index from the three-document fixture
    index = BM25Index(FIXTURE_CHUNKS)

    # Expected number of non-empty documents and average length (3 + 2 + 2) / 3
    assert index.total_docs == 3
    assert close(index.avg_doc_length, 7 / 3)

    # 1. Verify IDF: ln(1 + (N - DF + 0.5) / (DF + 0.5))
    assert index.df["quantum"] == 2
    assert close(index.idf["quantum"], math.log(1.6))      # DF=2 -> ln(1 + 1.5/2.5)
    assert close(index.idf["computing"], math.log(8 / 3))  # DF=1 -> ln(1 + 2.5/1.5)
    assert close(index.idf["quantum"], 0.470004)
    assert close(index.idf["computing"], 0.980829)

    # 2. Verify postings hold RAW term counts (not relative frequency)
    quantum_postings = dict(index.postings["quantum"])
    assert quantum_postings == {"D1": 2, "D2": 1}

    # 3. Verify chunk lengths are token counts
    assert index.doc_lengths == {"D1": 3, "D2": 2, "D3": 2}

    # 4. Verify metadata
    assert index.doc_metadata["D1"]["paper_id"] == "paper_001"
    assert index.doc_metadata["D1"]["title"] == "Quantum Physics"

    # 5. Verify searching and BM25 ranking (k1=1.5, b=0.75)
    hits_quantum = index.search("quantum", top_k=3)
    assert len(hits_quantum) == 2
    assert hits_quantum[0]["chunk_id"] == "D1"
    assert close(hits_quantum[0]["score"], 0.614958)
    assert hits_quantum[1]["chunk_id"] == "D2"
    assert close(hits_quantum[1]["score"], 0.502294)

    # Query matching D2 on both terms: the rare term "computing" makes D2 win
    hits_d2 = index.search(["quantum", "computing"], top_k=2)
    assert len(hits_d2) == 2
    assert hits_d2[0]["chunk_id"] == "D2"
    assert close(hits_d2[0]["score"], 1.550508)
    assert hits_d2[1]["chunk_id"] == "D1"
    assert close(hits_d2[1]["score"], 0.614958)

    # A term repeated in the query counts once per repetition
    hits_repeat = index.search(["quantum", "quantum", "computing"], top_k=2)
    assert close(hits_repeat[0]["score"], 2.052802)
    assert close(hits_repeat[1]["score"], 1.229916)

    # Result format matches the TF-IDF search (citation key included)
    assert hits_quantum[0]["citation"] == "[D1]"
    assert set(hits_quantum[0]) == {"chunk_id", "citation", "paper_id", "chunk_index", "title", "score", "text"}

    # Search out-of-vocabulary terms, empty query and top_k limit
    assert index.search(["unknownterm"], top_k=5) == []
    assert index.search([], top_k=5) == []
    assert len(index.search("quantum", top_k=1)) == 1

    print("All BM25 index and search tests passed!")


def test_idf_is_never_negative():
    # A term in every chunk would give a negative IDF in the classic formula
    chunks = [
        {"chunk_id": f"C{i}", "tokens": ["common", f"rare{i}"]} for i in range(10)
    ]
    index = BM25Index(chunks)
    assert index.df["common"] == 10
    assert index.idf["common"] > 0
    assert index.idf["common"] < index.idf["rare0"]


def test_length_normalization_and_saturation():
    # Both chunks contain "quantum" once, but D_long is much longer
    chunks = [
        {"chunk_id": "D_short", "tokens": ["quantum", "x1"]},
        {"chunk_id": "D_long", "tokens": ["quantum"] + [f"w{i}" for i in range(20)]},
    ]
    index = BM25Index(chunks)
    hits = index.search(["quantum"])
    assert hits[0]["chunk_id"] == "D_short"  # shorter chunk ranks higher

    # b = 0 switches length normalization off: same tf -> same score
    index.set_parameters(1.5, 0.0)
    hits = index.search(["quantum"])
    assert close(hits[0]["score"], hits[1]["score"])

    # Saturation: score never goes above IDF * (k1 + 1), however many repeats
    chunks = [
        {"chunk_id": "A", "tokens": ["quantum"] * 5 + ["pad"]},
        {"chunk_id": "B", "tokens": ["quantum"] * 50 + ["pad"]},
        {"chunk_id": "C", "tokens": ["other", "pad"]},
    ]
    index = BM25Index(chunks, k1=1.5, b=0.0)
    scores = {h["chunk_id"]: h["score"] for h in index.search(["quantum"])}
    assert scores["B"] > scores["A"]
    assert scores["B"] < index.idf["quantum"] * (1.5 + 1)
    assert scores["B"] < 2 * scores["A"]  # 10x more repeats does not mean 10x the score


def test_matches_brute_force_on_random_corpus():
    rng = random.Random(42)
    vocabulary = [f"term{i}" for i in range(30)]
    chunks = []
    for i in range(60):
        length = rng.randint(1, 40)
        chunks.append({"chunk_id": f"R{i}", "tokens": [rng.choice(vocabulary) for _ in range(length)]})

    for k1, b in [(1.5, 0.75), (1.2, 0.5), (2.0, 1.0), (0.0, 0.75)]:
        index = BM25Index(chunks, k1=k1, b=b)
        for _ in range(20):
            query = [rng.choice(vocabulary + ["missing"]) for _ in range(rng.randint(1, 4))]
            expected = brute_force_scores(chunks, query, k1, b)
            hits = index.search(query, top_k=len(chunks))
            assert {h["chunk_id"] for h in hits} == set(expected)
            for hit in hits:
                assert close(hit["score"], expected[hit["chunk_id"]])
            scores = [h["score"] for h in hits]
            assert scores == sorted(scores, reverse=True)


def test_skips_empty_chunks_and_rejects_duplicates():
    chunks = FIXTURE_CHUNKS + [{"chunk_id": "EMPTY", "tokens": []}]
    index = BM25Index(chunks)
    assert index.total_docs == 3
    assert "EMPTY" not in index.doc_lengths

    assert raises(ValueError, lambda: BM25Index(FIXTURE_CHUNKS + [FIXTURE_CHUNKS[0]]))


def test_invalid_parameters():
    assert raises(ValueError, lambda: BM25Index(k1=-1))
    assert raises(ValueError, lambda: BM25Index(b=1.5))
    assert raises(TypeError, lambda: BM25Index(FIXTURE_CHUNKS).search(123))


def test_build_from_jsonl_and_save_load():
    with tempfile.TemporaryDirectory() as folder:
        tmp_path = Path(folder)
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            "\n".join(json.dumps(chunk) for chunk in FIXTURE_CHUNKS) + "\n", encoding="utf-8"
        )

        built = BM25Index()
        built.build(chunks_file)
        index_file = tmp_path / "nested" / "bm25_index.json"
        built.save(index_file)

        loaded = BM25Index()
        loaded.load(index_file)
        assert loaded.total_docs == built.total_docs
        assert close(loaded.avg_doc_length, built.avg_doc_length)
        assert (loaded.k1, loaded.b) == (built.k1, built.b)
        assert loaded.search(["quantum", "computing"]) == built.search(["quantum", "computing"])

        # Tuning after load changes the scores but needs no rebuild
        loaded.set_parameters(1.2, 0.5)
        assert loaded.search(["quantum"])[0]["score"] != built.search(["quantum"])[0]["score"]

        assert raises(FileNotFoundError, lambda: BM25Index().build(tmp_path / "missing.jsonl"))


def test_load_rejects_tfidf_index():
    with tempfile.TemporaryDirectory() as folder:
        fake = Path(folder) / "tfidf_index.json"
        fake.write_text(json.dumps({"version": "1.0", "total_docs": 1}), encoding="utf-8")
        assert raises(ValueError, lambda: BM25Index().load(fake))


if __name__ == "__main__":
    test_bm25_index()
    test_idf_is_never_negative()
    test_length_normalization_and_saturation()
    test_matches_brute_force_on_random_corpus()
    test_skips_empty_chunks_and_rejects_duplicates()
    test_invalid_parameters()
    test_build_from_jsonl_and_save_load()
    test_load_rejects_tfidf_index()
    print("All extra BM25 checks passed!")