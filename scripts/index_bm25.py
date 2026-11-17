"""BM25 Inverted Indexer and Search Engine for ChatBot with Citation.

Exact formulas implemented (Okapi BM25 with the smoothed Lucene-style IDF):
  - tf(t, d)    = count(t, d)                         (raw term count in the chunk)
  - |d|         = totalTokens(d)                      (chunk length)
  - avgdl       = sum_d |d| / N                       (average chunk length)
  - DF(t)       = Number of chunks containing term t
  - IDF(t)      = ln(1 + (N - DF(t) + 0.5) / (DF(t) + 0.5))   (never negative)
  - TFpart(t,d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
  - BM25(q, d)  = sum over query terms t of  IDF(t) * TFpart(t, d)

Defaults: k1 = 1.5 (term frequency saturation), b = 0.75 (length normalization).
A term repeated in the query counts once per repetition.
The index stores raw counts and lengths, so k1 and b can be changed without rebuilding.
"""

import argparse
from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from preprocess import tokenize
except ImportError:
    try:
        from scripts.preprocess import tokenize
    except ImportError:
        tokenize = None

DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "physics-quant-ph.chunks.jsonl"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "bm25_index.json"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Indexer:
    def __init__(self, chunks=None, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = DEFAULT_K1
        self.b = DEFAULT_B
        self.set_parameters(k1, b)

        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.vocabulary = set()
        self.df = {}             # term -> int (document frequency)
        self.idf = {}            # term -> idf float
        self.postings = {}       # term -> list of [chunk_id, raw term count]
        self.doc_lengths = {}    # chunk_id -> int (number of tokens |d|)
        self.doc_metadata = {}   # chunk_id -> dict(paper_id, chunk_index, title, text)

        if chunks is not None:
            self.build(chunks)

    def set_parameters(self, k1: float, b: float):
        """Set the BM25 tuning parameters. k1 >= 0 and 0 <= b <= 1."""
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b

    def build(self, chunks_input):
        """Build the BM25 inverted index.

        Args:
            chunks_input: Either a Path/str pointing to a .jsonl file,
                          or a list/iterable of chunk dictionary records.
        """
        # Support both file path and in-memory list/generator of dictionaries
        if isinstance(chunks_input, (str, Path)):
            chunks_path = Path(chunks_input)
            if not chunks_path.exists():
                raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

            def record_generator():
                with chunks_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            records = record_generator()
        else:
            records = chunks_input

        postings = defaultdict(list)
        self.doc_lengths = {}
        self.doc_metadata = {}

        # Step 1: Read every chunk once: raw term counts, chunk length, citation metadata
        for record in records:
            chunk_id = record["chunk_id"]
            tokens = record.get("tokens", [])
            if not tokens:
                continue
            if chunk_id in self.doc_lengths:
                raise ValueError(f"Duplicate chunk_id found: {chunk_id}")

            # Chunk length |d| = number of tokens
            self.doc_lengths[chunk_id] = len(tokens)

            # One posting per (term, chunk): [chunk_id, count(t, d)]
            for term, count in Counter(tokens).items():
                postings[term].append([chunk_id, count])

            # Save metadata for citation linking
            self.doc_metadata[chunk_id] = {
                "paper_id": record.get("paper_id"),
                "chunk_index": record.get("chunk_index"),
                "title": record.get("title"),
                "text": record.get("text")
            }

        self.postings = dict(postings)
        self.total_docs = len(self.doc_lengths)
        self.vocabulary = set(self.postings.keys())
        N = self.total_docs

        # Step 2: Average chunk length avgdl
        if N > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / N
        else:
            self.avg_doc_length = 0.0

        # Step 3: Document frequency and smoothed IDF: ln(1 + (N - DF + 0.5) / (DF + 0.5))
        self.df = {term: len(plist) for term, plist in self.postings.items()}
        self.idf = {
            term: math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def search(self, query, top_k: int = 5):
        """Search the index for a query string or token list using BM25.

        Args:
            query: Either a raw query string (tokenized with preprocess.tokenize)
                   or a pre-tokenized list of stems.
            top_k: Maximum number of relevant chunks to return.

        Returns:
            List of dictionaries containing chunk_id, paper_id, chunk_index,
            title, BM25 score, and raw text for citation.
            (Same keys as the TF-IDF search, so the chatbot can use either.)
        """
        # Step 1: Tokenize query if passed as string
        if isinstance(query, str):
            if tokenize is None:
                raise ImportError("preprocess.tokenize is required for string queries.")
            tokens = tokenize(query)
        elif isinstance(query, list):
            tokens = query
        else:
            raise TypeError("query must be a string or list of tokens.")

        if not tokens or self.total_docs == 0:
            return []

        # Step 2: Add up BM25 scores, only for chunks that contain a query term
        scores = defaultdict(float)
        for term, q_count in Counter(tokens).items():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for chunk_id, tf in self.postings[term]:
                # Long chunks are penalized, short chunks are favored (controlled by b)
                length_norm = 1.0 - self.b + self.b * self.doc_lengths[chunk_id] / self.avg_doc_length
                # Saturation: more repeats of a term help less and less (controlled by k1)
                tf_part = tf * (self.k1 + 1.0) / (tf + self.k1 * length_norm)
                scores[chunk_id] += q_count * idf * tf_part

        # If no query terms exist in vocabulary, return empty
        if not scores:
            return []

        # Step 3: Rank descending by score (ties broken by chunk_id so results are stable)
        top_results = heapq.nsmallest(top_k, scores.items(), key=lambda item: (-item[1], item[0]))

        # Step 4: Package results with citation metadata
        output = []
        for chunk_id, score in top_results:
            meta = self.doc_metadata.get(chunk_id, {})
            output.append({
                "chunk_id": chunk_id,
                "citation": f"[{chunk_id}]",
                "paper_id": meta.get("paper_id"),
                "chunk_index": meta.get("chunk_index"),
                "title": meta.get("title"),
                "score": round(score, 6),
                "text": meta.get("text", "")
            })
        return output

    def save(self, index_path: Path):
        """Save the inverted index, IDF table, chunk lengths, and metadata to JSON."""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Saving index to {index_path}...")

        data = {
            "model": "bm25",
            "version": "1.0",
            "k1": self.k1,
            "b": self.b,
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "vocab_size": len(self.idf),
            "df": self.df,
            "idf": self.idf,
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "doc_metadata": self.doc_metadata
        }

        with index_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        file_size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"    -> Saved successfully ({file_size_mb:.2f} MB).")

    def load(self, index_path: Path):
        """Load an existing index from JSON."""
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at: {index_path}")

        print(f"[*] Loading index from {index_path}...")
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Stop early if someone passes the TF-IDF index file by mistake
        if data.get("model") != "bm25":
            raise ValueError(f"{index_path} is not a BM25 index file.")

        self.set_parameters(data["k1"], data["b"])
        self.total_docs = data["total_docs"]
        self.avg_doc_length = data["avg_doc_length"]
        self.df = data["df"]
        self.idf = data["idf"]
        self.postings = data["postings"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_metadata = data["doc_metadata"]
        self.vocabulary = set(self.idf.keys())
        print(f"    -> Successfully loaded index: {self.total_docs} chunks, {len(self.vocabulary)} terms.")

    def summary(self):
        """Print high-level statistics of the built index."""
        print("\n" + "=" * 60)
        print("BM25 INDEX SUMMARY")
        print("=" * 60)
        print(f"Total Chunks (N)       : {self.total_docs}")
        print(f"Vocabulary Size (|V|)  : {len(self.idf)} unique stems")
        print(f"Total Postings Entries : {sum(len(p) for p in self.postings.values())}")
        print(f"Parameters             : k1={self.k1}, b={self.b}")

        if self.doc_lengths:
            lengths = list(self.doc_lengths.values())
            print(f"Chunk Length (|d|)     : min={min(lengths)}, max={max(lengths)}, avgdl={self.avg_doc_length:.2f} tokens")

        sorted_by_idf = sorted(self.idf.items(), key=lambda x: x[1])
        print("\n--- Top 10 Most Frequent Terms (Lowest IDF) ---")
        for term, val in sorted_by_idf[:10]:
            df = self.df.get(term, len(self.postings.get(term, [])))
            print(f"  {term:<15} : IDF = {val:.4f}  (in {df} chunks, {df/self.total_docs*100:.1f}%)")

        print("\n--- Top 10 Rarest Terms (Highest IDF) ---")
        for term, val in sorted_by_idf[-10:]:
            df = self.df.get(term, len(self.postings.get(term, [])))
            print(f"  {term:<15} : IDF = {val:.4f}  (in {df} chunks)")
        print("=" * 60 + "\n")

    def inspect_term(self, term: str):
        """Verify IDF, posting count, and sample postings for a specific term."""
        print(f"\n[Term Verification: '{term}']")
        if term not in self.idf:
            print(f"  Term '{term}' is NOT in vocabulary.")
            return

        idf_val = self.idf[term]
        postings = self.postings.get(term, [])
        df = self.df.get(term, len(postings))
        print(f"  DF Value       : {df}")
        print(f"  IDF Value      : {idf_val:.6f}")
        print(f"  Sample Postings (first 5 of {len(postings)}):")
        for chunk_id, tf in postings[:5]:
            meta = self.doc_metadata.get(chunk_id, {})
            title = meta.get("title") or "No Title"
            print(f"    - chunk_id: {chunk_id:<25} tf: {tf:<3} |d|: {self.doc_lengths.get(chunk_id)} | Title: {title[:40]}...")

    def inspect_document(self, chunk_id: str):
        """Verify chunk length, metadata, and snippet for a specific chunk."""
        print(f"\n[Document Verification: '{chunk_id}']")
        if chunk_id not in self.doc_metadata:
            print(f"  Chunk '{chunk_id}' not found in index.")
            return

        meta = self.doc_metadata[chunk_id]
        doc_len = self.doc_lengths.get(chunk_id, 0)
        print(f"  Paper ID     : {meta.get('paper_id')}")
        print(f"  Chunk Index  : {meta.get('chunk_index')}")
        print(f"  Title        : {meta.get('title')}")
        print(f"  Chunk Length : {doc_len} tokens (average is {self.avg_doc_length:.2f})")
        print(f"  Text Snippet : {meta.get('text', '')[:160]}...")


# Alias BM25Index to BM25Indexer so both naming conventions work (same as TFIDFIndex)
BM25Index = BM25Indexer


def format_search_results(results, query: str):
    """Format and print search results as citation cards."""
    print("\n" + "=" * 70)
    print(f"BM25 SEARCH RESULTS FOR: \"{query}\"")
    print("=" * 70)
    if not results:
        print("No matching documents found.")
        print("=" * 70 + "\n")
        return

    for rank, hit in enumerate(results, start=1):
        chunk_id = hit["chunk_id"]
        title = hit.get("title", "Untitled")
        score = hit["score"]
        snippet = hit.get("text", "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."

        print(f"\n[{rank}] Score: {score:.5f} | Citation: [{chunk_id}]")
        print(f"    Title  : {title}")
        print(f"    Passage: {snippet}")
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Path to preprocessed chunks JSONL")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to BM25 index JSON")
    parser.add_argument("--build", action="store_true", help="Build and save the index")
    parser.add_argument("--verify", action="store_true", help="Load index and print summary verification")
    parser.add_argument("--term", type=str, help="Inspect a specific term (e.g. --term entangl)")
    parser.add_argument("--doc", type=str, help="Inspect a specific chunk_id (e.g. --doc arxiv:0704.0034#0)")
    parser.add_argument("--query", type=str, help="Query string to search in the index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top search results to retrieve (default: 5)")
    parser.add_argument("--k1", type=float, help=f"Term frequency saturation (default: {DEFAULT_K1})")
    parser.add_argument("--b", type=float, help=f"Length normalization 0..1 (default: {DEFAULT_B})")
    parser.add_argument("--interactive", action="store_true", help="Start interactive search prompt")

    args = parser.parse_args()

    if not (args.build or args.verify or args.term or args.doc or args.query or args.interactive):
        parser.print_help()
        return

    indexer = BM25Indexer()

    if args.build:
        print(f"[*] Reading chunk records from: {args.chunks}...")
        indexer.build(args.chunks)
        print(f"    -> Scanned {indexer.total_docs} chunks across {len(indexer.vocabulary)} unique terms.")
        print("[*] Computing smoothed IDF values: ln(1 + (N - DF + 0.5) / (DF + 0.5))...")
        print(f"    -> Inverted index created with {len(indexer.postings)} posting lists.")
        print(f"    -> Average chunk length (avgdl) = {indexer.avg_doc_length:.2f} tokens.")
        print("[*] Indexing stage complete!")
        # k1 and b are only used when searching, so they can also be set after loading
        if args.k1 is not None or args.b is not None:
            indexer.set_parameters(args.k1 if args.k1 is not None else indexer.k1,
                                   args.b if args.b is not None else indexer.b)
        indexer.save(args.index)
        indexer.summary()

    if args.verify or args.term or args.doc or args.query or args.interactive:
        if not args.build:
            indexer.load(args.index)
            # Try other k1 / b values without rebuilding the index
            if args.k1 is not None or args.b is not None:
                indexer.set_parameters(args.k1 if args.k1 is not None else indexer.k1,
                                       args.b if args.b is not None else indexer.b)
                print(f"    -> Using k1={indexer.k1}, b={indexer.b}")

        if args.verify and not args.build:
            indexer.summary()

        if args.term:
            indexer.inspect_term(args.term)

        if args.doc:
            indexer.inspect_document(args.doc)

        if args.query:
            results = indexer.search(args.query, top_k=args.top_k)
            format_search_results(results, args.query)

        if args.interactive:
            print("\n" + "=" * 60)
            print("INTERACTIVE BM25 CITATION SEARCH")
            print("Type your question/query and press Enter. Type 'exit' to quit.")
            print("=" * 60)
            while True:
                try:
                    q = input("\nQuery: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q or q.lower() in ("exit", "quit", "q"):
                    break
                hits = indexer.search(q, top_k=args.top_k)
                format_search_results(hits, q)


if __name__ == "__main__":
    main()