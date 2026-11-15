"""TF-IDF Inverted Indexer and Search Engine for ChatBot with Citation.

Exact formulas implemented:
  - TF(t, d) = count(t, d) / totalTokens(d)    (Relative Term Frequency)
  - DF(t)    = Number of chunks containing term t
  - IDF(t)   = ln(1 + N / DF(t))               (Smoothed Inverse Document Frequency)
  - TF-IDF   = TF(t, d) * IDF(t)
  - Doc norm = sqrt(sum_t (TF-IDF(t, d)^2))    (Euclidean length ||d||)
  - Cosine   = (q . d) / (||q|| * ||d||)       (Cosine Similarity Ranking)
"""

import argparse
from collections import Counter, defaultdict
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
DEFAULT_INDEX = PROJECT_ROOT / "data" / "tfidf_index.json"


class TFIDFIndexer:
    def __init__(self, chunks=None):
        self.total_docs = 0
        self.vocabulary = set()
        self.df = {}             # term -> int (document frequency)
        self.idf = {}            # term -> idf float
        self.postings = {}       # term -> list of [chunk_id, tf_idf_weight]
        self.doc_lengths = {}    # chunk_id -> float (Euclidean norm ||d||)
        self.doc_metadata = {}   # chunk_id -> dict(paper_id, chunk_index, title, text)

        if chunks is not None:
            self.build(chunks)

    def build(self, chunks_input):
        """Build the TF-IDF inverted index.
        
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

        df_counter = Counter()
        doc_term_frequencies = {}  # chunk_id -> {term: relative_tf}
        self.total_docs = 0
        self.doc_metadata.clear()

        # Step 1: Compute Term Frequency & Document Frequency
        for record in records:
            chunk_id = record["chunk_id"]
            tokens = record.get("tokens", [])
            if not tokens:
                continue

            self.total_docs += 1
            counts = Counter(tokens)
            total_tokens = len(tokens)

            # Relative TF: count(t, d) / totalTokens(d)
            rel_tf = {term: count / total_tokens for term, count in counts.items()}
            doc_term_frequencies[chunk_id] = rel_tf

            # Document frequency: count term once per chunk
            for term in rel_tf:
                df_counter[term] += 1

            # Save metadata for citation linking
            self.doc_metadata[chunk_id] = {
                "paper_id": record.get("paper_id"),
                "chunk_index": record.get("chunk_index"),
                "title": record.get("title"),
                "text": record.get("text")
            }

        self.df = dict(df_counter)
        self.vocabulary = set(df_counter.keys())
        N = self.total_docs

        # Step 2: Compute Smoothed IDF: ln(1 + N / DF)
        self.idf = {
            term: math.log(1.0 + (N / count))
            for term, count in df_counter.items()
        }

        # Step 3: Compute TF-IDF weights, build inverted index, and compute L2 norms
        postings = defaultdict(list)
        norm_accumulators = defaultdict(float)

        for chunk_id, tf_map in doc_term_frequencies.items():
            for term, tf in tf_map.items():
                weight = tf * self.idf[term]
                postings[term].append([chunk_id, round(weight, 6)])
                norm_accumulators[chunk_id] += weight * weight

        self.postings = dict(postings)
        self.doc_lengths = {
            chunk_id: round(math.sqrt(acc), 6)
            for chunk_id, acc in norm_accumulators.items()
        }

    def search(self, query, top_k: int = 5):
        """Search the index for a query string or token list using Cosine Similarity.
        
        Args:
            query: Either a raw query string (tokenized with preprocess.tokenize)
                   or a pre-tokenized list of stems.
            top_k: Maximum number of relevant chunks to return.
            
        Returns:
            List of dictionaries containing chunk_id, paper_id, chunk_index,
            title, similarity score, and raw text for citation.
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

        if not tokens:
            return []

        # Step 2: Calculate Query TF, Weights, and Query Norm ||q||
        q_counts = Counter(tokens)
        q_total_tokens = len(tokens)

        q_weights = {}
        q_norm_sq = 0.0

        for term, count in q_counts.items():
            if term in self.idf:
                tf = count / q_total_tokens
                weight = tf * self.idf[term]
                q_weights[term] = weight
                q_norm_sq += weight * weight

        # If no query terms exist in vocabulary, return empty
        if not q_weights or q_norm_sq == 0.0:
            return []

        q_norm = math.sqrt(q_norm_sq)

        # Step 3: Accumulate dot products (q . d) via inverted index postings
        dot_products = defaultdict(float)
        for term, q_w in q_weights.items():
            for chunk_id, doc_w in self.postings.get(term, []):
                dot_products[chunk_id] += q_w * doc_w

        # Step 4: Compute Cosine Similarity = (q . d) / (||q|| * ||d||)
        scored_candidates = []
        for chunk_id, dot in dot_products.items():
            doc_norm = self.doc_lengths.get(chunk_id, 0.0)
            if doc_norm > 0:
                sim = dot / (q_norm * doc_norm)
                scored_candidates.append((chunk_id, sim))

        # Step 5: Rank descending by similarity score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_candidates[:top_k]

        # Step 6: Package results with citation metadata
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
        """Save the inverted index, IDF table, document lengths, and metadata to JSON."""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Saving index to {index_path}...")
        
        data = {
            "version": "1.0",
            "total_docs": self.total_docs,
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
        
        self.total_docs = data["total_docs"]
        self.df = data.get("df", {})
        self.idf = data["idf"]
        self.postings = data["postings"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_metadata = data["doc_metadata"]
        self.vocabulary = set(self.idf.keys())
        print(f"    -> Successfully loaded index: {self.total_docs} chunks, {len(self.vocabulary)} terms.")

    def summary(self):
        """Print high-level statistics of the built index."""
        print("\n" + "=" * 60)
        print("TF-IDF INDEX SUMMARY")
        print("=" * 60)
        print(f"Total Chunks (N)       : {self.total_docs}")
        print(f"Vocabulary Size (|V|)  : {len(self.idf)} unique stems")
        print(f"Total Postings Entries : {sum(len(p) for p in self.postings.values())}")
        
        if self.doc_lengths:
            lengths = list(self.doc_lengths.values())
            avg_len = sum(lengths) / len(lengths)
            print(f"Document Norms (||d||) : min={min(lengths):.4f}, max={max(lengths):.4f}, avg={avg_len:.4f}")
        
        sorted_by_idf = sorted(self.idf.items(), key=lambda x: x[1])
        print("\n--- Top 10 Most Frequent Terms (Lowest IDF) ---")
        for term, val in sorted_by_idf[:10]:
            df = len(self.postings.get(term, []))
            print(f"  {term:<15} : IDF = {val:.4f}  (in {df} chunks, {df/self.total_docs*100:.1f}%)")

        print("\n--- Top 10 Rarest Terms (Highest IDF) ---")
        for term, val in sorted_by_idf[-10:]:
            df = len(self.postings.get(term, []))
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
        for chunk_id, weight in postings[:5]:
            meta = self.doc_metadata.get(chunk_id, {})
            title = meta.get("title", "No Title")
            print(f"    - chunk_id: {chunk_id:<25} weight: {weight:.6f} | Title: {title[:40]}...")

    def inspect_document(self, chunk_id: str):
        """Verify document norm, metadata, and top-weighted terms for a specific chunk."""
        print(f"\n[Document Verification: '{chunk_id}']")
        if chunk_id not in self.doc_metadata:
            print(f"  Chunk '{chunk_id}' not found in index.")
            return

        meta = self.doc_metadata[chunk_id]
        doc_len = self.doc_lengths.get(chunk_id, 0.0)
        print(f"  Paper ID     : {meta.get('paper_id')}")
        print(f"  Chunk Index  : {meta.get('chunk_index')}")
        print(f"  Title        : {meta.get('title')}")
        print(f"  Vector Norm  : {doc_len:.6f}")
        print(f"  Text Snippet : {meta.get('text', '')[:160]}...")


# Alias TFIDFIndex to TFIDFIndexer so both naming conventions work seamlessly
TFIDFIndex = TFIDFIndexer


def format_search_results(results, query: str):
    """Format and print search results as citation cards."""
    print("\n" + "=" * 70)
    print(f"SEARCH RESULTS FOR: \"{query}\"")
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
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to TF-IDF index JSON")
    parser.add_argument("--build", action="store_true", help="Build and save the index")
    parser.add_argument("--verify", action="store_true", help="Load index and print summary verification")
    parser.add_argument("--term", type=str, help="Inspect a specific term (e.g. --term entangl)")
    parser.add_argument("--doc", type=str, help="Inspect a specific chunk_id (e.g. --doc arxiv:0704.0034#0)")
    parser.add_argument("--query", type=str, help="Query string to search in the index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top search results to retrieve (default: 5)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive search prompt")

    args = parser.parse_args()

    if not (args.build or args.verify or args.term or args.doc or args.query or args.interactive):
        parser.print_help()
        return

    indexer = TFIDFIndexer()

    if args.build:
        print(f"[*] Reading chunk records from: {args.chunks}...")
        indexer.build(args.chunks)
        print(f"    -> Scanned {indexer.total_docs} chunks across {len(indexer.vocabulary)} unique terms.")
        print("[*] Computing smoothed IDF values: ln(1 + N / DF)...")
        print(f"    -> Inverted index created with {len(indexer.postings)} posting lists.")
        print(f"    -> Computed Euclidean vector norms for all {len(indexer.doc_lengths)} documents.")
        print("[*] Indexing stage complete!")
        indexer.save(args.index)
        indexer.summary()

    if args.verify or args.term or args.doc or args.query or args.interactive:
        if not args.build:
            indexer.load(args.index)

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
            print("INTERACTIVE TF-IDF CITATION SEARCH")
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
