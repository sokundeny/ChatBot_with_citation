"""Split papers into chunks and tokenize them for keyword search (TF-IDF, BM25).

Input : data/physics-quant-ph.jsonl          one paper per line (made by ingest.py)
Output: data/physics-quant-ph.chunks.jsonl   one chunk per line

Each paper becomes:
    chunk 0        title + abstract
    chunk 1, 2...  the full text, packed into chunks of whole sentences (at most 200 words);
                   each chunk repeats the last sentences (up to 40 words) of the one before

Each chunk keeps its raw text (for dense search and citations) and its tokens
(for TF-IDF and BM25).

    python scripts/preprocess.py
"""
import argparse
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import snowballstemmer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "physics-quant-ph.jsonl"
CHUNK_SIZE = 200  # maximum words per full-text chunk
OVERLAP = 40      # maximum words of whole sentences repeated from the previous chunk

# Words ending with "." that usually do not end a sentence ("Fig. 2", "et al. Smith", "Phys. Rev.").
ABBREVIATIONS = {"fig", "figs", "eq", "eqs", "ref", "refs", "sec", "al", "vs", "cf", "dr", "prof",
                 "phys", "rev", "lett", "vol", "no", "nucl", "math", "opt", "appl", "mod", "sci", "proc"}

GREEK_CHARACTER = r"[Ͱ-Ͽ]"  # the Greek block of Unicode

# LaTeX commands that only format text or stand for a math symbol: dropped instead of turned into words.
# (The same symbols written in Unicode, like "×" or "≤", are already dropped when splitting into words.)
LATEX_TO_DROP = {
    # fonts and styles
    "it", "bf", "rm", "em", "emph", "cal", "mathcal", "mathbf", "mathrm", "textbf", "textit", "mathversion",
    # accents, layout and references
    "hat", "check", "ddot", "dot", "tilde", "bar", "vec", "v", "frac", "left", "right", "cite", "spread",
    # math symbols
    "times", "otimes", "cdot", "ge", "geq", "le", "leq", "gg", "ll", "sim", "simeq", "approx",
    "to", "rightarrow", "longrightarrow", "cup", "cap", "in", "pm", "infty", "partial",
}
STEMMER = snowballstemmer.stemmer("english")
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves also et al via thus
""".split())


# --------------------------------------------------------------------------------------
# Tokenizing: text -> list of search tokens
# --------------------------------------------------------------------------------------

def normalize_unicode(text):
    """Turn look-alike characters into one standard form: "µ" -> "μ", "ﬁ" -> "fi", "²" -> "2"."""
    return unicodedata.normalize("NFKC", text)


def remove_accents(text):
    """Remove accents so a typed "schrodinger" matches "schrödinger": "ö" -> "o", "é" -> "e"."""
    separated = unicodedata.normalize("NFD", text)  # "ö" becomes "o" + a separate accent mark
    return "".join(character for character in separated if not unicodedata.combining(character))


def latex_command_to_word(match):
    command = match.group(1)
    return " " if command in LATEX_TO_DROP else f" {command} "


def latex_to_words(text):
    r"""Turn LaTeX commands into words: "\rho" -> " rho ". Formatting like "\frac" or "\times" is dropped.

    Run after lowercasing.
    """
    return re.sub(r"\\([a-z]+)", latex_command_to_word, text)


def greek_letter_name(character):
    """Return the name of a Greek letter: "ρ" -> "rho", "Ψ" -> "psi"; "" if it has none."""
    unicode_name = unicodedata.name(character, "")  # e.g. "GREEK SMALL LETTER RHO"
    match = re.match(r"GREEK (?:SMALL |CAPITAL )?(?:LETTER )?(\w+)", unicode_name)
    if not match:
        return ""
    name = match.group(1).lower()
    if name == "lamda":  # Unicode spells it "lamda", LaTeX spells it "lambda"
        name = "lambda"
    return name


def greek_to_words(text):
    """Replace Greek letters with their names, so "ρ" matches LaTeX "\\rho" and a typed "rho"."""
    return re.sub(GREEK_CHARACTER, lambda match: f" {greek_letter_name(match.group())} ", text)


def split_words(text):
    """Split text into words: runs of letters and digits. Every other character separates words."""
    return re.findall(r"[^\W_]+", text)


def is_useful(word):
    """Single letters, numbers and stopwords carry little meaning for search."""
    return len(word) > 1 and not word.isdigit() and word not in STOPWORDS


def remove_noise(words):
    return [word for word in words if is_useful(word)]


def stem(words):
    """Cut words to their stem: "entangled", "entanglement" -> "entangl"."""
    return [STEMMER.stemWord(word) for word in words]


def tokenize(text):
    """Turn text into search tokens. Use the same function for documents and queries."""
    text = normalize_unicode(text)
    text = text.lower()
    text = remove_accents(text)
    text = latex_to_words(text)
    text = greek_to_words(text)
    words = split_words(text)
    words = remove_noise(words)
    return stem(words)


# --------------------------------------------------------------------------------------
# Chunking: paper -> list of chunks
# --------------------------------------------------------------------------------------

def simplify_word(word):
    """Lowercase and drop punctuation, so "congestion-free" and "congestionfree" compare equal."""
    return re.sub(r"\W", "", word.lower())


def remove_abstract_copy(abstract, full_text):
    """Remove the abstract from the start of the full text, because chunk 0 already contains it.

    The full text comes from the PDF and the abstract from arXiv's LaTeX, so the two copies differ
    slightly ("$\\times$" vs "×", "congestion-free" vs "congestionfree"). They are compared word by
    word, and the full text is cut where its copy of the abstract ends.
    """
    abstract_words = [simplify_word(word) for word in abstract.split()]
    full_words = full_text.split()
    search_length = int(len(abstract_words) * 1.5) + 50  # the PDF copy can be longer (extra math, page headers)
    start_of_full_text = [simplify_word(word) for word in full_words[:search_length]]

    matcher = SequenceMatcher(None, abstract_words, start_of_full_text, autojunk=False)
    matches = [block for block in matcher.get_matching_blocks() if block.size >= 3]
    matched_words = sum(block.size for block in matches)
    # Treat it as a copy only if almost all of the abstract matches, close to the start of the full text.
    # In this data, real copies match 86% or more; introductions that reword the abstract match 79% or less.
    if not matches or matched_words < 0.85 * len(abstract_words) or matches[0].b > 30:
        return full_text
    end_of_copy = matches[-1].b + matches[-1].size
    # The last words of the two copies often differ ("$\psi$." vs "ψ."), so the match can stop a few
    # words early. If it stopped in the middle of a sentence, move the cut to the end of that sentence.
    if not ends_sentence(full_words[end_of_copy - 1]):
        for position in range(end_of_copy, min(end_of_copy + 15, len(full_words))):
            if ends_sentence(full_words[position]):
                end_of_copy = position + 1
                break
    return " ".join(full_words[end_of_copy:])


def ends_sentence(word):
    return word.endswith((".", "!", "?"))


def ends_with_abbreviation(sentence):
    """True for pieces like "... see Fig." or "... J." that end with an abbreviation, not a sentence."""
    words = sentence.split()
    if not words:
        return False
    last_word = words[-1]
    return last_word.rstrip(".").lower() in ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", last_word) is not None


def split_sentences(text):
    """Split text into sentences: a sentence ends with . ! or ? followed by a space and a capital letter."""
    sentences = []
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        if sentences and ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + piece  # "Fig. 2" or "J. Smith": not a real sentence end
        else:
            sentences.append(piece)
    return sentences


def word_count(sentences):
    return sum(len(sentence) for sentence in sentences)


def last_sentences(sentences, max_words):
    """The last whole sentences of a chunk, together at most `max_words` words, to repeat in the next chunk."""
    kept = []
    for sentence in reversed(sentences):
        if word_count(kept) + len(sentence) > max_words:
            break
        kept.insert(0, sentence)
    return kept


def split_into_chunks(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Pack whole sentences into chunks of at most `size` words.

    Each new chunk starts by repeating the last sentences of the previous chunk (at most `overlap`
    words), so sentences near a boundary keep their context. A "sentence" longer than `size` words
    (usually a long equation or a reference list without full stops) is cut into `size`-word pieces.
    """
    sentences = []  # each sentence is a list of words
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) <= size:
            sentences.append(words)
            continue
        pieces = math.ceil(len(words) / size)
        piece_size = math.ceil(len(words) / pieces)  # equal pieces, so no 1-word leftovers
        for start in range(0, len(words), piece_size):
            sentences.append(words[start:start + piece_size])

    chunks = []
    current = []  # sentences of the chunk being built
    for sentence in sentences:
        if current and word_count(current) + len(sentence) > size:  # this sentence does not fit any more
            chunks.append(current)
            repeated = last_sentences(current, overlap)
            if len(repeated) == len(current) or word_count(repeated) + len(sentence) > size:
                repeated = []  # never repeat a whole chunk, and never overfill the next one
            current = repeated
        current.append(sentence)
    if current:
        chunks.append(current)
    return [" ".join(word for sentence in chunk for word in sentence) for chunk in chunks]


def make_chunks(paper, size=CHUNK_SIZE, overlap=OVERLAP):
    """Turn one paper into chunk records: chunk 0 = title + abstract, then chunks of the full text."""
    full_text = remove_abstract_copy(paper["abstract"], paper.get("full_text") or "")

    pieces = [paper["text"]]  # chunk 0: title + abstract
    if full_text.strip():
        pieces += split_into_chunks(full_text, size, overlap)

    chunks = []
    for index, text in enumerate(pieces):
        tokens = tokenize(text)
        if not tokens:  # PDF junk such as rows of dots: nothing to search
            continue
        chunks.append({
            "chunk_id": f"{paper['paper_id']}#{index}",
            "paper_id": paper["paper_id"],
            "chunk_index": index,
            "title": paper["title"],
            "text": text,
            "tokens": tokens,
        })
    return chunks


# --------------------------------------------------------------------------------------
# Command line: read papers, write chunks, print a summary
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Defaults to INPUT with .chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Maximum words per full-text chunk")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help="Maximum words of whole sentences repeated from the previous chunk")
    args = parser.parse_args()
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("overlap must be >= 0 and smaller than chunk-size")
    output = args.output or args.input.with_suffix(".chunks.jsonl")

    paper_count = chunk_count = token_count = 0
    vocabulary = set()
    with args.input.open(encoding="utf-8") as infile, output.open("w", encoding="utf-8") as outfile:
        for line in infile:
            paper = json.loads(line)
            paper_count += 1
            for chunk in make_chunks(paper, args.chunk_size, args.overlap):
                outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1
                token_count += len(chunk["tokens"])
                vocabulary.update(chunk["tokens"])

    print(f"Papers          : {paper_count}")
    print(f"Chunks          : {chunk_count} ({chunk_count / max(paper_count, 1):.1f} per paper)")
    print(f"Tokens per chunk: {token_count / max(chunk_count, 1):.0f} on average")
    print(f"Vocabulary      : {len(vocabulary)} different tokens")
    print(f"Saved to        : {output}")


if __name__ == "__main__":
    main()
