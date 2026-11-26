import re
import json
from pathlib import Path
import pymupdf

RAW = Path("data/raw")
OUT = Path("data/chunks.jsonl")
CHUNK_WORDS = 300   # size of each chunk
OVERLAP = 50        # words shared between neighbouring chunks


def clean_page(text):
    """Remove empty lines and lines that are only a page number."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        lines.append(line)
    return " ".join(lines)


def load_words(pdf_path):
    """Read a PDF and return a list of (word, page_number)."""
    doc = pymupdf.open(pdf_path)
    words = []
    for page_num, page in enumerate(doc, start=1):
        text = clean_page(page.get_text())
        for w in text.split():
            words.append((w, page_num))
    return words


def make_chunks(words, source):
    """Cut the words into overlapping chunks."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) < 50:
            break
        chunks.append({
            "id": f"{source}-{len(chunks)}",
            "source": source,
            "page": piece[0][1],
            "text": " ".join(w for w, _ in piece),
        })
    return chunks


all_chunks = []
for pdf in sorted(RAW.glob("*.pdf")):
    words = load_words(pdf)
    chunks = make_chunks(words, pdf.stem)
    print(pdf.name, "->", len(words), "words,", len(chunks), "chunks")
    all_chunks.extend(chunks)

with open(OUT, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print("Saved", len(all_chunks), "chunks to", OUT)