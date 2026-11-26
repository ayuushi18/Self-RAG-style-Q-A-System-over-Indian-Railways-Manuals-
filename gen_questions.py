import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

MODEL = "gemini-3.6-flash"   # keep whatever name worked for you
CHUNKS = Path("data/chunks.jsonl")
OUT = Path("data/draft_questions.jsonl")
SKIPPED = Path("data/skipped_chunks.txt")
HOW_MANY = 120               # total drafts wanted
BATCH_SIZE = 5               # passages per request
SECONDS_BETWEEN_CALLS = 6

PROMPT = """You are helping build a test set for a question answering system about Indian Railways rules.
Below are numbered passages. For EACH passage, write ONE question that a real person could ask, which can be answered using ONLY that passage.

Rules:
- The question must make sense on its own. Never say "according to the passage" or "in this text".
- The answer must be short (1-2 sentences) and fully supported by that passage.
- If a passage has no clear fact worth asking about (a table of numbers, a list of names), return {"passage": N, "skip": true} for it.

Return JSON only: a list with one item per passage, in this form:
[{"passage": 1, "question": "...", "answer": "..."}, {"passage": 2, "skip": true}]
"""

client = genai.Client()


def ask_gemini(prompt):
    """Ask Gemini. Retry if busy (503). Stop if the daily limit is reached (429)."""
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("Daily limit reached. Run this again in a few hours or tomorrow.")
                return "LIMIT"
            wait = 10 * (attempt + 1)
            print("Retry", attempt + 1, "in", wait, "seconds ->", str(e)[:80])
            time.sleep(wait)
    return None


def is_mostly_text(text):
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) >= 0.75


chunks = []
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

random.seed(42)
candidates = [c for c in chunks if is_mostly_text(c["text"])]
random.shuffle(candidates)

# Chunks we already used or skipped
done = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["chunk_id"])
skipped = set()
if SKIPPED.exists():
    skipped = set(SKIPPED.read_text(encoding="utf-8").split())

saved = len(done)
todo = [c for c in candidates if c["id"] not in done and c["id"] not in skipped]
position = 0

with open(OUT, "a", encoding="utf-8") as out, open(SKIPPED, "a", encoding="utf-8") as skip_file:
    while saved < HOW_MANY and position < len(todo):
        batch = todo[position:position + BATCH_SIZE]
        position += BATCH_SIZE

        text = PROMPT
        for n, c in enumerate(batch, start=1):
            text += f"\n\nPASSAGE {n}:\n{c['text']}"

        result = ask_gemini(text)
        if result == "LIMIT":
            break
        if not isinstance(result, list):
            print("Bad reply, skipping this batch")
            continue

        for item in result:
            n = item.get("passage")
            if not isinstance(n, int) or not 1 <= n <= len(batch):
                continue
            chunk = batch[n - 1]
            if item.get("skip") or "question" not in item:
                skip_file.write(chunk["id"] + "\n")
                skip_file.flush()
                continue
            row = {
                "chunk_id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "question": item["question"],
                "answer": item["answer"],
                "label": "answerable",
                "reviewed": False,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            saved += 1
            print(saved, "-", item["question"])
        time.sleep(SECONDS_BETWEEN_CALLS)

print("Done. Total drafts:", saved)