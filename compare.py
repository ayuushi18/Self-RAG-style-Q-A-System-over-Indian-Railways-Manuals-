import json
from pathlib import Path

FILES = {
    "Normal RAG": Path("data/results_normal_rag.jsonl"),
    "Prompt Self-RAG": Path("data/results_prompt_selfrag.jsonl"),
    "Fine-tuned Self-RAG": Path("data/results_finetuned_selfrag.jsonl")
}

PHRASES = [
    "cannot find", "not mention", "does not", "doesn't", "do not",
    "no information", "not specified", "not provided", "not stated",
    "not state", "not explicitly", "cannot", "can't", "unable",
    "not contain", "not included", "not given", "not covered", "not available",
]


def abstains(answer):
    a = (answer or "").lower()
    return any(p in a for p in PHRASES)


def load(path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


for name, path in FILES.items():
    rows = load(path)
    ans = [r for r in rows if r["label"] == "answerable"]
    una = [r for r in rows if r["label"] == "not_answerable"]

    made_up = [r for r in una if not abstains(r["model_answer"])]
    over_refused = [r for r in ans if abstains(r["model_answer"])]

    print("=" * 60)
    print(name, "-", len(rows), "results")
    print(f"  Not-answerable: made up an answer for {len(made_up)}/{len(una)}")
    print(f"  Answerable:     refused or hedged on {len(over_refused)}/{len(ans)}")

    if name != "Normal RAG" and rows:
        kept_gold = sum(r["gold_chunk"] in (r.get("relevant") or []) for r in ans)
        print(f"  Relevance check kept the right chunk for {kept_gold}/{len(ans)} answerable questions")
        print("  Support check results:",
              {s: sum(r.get("support") == s for r in rows)
               for s in ("fully", "partially", "no", "none")})
        print("\n  Answerable questions it refused (check if these were wrong refusals):")
        for r in over_refused[:5]:
            print("   -", r["question"])