import json
from pathlib import Path

from llm import ask

FILES = {
    "Normal RAG": Path("data/results_normal_rag.jsonl"),
    "Prompt Self-RAG": Path("data/results_prompt_selfrag.jsonl"),
    "Fine-tuned Self-RAG": Path("data/results_finetuned_selfrag.jsonl"),
    "Hybrid Self-RAG": Path("data/results_hybrid_selfrag.jsonl"),
    "Azure Self-RAG": Path("data/results_azure_selfrag.jsonl"),
}

ANSWERABLE_SYSTEM = (
    "You grade answers to questions about Indian Railways rules. You are given a question, "
    "the correct reference answer, and a model's answer. Reply with exactly one word:\n"
    "CORRECT = the model's answer states the same key facts as the reference answer "
    "(the wording may differ, and extra correct detail is fine).\n"
    "REFUSED = the model says it cannot find or does not know the answer.\n"
    "INCORRECT = anything else (wrong, missing the key fact, or contradicting the reference)."
)

UNANSWERABLE_SYSTEM = (
    "You grade answers to questions that CANNOT be answered from the documents. "
    "You are given a question and a model's answer. Reply with exactly one word:\n"
    "ABSTAINED = the model says the information is not available, not stated or cannot be found "
    "(even if it adds related background).\n"
    "ANSWERED = the model gives a specific answer (a number, period, amount or rule) "
    "as if the documents contain it."
)


def read_rows(path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def judge(row):
    if row["label"] == "answerable":
        user = (
            f"Question: {row['question']}\n\n"
            f"Reference answer: {row['gold_answer']}\n\n"
            f"Model's answer: {row['model_answer']}"
        )
        reply = ask([{"role": "system", "content": ANSWERABLE_SYSTEM},
                     {"role": "user", "content": user}])
        valid = ("CORRECT", "REFUSED", "INCORRECT")
    else:
        user = f"Question: {row['question']}\n\nModel's answer: {row['model_answer']}"
        reply = ask([{"role": "system", "content": UNANSWERABLE_SYSTEM},
                     {"role": "user", "content": user}])
        valid = ("ABSTAINED", "ANSWERED")

    word = (reply or "").strip().upper().split()[:1]
    return word[0].strip(".") if word and word[0].strip(".") in valid else "JUDGE_FAILED"


summary = {}
for name, path in FILES.items():
    out_path = Path("data/judgments_" + name.lower().replace(" ", "_").replace("-", "") + ".jsonl")
    done = {r["id"]: r for r in read_rows(out_path)}

    with open(out_path, "a", encoding="utf-8") as out:
        for row in read_rows(path):
            if row["id"] in done:
                continue
            verdict = judge(row)
            rec = {"id": row["id"], "label": row["label"], "question": row["question"],
                   "model_answer": row["model_answer"], "gold_answer": row["gold_answer"],
                   "verdict": verdict}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done[row["id"]] = rec
        print(name, "graded")

    verdicts = [r for r in done.values()]
    ans = [r for r in verdicts if r["label"] == "answerable"]
    una = [r for r in verdicts if r["label"] == "not_answerable"]
    count = lambda rows, v: sum(r["verdict"] == v for r in rows)
    correct, abstained = count(ans, "CORRECT"), count(una, "ABSTAINED")
    summary[name] = {
        "correct": correct, "incorrect": count(ans, "INCORRECT"),
        "refused": count(ans, "REFUSED"), "n_ans": len(ans),
        "abstained": abstained, "answered": count(una, "ANSWERED"), "n_una": len(una),
        "failed": sum(r["verdict"] == "JUDGE_FAILED" for r in verdicts),
        "balanced": 100 * (correct / max(len(ans), 1) + abstained / max(len(una), 1)) / 2,
    }

print("\n" + "=" * 60)
for name, s in summary.items():
    print(name)
    print(f"  Answerable ({s['n_ans']}):     correct {s['correct']}, "
          f"incorrect {s['incorrect']}, wrongly refused {s['refused']}")
    print(f"  Not answerable ({s['n_una']}): correctly abstained {s['abstained']}, "
          f"made up an answer {s['answered']}")
    print(f"  Balanced score: {s['balanced']:.0f}%   (grading failures: {s['failed']})")