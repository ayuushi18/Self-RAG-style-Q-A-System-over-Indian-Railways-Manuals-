import json

from hybrid import hybrid_search
from llm import ask

NOT_FOUND = "I cannot find this in the documents."


def number_passages(passages):
    return "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))


def need_retrieval(question):
    """Check 1: does this question need the documents?"""
    reply = ask([
        {"role": "system", "content": (
            "You decide whether a question needs looking up in documents. "
            "The documents are Indian Railways rule manuals (commercial and operating rules). "
            "Reply with one word: Yes or No."
        )},
        {"role": "user", "content": question},
    ])
    return not (reply or "").strip().lower().startswith("no")


def judge_relevance(question, passages):
    """Check 2: is each passage relevant? Returns a list of True/False."""
    reply = ask([
        {"role": "system", "content": (
            "For each passage, decide if it is relevant to the question, meaning it contains "
            "information that helps answer this exact question. A passage on the same general "
            "topic that does not help answer the exact question is NOT relevant. "
            "Reply with only a JSON list of true/false values, one per passage, in order. "
            "Example: [true, false, true]"
        )},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{number_passages(passages)}"},
    ])
    try:
        text = reply or ""
        flags = json.loads(text[text.find("["): text.rfind("]") + 1])
        if len(flags) == len(passages):
            return [bool(f) for f in flags]
    except Exception:
        pass
    return [True] * len(passages)  # if the reply is unreadable, keep everything


def generate(question, passages):
    """Step 3: write the answer (same instructions as normal RAG)."""
    return ask([
        {"role": "system", "content": (
            "You answer questions about Indian Railways rules. "
            "Use the passages provided to answer. Keep the answer to 1-3 sentences."
        )},
        {"role": "user", "content": f"Passages:\n{number_passages(passages)}\n\nQuestion: {question}"},
    ])


def check_support(question, answer, passages):
    """Check 4: is the answer supported? Returns fully / partially / no."""
    reply = ask([
        {"role": "system", "content": (
            "You check whether an answer is supported by the passages. Reply with one word:\n"
            "Fully = every claim in the answer, including numbers, is stated in the passages, "
            "and the answer directly answers the question that was asked.\n"
            "Partially = some of it is supported, or the passages cover a related case but "
            "not exactly what was asked.\n"
            "No = the passages do not support the answer."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n\nAnswer: {answer}\n\nPassages:\n{number_passages(passages)}"
        )},
    ])
    word = (reply or "").strip().lower()
    if word.startswith("full"):
        return "fully"
    if word.startswith("part"):
        return "partially"
    return "no"


def selfrag_answer(question, k=5):
    """The full pipeline. Returns the final answer and a record of every check."""
    trace = {"question": question}

    trace["retrieve"] = need_retrieval(question)
    if not trace["retrieve"]:
        trace["final"] = ask([{"role": "user", "content": question}])
        trace["retrieved"] = []
        return trace

    results = hybrid_search(question, k=k)
    trace["retrieved"] = [c["id"] for c in results]

    flags = judge_relevance(question, results)
    relevant = [c for c, f in zip(results, flags) if f]
    trace["relevant"] = [c["id"] for c in relevant]

    if not relevant:
        trace["support"] = "none"
        trace["draft"] = None
        trace["final"] = NOT_FOUND
        return trace

    draft = generate(question, relevant)
    trace["draft"] = draft
    trace["support"] = check_support(question, draft, relevant)
    trace["final"] = draft if trace["support"] == "fully" else NOT_FOUND
    return trace


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type q to quit): ")
        if q.strip().lower() == "q":
            break
        t = selfrag_answer(q)
        print("\nRetrieve needed?  ", t["retrieve"])
        print("Retrieved chunks: ", t["retrieved"])
        print("Relevant chunks:  ", t.get("relevant"))
        print("Draft answer:     ", t.get("draft"))
        print("Support check:    ", t.get("support"))
        print("\nFINAL ANSWER:", t["final"])