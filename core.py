import json

from llm import ask

NOT_FOUND = "I cannot find this in the documents."


def numbered(passages):
    return "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))


def need_retrieval(question):
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
    reply = ask([
        {"role": "system", "content": (
            "For each passage, decide if it is relevant to the question, meaning it contains "
            "information that helps answer this exact question. A passage on the same general "
            "topic that does not help answer the exact question is NOT relevant. "
            "Reply with only a JSON list of true/false values, one per passage, in order. "
            "Example: [true, false, true]"
        )},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{numbered(passages)}"},
    ])
    try:
        text = reply or ""
        flags = json.loads(text[text.find("["): text.rfind("]") + 1])
        if len(flags) == len(passages):
            return [bool(f) for f in flags]
    except Exception:
        pass
    return [True] * len(passages)


def generate(question, passages):
    return ask([
        {"role": "system", "content": (
            "You answer questions about Indian Railways rules. "
            "Use the passages provided to answer. Keep the answer to 1-3 sentences."
        )},
        {"role": "user", "content": f"Passages:\n{numbered(passages)}\n\nQuestion: {question}"},
    ])


def check_support(question, answer, passages):
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
            f"Question: {question}\n\nAnswer: {answer}\n\nPassages:\n{numbered(passages)}"
        )},
    ])
    word = (reply or "").strip().lower()
    if word.startswith("full"):
        return "fully"
    if word.startswith("part"):
        return "partially"
    return "no"


def selfrag_events(question, search, k=5):
    """Self-RAG pipeline that reports progress after every step."""
    yield {"step": "working", "label": "Deciding whether the manuals are needed..."}
    needs = need_retrieval(question)
    yield {"step": "retrieve_decision", "needs": needs}

    if not needs:
        yield {"step": "working", "label": "Answering without the manuals..."}
        answer = ask([{"role": "user", "content": question}])
        yield {"step": "final", "answer": answer or "(No answer returned)",
               "accepted": True, "draft": None, "support": "none"}
        return

    yield {"step": "working", "label": "Searching the manuals..."}
    results = search(question, k=k)
    yield {"step": "retrieved", "chunks": results, "compact": True}

    yield {"step": "working", "label": "Checking which passages are relevant..."}
    flags = judge_relevance(question, results)
    yield {"step": "relevance", "chunks": results, "flags": flags}
    relevant = [c for c, f in zip(results, flags) if f]

    if not relevant:
        yield {"step": "final", "answer": NOT_FOUND, "accepted": False,
               "reason": "None of the retrieved passages were relevant.",
               "draft": None, "support": "none"}
        return

    yield {"step": "working", "label": "Writing a draft answer..."}
    draft = generate(question, relevant) or ""
    yield {"step": "draft", "text": draft}

    yield {"step": "working", "label": "Checking the draft against the passages..."}
    label = check_support(question, draft, relevant)
    yield {"step": "support", "label": label}

    if label == "fully":
        yield {"step": "final", "answer": draft, "accepted": True,
               "draft": draft, "support": label}
    else:
        yield {"step": "final", "answer": NOT_FOUND, "accepted": False,
               "reason": "The draft answer was not fully supported by the passages, so it was rejected.",
               "draft": draft, "support": label}


def normal_events(question, search, k=5):
    """Normal RAG: search, then answer. No checks."""
    yield {"step": "working", "label": "Searching the manuals..."}
    results = search(question, k=k)
    yield {"step": "retrieved", "chunks": results, "compact": False}

    yield {"step": "working", "label": "Writing an answer..."}
    answer = generate(question, results)
    yield {"step": "final", "answer": answer or "(No answer returned)",
           "accepted": True, "draft": None, "support": "none"}