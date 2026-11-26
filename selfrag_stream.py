from hybrid import hybrid_search
from llm import ask
from selfrag_prompt import (
    need_retrieval, judge_relevance, generate, check_support, NOT_FOUND,
)


def selfrag_events(question, k=5):
    """Self-RAG pipeline that reports progress after every step."""
    yield {"step": "working", "label": "Deciding whether the manuals are needed..."}
    needs = need_retrieval(question)
    yield {"step": "retrieve_decision", "needs": needs}

    if not needs:
        yield {"step": "working", "label": "Answering without the manuals..."}
        answer = ask([{"role": "user", "content": question}])
        yield {"step": "final", "answer": answer or "(No answer returned)",
               "accepted": True}
        return

    yield {"step": "working", "label": "Searching the manuals..."}
    results = hybrid_search(question, k=k)
    yield {"step": "retrieved", "chunks": results, "compact": True}

    yield {"step": "working", "label": "Checking which passages are relevant..."}
    flags = judge_relevance(question, results)
    yield {"step": "relevance", "chunks": results, "flags": flags}
    relevant = [c for c, f in zip(results, flags) if f]

    if not relevant:
        yield {"step": "final", "answer": NOT_FOUND, "accepted": False,
               "reason": "None of the retrieved passages were relevant."}
        return

    yield {"step": "working", "label": "Writing a draft answer..."}
    draft = generate(question, relevant) or ""
    yield {"step": "draft", "text": draft}

    yield {"step": "working", "label": "Checking the draft against the passages..."}
    label = check_support(question, draft, relevant)
    yield {"step": "support", "label": label}

    if label == "fully":
        yield {"step": "final", "answer": draft, "accepted": True}
    else:
        yield {"step": "final", "answer": NOT_FOUND, "accepted": False,
               "reason": "The draft answer was not fully supported by the passages, so it was rejected."}


def normal_events(question, k=5):
    """Normal RAG: search, then answer. No checks."""
    yield {"step": "working", "label": "Searching the manuals..."}
    results = hybrid_search(question, k=k)
    yield {"step": "retrieved", "chunks": results, "compact": False}

    yield {"step": "working", "label": "Writing an answer..."}
    answer = generate(question, results)
    yield {"step": "final", "answer": answer or "(No answer returned)",
           "accepted": True}