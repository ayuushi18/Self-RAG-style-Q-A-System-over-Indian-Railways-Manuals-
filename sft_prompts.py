def numbered(passages):
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(passages, start=1))


def retrieve_prompt(question):
    return (
        "TASK: RETRIEVE\n"
        "Decide whether this question needs the Indian Railways rule manuals to be answered. "
        "Reply Yes or No.\n\n"
        f"Question: {question}"
    )


def relevance_prompt(question, passage):
    return (
        "TASK: RELEVANCE\n"
        "Decide whether the passage contains information that helps answer this exact question. "
        "Reply Relevant or Irrelevant.\n\n"
        f"Question: {question}\n\nPassage: {passage}"
    )


def generate_prompt(question, passages):
    return (
        "TASK: ANSWER\n"
        "Answer the question using the passages. Keep it to 1-3 sentences.\n\n"
        f"Passages:\n{numbered(passages)}\n\nQuestion: {question}"
    )


def support_prompt(question, answer, passages):
    return (
        "TASK: SUPPORT\n"
        "Decide whether every claim in the answer, including numbers, is stated in the passages, "
        "and the answer directly answers the question that was asked. "
        "Reply Supported or Unsupported.\n\n"
        f"Question: {question}\n\nAnswer: {answer}\n\nPassages:\n{numbered(passages)}"
    )