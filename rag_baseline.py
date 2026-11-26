from hybrid import hybrid_search
from llm import ask

SYSTEM = (
    "You answer questions about Indian Railways rules. "
    "Use the passages provided to answer. Keep the answer to 1-3 sentences."
)


def answer(question, k=5):
    """Normal RAG: retrieve the top k chunks, then answer from them."""
    results = hybrid_search(question, k=k)
    context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(results, start=1))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]
    return ask(messages), results


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type q to quit): ")
        if question.strip().lower() == "q":
            break
        reply, results = answer(question)
        print("\nANSWER:", reply)
        print("Sources:", ", ".join(f"{c['id']} (p.{c['page']})" for c in results))