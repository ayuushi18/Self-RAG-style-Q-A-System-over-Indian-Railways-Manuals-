import streamlit as st

from core import selfrag_events, normal_events
from azure_search import hybrid_search_azure

st.set_page_config(page_title="Railway Rules Self-RAG", page_icon="🚆", layout="wide")

EXAMPLES = {
    "Answerable: clerkage on a wait-listed ticket":
        "What is the clerkage charge for cancelling a wait-listed ticket?",
    "Trap: season ticket price":
        "A monthly season ticket is priced at how many times the single fare?",
    "Trap: wharfage free time":
        "How many days of free time are allowed before wharfage starts on parcels?",
    "Known weak spot: 1,000 km ticket":
        "For how many days is a 1,000 km single journey ticket valid?",
    "No lookup needed: a greeting":
        "Hello! What can you help me with?",
}


def set_question(text):
    st.session_state["question"] = text


def show_chunks(chunks, flags=None):
    for i, c in enumerate(chunks):
        mark = ""
        if flags is not None:
            mark = "✅ relevant" if flags[i] else "❌ not relevant"
        with st.expander(f"[{i + 1}] {c['source']}, page {c['page']}   {mark}"):
            text = c["text"]
            st.write(text[:700] + ("..." if len(text) > 700 else ""))


st.title("🚆 Railway rules Q&A with self-checks")
st.caption(
    "Answers come from two Indian Railways rule manuals. "
    "In Self-RAG mode the system checks its own work before it answers."
)

with st.sidebar:
    mode = st.radio("Mode", ["Self-RAG (with checks)", "Normal RAG (no checks)"])
    st.markdown("**Try an example**")
    for label, text in EXAMPLES.items():
        st.button(label, on_click=set_question, args=(text,), use_container_width=True)

question = st.text_input("Ask a question about railway rules", key="question")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    if mode.startswith("Self"):
        events = selfrag_events(question, hybrid_search_azure)
    else:
        events = normal_events(question, hybrid_search_azure)

    working = st.empty()

    for ev in events:
        step = ev["step"]

        if step == "working":
            working.info("⏳ " + ev["label"])

        elif step == "retrieve_decision":
            if ev["needs"]:
                st.markdown("### 1. Retrieve? Yes, look it up in the manuals")
            else:
                st.markdown("### 1. Retrieve? No, this needs no lookup")

        elif step == "retrieved":
            if ev.get("compact"):
                st.caption(f"Search found {len(ev['chunks'])} candidate passages.")
            else:
                st.markdown("### Search results")
                show_chunks(ev["chunks"])

        elif step == "relevance":
            flags = ev["flags"]
            st.markdown("### 2. Which passages are relevant?")
            st.write(f"{sum(flags)} of {len(flags)} passages judged relevant.")
            show_chunks(ev["chunks"], flags)

        elif step == "draft":
            st.markdown("### 3. Draft answer")
            st.write(ev["text"])

        elif step == "support":
            st.markdown("### 4. Is the draft supported by the passages?")
            verdicts = {
                "fully": "✅ Fully supported",
                "partially": "⚠️ Only partly supported",
                "no": "❌ Not supported",
            }
            st.write(verdicts.get(ev["label"], ev["label"]))

        elif step == "final":
            working.empty()
            st.markdown("---")
            st.markdown("### Final answer")
            if ev["accepted"]:
                st.success(ev["answer"])
            else:
                st.warning(ev["answer"])
                if ev.get("reason"):
                    st.caption(ev["reason"])