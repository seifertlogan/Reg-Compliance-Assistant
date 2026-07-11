import streamlit as st
from rag_query import get_collection, answer_question

st.set_page_config(page_title="Reg Compliance Assistant", page_icon="🏦", layout="centered")

st.title("🏦 Reg Compliance Assistant")
st.caption(
    "A RAG-powered assistant that answers questions about Reg E and Reg CC "
    "using only retrieved regulation text, with section citations. "
    "Built as a demo — not legal or compliance advice."
)

if "collection" not in st.session_state:
    with st.spinner("Loading vector store..."):
        try:
            st.session_state.collection = get_collection()
        except Exception as e:
            st.error(
                f"Could not load the vector store: {e}\n\n"
                "Run `python ingest.py` first to build the ChromaDB index."
            )
            st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

example_questions = [
    "How much can a consumer be liable for if they don't report a lost debit card for 10 days?",
    "How many business days does a bank have to investigate an error claim?",
    "When can a bank hold funds longer than the standard availability schedule?",
]

st.markdown("**Try an example:**")
cols = st.columns(len(example_questions))
clicked_example = None
for col, q in zip(cols, example_questions):
    if col.button(q, use_container_width=True):
        clicked_example = q

question = st.text_input(
    "Ask a question about Reg E or Reg CC:",
    value=clicked_example or "",
    placeholder="e.g. When must a bank make a $50 cash deposit available?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving relevant sections and generating answer..."):
        try:
            answer, sources = answer_question(question, st.session_state.collection)
            st.session_state.history.insert(0, (question, answer, sources))
        except RuntimeError as e:
            st.error(str(e))

for q, a, sources in st.session_state.history:
    st.markdown("---")
    st.markdown(f"**Q: {q}**")
    st.markdown(a)
    with st.expander(f"Sources used ({len(sources)})"):
        for s in sources:
            st.markdown(f"- **{s['source']}**, Section {s['section']} — {s['title']}")
