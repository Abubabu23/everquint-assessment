import streamlit as st

from search import HybridSearcher
from summarize import baseline_summary, clean_text, llm_summary

st.set_page_config(page_title="AI/ML Document Search", page_icon="🔎", layout="wide")


@st.cache_resource
def get_searcher():
    return HybridSearcher()


@st.cache_data(show_spinner=False)
def get_llm_summary(doc_id, text, sentences):
    return llm_summary(text, sentences=sentences)


searcher = get_searcher()
doc_text = {d["id"]: clean_text(d["text"]) for d in searcher.docs}

st.title("🔎 AI/ML Document Search")
st.caption(f"{len(searcher.docs)} Wikipedia articles | BM25 + MiniLM embeddings + RRF")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Search mode", ["hybrid", "bm25", "dense"], index=0)
    k = st.slider("Results", 1, 10, 5)
    n_sent = st.slider("Summary length (sentences)", 1, 8, 3)

query = st.text_input("Search query", placeholder="e.g. how does backpropagation work")

if query.strip():
    results = searcher.search(query, k=k, mode=mode)
    st.subheader(f"Top {len(results)} results ({mode})")

    for i, r in enumerate(results, 1):
        with st.expander(f"{i}. {r['title']}  (score {r['score']:.4f})", expanded=(i == 1)):
            st.markdown(f"[{r['url']}]({r['url']})")
            st.write(clean_text(r["snippet"]))

            if st.button("Summarize", key=f"sum_{r['doc_id']}_{i}"):
                text = doc_text[r["doc_id"]]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Baseline (lead-{n_sent})**")
                    st.write(baseline_summary(text, n=n_sent))
                with col2:
                    st.markdown(f"**LLM ({n_sent} sentences)**")
                    with st.spinner("LLM summarizing..."):
                        try:
                            st.write(get_llm_summary(r["doc_id"], text, n_sent))
                        except Exception as e:
                            st.error(f"LLM error: {e}")
else:
    st.info("Type your query.")