
import os
import time
from urllib.parse import quote

import streamlit as st

from search import SearchEngine
from summarize import resolve_length, summarize_results

PAGE_SIZE = 5           # results per page
FETCH = 50              # how many results are retrieved (and paginated)
TOPIC_LABEL = {"tamil_nadu": "Tamil Nadu", "ai": "AI / ML"}

st.set_page_config(page_title="Document Search & Summarization", page_icon="🔎", layout="wide")

try:                                                    # API key from Streamlit secrets (hosted) or env variable (local)
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except Exception:
    pass


@st.cache_resource(show_spinner="Loading the search index (the first load takes a while) ...")
def load_engine():
    return SearchEngine()


def suggest(text, titles, limit=6):
    """Auto-suggestion: article titles that start with / contain what the user typed."""
    t = text.strip().lower()
    if len(t) < 2:
        return []
    starts = [x for x in titles if x.lower().startswith(t)]
    contains = [x for x in titles if t in x.lower() and x not in starts]
    return (starts + contains)[:limit]


def set_query(value):
    st.session_state["query"] = value


def go_page(delta):
    st.session_state["page"] += delta


engine = load_engine()
titles = [d["title"] for d in engine.docs]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Search method", ["hybrid", "bm25", "embedding"],
                          help="hybrid = BM25 keywords + embeddings (merged with Reciprocal Rank Fusion)")
    make_summary = st.checkbox("Generate summary", value=True)
    n_docs = st.slider("Documents to summarize", 1, 5, 3)
    choice = st.radio("Summary length", ["short", "medium", "long", "custom"], index=1, horizontal=True)
    length = st.slider("Words", 20, 400, 100, step=10) if choice == "custom" else choice
    st.caption(f"Target length: about {resolve_length(length)} words")

# ---------------- main ----------------
st.title("🔎 Document Search & Summarization")
st.caption(f"{len(engine.docs)} Wikipedia articles (Tamil Nadu + AI/ML) · BM25 + embedding search · LLM summaries")

query = st.text_input("Search", key="query", placeholder="e.g. Chola temple architecture")

suggestions = suggest(query, titles)
if suggestions:
    st.caption("Suggestions (press Enter after typing to refresh):")
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        cols[i % 3].button(s, key=f"sg_{i}", on_click=set_query, args=(s,), use_container_width=True)

if st.button("Search", type="primary") and query.strip():
    q = query.strip()
    t0 = time.time()
    results = engine.search(q, top_n=FETCH, method=method)
    state = {"results": results, "page": 0, "last_query": q, "search_s": time.time() - t0,
             "summary": None, "summary_error": None, "summary_s": None, "summary_len": None,
             "summary_docs": [r["title"] for r in results[:n_docs]]}
    if make_summary and results:
        with st.spinner("Summarizing the top results ..."):
            try:
                t1 = time.time()
                state["summary"] = summarize_results(engine, q, results[:n_docs], length)
                state["summary_s"] = time.time() - t1
                state["summary_len"] = resolve_length(length)
            except Exception as e:                      # e.g. API quota: search results are still shown
                state["summary_error"] = str(e)
    st.session_state.update(state)

results = st.session_state.get("results")
if results:
    st.markdown(f"**{len(results)} results for “{st.session_state['last_query']}”** "
                f"· search took {st.session_state['search_s'] * 1000:.0f} ms")

    if st.session_state.get("summary"):
        st.subheader("Summary")
        st.info(st.session_state["summary"])
        words = len(st.session_state["summary"].split())
        st.caption(f"{words} words (target {st.session_state['summary_len']}) · "
                   f"generated in {st.session_state['summary_s']:.1f}s · based on: "
                   + ", ".join(st.session_state["summary_docs"]))
    elif st.session_state.get("summary_error"):
        st.error("Summary could not be generated (the search results below are still valid): "
                 + st.session_state["summary_error"][:300])

    st.subheader("Results")
    pages = (len(results) + PAGE_SIZE - 1) // PAGE_SIZE
    page = min(st.session_state["page"], pages - 1)
    for r in results[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]:
        with st.container(border=True):
            url = "https://en.wikipedia.org/wiki/" + quote(r["title"].replace(" ", "_"))
            st.markdown(f"### [{r['title']}]({url})")
            st.caption(f"{TOPIC_LABEL.get(r['topic'], r['topic'])} · score {r['score']}")
            st.write(r["excerpt"][:600] + " ...")

    c1, c2, c3 = st.columns([1, 2, 1])
    c1.button("← Previous", disabled=page == 0, on_click=go_page, args=(-1,))
    c2.markdown(f"<div style='text-align:center'>Page {page + 1} of {pages}</div>", unsafe_allow_html=True)
    c3.button("Next →", disabled=page >= pages - 1, on_click=go_page, args=(1,))
elif "last_query" in st.session_state:
    st.warning("No results.")