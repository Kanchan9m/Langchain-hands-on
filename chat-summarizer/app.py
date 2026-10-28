

import traceback
from io import BytesIO
import validators
import streamlit as st

from langchain.prompts import PromptTemplate
from lanchain_groq import ChatGroq
from lanchain.chains.summarize import load_summarize_chain
from lanchain_community.document_loaders import UnstructuresURLLoader

st.set_page_config(
    page_title="Website Summarizer - Langchain + Groq",
    page_icon="🦜",
    layout="centeres",
    initial_sidebar_state="expanded",
)

st.title("🦜 Website Summarizer — LangChain + Groq")

st.markdown(
    """
A lightweight Streamlit app that fetches text from a website and summarizes it using Groq (LLaMA).

**How to use:**
1. Paste a website URL (must be reachable and allow scraping).
2. Enter your Groq API key in the sidebar.
3. Choose summary length and options, then click **Summarize**.
"""
)

#sidebar
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    summary_words = st.slider("Approx. summary length (words)", min_value=50, max_value=1000, value=300, step=50)
    show_extracted = st.checkbox("Show extracted page text before summarizing", value=False)
    chain_type = st.selectbox("Summarization chain type", options=["stuff", "map_reduce", "refine"], index=0)
    st.markdown("---")
    st.caption("This app summarizes websites only (no YouTube). Use responsibly and respect site robots.txt and copyright.")
    # Dev mode toggle
    dev_mode = st.checkbox("Dev mode (show debug)", value=False)

# Keep dev_mode in session_state for other places to inspect
st.session_state["dev_mode"] = dev_mode

generic_url = st.text_input("Website URL", placeholder="https://example.com/article-to-summarize", label_visibility="visible")

col1, col2 = st.columns([3, 1])
with col1:
    summarize_btn = st.button("Summarize website")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

@st.cache_data(show_spinner=False)
def load_url_content(url: str):
    """
    Load website content using UnstructuredURLLoader and return list of LangChain docs.
    """
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/116.0.0.0 Safari/537.36"
        },
    )
    docs = loader.load()
    return docs

def make_prompt(words: int) -> PromptTemplate:
    """
    Build a PromptTemplate. The summarization chain will supply the page text into the `{text}` variable.
    We bake the 'words' guidance directly into the template string so the chain only needs `text`.
    """
    template = (
        f"Provide a concise, clear summary of the following content in about {words} words.\n\n"
        "Content:\n{text}\n\n"
        "Include the key points and any conclusions. Keep the tone neutral."
    )
    # The chain will pass the page text as 'text', so the template only expects "text".
    return PromptTemplate(template=template, input_variables=["text"])

