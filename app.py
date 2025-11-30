# app.py
import streamlit as st
import yake
from rake_nltk import Rake
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import tempfile
import PyPDF2
import pandas as pd

# ---------------------------
# Setup (downloads once)
# ---------------------------
nltk.download("punkt")
nltk.download("stopwords")



# ---------------------------
# Extraction Functions
# ---------------------------
def yake_extract(text, top_n=15):
    kw_extractor = yake.KeywordExtractor(top=top_n, stopwords=None)
    kw = kw_extractor.extract_keywords(text)
    return [k for k, score in kw]

def rake_extract(text, top_n=15):
    r = Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    return phrases[:top_n]

def spacy_np_extract(text, top_n=15):
    doc = nlp(text)
    # collect noun chunks and lemmatize them for normalization
    chunks = []
    for chunk in doc.noun_chunks:
        lemma = " ".join([token.lemma_ for token in chunk if not token.is_punct and not token.is_space])
        if lemma:
            chunks.append(lemma.lower())
    # preserve order & uniqueness
    unique = list(dict.fromkeys(chunks))
    return unique[:top_n]

def tfidf_extract(text, top_n=15):
    # simple TF-IDF on the single document: we split into sentences as "corpus"
    sentences = [s for s in text.split(".") if s.strip()]
    if not sentences:
        sentences = [text]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=500)
    X = vectorizer.fit_transform(sentences)
    # compute average tfidf across sentences for each term
    scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    term_scores = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [t for t, s in term_scores[:top_n]]

def rule_freq_extract(text, top_n=15):
    # rule-based frequency after cleaning using spaCy tokenization
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t) > 2]
    most = [w for w, c in Counter(tokens).most_common(top_n)]
    return most

def hybrid_extract(text, top_n=20):
    # combine many sources, score by occurrence and method-rank
    sources = []
    sources += yake_extract(text, top_n * 2)
    sources += rake_extract(text, top_n * 2)
    sources += spacy_np_extract(text, top_n * 2)
    sources += tfidf_extract(text, top_n * 2)
    sources += rule_freq_extract(text, top_n * 2)
    # scoring: earlier occurrences get slightly higher weight (rank-based)
    score = {}
    for src in sources:
        # normalize
        k = src.strip().lower()
        if not k:
            continue
        score[k] = score.get(k, 0) + 1
    # sort by score then alphabetically
    ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, s in ranked][:top_n]

# ---------------------------
# Utility: PDF -> text
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception:
        return ""

# ---------------------------
# Streamlit UI + Styling
# ---------------------------
st.set_page_config(page_title="Keyword Extractor (Stable)", layout="wide", page_icon="🔎")

st.markdown(
    """
    <style>
    body { background: linear-gradient(135deg,#0f2027,#2c5364); color: #f7f7fb; }
    .title { font-size:2.6rem; font-weight:800; text-align:center;
             background: -webkit-linear-gradient(#fff,#d0c4ff);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom:8px;}
    .subtitle { text-align:center; color:#dfe9f3; margin-bottom:22px;}
    .glass { background: rgba(255,255,255,0.06); padding:22px; border-radius:14px; border:1px solid rgba(255,255,255,0.06);
            box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
    .kw { display:inline-block; padding:8px 12px; margin:6px; border-radius:12px; background: rgba(255,255,255,0.06); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>🔎 Keyword Extraction — Stable Edition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>YAKE • RAKE • spaCy • TF–IDF • Rule-based frequency • Hybrid</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a PDF (optional) or paste text below", type=["pdf", "txt", "md"])
    text_input = st.text_area("Or paste / type your article/report here", height=300)

    if uploaded is not None:
        if uploaded.type == "application/pdf":
            text_from_pdf = extract_text_from_pdf(uploaded)
            if text_from_pdf.strip():
                st.success("PDF successfully read. Text loaded into the editor.")
                # prefill the text area only if empty to avoid overriding user's typed text
                if not text_input.strip():
                    text_input = text_from_pdf
            else:
                st.warning("Could not extract text from PDF. Please paste text manually.")
        else:
            # text file
            bytes_data = uploaded.read()
            try:
                text_from_file = bytes_data.decode("utf-8")
            except:
                text_from_file = ""
            if text_from_file:
                if not text_input.strip():
                    text_input = text_from_file

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.write("### Settings")
    method = st.selectbox("Extraction method", ["Hybrid (recommended)", "YAKE", "RAKE", "spaCy Noun Phrases", "TF-IDF", "Rule Frequency"])
    num = st.slider("Number of keywords", 5, 50, 20)
    st.write("---")
    st.write("Download results:")
    download_name = st.text_input("CSV filename (without extension)", value="keywords")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

if st.button("🔍 Extract Keywords"):
    doc_text = text_input.strip()
    if not doc_text:
        st.warning("Please paste text or upload a PDF.")
    else:
        with st.spinner("Extracting keywords..."):
            if method == "YAKE":
                kws = yake_extract(doc_text, top_n=num)
            elif method == "RAKE":
                kws = rake_extract(doc_text, top_n=num)
            elif method == "spaCy Noun Phrases":
                kws = spacy_np_extract(doc_text, top_n=num)
            elif method == "TF-IDF":
                kws = tfidf_extract(doc_text, top_n=num)
            elif method == "Rule Frequency":
                kws = rule_freq_extract(doc_text, top_n=num)
            else:  # Hybrid
                kws = hybrid_extract(doc_text, top_n=num)

        # display
        st.markdown("### 🎯 Extracted Keywords")
        for k in kws:
            st.markdown(f"<span class='kw'>{k}</span>", unsafe_allow_html=True)

        # provide CSV download
        df = pd.DataFrame({"keyword": kws})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇️ Download CSV", data=csv, file_name=f"{download_name}.csv", mime="text/csv")

