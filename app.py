import streamlit as st
import nltk
from rake_nltk import Rake
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -----------------------------
# NLTK downloads (fixed)
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")   # 🔥 Required fix for RAKE


# -----------------------------
# RULE-BASED: RAKE
# -----------------------------
def rake_extract(text, top_n=10):
    r = Rake()
    r.extract_keywords_from_text(text)
    scores = r.get_ranked_phrases_with_scores()
    return [kw for score, kw in scores[:top_n]]


# -----------------------------
# RULE-BASED: YAKE
# -----------------------------
def yake_extract(text, top_n=10):
    y = yake.KeywordExtractor(lan="en", n=1, top=top_n)
    return [kw for kw, score in y.extract_keywords(text)]


# -----------------------------
# ML-BASED: TF-IDF
# -----------------------------
def tfidf_extract(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform([text])
    scores = X.toarray()[0]

    indices = np.argsort(scores)[::-1][:top_n]
    features = vectorizer.get_feature_names_out()

    return [features[i] for i in indices]


# -----------------------------
# HYBRID METHOD (RAKE + YAKE + TF-IDF)
# -----------------------------
def hybrid_extract(text, top_n=10):
    rake_kw = rake_extract(text, top_n * 2)
    yake_kw = yake_extract(text, top_n * 2)
    tfidf_kw = tfidf_extract(text, top_n * 2)

    merged = rake_kw + yake_kw + tfidf_kw
    unique = list(dict.fromkeys(merged))  # remove duplicates
    return unique[:top_n]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Keyword Extraction App", page_icon="📰", layout="wide")

st.markdown("""
    <h1 style='text-align:center;color:#4A90E2;'>📰 Keyword Extraction for Investigative Journalism</h1>
    <p style='text-align:center;color:#999;font-size:18px;'>
        Extract meaningful keywords using Rule-based + ML methods.
    </p>
""", unsafe_allow_html=True)

st.write("")

text = st.text_area("Enter your text here:", height=250, placeholder="Paste an article, investigation report, or long document...")

method = st.selectbox(
    "Select Keyword Extraction Method",
    ["RAKE (Rule-based)", "YAKE (Rule-based)", "TF-IDF (ML-based)", "Hybrid (Recommended)"]
)

num = st.slider("Number of keywords", 5, 50, 15)

if st.button("Extract Keywords"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        if method == "RAKE (Rule-based)":
            keywords = rake_extract(text, num)
        elif method == "YAKE (Rule-based)":
            keywords = yake_extract(text, num)
        elif method == "TF-IDF (ML-based)":
            keywords = tfidf_extract(text, num)
        else:
            keywords = hybrid_extract(text, num)

        st.subheader("🔑 Extracted Keywords:")
        for i, kw in enumerate(keywords, 1):
            st.write(f"**{i}.** {kw}")
