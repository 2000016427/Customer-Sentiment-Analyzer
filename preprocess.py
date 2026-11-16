"""Text cleaning shared by training and the Streamlit app."""
import re

import nltk

# Negations carry sentiment ("not good"), so never drop them as stopwords.
KEEP_WORDS = {"no", "not", "nor", "never", "against", "but"}

_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "if", "of", "at", "by", "for", "with", "about",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "am", "as", "until", "while", "into", "through", "during", "before", "after",
}


def _load_stopwords():
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            words = set(stopwords.words("english"))
        except Exception:
            words = _FALLBACK_STOPWORDS
    return words - KEEP_WORDS


STOP_WORDS = _load_stopwords()


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation, expand n't, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags (IMDB has <br />)
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"n't\b", " not", text)          # don't -> do not
    text = re.sub(r"[^a-z\s]", " ", text)          # punctuation / digits
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
