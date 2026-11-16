"""Streamlit UI for the AI-Powered Customer Sentiment Analyzer.

Run:  streamlit run app.py
"""
import os

import joblib
import pandas as pd
import streamlit as st

import preprocess  # noqa: F401  (needed so the pickled pipeline can find clean_text)

MODEL_PATH = "model.joblib"
EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐"}

st.set_page_config(page_title="Customer Sentiment Analyzer", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict(model, texts, threshold):
    """Return DataFrame with sentiment, confidence and per-class probabilities."""
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    has_neutral = "neutral" in classes
    rows = []
    for p in probs:
        top = int(p.argmax())
        label, conf = classes[top], float(p[top])
        # Binary model (e.g. IMDB): treat low-confidence predictions as Neutral.
        if not has_neutral and conf < threshold:
            label = "neutral"
        rows.append({"sentiment": label, "confidence": conf,
                     **{f"P({c})": float(v) for c, v in zip(classes, p)}})
    return pd.DataFrame(rows)


st.title("💬 Customer Sentiment Analyzer")
st.caption("Paste a customer review and get the predicted sentiment with a confidence score.")

model = load_model()
if model is None:
    st.error("`model.joblib` not found. Train a model first:\n\n"
             "`python train_model.py --data <your_dataset.csv> --text-col <col> --label-col <col>`")
    st.stop()

is_binary = "neutral" not in list(model.classes_)
threshold = 0.65
with st.sidebar:
    st.header("Settings")
    if is_binary:
        threshold = st.slider("Neutral threshold", 0.5, 0.95, 0.65, 0.01,
                              help="Model is binary; reviews with confidence below this are labelled Neutral.")
    else:
        st.write("Model classes:", ", ".join(model.classes_))

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV)"])

with tab_single:
    review = st.text_area("Customer review", height=160,
                          placeholder="e.g. The product arrived late and the quality was disappointing...")
    if st.button("Analyze", type="primary"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            res = predict(model, [review], threshold).iloc[0]
            label = res["sentiment"]
            st.subheader(f"{EMOJI.get(label, '')} {label.capitalize()}")
            st.metric("Confidence", f"{res['confidence']:.1%}")
            st.progress(min(res["confidence"], 1.0))
            prob_df = (res.filter(like="P(").rename(lambda k: k[2:-1]).astype(float)
                       .to_frame("probability"))
            st.bar_chart(prob_df)

with tab_batch:
    file = st.file_uploader("Upload a CSV of reviews", type="csv")
    if file:
        df = pd.read_csv(file)
        col = st.selectbox("Column containing review text", df.columns)
        if st.button("Analyze all"):
            with st.spinner("Analyzing..."):
                out = predict(model, df[col].fillna("").astype(str).tolist(), threshold)
            result = pd.concat([df.reset_index(drop=True), out], axis=1)
            counts = result["sentiment"].value_counts()
            st.bar_chart(counts)
            st.dataframe(result, use_container_width=True)
            st.download_button("Download results", result.to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")
