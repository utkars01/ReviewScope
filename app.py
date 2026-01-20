import streamlit as st
import pandas as pd

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda

st.set_page_config(
    page_title="ReviewScope",
    layout="wide"
)


st.title("ReviewScope – Smart Review Analysis Platform")
st.caption("Analyze customer reviews to discover sentiment trends and hidden topics")

st.write("Upload a CSV file with a 'review' column to analyze text.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Waiting for CSV file upload")
else:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain a column named 'review'")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        df["clean_text"] = df["review"].apply(clean_text)
        df["sentiment"] = df["review"].apply(get_sentiment)

        st.subheader("Sentiment Results")
        st.dataframe(df[["review", "sentiment"]].head(10))

        st.subheader("Topic Modeling")
        lda, topics, coherence = train_lda(df["clean_text"])

        st.write("Coherence Score:", coherence)
        for t in topics:
            st.write(t)
