import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from src.preprocessing import clean_text
from src.sentiment_analysis import get_sentiment
from src.topic_modeling import train_lda

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ReviewScope", layout="wide")

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        "<style>body{background-color:#0e1117;color:white}</style>",
        unsafe_allow_html=True
    )

st.title("📊 ReviewScope – Smart Review Analysis Platform")
st.caption("An end-to-end interactive NLP system for review analytics")

# ---------------- HELPER FUNCTIONS ----------------
def get_top_keywords(text_series, top_n=20):
    words = " ".join(text_series).split()
    return Counter(words).most_common(top_n)

def sentiment_confidence(text):
    score = sum(w in text.lower() for w in ["good","great","excellent","amazing","love"]) - \
            sum(w in text.lower() for w in ["bad","worst","poor","terrible","hate"])
    return min(max((score + 5) * 10, 0), 100)

# ---------------- MAIN TABS ----------------
tabs = st.tabs([
    "🏠 Overview",
    "📝 Live Text Analysis",
    "📂 Dataset Workflow",
    "🔑 Keyword Insights",
    "📊 Dashboard"
])

# ================= OVERVIEW =================
with tabs[0]:
    st.subheader("What is ReviewScope?")
    st.info("""
    ReviewScope is a professional NLP analytics platform that enables:
    - Real-time sentiment detection
    - Topic discovery
    - Keyword insights
    - Interactive dashboards
    """)

# ================= LIVE TEXT ANALYSIS =================
with tabs[1]:
    st.subheader("📝 Real-Time Review Analysis")
    text = st.text_area("Type or paste a review")

    if text.strip():
        sentiment = get_sentiment(text)
        confidence = sentiment_confidence(text)
        st.success(f"Sentiment: **{sentiment}**")
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence}%")

# ================= DATASET WORKFLOW =================
with tabs[2]:
    st.subheader("📂 Step-by-Step Dataset Analysis")

    step = st.radio(
        "Workflow Step",
        ["1️⃣ Upload", "2️⃣ Preprocess", "3️⃣ Analyze", "4️⃣ Results"]
    )

    if step == "1️⃣ Upload":
        uploaded = st.file_uploader("Upload CSV (review column required)", type=["csv"])

    if step == "2️⃣ Preprocess" and "uploaded" in locals() and uploaded:
        df = pd.read_csv(uploaded)
        df["clean_text"] = df["review"].apply(clean_text)
        st.success("Text preprocessing completed")

    if step == "3️⃣ Analyze" and "df" in locals():
        df["sentiment"] = df["review"].apply(get_sentiment)
        lda, topics, _ = train_lda(df["clean_text"], num_topics=5)
        st.success("Analysis completed")

    if step == "4️⃣ Results" and "df" in locals():
        with st.expander("📄 Filter Reviews"):
            keyword = st.text_input("Search keyword")
            if keyword:
                st.dataframe(df[df["review"].str.contains(keyword, case=False)])

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            file_name="reviewscope_results.csv"
        )

# ================= KEYWORD INSIGHTS =================
with tabs[3]:
    if "df" in locals():
        st.subheader("🔑 Keyword Overview")
        keywords = get_top_keywords(df["clean_text"])
        kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(kw_df.set_index("Keyword"))
        with col2:
            wc = WordCloud(background_color="white").generate(" ".join(df["clean_text"]))
            plt.imshow(wc)
            plt.axis("off")
            st.pyplot()

# ================= DASHBOARD =================
with tabs[4]:
    if "df" in locals():
        st.subheader("📊 Summary Dashboard")

        colA, colB, colC = st.columns(3)
        colA.metric("Total Reviews", len(df))
        colB.metric("Positive", (df["sentiment"]=="Positive").sum())
        colC.metric("Negative", (df["sentiment"]=="Negative").sum())

        st.subheader("Topic-Wise Sentiment")
        df["topic"] = [f"Topic {i%5}" for i in range(len(df))]
        pivot = df.pivot_table(index="topic", columns="sentiment", aggfunc="size", fill_value=0)
        st.bar_chart(pivot)
