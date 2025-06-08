import streamlit as st
import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Title and Sidebar
st.set_page_config(page_title="IMDB Review Sentiment Analyzer", layout="wide")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Data Insights", "🧠 Sentiment Prediction"])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    return df

# Clean reviews
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Train model and prepare data
@st.cache_resource
def train_model(df):
    df["clean_review"] = df["review"].apply(clean_text)
    X = df["clean_review"]
    y = df["sentiment"].map({"positive": 1, "negative": 0})

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))

    return model, vectorizer, accuracy, cm

# Load data and train
df = load_data()
model, vectorizer, accuracy, cm = train_model(df)

# --- Page: Home ---
if page == "🏠 Home":
    st.title("🎬 IMDB Movie Review Sentiment Analyzer")
    st.markdown("Analyze movie reviews and predict whether the sentiment is **positive** or **negative**.")

    review_input = st.text_area("✍️ Enter a Movie Review", height=200)

    if st.button("🔍 Analyze"):
        if review_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            cleaned = clean_text(review_input)
            vec_input = vectorizer.transform([cleaned])
            prediction = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input)[0][prediction]

            st.subheader("🔎 Prediction Result")
            if prediction == 1:
                st.success(f"✅ Sentiment: Positive ({prob*100:.2f}% confidence)")
            else:
                st.error(f"❌ Sentiment: Negative ({prob*100:.2f}% confidence)")

# --- Page: Data Insights ---
elif page == "📈 Data Insights":
    st.title("📊 IMDB Data Insights")

    st.subheader("⚙️ Dataset Overview")
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(10))

    st.subheader("🔁 Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax1)
    ax1.set_title("Sentiment Counts")
    st.pyplot(fig1)

    st.subheader("🧠 Model Performance: Confusion Matrix")
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    st.subheader(f"📈 Model Accuracy: {accuracy*100:.2f}%")

# --- Page: Custom Prediction ---
elif page == "🧠 Sentiment Prediction":
    st.title("🧠 Try Out Custom Review")

    user_input = st.text_area("Type your review here:")
    if st.button("Predict Sentiment"):
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        result = model.predict(vec_input)[0]
        prob = model.predict_proba(vec_input)[0][result]

        if result == 1:
            st.success(f"✅ Positive ({prob*100:.2f}%)")
        else:
            st.error(f"❌ Negative ({prob*100:.2f}%)")

