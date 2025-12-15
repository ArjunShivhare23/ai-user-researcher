import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI User Researcher", page_icon="🔍", layout="wide")

st.title("🔍 AI User Researcher: Feedback Analyzer")
st.markdown("### Upload raw user interviews and instantly find hidden patterns.")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Input Data")
    default_text = """I love the dark mode, it looks great.
The app crashes when I try to download songs.
Offline mode is hard to find, please fix the button.
The subscription is too expensive for students.
I want to share lyrics on Instagram but I can't.
Downloads are slow and the icon is confusing.
The UI is beautiful but the app is buggy."""
    
    # Allow user to paste their own data
    raw_text = st.text_area("Paste Interviews (One per line):", value=default_text, height=200)
    interviews = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    st.divider()
    st.header("2. Search Data")
    search_query = st.text_input("Semantic Search:", placeholder="e.g., 'Money issues'")

# --- MAIN LOGIC ---
if interviews:
    # 1. TRAIN THE ENGINE (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(interviews)
    doc_vectors = tfidf_matrix.toarray()

    # --- TAB 1: CLUSTER HEATMAP ---
    tab1, tab2 = st.tabs(["🔥 Pattern Heatmap", "🔎 Semantic Search"])
    
    with tab1:
        st.subheader("Cluster Analysis: Which users have similar problems?")
        
        # Calculate Similarity
        similarity_matrix = cosine_similarity(doc_vectors)
        
        # Draw Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        
        st.info("💡 **How to read this:** Red squares mean two users said nearly the same thing.")

    # --- TAB 2: SEARCH ENGINE ---
    with tab2:
        st.subheader(f"Results for: '{search_query}'")
        
        if search_query:
            # Vectorize the query
            query_vector = vectorizer.transform([search_query]).toarray()
            
            # Calculate Match Scores
            scores = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Show Table
            results = pd.DataFrame({'User Feedback': interviews, 'Match Score': scores})
            results = results.sort_values(by='Match Score', ascending=False)
            
            # Highlight high matches
            st.dataframe(
                results.style.background_gradient(cmap="Greens", subset=["Match Score"]),
                use_container_width=True
            )
        else:
            st.write("👈 Enter a search term in the sidebar to test the AI.")

else:
    st.warning("Please paste some text in the sidebar to begin.")

st.write("---")
st.caption("Built by Arjun Shivhare | Powered by Scikit-Learn NLP")
