
import nltk
nltk.download('stopwords')  # Ensure stopwords are downloaded

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import streamlit as st


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.lower().strip()


# Function to extract meaningful keywords
def extract_keywords(text, num_keywords=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:num_keywords]


# Function to load and preprocess the CSV file
def load_csv(file):
    try:
        data = pd.read_csv(file)
        if not {'Title', 'URL', 'Keywords'}.issubset(data.columns):
            raise ValueError("CSV file must have 'Title', 'URL', and 'Keywords' columns.")
        data = data.groupby(['Title', 'URL'], as_index=False).agg({'Keywords': lambda x: ' '.join(set(x))})
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None


# Function to calculate relevance for content
def calculate_relevance(content, blog_data):
    content_cleaned = preprocess_text(content)
    blog_data['Combined'] = blog_data.apply(
        lambda x: f"{preprocess_text(x['Keywords'])} {preprocess_text(x['Title'])}", axis=1
    )
    combined_text = blog_data['Combined'].tolist()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined_text + [content_cleaned])

    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    return similarities


# Function to suggest internal links
def suggest_internal_links(content, blog_data, threshold=0.20, top_n=10):
    relevance_scores = calculate_relevance(content, blog_data)

    suggestions = []
    for idx, score in enumerate(relevance_scores):
        if score >= threshold:
            matched_keywords = list(set(extract_keywords(blog_data.iloc[idx]["Combined"])))
            matched_keywords = [kw for kw in matched_keywords if len(kw.split()) > 1]

            suggestions.append({
                "Blog Title": blog_data.iloc[idx]["Title"],
                "Matched Keywords": ', '.join(matched_keywords),
                "Suggested URL": blog_data.iloc[idx]["URL"],
                "Relevance Score (%)": f"{score * 100:.2f}",
            })

    suggestions = sorted(suggestions, key=lambda x: -float(x["Relevance Score (%)"]))[:top_n]
    return suggestions


# Streamlit App
def main():
    st.title("Internal Link Suggestion Tool")
    st.write("Upload your blog database and enter your new blog content to get suggestions for internal links.")

    uploaded_file = st.file_uploader("Upload Blog Database (CSV format)", type=["csv"])
    if uploaded_file:
        blog_data = load_csv(uploaded_file)
        if blog_data is not None:
            st.success("Blog database loaded successfully!")

            new_blog_content = st.text_area("Enter New Blog Content", height=200)
            if st.button("Get Suggestions"):
                if new_blog_content.strip():
                    suggestions = suggest_internal_links(new_blog_content, blog_data)
                    if suggestions:
                        st.subheader("Suggested Internal Links:")
                        for idx, suggestion in enumerate(suggestions, 1):
                            st.write(f"**{idx}. {suggestion['Blog Title']} ({suggestion['Relevance Score (%)']}%)**")
                            st.write(f"   URL: [Link]({suggestion['Suggested URL']})")
                            st.write(f"   Matched Keywords: {suggestion['Matched Keywords']}\n")
                    else:
                        st.warning("No relevant links found. Try adjusting the content or lowering the relevance threshold.")
                else:
                    st.error("Please enter blog content to get suggestions.")

            if 'suggestions' in locals() and suggestions:
                suggestions_df = pd.DataFrame(suggestions)
                st.download_button(
                    label="Download Suggestions as CSV",
                    data=suggestions_df.to_csv(index=False),
                    file_name="suggested_links.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
