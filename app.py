import streamlit as st
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from rake_nltk import Rake
import nltk
import os

# Ensure NLTK data is available
def ensure_nltk_data():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")  # Custom nltk_data directory
    nltk.data.path.append(nltk_data_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    # Ensure punkt and stopwords are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

ensure_nltk_data()

# Helper Functions
@st.cache_data
def fetch_sitemap_urls(sitemaps):
    urls = []
    for sitemap in sitemaps:
        try:
            root = ET.fromstring(requests.get(sitemap).content)
            urls.extend(url.text for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"))
        except Exception as e:
            st.error(f"Error with sitemap {sitemap}: {e}")
    return urls

@st.cache_data
def scrape_blog_data(urls):
    def fetch_blog_data(url, word_limit=1000):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.get_text(strip=True) if soup.title else "Title not found"
            content = soup.find('div', class_='main-content') or soup.find('article') or soup.find('section')
            text = content.get_text(" ").strip() if content else "Content not found"
            return {"url": url, "title": title, "content": " ".join(text.split()[:word_limit])}
        except Exception as e:
            return {"url": url, "title": "Error", "content": f"Error: {e}"}
    
    return [fetch_blog_data(url) for url in urls]

@st.cache_data
def generate_keywords(scraped_df):
    def extract_keywords_with_rake(text, num_keywords=10):
        rake = Rake()
        try:
            rake.extract_keywords_from_text(str(text))
            return ", ".join(rake.get_ranked_phrases()[:num_keywords])
        except Exception as e:
            return f"Error: {e}"

    scraped_df['keywords'] = scraped_df['content'].apply(lambda x: extract_keywords_with_rake(x))
    return scraped_df

@st.cache_data
def preprocess_text(text):
    stop_words = set(stopwords.words('english')).union({'https', 'com', 'blog', 'www'})
    stop_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, stop_words)) + r')\b')
    text = re.sub(r'\W+', ' ', str(text).lower())
    return stop_pattern.sub('', text)

@st.cache_data
def suggest_internal_links(content, blog_data, title_weight=2, threshold=0.15):
    content_cleaned = preprocess_text(content)
    blog_data['processed_keywords'] = blog_data['keywords'].apply(preprocess_text)
    blog_data['processed_title'] = blog_data['title'].apply(preprocess_text)
    combined_data = blog_data['processed_keywords'] + " " + blog_data['processed_title'] * title_weight
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    vectors = vectorizer.fit_transform(combined_data.tolist() + [content_cleaned])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    blog_data['relevance'] = similarities
    suggestions = blog_data[blog_data['relevance'] >= threshold].nlargest(50, 'relevance')  # Fetch top 50 rows
    suggestions['relevance (%)'] = (suggestions['relevance'] * 100).round(2)
    return suggestions[['title', 'url', 'relevance (%)']]

# Streamlit Application
st.title("Content Scraper and Link Suggester")

# Step 1: Sitemap Selection
SITEMAP_OPTIONS = [
    "https://acviss.com/page-sitemap.xml",
    "https://blog.acviss.com/sitemap-post.xml",
    "https://blog.acviss.com/sitemap-home.xml",
]

st.header("Step 1: Select Sitemap")
selected_sitemaps = st.multiselect(
    "Choose one or more sitemaps to crawl:",
    options=SITEMAP_OPTIONS,
    default=SITEMAP_OPTIONS  # Select all by default
)

if not selected_sitemaps:
    st.warning("Please select at least one sitemap.")
else:
    if st.button("Fetch URLs"):
        urls = fetch_sitemap_urls(selected_sitemaps)
        if urls:
            st.write(f"Extracted {len(urls)} URLs")
            st.session_state['urls'] = urls
            st.dataframe(urls)
        else:
            st.error("No URLs extracted. Check the sitemap format.")

# Step 2: Scrape Blog Data
st.header("Step 2: Scrape Blog Data")
if "urls" in st.session_state:
    if st.button("Scrape Blogs"):
        scraped_data = scrape_blog_data(st.session_state['urls'])
        scraped_df = pd.DataFrame(scraped_data)
        st.session_state['scraped_data'] = scraped_df
        st.write("Scraped Blog Data")
        st.dataframe(scraped_df)
        scraped_df.to_csv("scraped_data.csv", index=False)
else:
    st.warning("Please fetch URLs first!")

# Step 3: Extract Keywords
st.header("Step 3: Extract Keywords")
if "scraped_data" in st.session_state:
    if st.button("Generate Keywords"):
        scraped_df = generate_keywords(st.session_state['scraped_data'])
        st.session_state['scraped_data'] = scraped_df
        st.write("Updated Blog Data with Keywords")
        st.dataframe(scraped_df)
        scraped_df.to_csv("updated_scraped_data.csv", index=False)
else:
    st.warning("Please scrape blogs first!")

# Step 4: Suggest Internal Links
st.header("Step 4: Suggest Internal Links")
if "scraped_data" in st.session_state:
    blog_content = st.text_area("Enter New Blog Content")
    if st.button("Suggest Links"):
        suggestions = suggest_internal_links(blog_content, st.session_state['scraped_data'])
        if not suggestions.empty:
            st.write(f"Suggested Internal Links ({len(suggestions)} results):")
            st.dataframe(suggestions)
            csv = suggestions.to_csv(index=False)
            st.download_button(
                label="Download Suggestions as CSV",
                data=csv,
                file_name="suggestions.csv",
                mime="text/csv"
            )
        else:
            st.warning("No relevant links found!")
else:
    st.warning("Please generate keywords first!")
