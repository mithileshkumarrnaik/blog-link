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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import os

# Ensure NLTK data is available
def ensure_nltk_data():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")  # Custom nltk_data directory
    nltk.data.path.append(nltk_data_path)

    # Create the directory if it doesn't exist
    os.makedirs(nltk_data_path, exist_ok=True)

    # Download necessary NLTK resources
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', download_dir=nltk_data_path)

ensure_nltk_data()

# Helper Function to Read Included and Excluded URLs
@st.cache_data
def read_url_filters():
    """
    Reads the included and excluded URL files from the Git repository.
    Returns two lists: included_urls and excluded_urls.
    """
    included_urls = []
    excluded_urls = []

    # Read included_urls.txt
    try:
        with open("included_urls.txt", "r") as file:
            included_urls = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        st.warning("included_urls.txt not found. Proceeding without inclusion filter.")

    # Read excluded_urls.txt
    try:
        with open("excluded_urls.txt", "r") as file:
            excluded_urls = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        st.warning("excluded_urls.txt not found. Proceeding without exclusion filter.")

    return included_urls, excluded_urls

# Helper Function to Process URLs Based on Included and Excluded Lists
@st.cache_data
def process_included_excluded_urls(all_urls, included_urls, excluded_urls):
    """
    Filters URLs based on included and excluded lists.
    """
    if included_urls:
        # Only include URLs specified in included_urls.txt
        filtered_urls = [url for url in all_urls if url in included_urls]
    else:
        # If no included_urls, exclude URLs in excluded_urls.txt
        filtered_urls = [url for url in all_urls if url not in excluded_urls]

    return filtered_urls

# Helper Function to Fetch Sitemap URLs
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

# Helper Function to Scrape Blog Data
@st.cache_data
def scrape_blog_data(urls):
    def fetch_blog_data(url, word_limit=1000):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
    
            # Extract title
            title = soup.title.get_text(strip=True) if soup.title else "Title not found"
    
            # Try extracting content using various common tags or classes
            content = (
                soup.find('div', class_='main-content') or
                soup.find('article') or
                soup.find('section') or
                soup.find('div', id='content') or  # Add more tags/classes as needed
                soup.find('body')  # Fallback to entire body
            )
    
            # Get text content
            text = content.get_text(" ").strip() if content else "Content not found"
    
            # Limit the text to the specified word limit
            return {"url": url, "title": title, "content": " ".join(text.split()[:word_limit])}
        except Exception as e:
            return {"url": url, "title": "Error", "content": f"Error: {e}"}
    
    return [fetch_blog_data(url) for url in urls]

# Helper Function to Generate Keywords
@st.cache_data
def generate_keywords(scraped_df):
    # Function to clean text by removing HTML tags and special characters
    def clean_text(text):
        text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text.strip()

    # Function to extract keywords using RAKE
    def extract_keywords_with_rake(text, num_keywords=10):
        if len(text.split()) < 10:  # Skip short or insufficient content
            return "Insufficient content"
        rake = Rake()
        try:
            rake.extract_keywords_from_text(text)
            return ", ".join(rake.get_ranked_phrases()[:num_keywords])
        except Exception as e:
            return f"Error: {e}"

    # Clean content and extract keywords
    scraped_df['clean_content'] = scraped_df['content'].apply(clean_text)
    scraped_df['keywords'] = scraped_df['clean_content'].apply(lambda x: extract_keywords_with_rake(x))

    # Return a DataFrame with relevant columns
    return scraped_df[['url', 'title', 'keywords']]

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
st.title("Content Scraper and Link Suggester with URL Inclusion/Exclusion")

# Step 1: Sitemap Selection
SITEMAP_OPTIONS = [
    "https://acviss.com/page-sitemap.xml",
    "https://blog.acviss.com/sitemap-post.xml",
    "https://blog.acviss.com/sitemap-home.xml",
]

st.header("Step 1: Select Sitemap and Apply Filters")
selected_sitemaps = st.multiselect(
    "Choose one or more sitemaps to crawl:",
    options=SITEMAP_OPTIONS,
    default=SITEMAP_OPTIONS
)

if not selected_sitemaps:
    st.warning("Please select at least one sitemap.")
else:
    if st.button("Fetch and Filter URLs"):
        all_urls = fetch_sitemap_urls(selected_sitemaps)

        # Read included and excluded URLs from Git files
        included_urls, excluded_urls = read_url_filters()

        # Apply inclusion and exclusion filters
        filtered_urls = process_included_excluded_urls(all_urls, included_urls, excluded_urls)

        if filtered_urls:
            st.write(f"Filtered {len(filtered_urls)} URLs from the sitemap(s):")
            st.session_state['urls'] = filtered_urls
            st.dataframe(filtered_urls)
        else:
            st.error("No URLs remain after applying filters.")

# Step 2: Scrape Blog Data
st.header("Step 2: Scrape Blog Data")
if "urls" in st.session_state:
    if st.button("Scrape Blogs"):
        scraped_data = scrape_blog_data(st.session_state['urls'])
        if scraped_data:
            scraped_df = pd.DataFrame(scraped_data)
            st.session_state['scraped_data'] = scraped_df
            st.write("Scraped Blog Data")
            st.dataframe(scraped_df)
        else:
            st.error("No data was scraped. Please check your URLs.")
else:
    st.warning("Please fetch URLs first!")

# Step 3: Extract Keywords
st.header("Step 3: Extract Keywords")
if "scraped_data" in st.session_state:
    if st.button("Generate Keywords"):
        # Generate keywords from the scraped data
        scraped_df = generate_keywords(st.session_state['scraped_data'])
        st.session_state['scraped_data'] = scraped_df

        # Display the results
        st.write("Extracted Keywords:")
        st.dataframe(scraped_df[['url', 'title', 'keywords']])

        # Option to download the updated DataFrame as CSV
        csv = scraped_df.to_csv(index=False)
        st.download_button(
            label="Download Keywords as CSV",
            data=csv,
            file_name="keywords_data.csv",
            mime="text/csv"
        )
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
