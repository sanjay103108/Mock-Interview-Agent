import uuid
import tiktoken
import trafilatura
import chromadb
from ddgs import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from numpy.linalg import norm

# -----------------------------
# 0. GPU check
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# 1. Embedding model
# -----------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

# -----------------------------
# 2. ChromaDB PersistentClient
# -----------------------------
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="cs_interview_data")

# -----------------------------
# 3. Trusted CS sites
# -----------------------------
TRUSTED_SITES = [
    "geeksforgeeks.org",
    "leetcode.com",
    "interviewbit.com",
    "hellointerview.io",
    "hackerrank.com",
    "educative.io",
    "freecodecamp.org",
    "cs50.harvard.edu",
    "ocw.mit.edu",
    "stackoverflow.com",
    "medium.com"
]

def build_query(topic):
    site_filter = " OR ".join([f"site:{s}" for s in TRUSTED_SITES])
    return f"{topic} interview questions AND answers {site_filter}"

def search_duckduckgo(query, max_results=20):
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            urls.append(r["href"])
    print(f"\nFound {len(urls)} URLs:")
    for u in urls:
        print(u)
    return urls

# -----------------------------
# 4. Scrape & clean
# -----------------------------
def scrape_page(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        return text
    return None

# -----------------------------
# 5. Chunking
# -----------------------------
def chunk_text(text, max_tokens=500, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(enc.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks

# -----------------------------
# 6. Page relevance scoring
# -----------------------------
def page_relevance(page_text, topic_text):
    emb_page = embed_texts([page_text])[0]
    emb_topic = embed_texts([topic_text])[0]
    sim = np.dot(emb_page, emb_topic) / (norm(emb_page) * norm(emb_topic))
    return sim

# -----------------------------
# 7. Embed & store
# -----------------------------
def embed_and_store(chunks, topic, source_url):
    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"topic": topic, "source_url": source_url} for _ in chunks]
    )

# -----------------------------
# 8. Full pipeline
# -----------------------------
def build_topic_knowledge_base(topic, max_pages=7, relevance_threshold=0.4):
    print(f"\n🔸 Searching for topic: {topic}")
    query = build_query(topic)
    urls = search_duckduckgo(query, max_results=max_pages*3)  # fetch extra to filter

    kept_urls = []
    for url in urls:
        print(f"\n🌐 Scraping: {url}")
        text = scrape_page(url)
        if text:
            sim = page_relevance(text, topic)
            print(f"   Relevance score: {sim:.2f}")
            # Keep all pages for debugging
            kept_urls.append((url, text))
        else:
            print(f"❌ Failed to extract content from: {url}")

    # Chunk and store
    for url, text in kept_urls[:max_pages]:
        chunks = chunk_text(text)
        embed_and_store(chunks, topic, url)
        print(f"💾 Stored {len(chunks)} chunks from {url}")

    print(f"\n🧠 Knowledge base built for topic: {topic}")

# -----------------------------
# 9. Example Run
# -----------------------------
if __name__ == "__main__":
    topic = "Operating Systems"
    build_topic_knowledge_base(topic, max_pages=7, relevance_threshold=0.4)
