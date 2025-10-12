import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from collections import defaultdict
import faiss

import llama_index
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from trafilatura import fetch_url, extract

# --- Configuration ---
PDF_PATH = "Split_Chapters"
TXT_PATH = "Combined_Descriptions"
INDEX_ROOT_DIR = "./storage"  # Main directory to store all sub-indexes
SUMMARIES_FILE = os.path.join(INDEX_ROOT_DIR, "summaries.json")
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATION_MODEL = 'gemini-2.5-pro' # For generating summaries

URLS_TO_ADD = [
    "https://cobbtuning.atlassian.net/wiki/spaces/PRS/pages/143753246/Volkswagen+MQB+Tuning+Guide",
    "https://cobbtuning.atlassian.net/wiki/spaces/PRS/pages/725221419/VW+Reference+Torque+Set+Point+Calculations",
]

# --- Helper Functions ---

def generate_summary(text: str, model) -> str:
    """Generates a concise summary for a given block of text."""
    prompt = f"""
    Based on the following technical document content, provide a concise, one-paragraph summary.
    This summary will be used by a router to decide if this document is relevant to a user's question.
    Focus on the core purpose, systems, and key technical concepts described.

    --- CONTENT ---
    {text[:20000]}
    --- END CONTENT ---

    SUMMARY:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"    -> ERROR: Could not generate summary: {e}")
        return "No summary could be generated for this document."

# --- Main Indexing Function ---
def build_and_save_hierarchical_index():
    """
    Builds a hierarchical RAG index. It creates a separate sub-index for each
    document chapter and web page, and generates an AI summary for each.
    """
    # --- 1. Initial Setup ---
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        return

    try:
        genai.configure(api_key=api_key)
        print("Google Generative AI configured successfully.")
    except Exception as e:
        print(f"Error configuring Google AI: {e}")
        return

    # Initialize models
    summary_model = genai.GenerativeModel(GENERATION_MODEL)
    embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
    llama_index.core.Settings.embed_model = embed_model
    llama_index.core.Settings.chunk_size = 1024
    llama_index.core.Settings.chunk_overlap = 100

    # Ensure directories exist
    os.makedirs(INDEX_ROOT_DIR, exist_ok=True)
    for path in [PDF_PATH, TXT_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}. Please add your documents here.")

    # --- 2. Load and Group Local Documents by Chapter ---
    print("\n--- Loading and Grouping Local Documents ---")

    def get_file_metadata(file_path: str) -> dict:
        file_name = os.path.basename(file_path)
        # Use the filename (without extension) as the chapter identifier
        chapter = os.path.splitext(file_name)[0]
        doc_type = "Diagram Description" if "Combined_Descriptions" in file_path else "Technical Guide"
        return {"chapter": chapter, "document_type": doc_type, "source_filename": file_name}

    local_documents = SimpleDirectoryReader(PDF_PATH, file_metadata=get_file_metadata).load_data() + \
                      SimpleDirectoryReader(TXT_PATH, file_metadata=get_file_metadata).load_data()

    grouped_docs = defaultdict(list)
    for doc in local_documents:
        grouped_docs[doc.metadata['chapter']].append(doc)

    print(f"Found {len(local_documents)} documents, grouped into {len(grouped_docs)} chapters.")

    # --- 3. Process and Index Each Chapter ---
    chapter_summaries = {}
    print("\n--- Generating Summaries and Building Sub-Indexes for Chapters ---")
    # Get embedding dimension from the model
    embedding_dim = len(embed_model.get_text_embedding("test"))

    for chapter, docs in grouped_docs.items():
        print(f"  -> Processing Chapter: {chapter}")

        full_text = "\n\n".join([doc.get_content() for doc in docs])
        print("     - Generating AI summary...")
        summary = generate_summary(full_text, summary_model)
        chapter_summaries[chapter] = summary

        print("     - Building vector index...")
        chapter_index_dir = os.path.join(INDEX_ROOT_DIR, f"index_{chapter}")
        # Initialize FAISS index and VectorStore
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
        index.storage_context.persist(persist_dir=chapter_index_dir)
        print(f"     - Index for chapter '{chapter}' saved to {chapter_index_dir}")

    # --- 4. Process and Index Each Web Page ---
    print("\n--- Processing and Indexing Web Pages ---")
    for url in URLS_TO_ADD:
        print(f"  -> Processing URL: {url}")
        try:
            downloaded = fetch_url(url)
            main_content = extract(downloaded, include_comments=False, include_tables=True)
            if not main_content:
                print("     - Could not extract content.")
                continue

            doc = Document(text=main_content, metadata={"source_url": url, "document_type": "Web Page"})

            print("     - Generating AI summary...")
            url_key = f"web_{os.path.basename(url)}"
            summary = generate_summary(main_content, summary_model)
            chapter_summaries[url_key] = summary

            print("     - Building vector index...")
            web_index_dir = os.path.join(INDEX_ROOT_DIR, f"index_{url_key}")
            # Initialize FAISS index and VectorStore
            faiss_index = faiss.IndexFlatL2(embedding_dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents([doc], storage_context=storage_context, show_progress=True)
            index.storage_context.persist(persist_dir=web_index_dir)
            print(f"     - Index for URL '{url_key}' saved to {web_index_dir}")

        except Exception as e:
            print(f"     - An error occurred: {e}")

    # --- 5. Save the Summaries Metadata File ---
    print(f"\n--- Saving Summaries Metadata ---")
    with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
        json.dump(chapter_summaries, f, indent=4)
    print(f"Chapter and web page summaries saved to: {SUMMARIES_FILE}")

    print("\n✅ Hierarchical RAG Indexing Complete.")

if __name__ == "__main__":
    build_and_save_hierarchical_index()
