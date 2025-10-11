import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from collections import defaultdict

import llama_index
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.google import GooglePairedEmbeddings
from llama_index.vector_stores.faiss import FaissVectorStore
from trafilatura import fetch_url, extract

# --- Configuration ---
PDF_PATH = "Split_Chapters"
TXT_PATH = "Combined_Descriptions"
INDEX_ROOT_DIR = "./storage"  # Main directory to store all sub-indexes
SUMMARIES_FILE = os.path.join(INDEX_ROOT_DIR, "summaries.json")
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATION_MODEL = 'gemini-1.5-pro-latest' # For generating summaries

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
    embed_model = GooglePairedEmbeddings(
        model_name=EMBEDDING_MODEL, api_key=api_key,
        query_task_type="retrieval_query", doc_task_type="retrieval_document"
    )
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
        chapter_match = re.search(r"Chapter_([^_]+)", file_name)
        chapter = chapter_match.group(1) if chapter_match else "Unknown"
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
    for chapter, docs in grouped_docs.items():
        print(f"  -> Processing Chapter: {chapter}")

        # Combine text for summary
        full_text = "\n\n".join([doc.get_content() for doc in docs])

        # Generate summary
        print("     - Generating AI summary...")
        summary = generate_summary(full_text, summary_model)
        chapter_summaries[chapter] = summary

        # Create and persist a specific index for this chapter
        print("     - Building vector index...")
        chapter_index_dir = os.path.join(INDEX_ROOT_DIR, f"index_{chapter}")
        vector_store = FaissVectorStore.from_defaults()
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

            # Generate summary for the web page
            print("     - Generating AI summary...")
            url_key = f"web_{os.path.basename(url)}" # Create a simple key from URL
            summary = generate_summary(main_content, summary_model)
            chapter_summaries[url_key] = summary

            # Create and persist a specific index for this web page
            print("     - Building vector index...")
            web_index_dir = os.path.join(INDEX_ROOT_DIR, f"index_{url_key}")
            vector_store = FaissVectorStore.from_defaults()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents([doc], storage_context=storage_context, show_progress=True)
            index.storage_context.persist(persist_dir=web_index_dir)
            print(f"     - Index for URL '{url_key}' saved to {web_index_dir}")

        except Exception as e:
            print(f"     - An error occurred: {e}")

    # --- 5. Save the Summaries Metadata File ---
    print(f"\n--- Saving Summaries Metadata ---")
    with open(SUMMARIES_FILE, "w") as f:
        json.dump(chapter_summaries, f, indent=4)
    print(f"Chapter and web page summaries saved to: {SUMMARIES_FILE}")

    print("\n✅ Hierarchical RAG Indexing Complete.")

if __name__ == "__main__":
    build_and_save_hierarchical_index()
