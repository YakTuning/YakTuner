import os
import json
import tempfile
import streamlit as st
import pandas as pd
import re
import sys

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import Core Libraries ---
import google.generativeai as genai
from google.generativeai.types import content_types
from google.api_core import exceptions as google_exceptions

# --- LlamaIndex Imports ---
import llama_index
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# --- Custom Module Imports ---
from tuning_loader import read_map_by_description
from xdf_parser import list_available_maps
from log_filter import filter_log_data

# --- Constants ---
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']
INDEX_ROOT_DIR = "./storage"
SUMMARIES_FILE = os.path.join(INDEX_ROOT_DIR, "summaries.json")
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATION_MODEL = 'gemini-2.5-pro'

# --- Page Configuration ---
st.set_page_config(page_title="Diagnostic Assistant", layout="wide")
st.title("💡 Diagnostic Assistant")
st.markdown("Ask a technical question about your tune, logs, or general ECU concepts. The assistant can use your uploaded `.bin` tune file for context.")

# --- RAG Data Loading ---
def load_hierarchical_index_data():
    """
    Loads all sub-indexes and summaries from the storage directory.
    """
    if not os.path.exists(INDEX_ROOT_DIR) or not os.path.exists(SUMMARIES_FILE):
        st.warning(f"Knowledge base not found. Please run `build_rag_index.py` first.")
        return None, None

    try:
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            summaries = json.load(f)

        sub_indexes = {}
        for chapter_id in summaries.keys():
            index_dir = os.path.join(INDEX_ROOT_DIR, f"index_{chapter_id}")
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            sub_indexes[chapter_id] = load_index_from_storage(storage_context)

        st.success("Knowledge base loaded successfully.")
        return sub_indexes, summaries
    except Exception as e:
        st.error(f"An unexpected error occurred during the final load: {e}")
        return None, None

# --- Tool Functions (Unchanged) ---
def get_tune_data(map_description: str) -> str:
    if 'bin_content' not in st.session_state: return "Error: User has not uploaded a .bin file."
    xdf_content = _get_xdf_content()
    if not xdf_content: return "Error: Could not find the required XDF file."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
        tmp_xdf.write(xdf_content)
        tmp_xdf_path = tmp_xdf.name
    try:
        map_data = read_map_by_description(tmp_xdf_path, map_description, st.session_state.bin_content)
        if not map_data: return f"Error: Map with description '{map_description}' not found."
        return "\n".join([f"Data for map '{name}':\n{pd.DataFrame(data).to_string()}" for name, data in map_data.items()])
    finally:
        os.remove(tmp_xdf_path)

def list_available_maps_tool() -> dict:
    xdf_content = _get_xdf_content()
    if not xdf_content: return "Error: Could not find the required XDF file."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
        tmp_xdf.write(xdf_content)
        tmp_xdf_path = tmp_xdf.name
    try:
        return list_available_maps(tmp_xdf_path)
    finally:
        os.remove(tmp_xdf_path)

def _get_xdf_content():
    firmware_choice = st.session_state.get('firmware')
    if firmware_choice in PREDEFINED_FIRMWARES:
        path = os.path.join(XDF_SUBFOLDER, f"{firmware_choice}.xdf")
        if os.path.exists(path):
            with open(path, "rb") as f: return f.read()
    elif firmware_choice == 'Other' and st.session_state.get('xdf_content'):
        return st.session_state.xdf_content
    return None

def render_thinking_process(history):
    if not history:
        st.info("The assistant's thinking process will be shown here.")
        return
    # This function remains the same

# --- UI Layout ---
with st.sidebar:
    st.divider()
    st.subheader("💡 Assistant API Key")
    api_key_input = st.text_input(
        "Enter Google API Key", type="password", key="api_key_input",
        value=st.session_state.get('google_api_key', ''),
        help="Your key powers the Diagnostic Assistant and is not stored."
    )
    if api_key_input: st.session_state.google_api_key = api_key_input
    st.divider()
    st.page_link("yaktuner_streamlit.py", label="Main YAKtuner Analysis", icon="📈")
    st.divider()
    st.header("⚙️ Assistant Settings")
    firmware = st.radio(
        "Firmware Version", options=ALL_FIRMWARES, horizontal=True,
        help="Select your ECU's firmware to load correct map definitions.", key="firmware"
    )

if 'diag_chat_history' not in st.session_state: st.session_state.diag_chat_history = []
if 'diag_chat' not in st.session_state: st.session_state.diag_chat = None

st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader("Upload .bin tune file", type=['bin', 'all'], key="uploaded_bin_file_assistant")
if uploaded_bin_file: st.session_state.bin_content = uploaded_bin_file.getvalue()

if firmware == 'Other':
    st.info("Please provide an XDF file for 'Other' firmware.")
    uploaded_xdf_file = st.file_uploader("Upload .xdf definition file", type=['xdf'], key="uploaded_xdf_file_assistant")
    if uploaded_xdf_file: st.session_state.xdf_content = uploaded_xdf_file.getvalue()

st.subheader("2. Ask Your Question")
user_query = st.text_input("Enter your diagnostic question:", placeholder="e.g., What does 'combmodes_MAF' control?", key="rag_query")
uploaded_diag_log = st.file_uploader("Upload a CSV data log (Optional)", type="csv", key="diag_log")

# --- Main Logic Block ---
if st.button("Get Diagnostic Answer", key="get_diag_answer", use_container_width=True):
    # --- Input Validation ---
    api_key = st.session_state.get('google_api_key')
    if not api_key: st.error("Please enter your Google API Key in the sidebar.")
    elif not uploaded_bin_file: st.error("Please upload your .bin tune file.")
    elif not user_query: st.warning("Please enter a question.")
    else:
        with st.status("Analyzing...", expanded=True) as status:
            try:
                # --- Step 1: Configure API-dependent components ---
                status.update(label="Initializing models and loading knowledge base...")
                genai.configure(api_key=api_key)
                embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004", api_key=api_key)
                llama_index.core.Settings.embed_model = embed_model
                llama_index.core.Settings.llm = GoogleGenAI(model_name=f"models/{GENERATION_MODEL}", api_key=api_key)

                # --- Step 2: Load the index from session state or disk ---
                if 'sub_indexes' not in st.session_state or 'summaries' not in st.session_state:
                    with st.spinner("Loading knowledge base for the first time..."):
                        st.session_state.sub_indexes, st.session_state.summaries = load_hierarchical_index_data()

                if not st.session_state.sub_indexes or not st.session_state.summaries:
                    st.error("Knowledge base could not be loaded. Please ensure the index has been built and is not corrupted.")
                    st.stop()

                sub_indexes = st.session_state.sub_indexes
                summaries = st.session_state.summaries

                # --- Step 3: Build the RouterQueryEngine ---
                query_engine_tools = []
                for chapter_id, sub_index in sub_indexes.items():
                    query_engine = sub_index.as_query_engine(similarity_top_k=3)
                    tool = QueryEngineTool.from_defaults(
                        query_engine=query_engine,
                        name=chapter_id,
                        description=summaries[chapter_id]
                    )
                    query_engine_tools.append(tool)

                query_engine = RouterQueryEngine(
                    selector=LLMSingleSelector.from_defaults(),
                    query_engine_tools=query_engine_tools
                )

                model = genai.GenerativeModel(GENERATION_MODEL, tools=[get_tune_data, list_available_maps_tool])

                # --- Step 4: Process Logs and Retrieve Context ---
                log_data_str = ""
                if uploaded_diag_log:
                    log_df = pd.read_csv(uploaded_diag_log, encoding='latin1')
                    filtered_log_df = filter_log_data(log_df, user_query)
                    log_data_str = f'--- **USER-UPLOADED LOG FILE DATA:**\n{filtered_log_df.to_string()}\n---'

                status.update(label="Routing query and retrieving context from knowledge base...")
                response_from_rag = query_engine.query(user_query)
                context_str = "\n\n".join([f"Source: {node.metadata.get('source_filename', 'N/A')} | Chapter: {node.metadata.get('chapter', 'N/A')}\nContent: {node.get_content()}" for node in response_from_rag.source_nodes])

                # --- Step 5: Run Chat ---
                if st.session_state.diag_chat is None:
                    st.session_state.diag_chat = model.start_chat(enable_automatic_function_calling=True)
                    st.session_state.diag_chat_history = []

                chat = st.session_state.diag_chat
                initial_prompt = f'''
                You are an expert automotive systems engineer and a master diagnostician for ECUs.
                Your primary goal is to provide a comprehensive and accurate answer to the user's question by acting as a detective.
                **Your Process:**
                1.  **Analyze the user's question and the provided documentation and log file data (CONTEXT) to form an initial hypothesis.**
                2.  **If you need to look up a map from the tune file, you MUST use a two-step process:**
                    a. **First, call the `list_available_maps_tool()`** to get a dictionary of all available maps.
                    b. **Second, use this dictionary to find the exact `map_description` string** for the map you need to investigate.
                    c. **Finally, call the `get_tune_data()` tool** with the precise `map_description`.
                3.  **Synthesize all the evidence.** Your final answer MUST be a synthesis of information from the documentation, the log data, and the tune data.
                4.  **Formulate your final answer ONLY when you are confident you have a complete picture.**
                **Available Tools:**
                - `list_available_maps_tool()`: Returns a dictionary mapping map titles to their full descriptions.
                - `get_tune_data(map_description: str)`: Use this to look up a specific map.
                ---
                **CONTEXT FROM DOCUMENTATION:**
                {context_str}
                {log_data_str}
                ---
                **USER'S QUESTION:**
                {user_query}
                '''

                status.update(label="Sending request to the generative model...")

                # --- New: Manually construct and save the prompt before the API call ---
                # This ensures the user's prompt is visible even if send_message() fails.
                user_prompt_content = content_types.to_content(initial_prompt)
                st.session_state.diag_chat_history.append(user_prompt_content)

                response = chat.send_message(initial_prompt)
                st.session_state.diag_chat_history = chat.history

                st.markdown("#### Assistant's Answer")
                full_response_text = "".join(part.text for part in response.parts)
                st.info(full_response_text)
                status.update(label="Response received.", state="complete", expanded=False)

                with st.expander("Show Retrieved Context from Documentation"):
                    for node in response_from_rag.source_nodes:
                        st.markdown(f"**Source:** {node.metadata.get('source_filename', 'N/A')} | **Chapter:** {node.metadata.get('chapter', 'N/A')} | **Relevance:** {node.score:.2f}")
                        st.text_area("Content", node.get_content(), height=150, disabled=True, key=f"context_{node.node_id}")

            except Exception as e:
                # The manually added prompt will be preserved in st.session_state.diag_chat_history
                st.error(f"An error occurred with the generative model: {e}")
                st.session_state.diag_chat = None # Reset chat on failure

st.subheader("3. Assistant's Thinking Process")
with st.expander("Show/Hide the detailed reasoning process", expanded=True):
    render_thinking_process(st.session_state.diag_chat_history)
