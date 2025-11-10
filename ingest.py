import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = "data/" 
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# ---------------------

def build_vector_store():
    """
    Loads all .md files, splits them into chunks,
    and serializes them into a FAISS vector store.
    """
    
    # 1. Clear existing index
    if os.path.exists(DB_FAISS_PATH):
        print(f"Deleting old index at {DB_FAISS_PATH}...")
        shutil.rmtree(DB_FAISS_PATH)
        
    print(f"Loading all .md files from {DATA_PATH}...")

    # 2. Load Documents
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        recursive=True,
        loader_cls=UnstructuredMarkdownLoader
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    if not documents:
        print(f"No .md files found in {DATA_PATH}. Aborting.")
        return

    print(f"Loaded {len(documents)} total markdown documents.")

    # 3. Split Documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks.")

    # 4. Load Embedding Model
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embedding model loaded.")

    # 5. Create and Save Vector Store
    print("Creating vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved to '{DB_FAISS_PATH}'")

if __name__ == "__main__":
    build_vector_store()