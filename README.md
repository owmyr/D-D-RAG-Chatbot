# Hybrid RAG Chatbot for D&D 5e

This project is a hybrid RAG (Retrieval-Augmented Generation) pipeline built in Python. This chatbot provides grounded, factual Q&A for the Dungeons & Dragons 5th Edition ruleset by leveraging a local, pre-processed Markdown dataset (E.G., Player's Handbook SRD from [srd.wiki](https://srd.wiki/))

It uses a local "Retrieve and Re-rank" strategy for high-accuracy context retrieval and the Google Gemini API for state-of-the-art, instruction-following generation.

## Features

* **Hybrid Pipeline:** Combines local, CPU-based components (FAISS, BGE) for data processing with a powerful cloud LLM (Gemini 2.5 Pro) for generation.
* **Retrieve and Re-rank:** Implements an advanced RAG strategy, retrieving 10 documents with `FAISS` and re-ranking them with a `CrossEncoder` (TinyBERT) to select the top 3 most relevant chunks.
* **Markdown-First Ingestion:** Utilizes `unstructured` to recursively load a pre-processed library of `.md` files, ensuring clean, semantic data for the vector store.
* **Grounded & Factual:** The prompt is strictly engineered to force the LLM to answer *only* from the provided context, preventing hallucinations.

## Tech Stack

* **Orchestration:** LangChain (LCEL)
* **LLM (Generation):** Google Gemini API (`gemini-2.5-pro`)
* **Embedding:** `BAAI/bge-small-en-v1.5` (Local, via `langchain-huggingface`)
* **Vector Store:** `FAISS` (Local, CPU-based)
* **Re-ranker:** `cross-encoder/ms-marco-TinyBERT-L-2` (Local)
* **Data Pipeline:** `unstructured[md]` + `RecursiveCharacterTextSplitter`

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Data:**
    * Place your pre-processed D&D Markdown folders (e.g., `Spells`, `Classes`, `Races`) into the `data/` directory.

5.  **Build the Vector Store:**
    * Run the ingestion script. This only needs to be done once (or when your data changes).
    ```bash
    python ingest.py
    ```

6.  **Set API Key:**
    * Set your Google AI API key as an environment variable.
    ```powershell
    $env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
    ```

7.  **Run the Chatbot:**
    ```bash
    python main.py
    ```

## Data Source

The pre-processed Markdown files used for the data ingestion pipeline were sourced from [srd.wiki](https://srd.wiki/). This project is possible thanks to their organized and accessible data.
