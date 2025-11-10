import sys
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from sentence_transformers.cross_encoder import CrossEncoder

# 1. Initialize LLM
print("Connecting to Google Gemini API...")
if os.getenv("GOOGLE_API_KEY") is None:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)
    
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro")
print("LLM (Gemini) loaded.")

# 2. Initialize Embedding Model
print("Loading Embedding Model (BGE-small)...")
embedding_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'}
)
print("Embedding Model loaded.")

# 3. Initialize Re-ranker
print("Loading Re-ranker model (TinyBERT)...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')
print("Re-ranker loaded.")

# 4. Load Local Vector Store
print("Loading local vector store...")
db = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)
print("Vector store loaded.")

# 5. Helper Functions
def format_docs(docs):
    """Joins the page_content of all documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def rerank_docs(data):
    """
    Re-ranks retrieved documents using the CrossEncoder
    to improve relevance.
    """
    if not data["context"]:
        print("No documents retrieved, skipping re-ranking.")
        return data
        
    print(f"Re-ranking {len(data['context'])} documents...")
    
    pairs = []
    for doc in data["context"]:
        pairs.append([data["question"], doc.page_content])
        
    scores = cross_encoder.predict(pairs)
    
    doc_scores = list(zip(scores, data["context"]))
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top 3 documents post-reranking
    top_docs = [doc for score, doc in doc_scores[:3]]
    
    data["context"] = top_docs
    print(f"Re-ranking complete. Top 3 docs selected.")
    return data

# 6. Prompt Template
chat_template_string = """You are a professional D&D Dungeon Master.
Answer the question based *only* on the following context.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(chat_template_string)

# 7. Retriever
retriever = db.as_retriever(search_kwargs={"k": 10})

# 8. RAG Chain Definition
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(rerank_docs)
    | {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Main Application Loop
if __name__ == "__main__":
    print("Device set to use cpu")
    print("RAG Chatbot is ready. Type your question and press Enter.")
    
    while True:
        try:
            query = input("Enter your Question: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            print("Answer:")
            for chunk in rag_chain.stream(query):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break