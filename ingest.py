# ingest.py (showing relevant parts for loading and chunking)
import os
from langchain_community.document_loaders import TextLoader # For loading .txt files
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data/" # This points to the directory you just created

# Function to load documents from the data directory
def load_documents(directory_path):
    documents = []
    print(f"Looking for documents in: {os.path.abspath(directory_path)}")
    # Corrected section in ingest.py -> load_documents
# ...
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename) # Define file_path here
        if filename.endswith(".txt"):
            print(f"Loading TXT document: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        elif filename.endswith(".pdf"): # Add this block for PDFs
            print(f"Loading PDF document: {file_path}")
            loader = PyPDFLoader(file_path) # PyPDFLoader takes the path
            documents.extend(loader.load()) # .load() returns a list of Document objects (often one per page)
# ...
    print(f"Loaded {len(documents)} documents successfully.")
    # Each "document" here is an object that LangChain uses, containing page_content and metadata.
    # If you had 3 .txt files, len(documents) will be 3.
    return documents

# Function to split documents into chunks
def split_text_into_chunks(documents):
    # RecursiveCharacterTextSplitter is good for general text.
    # It tries to split on "\n\n", then "\n", then " ", then by character.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Max size of a chunk (in characters)
        chunk_overlap=50,     # How many characters overlap between adjacent chunks
        length_function=len,  # How to measure length (standard len is fine for characters)
        is_separator_regex=False,
    )
    # .split_documents() takes a list of Document objects and returns a list of smaller Document objects (chunks)
    chunks = text_splitter.split_documents(documents)
    print(f"Split the {len(documents)} documents into {len(chunks)} chunks.")
    # For example, if animal_facts.txt is 800 characters long, it might become 2 chunks.
    # Chunk 1: characters 0-500
    # Chunk 2: characters 450-800 (due to 50 char overlap)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk.page_content)} characters") # See the size of each chunk
    return chunks

# The rest of ingest.py (embedding and storing) will use these chunks.

# ... (embedding and storing functions from previous response) ...

if __name__ == "__main__":
    # 1. Load documents
    docs = load_documents(DATA_PATH)
    if not docs:
        print("No documents found. Please add some .txt files to the 'data' directory.")
    else:
        # 2. Split documents into chunks
        text_chunks = split_text_into_chunks(docs)

        # ... (rest of the main block for embedding and storing) ...
        # For now, let's just confirm loading and chunking works.
        # You can comment out the embedding part temporarily if you just want to test this.
        # For example:
        # print("Loading and chunking complete. Embedding part is next.")

        # For the full run, uncomment the embedding part from the previous script:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        DB_FAISS_PATH = "vectorstore/db_faiss"

        def store_embeddings_part(chunks_to_embed, embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"):
            print(f"Using embedding model: {embeddings_model_name}")
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': 'cpu'})
            print("Creating vector store...")
            db = FAISS.from_documents(chunks_to_embed, embeddings)
            if not os.path.exists("vectorstore"):
                os.makedirs("vectorstore")
            db.save_local(DB_FAISS_PATH)
            print(f"Vector store created and saved to {DB_FAISS_PATH}")
            return db

        vector_store = store_embeddings_part(text_chunks)
        print("Ingestion complete!")