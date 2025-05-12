# query.py
import os
import traceback # <--- ADDED THIS IMPORT
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# GEMINI_MODEL_NAME = "gemini-pro" # Old, not available to you
GEMINI_MODEL_NAME = "gemini-2.0-flash" # <--- NEW - Use this one
# Or you could try:
# GEMINI_MODEL_NAME = "gemini-1.5-pro-latest" # If Flash doesn't work or you want more power

load_dotenv()

# --- Helper Functions ---
def load_vector_store(path, embeddings_model):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store not found at {path}. Please run ingest.py first.")
    print(f"Loading vector store from {path}...")
    db = FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
    return db

def create_llm():
    print(f"Initializing LLM with Google Gemini: {GEMINI_MODEL_NAME}")
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=0.3,
        # convert_system_message_to_human=True # Usually not needed for gemini-pro with good prompting
    )
    print("Google Gemini LLM initialized.")
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG Chain Definition ---
def setup_rag_chain(retriever, llm):
    # Inside the setup_rag_chain function in query.py

    system_message_content = """You are an expert question-answering assistant.
    Strictly use ONLY the provided context below to answer the user's question.
    If the context does not contain the answer, state 'The provided context does not contain the answer to this question.'
    Do not add any information not found in the context. Your answer should be concise.

    Context:
    {context}"""  # Notice {context} is now part of this system message

    # The human message will now just be the question
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message_content), # System message now includes the context placeholder
        ("human", "{question}")             # Human message is just the question placeholder
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain with Gemini created.")
    return rag_chain

# --- Main Execution ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not found. Please set it in your .env file.")
        exit()

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME,
                                           model_kwargs={'device': 'cpu'})
    except Exception as e:
        print(f"Error initializing HuggingFaceEmbeddings: {e}")
        print("Make sure 'langchain-huggingface' is installed and working, or adjust the import.")
        exit()

    try:
        db = load_vector_store(DB_FAISS_PATH, embeddings)
    except FileNotFoundError as e:
        print(e)
        exit()

    retriever = db.as_retriever(search_kwargs={"k": 3})
    print(f"Retriever created. Will retrieve k={retriever.search_kwargs['k']} documents.")

    llm = create_llm()
    qa_chain = setup_rag_chain(retriever, llm)

    print("\n--- Document Q&A System (with Google Gemini) ---")
    print("Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() == 'exit':
            break
        if not user_question.strip():
            continue

        print("Processing your question with Gemini...")
        
        try:

            answer = qa_chain.invoke(user_question)
            
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"\nError during Gemini LLM call: {e}")
            traceback.print_exc() # Now this will work

    print("Exiting Q&A system.")