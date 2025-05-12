# app_v2.py
import streamlit as st
import os
import json
import bcrypt # For password hashing
from datetime import datetime
from dotenv import load_dotenv

# --- Assuming your RAG logic is importable ---
from query import (
    load_vector_store, create_llm, setup_rag_chain,
    DB_FAISS_PATH, EMBEDDINGS_MODEL_NAME
)
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- Constants ---
USERS_FILE = "users.json"
CHAT_HISTORY_DIR = "chat_histories"

# --- Authentication Functions (Simplified) ---
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users_data):
    with open(USERS_FILE, "w") as f:
        json.dump(users_data, f, indent=4)

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def signup_user(username, password): # Basic signup
    users = load_users()
    if username in users:
        return False, "Username already exists."
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    users[username] = {"hashed_password": hashed_password}
    save_users(users)
    # Create chat history directory if it doesn't exist
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    return True, "Signup successful! Please log in."

# --- Chat History Functions ---
def get_user_chat_file(username):
    return os.path.join(CHAT_HISTORY_DIR, f"{username}_chat.json")

def load_chat_history(username):
    chat_file = get_user_chat_file(username)
    if not os.path.exists(chat_file):
        return []
    with open(chat_file, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return [] # Return empty if file is corrupted

def save_chat_history(username, history):
    chat_file = get_user_chat_file(username)
    # Create chat history directory if it doesn't exist
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    with open(chat_file, "w") as f:
        json.dump(history, f, indent=4)


# --- RAG Core Logic (Cached as before) ---
@st.cache_resource
def get_vector_store():
    # ... (same as your previous app.py)
    if not os.path.exists(DB_FAISS_PATH): return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs={'device': 'cpu'})
    return load_vector_store(DB_FAISS_PATH, embeddings)

@st.cache_resource
def get_qa_chain(_db):
    # ... (same as your previous app.py, ensuring API keys are checked)
    if not _db: return None
    retriever = _db.as_retriever(search_kwargs={"k": 3})
    llm = create_llm()
    return setup_rag_chain(retriever, llm)

# --- Main App Logic ---
st.set_page_config(page_title="Pro RAG Q&A", layout="wide")
st.title("Pro RAG Q&A System  Profesional")

# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state: # This will hold current user's messages
    st.session_state.messages = []
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False


# --- Authentication UI ---
if not st.session_state.authenticated:
    st.sidebar.header("User Account")
    
    if st.session_state.show_signup:
        st.sidebar.subheader("Sign Up")
        new_username = st.sidebar.text_input("Choose a Username", key="signup_user")
        new_password = st.sidebar.text_input("Choose a Password", type="password", key="signup_pass")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.sidebar.button("Create Account"):
            if new_password == confirm_password:
                if new_username and new_password:
                    success, message = signup_user(new_username, new_password)
                    if success:
                        st.sidebar.success(message)
                        st.session_state.show_signup = False # Go back to login
                        st.rerun()
                    else:
                        st.sidebar.error(message)
                else:
                    st.sidebar.error("Username and password cannot be empty.")
            else:
                st.sidebar.error("Passwords do not match.")
        if st.sidebar.button("Back to Login"):
            st.session_state.show_signup = False
            st.rerun()
    else:
        st.sidebar.subheader("Login")
        username_input = st.sidebar.text_input("Username", key="login_user")
        password_input = st.sidebar.text_input("Password", type="password", key="login_pass")
        if st.sidebar.button("Login"):
            users = load_users()
            if username_input in users and verify_password(password_input, users[username_input]["hashed_password"]):
                st.session_state.authenticated = True
                st.session_state.username = username_input
                st.session_state.messages = load_chat_history(username_input) # Load their history
                st.rerun() # Rerun to hide login and show app
            else:
                st.sidebar.error("Invalid username or password")
        
        if st.sidebar.button("Create New Account"):
            st.session_state.show_signup = True
            st.rerun()
else: # --- Main Application UI (If Authenticated) ---
    st.sidebar.header(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.messages = []
        # Clear cached RAG components if desired on logout, or keep them for next login
        # st.cache_resource.clear() # This would clear vector store and QA chain cache
        st.rerun()

    # --- Load RAG components only if authenticated and not already loaded ---
    if "db" not in st.session_state or st.session_state.db is None:
        with st.spinner("Loading knowledge base..."):
            st.session_state.db = get_vector_store()
            if st.session_state.db is None:
                st.error("Failed to load vector store. Please ensure `ingest.py` has been run.")
                st.stop()

    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
         if st.session_state.db: # Ensure db is loaded before trying to get qa_chain
             with st.spinner("Initializing AI Assistant..."):
                 st.session_state.qa_chain = get_qa_chain(st.session_state.db)
                 if st.session_state.qa_chain is None:
                     st.error("Failed to initialize AI assistant. Check API keys and configurations.")
                     st.stop()
         else:
             st.error("Knowledge base not loaded, cannot initialize AI assistant.")
             st.stop()
    
    qa_chain = st.session_state.qa_chain

    # --- Chat Interface ---
    st.header("Chat with Your Documents")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message: # Optional: display timestamp
                st.caption(f"Sent: {message['timestamp']}")


    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"Sent: {current_time}")


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    if qa_chain: # Ensure chain is loaded
                        response = qa_chain.invoke(prompt)
                    else:
                        response = "AI Assistant is not ready. Please check setup."
                except Exception as e:
                    st.error(f"Error processing your question: {e}")
                    response = "Sorry, I encountered an error."
                
                message_placeholder.markdown(response)
        
        assistant_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": assistant_time})
        save_chat_history(st.session_state.username, st.session_state.messages) # Save after each exchange