# GenAI-Powered Document Q&A System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A Streamlit web application demonstrating a Retrieval-Augmented Generation (RAG) pipeline for answering questions based on a custom set of documents. Built using LangChain, Google Gemini, FAISS/ChromaDB, and Sentence Transformers.

## Features

*   **Document Ingestion:** Supports processing of `.txt` and `.pdf` files.
*   **RAG Pipeline:** Implements core RAG steps: loading, chunking, embedding, vector storage, retrieval.
*   **Vector Search:** Uses FAISS for efficient similarity search to find relevant document chunks.
*   **LLM Integration:** Leverages the Google Gemini API (specifically `gemini-2.0-flash-latest`) for generating answers based on retrieved context.
*   **Grounded Answers:** Prompting techniques employed to ensure answers are based strictly on the provided document context.
*   **Interactive UI:** User-friendly chat interface built with Streamlit.
*   **User Authentication:** Basic (file-based) user login and signup. **(Note: Demo purposes only, not production-secure)**.
*   **Chat History:** Saves and loads chat history per user.

## Tech Stack

*   **Language:** Python 3.9+
*   **Core Libraries:**
    *   LangChain (`langchain`, `langchain-community`, `langchain-google-genai`, `langchain-huggingface`)
    *   Streamlit (for UI)
    *   Sentence Transformers (`sentence-transformers`, `HuggingFaceEmbeddings`)
    *   Google Generative AI (`google-generativeai`)
    *   FAISS (`faiss-cpu` or `faiss-gpu`) / ChromaDB (`chromadb`)
    *   PyPDF (`pypdf` for PDF loading)
    *   python-dotenv (for environment variables)
    *   bcrypt (for password hashing)
*   **LLM:** Google Gemini API (`gemini-1.5-flash-latest`)
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Vector Store:** FAISS / ChromaDB

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_GITHUB_REPO_URL>
    cd <your-repo-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    *   Create a `.env` file in the project root directory.
    *   Add your Google AI Studio API key:
      ```env
      GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE"
      ```

5.  **Prepare User Authentication (for `app_v2.py`):**
    *   Create a `users.json` file in the root directory.
    *   Generate a bcrypt hash for a test password (see comments in `app_v2.py` or run `generate_hash.py` if provided).
    *   Add user(s) to `users.json`:
        ```json
        {
            "testuser": {
                "hashed_password": "$2b$12$YOUR_ACTUAL_GENERATED_HASH_HERE"
            }
        }
        ```
    *   Create an empty directory named `chat_histories`.

6.  **Prepare Documents:**
    *   Place your `.txt` and `.pdf` documents inside the `data/` directory (create it if it doesn't exist).

7.  **Ingest Documents (Create Vector Store):**
    *   Run the ingestion script:
      ```bash
      python ingest.py
      ```
    *   This will create the `vectorstore/db_faiss` directory (or similar for ChromaDB). **Ensure this `vectorstore` directory is listed in your `.gitignore` file!**

## Usage

1.  **Ensure prerequisites** (virtual environment activated, `.env` set up, vector store created) are met.
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app_v2.py
    ```
3.  Open the provided URL (usually `http://localhost:8501`) in your browser.
4.  Log in using the credentials you set up in `users.json` (e.g., `testuser` / `password123`). Or create a new account via the UI.
5.  Ask questions related to the content of the documents in the `data/` folder via the chat interface.

## Configuration

*   **LLM Model:** Modify `GEMINI_MODEL_NAME` in `query.py` / `app_v2.py` if needed (ensure the model is available via your API key by running `list_google_models.py`).
*   **Embedding Model:** Configured in `ingest.py` and `query.py`.
*   **Retriever Settings:** The number of chunks (`k`) retrieved can be adjusted in `app_v2.py` where the retriever is initialized.

## Future Improvements

*   Replace file-based authentication with `streamlit-authenticator` or a database.
*   Use a database (e.g., SQLite, PostgreSQL) for chat history persistence.
*   Implement more sophisticated chunking and retrieval strategies.
*   Add document upload functionality directly via the UI.
*   Improve error handling and UI feedback.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (Optional: create a LICENSE file).

## Contact

Abbas Ali Patel
[p.abbasali5@gmail.com]

