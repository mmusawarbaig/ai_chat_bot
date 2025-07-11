import os
import gradio as gr
import hashlib
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

PAPERS_FOLDER = "./papers"
VECTORSTORE_PATH = "vectorstore.index"
PROCESSED_FILES_PATH = "processed_files.txt"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def check_ollama_running(url="http://localhost:11434"):
    """Check if the Ollama server is running and reachable."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama server is running.")
            return True
        print(f"⚠️ Ollama server responded with status code {response.status_code}.")
        return False
    except Exception as e:
        print(f"❌ Ollama server is not running or not reachable at {url}. Error: {e}")
        return False


if not check_ollama_running():
    print("\nPlease start Ollama before starting the RAG System.\n")
    exit(1)


def get_pdf_hash(filepath):
    """Return a hash of the PDF file contents."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_current_files_and_hashes(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder, filename)
            files.append((filename, get_pdf_hash(full_path)))
    return files


def load_processed_files():
    if not os.path.exists(PROCESSED_FILES_PATH):
        return []
    with open(PROCESSED_FILES_PATH, "r") as f:
        return [line.strip().split(",") for line in f.readlines()]


def save_processed_files(files_and_hashes):
    with open(PROCESSED_FILES_PATH, "w") as f:
        for filename, filehash in files_and_hashes:
            f.write(f"{filename},{filehash}\n")


def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def create_vector_store(splits):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)


def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def create_qa_pipeline(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="llama3.1:8b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa


# --- Main logic: load or build vectorstore ---
print("Checking for new or updated documents...")

current_files = get_current_files_and_hashes(PAPERS_FOLDER)
processed_files = [(filename, filehash) for filename, filehash in load_processed_files()]

if os.path.exists(VECTORSTORE_PATH):
    vectorstore = load_vectorstore(VECTORSTORE_PATH)
    # Find new or updated files
    new_files = [
        (filename, filehash)
        for filename, filehash in current_files
        if (filename, filehash) not in processed_files
    ]

    print(f"Found {len(new_files)} new or updated documents.")
    print("New files:", new_files)

    if new_files:
        print("Adding new or updated documents to vectorstore...")
        documents = load_documents_from_folder(PAPERS_FOLDER)
        # Only process new/updated docs
        docs_to_add = [
            doc for doc in documents
            if any(doc.metadata.get("source", "").endswith(f[0]) for f in new_files)
        ]

        splits = split_documents(docs_to_add)
        vectorstore.add_documents(splits)
        save_vectorstore(vectorstore, VECTORSTORE_PATH)
        save_processed_files(current_files)
    else:
        print("No new documents. Vectorstore is up to date.")
else:
    print("No vectorstore found. Building from scratch...")
    documents = load_documents_from_folder(PAPERS_FOLDER)
    splits = split_documents(documents)
    vectorstore = create_vector_store(splits)
    save_vectorstore(vectorstore, VECTORSTORE_PATH)
    save_processed_files(current_files)

print("Setup complete.")

qa_pipeline = create_qa_pipeline(vectorstore)


def answer_query(query):
    output = qa_pipeline.invoke(query)
    # If output is a dict, return only the 'result' field
    if isinstance(output, dict) and 'result' in output:
        return output['result']
    return output  # fallback


gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs="text",
    title="Research Paper Q&A with RAG",
    description="Enter a question. The system will retrieve relevant info from your research paper folder and answer your query using an LLM.",
    flagging_mode="never"
).launch(server_name="0.0.0.0", server_port=7860)