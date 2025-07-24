import os
import gradio as gr
import hashlib
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
import socket
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

PAPERS_FOLDER = "./papers" # "./papers_chat_bot_rag"


VECTORSTORE_PATH = "vectorstore.index"
PROCESSED_FILES_PATH = "processed_files.txt"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SERVER_PORT = 7860

print("*"*50)

if not os.path.exists(PAPERS_FOLDER):
    print(f"Creating papers folder at {PAPERS_FOLDER}")
    os.makedirs(PAPERS_FOLDER)


def fix_names(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if it's a PDF file
        if filename.endswith(".pdf"):
            # Build the full file path
            old_path = os.path.join(folder_path, filename)
            
            # Replace spaces, commas, and full stops (except the .pdf extension)
            name, ext = os.path.splitext(filename)
            new_name = name.replace(" ", "_").replace(",", "_").replace(".", "_") + ext
            
            # Skip renaming if old and new names are the same
            if filename == new_name:
                continue

            # Build new file path
            new_path = os.path.join(folder_path, new_name)


            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")


# --- Adding a reload function ---
def reload_vectorstore_and_pipeline():
    global vectorstore, qa_pipeline
    print("🔄 Reloading vectorstore and pipeline...")
    current_files = get_current_files_and_hashes(PAPERS_FOLDER)
    processed_files = [(filename, filehash) for filename, filehash in load_processed_files()]

    # Find new or updated files
    new_files = [
        (filename, filehash)
        for filename, filehash in current_files
        if (filename, filehash) not in processed_files
    ]

    if new_files:
        print(f"⚠️ {len(new_files)} new PDF(s) detected.")
        
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

    qa_pipeline = create_qa_pipeline(vectorstore)
    print("✅ Reload complete.")

class PDFChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            print(f"Detected new PDF. Reloading vectorstore and pipeline...")
            threading.Thread(target=reload_vectorstore_and_pipeline, daemon=True).start()

def start_folder_watcher(path):
    event_handler = PDFChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.daemon = True
    observer.start()

# Start the watcher in a background thread before main logic
# This will allow the watcher to run concurrently with the Gradio interface
# This is to keep an eye on the papers folder for new PDFs
start_folder_watcher(PAPERS_FOLDER)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

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
    
    # Fix file names in the folder
    fix_names(folder)


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
            full_path = os.path.join(folder_path, filename)
            if os.path.getsize(full_path) == 0:
                print(f"⚠️ Skipping empty PDF: {filename}")
                continue
            try:
                loader = PyMuPDFLoader(full_path)
                loaded = loader.load()
                # Filter out docs with invalid page_content
                valid_loaded = [
                    doc for doc in loaded
                    if isinstance(doc.page_content, str) and doc.page_content.strip()
                ]
                docs.extend(valid_loaded)
            except Exception as e:
                print(f"⚠️ Skipping problematic PDF: {filename} ({e})")
                continue
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
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa


# --- Chat function for Gradio ---
def chat_fn(message, history):
    # Filter out any turns where assistant reply is None
    safe_history = [(user, assistant if assistant is not None else "") for user, assistant in history if user is not None]
    output = qa_pipeline.invoke({"question": message, "chat_history": safe_history})
    return output["answer"]


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

    if new_files:
        print(f"⚠️ {len(new_files)} new PDF(s) detected.")
        # print("New files:", new_files)
        print("Adding new PDF(s) to vectorstore...")
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

print("✅ Setup complete.")
print("*"*50)

qa_pipeline = create_qa_pipeline(vectorstore)


def answer_query(query):
    output = qa_pipeline.invoke(query)
    # If output is a dict, return only the 'result' field
    if isinstance(output, dict) and 'result' in output:
        return output['result']
    return output  # fallback


with gr.Blocks(theme="soft") as demo:
    # gr.Image("Logo_of_Hochschule_Kaiserslautern.png", width=120, show_label=False)   
    gr.ChatInterface(
    fn=chat_fn,
    title="AI Chat Bot",
    description="Hi, How can I help you?",
    flagging_mode="never",
    type="messages"  
    )

demo.launch(server_name=get_local_ip(), server_port=SERVER_PORT)
