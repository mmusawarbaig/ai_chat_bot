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
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import uuid

PAPERS_FOLDER = "./papers"
VECTORSTORE_PATH = "vectorstore.index"
PROCESSED_FILES_PATH = "processed_files.txt"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
SERVER_PORT = 7860
DATABASE_URL = "sqlite:///rag_conversations.db"

print("*"*50)

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    title = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# Create tables
Base.metadata.create_all(bind=engine)

class ConversationManager:
    def __init__(self):
        self.current_session_id = None
        self.pending_conversation = False  # Track if conversation needs to be created

    def create_new_conversation(self, title=None):
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        if not title:
            title = f"New Conversation"
        
        db = SessionLocal()
        try:
            conversation = Conversation(session_id=session_id, title=title)
            db.add(conversation)
            db.commit()
            self.current_session_id = session_id
            self.pending_conversation = False
            return session_id, title
        finally:
            db.close()

    def start_new_conversation(self):
        """Start a new conversation (but don't save to DB until first message)"""
        self.current_session_id = None
        self.pending_conversation = True

    def save_message(self, role, content):
        """Save a message to the current conversation"""
        if not self.current_session_id and self.pending_conversation:
            # Create conversation only when first message is sent
            self.create_new_conversation()
        elif not self.current_session_id:
            self.create_new_conversation()
        
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == self.current_session_id
            ).first()
            
            if conversation:
                message = Message(
                    conversation_id=conversation.id,
                    role=role,
                    content=content
                )
                db.add(message)
                conversation.updated_at = datetime.utcnow()
                db.commit()
                
                # Generate title after first user message
                if role == "user":
                    message_count = db.query(Message).filter(
                        Message.conversation_id == conversation.id,
                        Message.role == "user"
                    ).count()
                    
                    if message_count == 1:  # First user message
                        threading.Thread(
                            target=self.generate_conversation_title, 
                            args=(conversation.id, content),
                            daemon=True
                        ).start()
        finally:
            db.close()

    def generate_conversation_title(self, conversation_id, first_message):
        """Generate a meaningful title for the conversation using LLM"""
        try:
            # Use a simple prompt to generate title
            title_prompt = f"Generate a short, descriptive title (max 50 characters) for a conversation that starts with this message: '{first_message[:200]}...'. Only return the title, nothing else."
            
            # Use the same LLM as the main pipeline
            llm = OllamaLLM(model="llama3.1:8b")
            generated_title = llm.invoke(title_prompt)
            
            # Clean up the title
            title = generated_title.strip().strip('"').strip("'")
            if len(title) > 50:
                title = title[:47] + "..."
            
            # Update the conversation title in database
            db = SessionLocal()
            try:
                conversation = db.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if conversation:
                    conversation.title = title
                    conversation.updated_at = datetime.utcnow()
                    db.commit()
                    print(f"✅ Generated title: {title}")
            finally:
                db.close()
                
        except Exception as e:
            print(f"⚠️ Failed to generate title: {e}")
            # Keep the default title if generation fails

    def load_conversation(self, session_id):
        """Load a conversation by session_id"""
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                messages = db.query(Message).filter(
                    Message.conversation_id == conversation.id
                ).order_by(Message.timestamp).all()
                
                history = []
                for msg in messages:
                    history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                
                self.current_session_id = session_id
                self.pending_conversation = False
                return history, conversation.title
            return [], None
        finally:
            db.close()

    def get_conversation_list(self):
        """Get list of all conversations with messages"""
        db = SessionLocal()
        try:
            # Only get conversations that have messages
            conversations = db.query(Conversation).join(Message).distinct().order_by(
                Conversation.updated_at.desc()
            ).all()
            
            return [(conv.session_id, conv.title, conv.updated_at) for conv in conversations]
        finally:
            db.close()

    def delete_conversation(self, session_id):
        """Delete a conversation"""
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                # Delete messages first
                db.query(Message).filter(
                    Message.conversation_id == conversation.id
                ).delete()
                # Delete conversation
                db.delete(conversation)
                db.commit()
                return True
            return False
        finally:
            db.close()

    def update_conversation_title(self, session_id, new_title):
        """Update conversation title"""
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                conversation.title = new_title
                conversation.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        finally:
            db.close()

# Initialize conversation manager
conv_manager = ConversationManager()

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

# --- Chat function for Gradio with persistent storage ---
def chat_fn(message, history):
    # If we're in a new conversation state, ignore any existing history
    if conv_manager.pending_conversation or not conv_manager.current_session_id:
        safe_history = []  # Start with empty history for new conversations
    else:
        # Convert Gradio history format to LangChain format only for existing conversations
        safe_history = []
        user_msg = None
        for msg in history:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"] if msg["content"] is not None else ""
                if user_msg is not None:
                    safe_history.append((user_msg, assistant_msg))
                    user_msg = None
    
    # Save user message (this will create conversation if needed)
    conv_manager.save_message("user", message)
    
    # Get response from QA pipeline with appropriate history
    output = qa_pipeline.invoke({"question": message, "chat_history": safe_history})
    response = output["answer"]
    
    # Save assistant response
    conv_manager.save_message("assistant", response)
    
    return response

# Conversation management functions for Gradio
def load_conversation_list():
    """Load conversation list for dropdown"""
    conversations = conv_manager.get_conversation_list()
    choices = [("Select a conversation...", None)]  # Add default option
    for session_id, title, updated_at in conversations:
        display_name = f"{title}"
        choices.append((display_name, session_id))
    return gr.Dropdown(choices=choices, value=None)

def on_conversation_select(selected_conversation):
    """Load selected conversation automatically when dropdown value changes"""
    if not selected_conversation:
        return [], "Ready to chat"
    
    history, title = conv_manager.load_conversation(selected_conversation)
    if history:
        return history, f"Loaded: {title}"
    return [], "Conversation not found"

def create_new_conversation():
    """Start new conversation"""
    conv_manager.start_new_conversation()
    # Return empty history and reset dropdown to default
    updated_dropdown = load_conversation_list()
    return [], "New conversation started", gr.Dropdown(choices=updated_dropdown.choices, value=None)

def delete_selected_conversation(selected_conversation):
    """Delete selected conversation"""
    if not selected_conversation:
        return "No conversation selected for deletion", load_conversation_list()
    
    if conv_manager.delete_conversation(selected_conversation):
        return "Conversation deleted successfully", load_conversation_list()
    return "Failed to delete conversation", load_conversation_list()

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

# Gradio interface with conversation management
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# AI Chat Bot with Conversation History")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Conversation Management")
            
            conversation_dropdown = gr.Dropdown(
                label="Select Conversation",
                choices=[("Select a conversation...", None)],
                value=None
            )
            
            with gr.Row():
                new_btn = gr.Button("New Chat", variant="primary", size="sm")
                delete_btn = gr.Button("Delete", variant="stop", size="sm")
            
            status_text = gr.Textbox(
                label="Status",
                value="Ready to chat",
                interactive=False
            )
        
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat_fn,
                title="",
                description="Hi, How can I help you?",
                flagging_mode="never",
                type="messages"
            )
    
    # Event handlers
    demo.load(
        lambda: load_conversation_list(), 
        outputs=[conversation_dropdown]
    )
    
    # Auto-load conversation when selected
    conversation_dropdown.change(
        on_conversation_select,
        inputs=[conversation_dropdown],
        outputs=[chatbot.chatbot, status_text]
    )
    
    def new_chat_click():
        conv_manager.start_new_conversation()
        updated_dropdown = load_conversation_list()
        return [], "New conversation started", gr.Dropdown(choices=updated_dropdown.choices, value=None)
    
    new_btn.click(
        new_chat_click,
        outputs=[chatbot.chatbot, status_text, conversation_dropdown]
    )
    
    delete_btn.click(
        delete_selected_conversation,
        inputs=[conversation_dropdown],
        outputs=[status_text, conversation_dropdown]
    )

demo.launch(server_name=get_local_ip(), server_port=SERVER_PORT)