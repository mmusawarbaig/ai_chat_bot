
```markdown
# 🧠 RAG-based Research Paper Q&A Interface

This project is a **Retrieval-Augmented Generation (RAG)** system with a **Gradio UI** that lets users query a folder of research papers (PDFs) and get intelligent answers using a Large Language Model (LLM).

---

## 🚀 Features

- 📄 Ingests and indexes research papers from a folder
- 🔍 Retrieves relevant context using semantic search (FAISS)
- 💬 Generates answers using OpenAI's GPT (via LangChain)
- 🧑‍💻 Simple Gradio web interface

---

## 📁 Project Structure

```

project/

│

├── app.py               # Main script

├── papers/              # Folder with PDF files

│   ├── paper1.pdf

│   └── paper2.pdf

└── README.md

## 📦 Setup

### 1. Clone the Repository

```
git clone https://github.com/your-username/custom_assistant.git
cd custom_assistant
```

### 2. Create Conda Environment

```bash
conda create -n rag_system_env python=3.12 -y
conda activate rag_system_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Downlaod and install Ollama:

Instructions are given on the following link:

[Ollama Docs]([https://ollama.com/download](ollama))

## ▶️ Run the App

```bash
conda activate rag_system_env 
python app.py
```

Then open the Gradio interface in your browser.

---

## ❓ How It Works

1. Loads and splits all PDFs from `./papers/`
2. Converts chunks into embeddings using `HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")`
3. Stores them in a FAISS vector store
4. At runtime, retrieves relevant documents based on your question
5. Passes context + query to LLM for final answer

---

Last Update: Jun 18th, 2025
