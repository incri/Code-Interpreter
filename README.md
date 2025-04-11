# 🚀 PromptPilot

> **Your AI-Powered Workspace Assistant for Smarter Prompts, Code, and Conversations.**

PromptPilot is a Streamlit-based app that transforms the way you interact with documents, code, and AI. Upload PDFs, chat intelligently with content, classify tasks automatically, and execute prompt-driven actions with seamless AI integration using Google Gemini and FAISS.

---

## ✨ Features

- 📂 **Multi-Workspace Environment**: Create and manage isolated workspaces for different projects or documents.
- 📄 **PDF Upload & Parsing**: Upload one or more PDFs, automatically indexed with semantic search using FAISS.
- 💬 **Smart Chat Assistant**: Ask questions, get answers, and navigate content intelligently with the help of Google Gemini.
- 🧠 **Prompt Classification**: Classifies user queries to determine task type (e.g., QA, code generation, summarization).
- 🔄 **Auto Response Handling**: Executes tasks dynamically based on the prompt category using specialized handlers.
- 🔍 **Semantic Search**: Relevant chunks of your uploaded content are fetched in real-time to ground responses.
- 🧰 **Modular Architecture**: Clean separation of logic into components like `workspace`, `parser`, `classifier`, and `handlers`.

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **AI Model**: Google Gemini API
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Document Parsing**: PyMuPDF, PDFMiner, or similar
- **Language**: Python 3.10+
- **Architecture**: Modular, extensible with workspace-based isolation


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PromptPilot.git
cd PromptPilot
```
### 2. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### 3. Add Environment Variables 
create .env and add
```bash
GOOGLE_API_KEY=
WORKSPACE_DIR=
```
### 4. Setup mongodb 
look at db file inside helper

### 5. Run the app
```bash
streamlit run main.py
```







