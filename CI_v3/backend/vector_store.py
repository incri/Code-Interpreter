import os
import json
from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

import google.generativeai as genai
import faiss

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")

genai.configure(api_key=GOOGLE_API_KEY)

os.makedirs(WORKSPACE_DIR, exist_ok=True)


def create_workspace(workspace_name: str):
    """Creates a new workspace with a local FAISS index."""
    DIMENSION = 768

    index_filename = f"{workspace_name.lower().replace(' ', '-')}.faiss"
    index_path = os.path.join(WORKSPACE_DIR, index_filename)

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Loaded existing FAISS index for workspace: {workspace_name}")
    else:
        index = faiss.IndexFlatL2(DIMENSION)
        faiss.write_index(index, index_path)
        print(f"Created new FAISS index for workspace: {workspace_name}")

    workspace_metadata = {
        "name": workspace_name,
        "index_path": index_path,
        "index_name": index_filename,
        "files": [],
    }

    metadata_path = os.path.join(WORKSPACE_DIR, f"{workspace_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(workspace_metadata, f, indent=4)

    return index_path


def list_workspaces() -> List[str]:
    """Lists all available workspaces."""
    return [
        f.split(".json")[0] for f in os.listdir(WORKSPACE_DIR) if f.endswith(".json")
    ]


def ingest_pdfs(workspace_name: str, pdf_paths: List[str]):
    """Processes and embeds PDFs into the specified workspace."""
    workspace_file = os.path.join(WORKSPACE_DIR, f"{workspace_name}.json")
    if not os.path.exists(workspace_file):
        raise ValueError("Workspace does not exist. Create it first.")

    with open(workspace_file, "r") as f:
        workspace_data = json.load(f)

    index_path = workspace_data["index_path"]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Load existing FAISS index or create a new one
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"Loaded existing FAISS index for workspace: {workspace_name}")
    else:
        vector_store = FAISS()
        print(f"Created new FAISS index for workspace: {workspace_name}")

    text_splitter = SemanticChunker(embeddings)
    all_documents = []

    for pdf_path in pdf_paths:
        pdf_loader = PyPDFLoader(pdf_path)
        raw_documents = pdf_loader.load()
        documents = text_splitter.split_documents(documents=raw_documents)
        all_documents.extend(documents)

        if pdf_path not in workspace_data["files"]:
            workspace_data["files"].append(pdf_path)

    with open(workspace_file, "w") as f:
        json.dump(workspace_data, f, indent=4)

    vector_store = FAISS.from_documents(all_documents, embeddings)
    vector_store.save_local(index_path)

    print(f"Total documents processed: {len(all_documents)}")
    print(f"Updated FAISS index for workspace: {workspace_name}")
