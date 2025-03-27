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
