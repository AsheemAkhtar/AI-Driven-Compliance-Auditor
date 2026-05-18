# create_kb.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from pypdf import PdfReader
import os

# =========================
# CONFIG
# =========================

QDRANT_URL = "https:-------"
QDRANT_API_KEY = "---------------------"
COLLECTION_NAME = "aml_regulations"

PDF_FOLDER = "./pdfs"

# LIGHT + FAST MODEL
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)

# =========================
# QDRANT
# =========================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# create collection if not exists
collections = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME not in collections:

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # bge-small-en-v1.5 dimension
            distance=Distance.COSINE
        )
    )

# =========================
# LOAD PDFs
# =========================

all_chunks = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

for file in os.listdir(PDF_FOLDER):

    if not file.endswith(".pdf"):
        continue

    path = os.path.join(PDF_FOLDER, file)

    pdf = PdfReader(path)

    text = ""

    for page in pdf.pages:
        text += page.extract_text() or ""

    chunks = splitter.split_text(text)

    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "source": file
        })

print(f"Total chunks: {len(all_chunks)}")

# =========================
# STORE IN QDRANT
# =========================

texts = [x["text"] for x in all_chunks]

metadatas = [
    {"source": x["source"]}
    for x in all_chunks
]

QdrantVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
)

print("Knowledge base created successfully.")