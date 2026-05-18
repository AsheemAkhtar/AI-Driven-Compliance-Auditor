from mangum import Mangum
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
import io
import uuid
import os
import boto3

from qdrant_client import QdrantClient

from app.rag import get_vectorstore, retrieve, ask_llm, compare_llm
from app.config import *

# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# S3 CLIENT
# -------------------------
s3 = boto3.client("s3")

BUCKET = os.getenv("S3_BUCKET")

# -------------------------
# QDRANT + VECTORSTORE
# -------------------------
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

vs = get_vectorstore(qdrant)

# -------------------------
# PDF TEXT EXTRACTION
# -------------------------
def extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n".join(pages_text).strip()

# -------------------------
# GET PDF FROM S3
# -------------------------
def get_pdf_text(document_id: str) -> str:
    obj = s3.get_object(Bucket=BUCKET, Key=f"{document_id}.pdf")
    file_bytes = obj["Body"].read()
    return extract_text(file_bytes)

# -------------------------
# UPLOAD → S3
# -------------------------
@app.post("/upload")
async def upload(file: UploadFile):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    pdf_bytes = await file.read()
    doc_id = str(uuid.uuid4())

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{doc_id}.pdf",
        Body=pdf_bytes,
        ContentType="application/pdf"
    )

    return {
        "document_id": doc_id,
        "message": "uploaded successfully"
    }

# -------------------------
# ASK
# -------------------------
@app.post("/ask")
async def ask(document_id: str, question: str):

    try:
        text = get_pdf_text(document_id)
    except Exception:
        raise HTTPException(404, "Document not found or expired")

    kb_docs = retrieve(vs, question, k=6)
    kb_context = "\n".join(d.page_content for d in kb_docs)

    answer = ask_llm(kb_context + "\n" + text, question)

    return {"answer": answer}

# -------------------------
# COMPARE
# -------------------------
@app.post("/compare")
async def compare(document_id: str):

    try:
        text = get_pdf_text(document_id)
    except Exception:
        raise HTTPException(404, "Document expired or not found")

    kb_docs = retrieve(vs, text[:3000], k=8)
    kb_context = "\n".join(d.page_content for d in kb_docs)

    result = compare_llm(text, kb_context)

    return {"result": result}

# -------------------------
# LAMBDA HANDLER
# -------------------------
handler = Mangum(app)