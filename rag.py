from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import *

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)


def get_vectorstore(qdrant_client):
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )


def retrieve(vs, query, k=4):
    return vs.similarity_search(query, k=k)


def ask_llm(context, question):
    prompt = f"""
Answer only using provided context.

CONTEXT:
{context}

QUESTION:
{question}
"""
    return llm.invoke(prompt).content


def compare_llm(doc, kb):

    prompt = f"""
You are an AML compliance auditor.

The KNOWLEDGE BASE is the official AML regulatory standard.
The DOCUMENT is the company policy being audited.

Your task:
- ONLY audit the DOCUMENT against relevant sections of the KNOWLEDGE BASE
- Determine the scope/topic of the DOCUMENT first
- ONLY evaluate requirements related to that scope
- Do NOT mark unrelated AML requirements as missing
- Example:
  - If DOCUMENT is about CDD, only evaluate CDD-related regulations
  - If DOCUMENT is about STR reporting, only evaluate STR-related regulations
  - If DOCUMENT is about sanctions, only evaluate sanctions-related regulations

Rules:
- Treat the KNOWLEDGE BASE as the source of truth
- NEVER criticize or evaluate the KNOWLEDGE BASE
- ONLY identify gaps in the DOCUMENT
- Mention regulation names/source names whenever possible
- Keep findings concise and professional
- Avoid generic statements
- Do not invent missing requirements unrelated to document scope

DOCUMENT:
{doc}

KNOWLEDGE BASE:
{kb}

Return STRICTLY in this format:

Compliant:
- ...

Partial:
- ...

Missing:
- ...

Suggestions:
- ...

Rules for output:
- Suggestions must ONLY improve the DOCUMENT
- Never say "KB is missing"
- Never compare both directions
- Only mention relevant compliance requirements
"""

    return llm.invoke(prompt).content