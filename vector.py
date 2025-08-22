import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import uuid

DB_DIR = "./chroma_langchain_db"  # unused when using Qdrant, retained for compatibility
CSV_PATH = "rise_employees_summary.csv"
DOCX_PATH = "report.docx"
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "hr_docs")

def _build_vectorstore():
    """Create or load the Chroma vector store using OpenAI embeddings.

    Requires OPENAI_API_KEY in environment or .env.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip('"').strip("'")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=your_key or set it in the environment."
        )

    # OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",  # or "text-embedding-3-large"
    )

    # Require Qdrant config and use it exclusively
    qdrant_url = os.environ.get("QDRANT_URL", "").strip()
    qdrant_key = (
        os.environ.get("QDRANT_API_KEY")
        or os.environ.get("QDRANT_API_KRY")
        or ""
    )
    qdrant_key = qdrant_key.strip('"').strip("'")

    if not qdrant_url or not qdrant_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY (or QDRANT_API_KRY) must be set to use Qdrant.")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Ensure collection exists with correct vector size and distance
    vector_size = int(os.environ.get("QDRANT_VECTOR_SIZE", "1536"))
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

def clean_str(v):
    try:
        import math
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

def csv_row_to_doc(row: dict, idx: int) -> Document:
    name = clean_str(row.get("Name"))
    dept = clean_str(row.get("Department"))
    salary = clean_str(row.get("SalaryPerDay"))
    advance = clean_str(row.get("AdvanceAmount"))
    to_pay = clean_str(row.get("AmountToBePaid"))

    page = (
        f"Employee record\n"
        f"Name: {name}\n"
        f"Department: {dept}\n"
        f"SalaryPerDay: {salary}\n"
        f"AdvanceAmount: {advance}\n"
        f"AmountToBePaid: {to_pay}"
    )

    meta = {k: clean_str(v) for k, v in row.items()}
    meta.update({
        "source": CSV_PATH,
        "row_index": idx,
        "doc_type": "employee_row",
    })
    return Document(page_content=page, metadata=meta)

def _ensure_index(vectorstore):
    """Add documents on first run; otherwise reuse existing index."""
    # Always upsert into Qdrant (ids prevent duplicates on re-runs)
    add_documents = True

    documents: list[Document] = []
    ids: list[str] = []

    # CSV -> Documents
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, engine="python")
        for i, row in enumerate(df.to_dict(orient="records")):
            doc = csv_row_to_doc(row, i)
            documents.append(doc)
            uid = uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION_NAME}/csv/{i}")
            ids.append(str(uid))
    else:
        print(f"Warning: CSV not found at {CSV_PATH}")

    # DOCX -> Documents
    if os.path.exists(DOCX_PATH):
        loader = Docx2txtLoader(DOCX_PATH)
        docx_docs = loader.load()
        for j, d in enumerate(docx_docs):
            meta = dict(d.metadata)
            meta.update({
                "source": DOCX_PATH,
                "doc_type": "report",
                "chunk_index": j,
            })
            documents.append(Document(page_content=d.page_content, metadata=meta))
            uid = uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION_NAME}/report/{j}")
            ids.append(str(uid))
    else:
        print(f"Warning: DOCX not found at {DOCX_PATH}")

    if documents:
        # Upsert using deterministic ids to avoid duplicates in Qdrant
        try:
            vectorstore.add_documents(documents=documents, ids=ids)
        except TypeError:
            vectorstore.add_documents(documents=documents)

        print(f"Indexed {len(documents)} documents into Qdrant")
    else:
        print("No documents to add.")


# Lazy-create the retriever so imports don't fail when API key is missing
_vs = _build_vectorstore()
_ensure_index(_vs)
# Improve retrieval to reduce hallucinations
# - higher k for more context
# - MMR for diverse results if supported
retriever = _vs.as_retriever(search_kwargs={"k": 8, "search_type": "mmr"})