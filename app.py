import os
import time
import uuid
import shutil
from pathlib import Path
from threading import Lock
from typing import Optional
import chromadb
import torch

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from langchain_core.documents import Document

from langchain_experimental.text_splitter import (
    SemanticChunker,
)

# Optional Google provider. Install with:
# pip install langchain-google-genai
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
VECTOR_DB_ROOT = APP_DIR / "vector_dbs"
STATIC_DIR = APP_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DB_ROOT.mkdir(exist_ok=True)

MODEL_NAME = os.getenv(
    "LOCAL_MODEL",
    "Qwen/Qwen2.5-3B-Instruct",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Existing database from your current localrag.py.
DEFAULT_CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    "./chroma_db_600",
)

DEFAULT_COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "pdf_rag_600",
)

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))

# Semantic chunking
SEMANTIC_BREAKPOINT_TYPE = os.getenv(
    "SEMANTIC_BREAKPOINT_TYPE",
    "percentile",
)

SEMANTIC_BREAKPOINT_THRESHOLD = float(
    os.getenv(
        "SEMANTIC_BREAKPOINT_THRESHOLD",
        "95",
    )
)

HNSW_SPACE = "cosine"
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(
    os.getenv("HNSW_EF_CONSTRUCTION", "100")
)


# ============================================================
# App state
# ============================================================

app = FastAPI(
    title="Local RAG / Chatbot",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

embedding_model = None
semantic_embedding_model = None
local_tokenizer = None
local_model = None

active_client = None
active_collection = None
active_chroma_path = None
active_collection_name = None
active_pdf_name = None

state_lock = Lock()
model_lock = Lock()


# ============================================================
# Models
# ============================================================

def load_local_models():
    global embedding_model
    global local_tokenizer
    global local_model

    if embedding_model is None:
        print("Loading embedding model...")
        start = time.perf_counter()

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
        )

        print(
            f"Embedding model loaded: "
            f"{time.perf_counter() - start:.2f}s"
        )

    if local_model is None or local_tokenizer is None:
        print("Loading local Qwen model...")
        start = time.perf_counter()

        local_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
        )

        print(
            f"Qwen loaded: "
            f"{time.perf_counter() - start:.2f}s"
        )
        print(f"Device: {local_model.device}")


def load_embedding_only():
    global embedding_model
    global semantic_embedding_model

    # Raw SentenceTransformer
    if embedding_model is None:
        print("Loading SentenceTransformer...")

        start = time.perf_counter()

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
        )

        print(
            f"SentenceTransformer loaded: "
            f"{time.perf_counter() - start:.2f}s"
        )

    # LangChain wrapper for SemanticChunker
    if semantic_embedding_model is None:
        print("Loading LangChain embedding wrapper...")

        semantic_embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )


# ============================================================
# Chroma
# ============================================================

def load_existing_collection():
    global active_client
    global active_collection
    global active_chroma_path
    global active_collection_name
    global active_pdf_name

    path = Path(DEFAULT_CHROMA_DIR)

    if not path.is_absolute():
        path = APP_DIR / path

    try:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(
            name=DEFAULT_COLLECTION_NAME
        )

        active_client = client
        active_collection = collection
        active_chroma_path = str(path)
        active_collection_name = DEFAULT_COLLECTION_NAME
        active_pdf_name = "Existing local database"

        print(
            f"Loaded Chroma collection "
            f"'{DEFAULT_COLLECTION_NAME}' "
            f"({collection.count()} chunks)"
        )

    except Exception as exc:
        print(
            "No existing Chroma collection loaded: "
            f"{exc}"
        )


# ============================================================
# PDF -> Chroma/HNSW
# ============================================================

def build_vector_db(pdf_path: Path):
    load_embedding_only()

    build_id = uuid.uuid4().hex[:10]

    db_path = VECTOR_DB_ROOT / f"chroma_{build_id}"

    collection_name = f"pdf_rag_{build_id}"

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # PDF extraction
    # --------------------------------------------------------

    start = time.perf_counter()

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "text": text.strip(),
                    "page": page_number,
                }
            )

    pdf_time = time.perf_counter() - start

    print(
        f"[PDF] {len(pages)} pages extracted "
        f"in {pdf_time:.2f}s"
    )

    if not pages:
        raise ValueError(
            "No extractable text was found in the PDF."
        )

    # --------------------------------------------------------
    # Semantic chunker
    # --------------------------------------------------------

    start = time.perf_counter()

    semantic_chunker = SemanticChunker(
        embeddings=semantic_embedding_model,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
    )

    texts = []
    metadatas = []

    global_chunk_id = 0

    # --------------------------------------------------------
    # Semantic chunking
    #
    # Process each page separately so that a chunk always
    # belongs to a known PDF page.
    # --------------------------------------------------------

    for page in pages:

        document = Document(
            page_content=page["text"],
            metadata={
                "source": pdf_path.name,
                "page": page["page"],
            },
        )

        page_chunks = semantic_chunker.split_documents(
            [document]
        )

        for chunk in page_chunks:

            text = chunk.page_content.strip()

            if not text:
                continue

            metadata = {
                "source": pdf_path.name,
                "page": page["page"],
                "chunk_id": global_chunk_id,
                "chunking": "semantic",
            }

            texts.append(text)
            metadatas.append(metadata)

            global_chunk_id += 1

    chunking_time = time.perf_counter() - start

    print(
        f"[SEMANTIC CHUNKING] "
        f"{len(texts)} chunks created "
        f"in {chunking_time:.2f}s"
    )

    # --------------------------------------------------------
    # Chunk statistics
    # --------------------------------------------------------

    if texts:

        lengths = [
            len(text)
            for text in texts
        ]

        print(
            f"[CHUNKS] "
            f"min={min(lengths)}, "
            f"max={max(lengths)}, "
            f"avg={sum(lengths) / len(lengths):.1f} chars"
        )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    start = time.perf_counter()

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embedding_time = time.perf_counter() - start

    print(
        f"[EMBEDDING] "
        f"{len(embeddings)} embeddings generated "
        f"in {embedding_time:.2f}s"
    )

    # --------------------------------------------------------
    # Chroma + HNSW
    # --------------------------------------------------------

    start = time.perf_counter()

    client = chromadb.PersistentClient(
        path=str(db_path)
    )

    collection = client.create_collection(
        name=collection_name,
        configuration={
            "hnsw": {
                "space": HNSW_SPACE,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "max_neighbors": HNSW_M,
            }
        },
    )

    ids = [
        f"chunk_{i}"
        for i in range(len(texts))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    chroma_time = time.perf_counter() - start

    print(
        f"[CHROMA/HNSW] DB created in "
        f"{chroma_time:.2f}s"
    )

    print(
        f"[TOTAL DB BUILD] "
        f"{time.perf_counter() - total_start:.2f}s"
    )

    return (
        client,
        collection,
        str(db_path),
        collection_name,
    )


# ============================================================
# Retrieval
# ============================================================

def retrieve_documents(query: str, k: int = TOP_K):
    if active_collection is None:
        raise RuntimeError(
            "No RAG database is loaded. Upload a PDF first."
        )

    start_total = time.perf_counter()

    start = time.perf_counter()

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    embedding_ms = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    results = active_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    hnsw_ms = (
        time.perf_counter() - start
    ) * 1000

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    total_ms = (
        time.perf_counter() - start_total
    ) * 1000

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
        "timing": {
            "embedding_ms": round(embedding_ms, 2),
            "hnsw_ms": round(hnsw_ms, 2),
            "retrieval_ms": round(total_ms, 2),
        },
    }


def build_context(documents, metadatas):
    parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):

        if metadata is None:
            metadata = {}

        page = metadata.get(
            "page",
            "Unknown",
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown",
        )

        parts.append(
            f"[Source {i}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_id}\n\n"
            f"{document}"
        )

    return "\n\n".join(parts)


# ============================================================
# Prompts
# ============================================================

RAG_SYSTEM_PROMPT = """You are a strict document question-answering system.

Your ONLY source of information is the provided context. Answer ONLY what the user asked, using ONLY information explicitly supported by the retrieved excerpts.

Rules:
1. Do not use pretrained knowledge, assumptions, or outside information.
2. If the context cannot answer the question, respond exactly:
"I could not find this information in the provided document."
3. Never invent facts, page numbers, or details.
4. Every factual statement from the context must include a page citation like "(p. 12)".
5. If excerpts conflict, state the conflict and cite the relevant pages.
6. Keep the answer concise.
7. Do not answer related questions that were not asked.
8. Do not add background information unless it is required to answer the question.
9. Do not infer missing information.
10. Do not repeat the context verbatim.
"""

CHAT_SYSTEM_PROMPT = """You are a helpful general-purpose AI assistant.

Answer the user's question directly and clearly.
You may use your general knowledge.
Do not claim that information comes from a document unless a document was provided.
Keep the response concise unless the user requests more detail.
"""


# ============================================================
# Local generation
# ============================================================

def generate_local(
    question: str,
    system_prompt: str,
    context: str = "",
):
    load_local_models()

    if context:
        user_prompt = (
            f"Context:\n\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
    else:
        user_prompt = question

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    with model_lock:
        start = time.perf_counter()

        inputs = local_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(local_model.device)

        tokenization_ms = (
            time.perf_counter() - start
        ) * 1000

        input_tokens = (
            inputs["input_ids"].shape[-1]
        )

        start = time.perf_counter()

        with torch.no_grad():
            outputs = local_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        generation_s = (
            time.perf_counter() - start
        )

        generated_tokens = outputs[0][
            input_tokens:
        ]

        answer = local_tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        output_tokens = len(generated_tokens)

    return answer, {
        "provider": "local",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokenization_ms": round(
            tokenization_ms, 2
        ),
        "generation_ms": round(
            generation_s * 1000, 2
        ),
        "tokens_per_second": round(
            output_tokens / generation_s, 2
        ) if generation_s > 0 else 0,
    }


# ============================================================
# Google generation
# ============================================================

def generate_google(
    question: str,
    system_prompt: str,
    context: str = "",
    api_key: Optional[str] = None,
):
    if not GOOGLE_AVAILABLE:
        raise RuntimeError(
            "Google provider is not installed. "
            "Run: pip install langchain-google-genai"
        )

    key = (
        api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not key:
        raise RuntimeError(
            "Google API key is required."
        )

    if context:
        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            f"{system_prompt}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    start = time.perf_counter()

    model = ChatGoogleGenerativeAI(
        model=os.getenv(
            "GOOGLE_MODEL",
            "gemini-3.6-flash",
        ),
        google_api_key=key,
        temperature=0.1,
    )

    response = model.invoke(prompt)

    generation_ms = (
        time.perf_counter() - start
    ) * 1000

    return str(response.content).strip(), {
        "provider": "google",
        "generation_ms": round(
            generation_ms, 2
        ),
    }


# ============================================================
# API schemas
# ============================================================

class ChatRequest(BaseModel):
    question: str
    mode: str = "rag"
    provider: str = "local"
    top_k: int = TOP_K
    api_key: Optional[str] = None


# ============================================================
# Routes
# ============================================================

@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html"
    )


@app.get("/api/status")
def status():
    return {
        "local_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "rag_loaded": active_collection is not None,
        "pdf": active_pdf_name,
        "collection": active_collection_name,
        "chunks": (
            active_collection.count()
            if active_collection is not None
            else 0
        ),
        "google_available": GOOGLE_AVAILABLE,
        "device": (
            str(local_model.device)
            if local_model is not None
            else "not loaded"
        ),
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global active_client
    global active_collection
    global active_chroma_path
    global active_collection_name
    global active_pdf_name

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    safe_name = (
        Path(file.filename).name
    )

    upload_id = uuid.uuid4().hex[:10]
    pdf_path = (
        UPLOAD_DIR
        / f"{upload_id}_{safe_name}"
    )

    try:
        with pdf_path.open("wb") as output:
            shutil.copyfileobj(
                file.file,
                output,
            )

        (
            client,
            collection,
            db_path,
            collection_name,
        ) = build_vector_db(pdf_path)

        with state_lock:
            active_client = client
            active_collection = collection
            active_chroma_path = db_path
            active_collection_name = collection_name
            active_pdf_name = safe_name

        return {
            "success": True,
            "filename": safe_name,
            "collection": collection_name,
            "chunks": collection.count(),
            "message": "PDF indexed successfully.",
        }

    except Exception as exc:
        if pdf_path.exists():
            pdf_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    mode = request.mode.lower()
    provider = request.provider.lower()

    if mode not in {"rag", "chat"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'rag' or 'chat'.",
        )

    if provider not in {"local", "google"}:
        raise HTTPException(
            status_code=400,
            detail="provider must be 'local' or 'google'.",
        )

    total_start = time.perf_counter()

    retrieval = None
    context = ""

    # --------------------------------------------------------
    # RAG mode
    # --------------------------------------------------------

    if mode == "rag":
        if active_collection is None:
            raise HTTPException(
                status_code=400,
                detail="Upload a PDF or load a RAG database first.",
            )

        k = max(1, min(request.top_k, 20))

        retrieval = retrieve_documents(
            question,
            k=k,
        )

        context = build_context(
            retrieval["documents"],
            retrieval["metadatas"],
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    if provider == "local":
        answer, generation = generate_local(
            question=question,
            system_prompt=(
                RAG_SYSTEM_PROMPT
                if mode == "rag"
                else CHAT_SYSTEM_PROMPT
            ),
            context=context,
        )
    else:
        answer, generation = generate_google(
            question=question,
            system_prompt=(
                RAG_SYSTEM_PROMPT
                if mode == "rag"
                else CHAT_SYSTEM_PROMPT
            ),
            context=context,
            api_key=request.api_key,
        )

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000

    sources = []

    for i, (metadata, distance,) in enumerate(zip(retrieval["metadatas"], retrieval["distances"],), start=1,):

        if metadata is None:
            metadata = {}

        sources.append(
            {
                "rank": i,
                "page": metadata.get(
                    "page",
                    "Unknown",
                ),
                "source": metadata.get(
                    "source",
                    "Unknown",
                ),
                "chunk_id": metadata.get(
                    "chunk_id",
                    "Unknown",
                ),
                "distance": round(
                    float(distance),
                    4,
                ),
            }
        )

    return {
        "answer": answer,
        "mode": mode,
        "provider": provider,
        "sources": sources,
        "timing": {
            "retrieval": (
                retrieval["timing"]
                if retrieval
                else None
            ),
            "generation": generation,
            "total_ms": round(
                total_ms,
                2,
            ),
        },
    }


@app.on_event("startup")
def startup():
    # Load the embedding model and existing RAG DB.
    # Qwen is lazy-loaded when Local is actually selected.
    load_embedding_only()
    load_existing_collection()
