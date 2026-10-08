# Local RAG Chatbot
A lightweight document question-answering application built with **FastAPI**, **ChromaDB**, **Sentence Transformers**, and **Qwen 2.5 3B Instruct**.
The application supports both:
- **RAG mode** — ask questions about an uploaded PDF
- **Normal Chat mode** — use the model as a general chatbot
- **Local LLM** — run Qwen locally
- **Google AI** — optionally use a Google/Gemini API
- **HNSW vector search** — fast similarity search using ChromaDB
- **Vanilla HTML/CSS/JavaScript frontend**

.env 

must contain:
'''
GOOGLE_API_KEY=API Key 
HF_TOKEN = TOKEN #skipable i think 
GOOGLE_MODEL="gemini-3.6-flash"

# Local RAG configuration
LOCAL_MODEL="Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR="./chroma_db_600"
COLLECTION_NAME="pdf_rag_600"
TOP_K=5
MAX_NEW_TOKENS=150

'''

---
## Features
### Document RAG
Upload a PDF and create a searchable vector database.
The pipeline is:
```text
PDF
 ↓
Text Extraction
 ↓
Semantic Chunker
 ↓
SentenceTransformer Embeddings
 ↓
ChromaDB
 ↓
HNSW Similarity Search
 ↓
Relevant Context
 ↓
Qwen / Google Model
 ↓
Answer

Chatbot Mode

The application can also operate without RAG:

User Question
      ↓
Qwen / Google Model
      ↓
Response

Model Providers

Two providers are supported:

Provider	Description
Local	      Qwen/Qwen2.5-3B-Instruct running locally
Google	Google Gemini API

Retrieval Configuration

The UI allows changing the number of retrieved chunks using Top-K.

Example:

Top-K = 3
Question
   ↓
Embedding
   ↓
HNSW
   ↓
Top 3 relevant chunks
   ↓
LLM

⸻

Project Structure

rag_fastapi_app/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── uploads/
│   └── uploaded PDFs
│
├── vector_dbs/
│   └── generated Chroma databases
│
└── chroma_db_600/
    └── existing Chroma database

⸻

Requirements

Python

Python 3.10+ is recommended.

Check your Python version:

python --version

or:

python3 --version

⸻

Installation

Clone or copy the project:

cd rag_fastapi_app

Create a virtual environment:

python -m venv .venv

Activate it.

macOS / Linux

source .venv/bin/activate

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

⸻

Requirements

The project uses:

fastapi
uvicorn
python-multipart
torch
transformers
sentence-transformers
chromadb
pypdf
langchain-core
langchain-text-splitters
langchain-google-genai
python-dotenv

Install them manually if required:

pip install fastapi uvicorn python-multipart
pip install torch transformers sentence-transformers
pip install chromadb pypdf
pip install langchain-core langchain-text-splitters
pip install langchain-google-genai python-dotenv

⸻

Running the Application

Start the FastAPI server:

python -m uvicorn app:app --host 127.0.0.1 --port 8000

For development with automatic reload:

python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

You should see:

Uvicorn running on http://127.0.0.1:8000

Open the application in your browser:

http://127.0.0.1:8000/

⸻

API Status

The backend provides a status endpoint:

http://127.0.0.1:8000/api/status

This can be used to verify that FastAPI is running.

Example:

{
  "rag_loaded": true,
  "chunks": 1117,
  "device": "mps"
}

The exact fields depend on the current backend configuration.

⸻

Using RAG Mode

1. Open the application.
2. Select RAG.
3. Select Local or Google.
4. Select a PDF.
5. Click Index PDF.
6. Wait for indexing to complete.
7. Enter a question.
8. Click Send.

The application extracts the PDF text, creates embeddings, stores them in ChromaDB, and performs HNSW similarity search when a question is asked.

⸻

PDF Processing

The current PDF pipeline uses pypdf for text extraction.

Text is then divided into chunks using:

Semantic Chunking

The default upload configuration is:

SEMANTIC_BREAKPOINT_TYPE : "percentile"
SEMANTIC_BREAKPOINT_THRESHOLD:"95"

These values can be changed in app.py.

⸻

Embedding Model

The default embedding model is:

sentence-transformers/all-MiniLM-L6-v2

It converts text into vector representations.

Example:

PDF chunk
   ↓
Embedding Model
   ↓
Vector

The query is embedded using the same model before similarity search.

⸻

Vector Database

The application uses:

ChromaDB

with:

HNSW

for approximate nearest-neighbor search.

The default similarity space is:

cosine

Typical HNSW configuration:

M = 16
ef_construction = 100

⸻

Local LLM

The default local model is:

Qwen/Qwen2.5-3B-Instruct

The model is loaded using Hugging Face Transformers.

On Apple Silicon, the application can use Apple’s Metal Performance Shaders backend when available.

For example:

mps

The model is loaded lazily when local generation is required.

⸻

Google Provider

The application can optionally use a Google model through:

langchain-google-genai

Select:

Google

in the UI and provide the API key.

Alternatively, configure an environment variable.

Example:

export GOOGLE_API_KEY="YOUR_API_KEY"

or:

export GEMINI_API_KEY="YOUR_API_KEY"

Do not commit API keys to GitHub.

⸻

Environment Variables

Create a .env file if desired:

GOOGLE_API_KEY=your_api_key_here

Keep .env out of version control.

Add this to .gitignore:

.env
.venv/
__pycache__/
uploads/
vector_dbs/

⸻

RAG Configuration

Important configuration values include:

TOP_K = 5

Number of chunks retrieved for each question.

CHUNK_SIZE = 600

Maximum chunk size used during PDF indexing.

CHUNK_OVERLAP = 100

Number of overlapping characters between adjacent chunks.

⸻

Retrieval Pipeline

For a RAG question:

User Question
      │
      ▼
SentenceTransformer
      │
      ▼
Query Embedding
      │
      ▼
ChromaDB / HNSW
      │
      ▼
Top-K Chunks
      │
      ▼
Context Construction
      │
      ▼
LLM
      │
      ▼
Answer

⸻

RAG Prompt

The local RAG model is instructed to answer using the retrieved document context rather than freely answering from its pretrained knowledge.

The intended behavior is:

Context Only
      ↓
Answer only what the context supports
      ↓
Avoid unsupported assumptions
      ↓
Return a fallback when information is unavailable

This helps reduce hallucination and off-topic responses.

⸻

Performance

The application reports timing information for requests.

Example:

Total 15713 ms
Embedding 30 ms
HNSW 8 ms
Generation 15637 ms
6.8 tok/s

The timings help identify the slowest component.

For a typical local Qwen RAG request, the generation stage can dominate the total latency.

Therefore:

Embedding       → usually small
HNSW search     → usually small
Generation      → potentially dominant

⸻

Performance Optimization

Useful parameters to experiment with:

Reduce Top-K

Instead of:

TOP_K = 5

try:

TOP_K = 3

This reduces the amount of retrieved context.

Reduce Maximum Output

For short document questions, avoid unnecessarily large output limits.

For example:

max_new_tokens = 120

instead of:

max_new_tokens = 300

Keep Models Loaded

Do not reload the embedding model or LLM for every request.

The application loads models once and reuses them.

⸻

Chunk Size vs Chunk Overlap

Chunking has a significant effect on RAG quality.

Example configurations to benchmark:

Configuration	Chunk Size	Overlap	Top-K
Baseline	1000	150	5
Test 1	800	100	5
Test 2	600	75	5
Test 3	500	50	5
Test 4	1000	150	3

Do not choose a configuration only because it produces the lowest latency.

Evaluate both:

Retrieval Quality
+
Answer Quality
+
Latency

⸻

API Endpoints

GET /

Returns the web application.

⸻

GET /api/status

Returns backend and RAG status.

⸻

POST /api/upload

Uploads and indexes a PDF.

Request:

multipart/form-data

Field:

file

⸻

POST /api/chat

Sends a question to the chatbot.

Example request:

{
  "question": "What is gradient descent?",
  "mode": "rag",
  "provider": "local",
  "top_k": 5
}

Possible modes:

rag
chat

Possible providers:

local
google

⸻

Troubleshooting

Could not connect to FastAPI server

Make sure FastAPI is running:

python -m uvicorn app:app --host 127.0.0.1 --port 8000

Then open:

http://127.0.0.1:8000/

Test:

http://127.0.0.1:8000/api/status

Do not open index.html directly using:

file:///

⸻

Could not import module "app"

Make sure you are inside the directory containing:

app.py

Then run:

python -m uvicorn app:app --reload

⸻

No module named ...

Activate the virtual environment:

source .venv/bin/activate

Then:

pip install -r requirements.txt

⸻

PyTorch / MPS Issues

Check PyTorch:

python -c "import torch; print(torch.__version__)"

Check MPS:

python -c "import torch; print(torch.backends.mps.is_available())"

If it returns:

True

MPS is available.

⸻

Google API Errors

Verify that:

GOOGLE_API_KEY

or:

GEMINI_API_KEY

is configured correctly.

Also make sure the Google model configured in app.py is currently available for your API account.

⸻

Development

Run with auto-reload:

python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

When modifying frontend files, refresh the browser.

For debugging, open browser developer tools:

macOS:
⌘ + Option + I

Check:

Console
Network

The Network tab is particularly useful for debugging:

/api/status
/api/upload
/api/chat

⸻

Recommended Git Structure

A clean repository can look like:

rag-fastapi-app/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── uploads/

Avoid committing generated vector databases, uploaded PDFs, model files, API keys, and virtual environments.

⸻

Technology Stack

Component	Technology
Backend	FastAPI
Server	Uvicorn
Frontend	HTML / CSS / JavaScript
PDF Extraction	pypdf
Text Splitting	LangChain Text Splitters
Embeddings	Sentence Transformers
Vector Database	ChromaDB
Vector Search	HNSW
Local LLM	Qwen2.5-3B-Instruct
Cloud LLM	Google Gemini
ML Framework	PyTorch
Local Apple GPU	MPS

⸻

Future Improvements

Possible improvements include:

* Structure-aware Markdown PDF extraction
* Heading-aware chunking
* Semantic chunking
* Reranking
* Hybrid BM25 + vector retrieval
* Retrieval evaluation
* RAGAS evaluation
* Recall@K evaluation
* MRR evaluation
* Answer faithfulness evaluation
* Streaming LLM responses
* Conversation memory
* Multi-PDF collections
* Document deletion
* Source text preview
* Page-level citations
* Token usage tracking
* Prefill/decode latency measurement
* Quantized local models
* Faster inference backends

⸻

RAG Evaluation

A fixed evaluation dataset should be used when comparing chunking configurations.

Useful metrics include:

Recall@1
Recall@3
Recall@5
MRR
Answer Correctness
Answer Faithfulness
Hallucination Rate
Embedding Latency
HNSW Latency
Generation Latency
Total Latency
Tokens / Second

The same questions should be evaluated across different:

chunk sizes
chunk overlaps
Top-K values
retrieval methods

This makes RAG optimization measurable rather than subjective.

⸻

License

This project is intended for educational, research, and development purposes.

Add an appropriate license before distributing the project publicly.

Save that content as:
```text
README.md

in the same directory as app.py.