import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.rag.vector_store import VectorStore
from backend.rag.document_loader import DocumentLoader
from backend.agent.gemini_client import GeminiClient
from backend.db.database import Database
from backend.api.routes import create_router

# Load environment variables
load_dotenv()

vector_store = VectorStore()
gemini_client = GeminiClient()
vector_store.gemini_client = gemini_client
db = Database()


def seed_sample_data():
    """Seeds sample research reports into the vector store for out-of-the-box demonstration."""
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    if not os.path.exists(sample_dir):
        return

    loader = DocumentLoader()
    for fname in os.listdir(sample_dir):
        fpath = os.path.join(sample_dir, fname)
        if os.path.isfile(fpath) and fname.endswith((".txt", ".md", ".pdf")):
            try:
                meta, chunks = loader.load_file(fpath, fname)
                vector_store.add_document(meta, chunks)
                print(f"[Startup] Indexed sample document: {fname} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"[Startup] Failed to index {fname}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Seed sample documents
    seed_sample_data()
    yield
    # Shutdown logic if needed


app = FastAPI(
    title="InsightPilot | Autonomous Research & Knowledge Agent",
    description="Production-grade AI agent featuring ReAct loops, Vector RAG, and dynamic knowledge synthesis.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API routes
api_router = create_router(vector_store, gemini_client, db=db)
app.include_router(api_router, prefix="/api")

# Static files setup
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n[InsightPilot] Launching at http://{host}:{port}\n")
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
