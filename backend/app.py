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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.rag.vector_store import VectorStore
from backend.rag.document_loader import DocumentLoader
from backend.agent.gemini_client import GeminiClient
from backend.db.database import Database
from backend.api.routes import create_router

# Load environment variables
load_dotenv()

app_mode = os.environ.get("APP_MODE", "local").strip().lower()
if app_mode not in {"local", "demo"}:
    raise RuntimeError("APP_MODE must be 'local' or 'demo'")

# The public demo has no mutable document or report store and makes no paid model calls.
vector_store = VectorStore(db_path=":memory:" if app_mode == "demo" else None)
gemini_client = GeminiClient(api_key="" if app_mode == "demo" else None)
db = None if app_mode == "demo" else Database()


def seed_sample_data():
    """Seeds sample research reports into the vector store for out-of-the-box demonstration."""
    root = os.path.dirname(os.path.dirname(__file__))
    directories = [os.path.join(root, "sample_data")]
    if app_mode == "demo":
        directories.append(os.path.join(root, "demo_data"))
    loader = DocumentLoader()
    for sample_dir in directories:
        if not os.path.isdir(sample_dir):
            continue
        for fname in os.listdir(sample_dir):
            fpath = os.path.join(sample_dir, fname)
            if os.path.isfile(fpath) and fname.endswith((".txt", ".md", ".pdf")):
                try:
                    if any(d.filename == fname for d in vector_store.list_documents()):
                        continue
                    meta, chunks = loader.load_file(fpath, fname)
                    pdf_bytes = None
                    if fname.endswith(".pdf"):
                        with open(fpath, "rb") as source:
                            pdf_bytes = source.read()
                    vector_store.add_document(meta, chunks, source_bytes=pdf_bytes)
                    print(f"[Startup] Indexed sample document: {fname} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"[Startup] Failed to index {fname}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Seed sample documents
    seed_sample_data()
    yield
    vector_store.close()


app = FastAPI(
    title="InsightPilot | Autonomous Research & Knowledge Agent",
    description="Research assistant with source-backed document and web retrieval.",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def demo_request_guard(request, call_next):
    # Reject writes before FastAPI parses multipart uploads or large report bodies.
    if app_mode == "demo":
        path = request.url.path
        method = request.method
        if (method == "POST" and path in {"/api/documents/upload", "/api/export/markdown"}) or (
            method == "DELETE" and (path.startswith("/api/documents/") or path.startswith("/api/reports/"))
        ):
            return JSONResponse({"detail": "Unavailable in the public demo"}, status_code=403)
        if method == "POST" and path == "/api/research/stream":
            if "content-length" not in request.headers:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            try:
                if int(request.headers["content-length"]) > 8192:
                    return JSONResponse({"detail": "Research request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
    return await call_next(request)

# The bundled frontend uses the same origin. Allow cross-origin requests only when configured.
allowed_origins = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "").split(",") if origin.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

# Attach API routes
api_router = create_router(vector_store, gemini_client, db=db, mode=app_mode)
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
    uvicorn.run("backend.app:app", host=host, port=port,
                reload=os.environ.get("DEBUG", "false").lower() == "true")
