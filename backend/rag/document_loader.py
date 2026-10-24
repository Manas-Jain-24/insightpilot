import os
import re
from typing import List, Tuple
from pypdf import PdfReader
from backend.models import DocumentChunk, DocumentMetadata
import uuid
import time


class DocumentLoader:
    """Handles parsing and semantic chunking of uploaded documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str, filename: str) -> Tuple[DocumentMetadata, List[DocumentChunk]]:
        ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(file_path)
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        pages_text: List[Tuple[int, str]] = []

        if ext == ".pdf":
            pages_text = self._load_pdf(file_path)
        else:
            # Plain text, markdown, csv, etc.
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            pages_text = [(1, content)]

        chunks = self._chunk_pages(doc_id, filename, pages_text)
        preview = chunks[0].content[:200] if chunks else ""

        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_type=ext.replace(".", "").upper(),
            size_bytes=file_size,
            chunks_count=len(chunks),
            upload_time=time.time(),
            preview_snippet=preview
        )

        return metadata, chunks

    def _load_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
        try:
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if text.strip():
                    pages.append((idx + 1, text))
        except Exception as e:
            raise ValueError("PDF text extraction failed") from e
        return pages

    def _clean_text(self, text: str) -> str:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_pages(self, doc_id: str, doc_name: str, pages: List[Tuple[int, str]]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page_num, text in pages:
            # Paragraph / sentence split
            sentences = re.split(r"(?<=[.!?])\s+", text)
            current_chunk = []
            current_len = 0

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                current_chunk.append(sent)
                current_len += len(sent)

                if current_len >= self.chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        id=f"{doc_id}_c{chunk_idx}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        page_number=page_num,
                        metadata={"source": doc_name, "page": page_num}
                    ))
                    chunk_idx += 1

                    # Handle overlap by keeping the last sentence
                    overlap_sentences = [current_chunk[-1]] if len(current_chunk) > 1 else []
                    current_chunk = overlap_sentences
                    current_len = sum(len(s) for s in current_chunk)

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    id=f"{doc_id}_c{chunk_idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_index=chunk_idx,
                    content=chunk_text,
                    page_number=page_num,
                    metadata={"source": doc_name, "page": page_num}
                ))
                chunk_idx += 1

        return chunks
