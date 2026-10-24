"""Checks the public deployment mode without changing the local test process."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.api.rate_limit import DemoRateLimiter


ROOT = Path(__file__).resolve().parents[1]


class TestDeployment(unittest.TestCase):
    def test_demo_mode_keeps_sample_research_and_blocks_writes(self):
        script = """
from fastapi.testclient import TestClient
from backend.app import app, db, vector_store

assert db is None
assert vector_store.db_path == ':memory:'
with TestClient(app) as client:
    health = client.get('/api/health').json()
    assert health['mode'] == 'demo'
    assert health['gemini_api_configured'] is False
    assert health['indexed_documents'] >= 3
    config = client.get('/api/config').json()
    assert config['uploads_enabled'] is False
    assert config['history_enabled'] is False
    docs = client.get('/api/documents').json()
    pdf_doc = next(doc for doc in docs if doc['filename'] == 'source_viewer_demo.pdf')
    chunk = next(chunk for chunk in vector_store.chunks if chunk.doc_id == pdf_doc['id'])
    assert client.get(f"/api/documents/{pdf_doc['id']}/chunks/{chunk.id}").json()['pdf_available']
    preview = client.get(f"/api/documents/{pdf_doc['id']}/pages/1/preview", params={
        'quote': 'The evidence viewer opens the original PDF page'
    })
    assert preview.status_code == 200
    assert preview.content.startswith(b'\\x89PNG')
    assert client.post('/api/documents/upload', files={'file': ('x.txt', b'test')}).status_code == 403
    assert client.delete('/api/documents/sample').status_code == 403
    assert client.delete('/api/reports/sample').status_code == 403
    assert client.post('/api/export/markdown', content=b'x' * 20000).status_code == 403
    assert client.post('/api/research/stream', json={'query': 'NIST quantum', 'api_key': 'test'}).status_code == 403
    assert client.post('/api/research/stream', content=b'x' * 9000).status_code == 413
    research = client.post('/api/research/stream', json={
        'query': 'How does the source viewer highlight a cited PDF passage?',
        'search_web': False
    })
    assert research.status_code == 200
    assert 'source_viewer_demo.pdf' in research.text
    assert client.get('/api/reports').json() == []
    assert client.get('/api/evaluation').status_code == 200
"""
        env = os.environ.copy()
        env["APP_MODE"] = "demo"
        env["GEMINI_API_KEY"] = ""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_demo_mode_uses_server_key_without_accepting_visitor_keys(self):
        script = """
from fastapi.testclient import TestClient
from backend.app import app
with TestClient(app) as client:
    assert client.get('/api/health').json()['gemini_api_configured'] is True
    config = client.get('/api/config').json()
    assert config['model_enabled'] is True
    assert config['visitor_api_keys_enabled'] is False
    assert client.post('/api/research/stream', json={'query': 'NIST quantum', 'api_key': 'visitor-key'}).status_code == 403
"""
        env = os.environ.copy()
        env["APP_MODE"] = "demo"
        env["GEMINI_API_KEY"] = "server-test-key"
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_demo_rate_limit_resets_after_window(self):
        limiter = DemoRateLimiter(window_seconds=60)
        with patch("backend.api.rate_limit.time.monotonic", return_value=100):
            self.assertTrue(limiter.allow("client", 2))
            self.assertTrue(limiter.allow("client", 2))
            self.assertFalse(limiter.allow("client", 2))
        with patch("backend.api.rate_limit.time.monotonic", return_value=161):
            self.assertTrue(limiter.allow("client", 2))
