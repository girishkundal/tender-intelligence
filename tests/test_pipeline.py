"""
Tests for the Tender Intelligence RAG Pipeline.
Run with: pytest tests/ -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Pipeline unit tests ───────────────────────────────────────────────────────

class TestDocumentLoading:
    """Test document loading and chunking."""

    def test_load_txt_document(self):
        """TextLoader should load a .txt file and return Document objects."""
        from app.rag_pipeline import load_document
        sample = Path("data/sample_tenders/tender_001_crossrail_tunnel.txt")
        if sample.exists():
            docs = load_document(str(sample))
            assert len(docs) > 0
            assert docs[0].page_content != ""
            assert "source_file" in docs[0].metadata

    def test_chunk_documents(self):
        """Chunker should produce chunks within expected size range."""
        from app.rag_pipeline import load_document, chunk_documents
        sample = Path("data/sample_tenders/tender_001_crossrail_tunnel.txt")
        if sample.exists():
            docs = load_document(str(sample))
            chunks = chunk_documents(docs)
            assert len(chunks) > 1
            for chunk in chunks:
                assert len(chunk.page_content) > 0

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        from app.rag_pipeline import load_document
        with pytest.raises(FileNotFoundError):
            load_document("nonexistent_file.txt")

    def test_unsupported_format(self):
        """Unsupported file types should be handled gracefully."""
        # Create a temp .docx file
        tmp = Path("tests/temp_test.docx")
        tmp.write_bytes(b"fake docx content")
        try:
            from app.rag_pipeline import load_document
            # Should either raise or fall back — depends on LangChain version
            # We just check it doesn't silently corrupt
            with pytest.raises(Exception):
                load_document(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)


class TestExtractionQuestions:
    """Test the structured extraction question set."""

    def test_extraction_questions_defined(self):
        """All expected extraction fields should be defined."""
        from app.rag_pipeline import EXTRACTION_QUESTIONS
        expected_fields = [
            "project_name", "client", "contract_reference",
            "contract_value", "contract_duration", "start_date",
            "scope_summary", "key_requirements", "evaluation_criteria",
            "key_risks", "boq_total",
        ]
        for field in expected_fields:
            assert field in EXTRACTION_QUESTIONS, f"Missing field: {field}"

    def test_questions_are_strings(self):
        """All questions should be non-empty strings."""
        from app.rag_pipeline import EXTRACTION_QUESTIONS
        for field, question in EXTRACTION_QUESTIONS.items():
            assert isinstance(question, str)
            assert len(question) > 5, f"Question too short for field: {field}"


class TestPromptTemplate:
    """Test that the LLM prompt template is correctly defined."""

    def test_prompt_has_required_variables(self):
        """Prompt template must accept 'context' and 'question' variables."""
        from app.rag_pipeline import TENDER_EXTRACTION_PROMPT
        assert "context" in TENDER_EXTRACTION_PROMPT.input_variables
        assert "question" in TENDER_EXTRACTION_PROMPT.input_variables

    def test_prompt_contains_grounding_instruction(self):
        """Prompt must instruct the LLM not to invent values."""
        from app.rag_pipeline import TENDER_EXTRACTION_PROMPT
        prompt_text = TENDER_EXTRACTION_PROMPT.template
        assert "Do not invent" in prompt_text or "only from the provided" in prompt_text


# ── API integration tests ─────────────────────────────────────────────────────

class TestAPIEndpoints:
    """Test FastAPI endpoints (requires running API or TestClient)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return 200 with expected fields."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "documents_loaded" in data
        assert "index_exists" in data
        assert data["status"] == "ok"

    def test_list_documents_empty(self, client):
        """Documents list should return empty array on fresh start."""
        r = client.get("/documents")
        assert r.status_code == 200
        assert "documents" in r.json()

    def test_ask_without_documents(self, client):
        """Asking a question without ingested documents should return 400."""
        r = client.post("/ask", json={"question": "What is the contract value?"})
        assert r.status_code == 400

    def test_ask_empty_question(self, client):
        """Empty question should return 422 validation error."""
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_ingest_unsupported_type(self, client):
        """Uploading an unsupported file type should return 422."""
        r = client.post(
            "/ingest",
            files={"file": ("test.docx", b"fake content", "application/octet-stream")},
        )
        assert r.status_code == 422


# ── Sample data tests ─────────────────────────────────────────────────────────

class TestSampleData:
    """Verify sample tender documents are present and readable."""

    def test_sample_tenders_exist(self):
        """All three sample tender files should be present."""
        samples = [
            "data/sample_tenders/tender_001_crossrail_tunnel.txt",
            "data/sample_tenders/tender_002_utility_microtunnel.txt",
            "data/sample_tenders/tender_003_monitoring_instrumentation.txt",
        ]
        for path in samples:
            assert Path(path).exists(), f"Missing sample tender: {path}"

    def test_sample_tenders_readable(self):
        """Sample tenders should be non-empty and UTF-8 readable."""
        import glob
        for path in glob.glob("data/sample_tenders/*.txt"):
            content = Path(path).read_text(encoding="utf-8")
            assert len(content) > 100, f"Sample tender too short: {path}"
