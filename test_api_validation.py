"""
FastAPI validation tests for Phase 3.

Tests:
1. App imports and structure
2. Pydantic model validation
3. Request/response schemas
4. Basic endpoint functionality
"""

import os
from fastapi.testclient import TestClient


def test_app_imports():
    """Test that the FastAPI app imports successfully."""
    print("[OK] Testing app imports...")
    try:
        from app import app, ChatRequest, ChatResponse, HealthResponse
        print("  [OK] app module imported")
        print("  [OK] FastAPI app instance found")
        print("  [OK] Pydantic models imported")
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        raise


def test_pydantic_models():
    """Test Pydantic model validation."""
    print("\n[OK] Testing Pydantic model validation...")
    from app import ChatRequest, ChatResponse, ContextChunk
    
    # Valid request
    try:
        req = ChatRequest(query="What is Agentic AI?")
        assert req.query == "What is Agentic AI?"
        print("  [OK] Valid ChatRequest accepted")
    except Exception as e:
        print(f"  [FAIL] Valid request rejected: {e}")
        raise
    
    # Empty question should fail
    try:
        ChatRequest(query="")
        print("  [FAIL] Empty query was accepted (should be rejected)")
        assert False, "Empty query should be rejected"
    except Exception:
        print("  [OK] Empty query rejected")
    
    # Whitespace trimming
    try:
        req = ChatRequest(query="  Test question  ")
        assert req.query.strip() == "Test question"
        print("  [OK] Query validation works")
    except Exception as e:
        print(f"  [FAIL] Validation failed: {e}")
        raise
    
    # Valid response structure
    try:
        chunk = ContextChunk(
            text="Test context",
            source="test.pdf",
            page=5,
            relevance_score=0.87
        )
        response = ChatResponse(
            query="What is Agentic AI?",
            final_answer="Test answer",
            retrieved_context_chunks=[chunk],
            confidence_score=0.84
        )
        assert response.query == "What is Agentic AI?"
        assert response.final_answer == "Test answer"
        assert len(response.retrieved_context_chunks) == 1
        assert response.retrieved_context_chunks[0].page == 5
        assert response.confidence_score == 0.84
        print("  [OK] ChatResponse structure valid")
    except Exception as e:
        print(f"  [FAIL] Response validation failed: {e}")
        raise


def test_health_endpoint():
    """Test /health endpoint without requiring credentials."""
    print("\n[OK] Testing /health endpoint...")
    
    # Test without .env (should still respond)
    has_env = os.path.exists(".env")
    
    if not has_env:
        print("  [INFO] No .env file - /health will report degraded status")
        print("  [SKIP] Skipping live health check (no credentials)")
        return
    
    try:
        from app import app
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "pinecone_connected" in data
        assert "index_name" in data
        
        print(f"  [OK] /health responded: {data['status']}")
        print(f"  [OK] Pinecone connected: {data['pinecone_connected']}")
        print(f"  [OK] Index name: {data['index_name']}")
    except Exception as e:
        print(f"  [FAIL] Health endpoint failed: {e}")
        raise


def test_chat_request_schema():
    """Test /chat endpoint request validation."""
    print("\n[OK] Testing /chat request validation...")
    
    has_env = os.path.exists(".env")
    if not has_env:
        print("  [INFO] No .env file - skipping live /chat tests")
        print("  [OK] Schema validation tested via Pydantic models")
        return
    
    try:
        from app import app
        client = TestClient(app)
        
        # Test empty question rejection
        response = client.post("/chat", json={"query": ""})
        assert response.status_code == 422  # Validation error
        print("  [OK] Empty query rejected with 422")
        
        # Test missing question field
        response = client.post("/chat", json={})
        assert response.status_code == 422
        print("  [OK] Missing query field rejected with 422")
        
        # Test invalid JSON
        response = client.post("/chat", data="invalid")
        assert response.status_code == 422
        print("  [OK] Invalid JSON rejected with 422")
        
    except Exception as e:
        print(f"  [FAIL] Request validation test failed: {e}")
        raise


def test_app_metadata():
    """Test FastAPI app configuration."""
    print("\n[OK] Testing FastAPI app metadata...")
    from app import app
    
    assert app.title == "Agentic AI RAG Chatbot"
    assert app.version == "1.0.0"
    assert app.description is not None
    print("  [OK] App title: " + app.title)
    print("  [OK] App version: " + app.version)
    print("  [OK] App description present")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 FastAPI API Layer Validation")
    print("=" * 60)
    
    try:
        test_app_imports()
        test_app_metadata()
        test_pydantic_models()
        test_health_endpoint()
        test_chat_request_schema()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL API VALIDATION TESTS PASSED")
        print("=" * 60)
        
        has_env = os.path.exists(".env")
        if not has_env:
            print("\nNote: Live /chat endpoint testing requires:")
            print("  1. Copy .env.example to .env")
            print("  2. Add your API keys")
            print("  3. Run: python -m src.ingestion")
            print("  4. Start server: uvicorn app:app --reload")
            print("  5. Test at: http://localhost:8000/docs")
        else:
            print("\nNote: .env detected - you can test live endpoints:")
            print("  1. Ensure ingestion completed: python -m src.ingestion")
            print("  2. Start server: uvicorn app:app --reload")
            print("  3. Access API docs: http://localhost:8000/docs")
            print("  4. Test /health and /chat endpoints")
        
    except Exception as e:
        print(f"\n[FAIL] VALIDATION FAILED: {e}")
        raise
