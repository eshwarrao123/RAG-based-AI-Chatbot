"""
Quick validation test for Phase 2 LangGraph RAG implementation.

Tests:
1. Graph compilation
2. State structure
3. Confidence calculation
4. Import validation
"""

import logging
from src.graph import build_rag_graph, RAGState, compute_confidence

logging.basicConfig(level=logging.INFO)

def test_graph_compilation():
    """Test that the graph compiles without errors."""
    print("[OK] Testing graph compilation...")
    graph = build_rag_graph()
    assert graph is not None
    print(f"  Graph type: {type(graph).__name__}")
    print("  [OK] Graph compiled successfully")

def test_state_structure():
    """Test that RAGState has the required fields."""
    print("\n[OK] Testing RAGState structure...")
    required_fields = ["question", "context", "retrieval_scores", "answer", "confidence_score"]
    state_fields = RAGState.__annotations__.keys()
    
    for field in required_fields:
        assert field in state_fields, f"Missing field: {field}"
        print(f"  [OK] Field '{field}' present")
    
    print("  [OK] RAGState structure valid")

def test_confidence_calculation():
    """Test confidence score calculation."""
    print("\n[OK] Testing confidence calculation...")
    
    # High confidence
    scores1 = [0.85, 0.82, 0.79, 0.76]
    conf1 = compute_confidence(scores1)
    assert conf1 == 0.805, f"Expected 0.805, got {conf1}"
    print(f"  [OK] High confidence: {conf1}")
    
    # Medium confidence
    scores2 = [0.65, 0.62, 0.58, 0.55]
    conf2 = compute_confidence(scores2)
    assert conf2 == 0.6, f"Expected 0.6, got {conf2}"
    print(f"  [OK] Medium confidence: {conf2}")
    
    # Low confidence
    scores3 = [0.45, 0.42, 0.40, 0.38]
    conf3 = compute_confidence(scores3)
    assert conf3 == 0.4125, f"Expected 0.4125, got {conf3}"
    print(f"  [OK] Low confidence: {conf3}")
    
    # No results
    scores4 = []
    conf4 = compute_confidence(scores4)
    assert conf4 == 0.0, f"Expected 0.0, got {conf4}"
    print(f"  [OK] Empty results: {conf4}")
    
    print("  [OK] Confidence calculation validated")

def test_imports():
    """Test that all required imports work."""
    print("\n[OK] Testing critical imports...")
    
    try:
        from langchain_pinecone import PineconeVectorStore
        print("  [OK] langchain_pinecone imported")
    except ImportError as e:
        print(f"  [FAIL] langchain_pinecone import failed: {e}")
        raise
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("  [OK] langchain_openai imported")
    except ImportError as e:
        print(f"  [FAIL] langchain_openai import failed: {e}")
        raise
    
    try:
        from langgraph.graph import StateGraph, START, END
        print("  [OK] langgraph imported")
    except ImportError as e:
        print(f"  [FAIL] langgraph import failed: {e}")
        raise
    
    try:
        from pinecone import Pinecone
        print("  [OK] pinecone imported")
    except ImportError as e:
        print(f"  [FAIL] pinecone import failed: {e}")
        raise
    
    print("  [OK] All imports successful")

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 LangGraph RAG Workflow Validation")
    print("=" * 60)
    
    try:
        test_imports()
        test_state_structure()
        test_confidence_calculation()
        test_graph_compilation()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL VALIDATION TESTS PASSED")
        print("=" * 60)
        print("\nNote: Live Pinecone/OpenAI tests require:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API keys")
        print("  3. Run: python -m src.ingestion")
        print("  4. Run: python -m src.graph")
        
    except Exception as e:
        print(f"\n[FAIL] VALIDATION FAILED: {e}")
        raise
