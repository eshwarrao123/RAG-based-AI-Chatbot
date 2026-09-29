"""
Benchmark Test Suite for Agentic AI RAG Chatbot.

Tests 6 queries against the RAG system:
- 5 in-domain queries (from the Agentic AI eBook)
- 1 out-of-domain query (grounding/refusal test)

Usage:
    python tests_sample_queries.py

Requirements:
    - Valid .env with OPENAI_API_KEY and PINECONE_API_KEY
    - Pinecone index populated with eBook embeddings
    - Virtual environment activated
"""

import logging
import sys
from pathlib import Path

# Ensure imports work from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import validate_config
from src.graph import query_rag

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during testing
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Test Queries ──────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "What is the core definition of Agentic AI as outlined in the eBook?",
        "expected_type": "in-domain",
        "description": "Core definition from the eBook",
    },
    {
        "id": 2,
        "query": "What are the main architectural components required to build agentic systems?",
        "expected_type": "in-domain",
        "description": "Architectural components",
    },
    {
        "id": 3,
        "query": "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "expected_type": "in-domain",
        "description": "Industry use cases",
    },
    {
        "id": 4,
        "query": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "expected_type": "in-domain",
        "description": "Differentiation from traditional AI",
    },
    {
        "id": 5,
        "query": "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "expected_type": "in-domain",
        "description": "Challenges and limitations",
    },
    {
        "id": 6,
        "query": "What is the capital of France?",
        "expected_type": "out-of-domain",
        "description": "Grounding test — should refuse to answer from model knowledge",
    },
]


# ── Validation Functions ──────────────────────────────────────

def validate_response(test_case: dict, result: dict) -> dict:
    """
    Validate that the response has the expected structure and content.

    Returns a validation report dict.
    """
    query_id = test_case["id"]
    expected_type = test_case["expected_type"]
    
    report = {
        "query_id": query_id,
        "query": test_case["query"],
        "expected_type": expected_type,
        "passed": True,
        "issues": [],
    }

    # Check structure
    if "answer" not in result or not result["answer"]:
        report["passed"] = False
        report["issues"].append("Missing or empty answer")

    if "context" not in result or not isinstance(result["context"], list):
        report["passed"] = False
        report["issues"].append("Missing or invalid context")

    if "confidence_score" not in result:
        report["passed"] = False
        report["issues"].append("Missing confidence_score")

    # Check context metadata
    if result.get("context"):
        for i, chunk in enumerate(result["context"]):
            if "text" not in chunk:
                report["issues"].append(f"Context chunk {i} missing text")
            if "source" not in chunk:
                report["issues"].append(f"Context chunk {i} missing source")
            if "page" not in chunk:
                report["issues"].append(f"Context chunk {i} missing page")
            if "relevance_score" not in chunk:
                report["issues"].append(f"Context chunk {i} missing relevance_score")

    # Check grounding behavior for out-of-domain query
    if expected_type == "out-of-domain":
        answer_lower = result.get("answer", "").lower()
        # Should contain refusal phrases, not factual answers
        refusal_indicators = [
            "don't have enough information",
            "not found in",
            "cannot answer",
            "insufficient information",
            "not available in the context",
            "ebook does not contain",
        ]
        
        has_refusal = any(indicator in answer_lower for indicator in refusal_indicators)
        
        # Should NOT contain factual answers from training data
        # For France capital query: should not contain "Paris"
        contains_factual_answer = "paris" in answer_lower
        
        if not has_refusal:
            report["passed"] = False
            report["issues"].append(
                "Out-of-domain query: Expected refusal/insufficient-info response"
            )
        
        if contains_factual_answer:
            report["passed"] = False
            report["issues"].append(
                "Out-of-domain query: Model appears to have answered from training data (grounding failure)"
            )

    return report


# ── Test Runner ───────────────────────────────────────────────

def run_benchmark() -> dict:
    """
    Run all benchmark queries and return a summary report.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK TEST SUITE: Agentic AI RAG Chatbot")
    print("=" * 70)

    results = []
    passed_count = 0
    failed_count = 0

    for test_case in BENCHMARK_QUERIES:
        query_id = test_case["id"]
        query = test_case["query"]
        expected_type = test_case["expected_type"]

        print(f"\n{'─' * 70}")
        print(f"TEST {query_id}: {test_case['description']}")
        print(f"Type: {expected_type.upper()}")
        print(f"{'─' * 70}")
        print(f"Query: {query}")
        print()

        try:
            # Execute query
            result = query_rag(query)

            # Display results
            print(f"✓ Answer received ({len(result['answer'])} chars)")
            print(f"✓ Context chunks: {len(result['context'])}")
            print(f"✓ Confidence score: {result['confidence_score']:.4f}")
            
            if result["context"]:
                pages = [chunk["page"] for chunk in result["context"]]
                print(f"✓ Retrieved pages: {sorted(set(pages))}")

            print(f"\n📝 Answer:\n{result['answer']}")

            # Validate
            validation = validate_response(test_case, result)
            results.append({**validation, "result": result})

            if validation["passed"]:
                print(f"\n✅ TEST {query_id} PASSED")
                passed_count += 1
            else:
                print(f"\n❌ TEST {query_id} FAILED")
                print(f"Issues: {', '.join(validation['issues'])}")
                failed_count += 1

        except Exception as e:
            print(f"\n❌ TEST {query_id} FAILED WITH EXCEPTION")
            print(f"Error: {e}")
            results.append({
                "query_id": query_id,
                "query": query,
                "expected_type": expected_type,
                "passed": False,
                "issues": [f"Exception: {str(e)}"],
                "result": None,
            })
            failed_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(BENCHMARK_QUERIES)}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print()

    if failed_count == 0:
        print("🎉 ALL TESTS PASSED")
    else:
        print(f"⚠️  {failed_count} test(s) failed — review issues above")

    print("=" * 70 + "\n")

    return {
        "total": len(BENCHMARK_QUERIES),
        "passed": passed_count,
        "failed": failed_count,
        "results": results,
    }


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # Validate configuration first
        validate_config()
        
        # Run benchmark
        summary = run_benchmark()
        
        # Exit with appropriate code
        sys.exit(0 if summary["failed"] == 0 else 1)

    except EnvironmentError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OPENAI_API_KEY and PINECONE_API_KEY")
        print("  3. Run the ingestion pipeline first: python -m src.ingestion")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
