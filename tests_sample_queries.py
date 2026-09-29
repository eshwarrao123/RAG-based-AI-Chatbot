"""
Benchmark Test Queries for the Agentic AI RAG Chatbot.

Contains 6 benchmark queries that test:
    - In-domain factual retrieval (queries 1-5)
    - Out-of-domain refusal behavior (query 6)

Usage:
    python tests_sample_queries.py              # Run against live API
    python tests_sample_queries.py --direct     # Run directly (no server)
"""

import json
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Benchmark Query Definitions ───────────────────────────

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "question": "What is Agentic AI?",
        "type": "in_domain",
        "description": "Core concept — should retrieve foundational definition",
        "expected_min_confidence": 0.70,
        "expected_keywords": ["autonomous", "agent", "AI", "decision"],
    },
    {
        "id": 2,
        "question": "What are the key components of an AI agent?",
        "type": "in_domain",
        "description": "Factual — should enumerate agent components",
        "expected_min_confidence": 0.70,
        "expected_keywords": ["planning", "tool", "memory", "reasoning"],
    },
    {
        "id": 3,
        "question": "How do multi-agent systems communicate?",
        "type": "in_domain",
        "description": "Specific topic — should reference hierarchical and peer-to-peer architectures",
        "expected_min_confidence": 0.65,
        "expected_keywords": ["communication", "hierarchical", "peer"],
    },
    {
        "id": 4,
        "question": "What is the difference between LLMs and AI agents?",
        "type": "in_domain",
        "description": "Comparison — should draw from the LLM vs Agent comparison section",
        "expected_min_confidence": 0.70,
        "expected_keywords": ["LLM", "agent"],
    },
    {
        "id": 5,
        "question": "What are the risks and challenges of Agentic AI?",
        "type": "in_domain",
        "description": "Analytical — should draw from safety/ethics sections",
        "expected_min_confidence": 0.65,
        "expected_keywords": ["risk", "challenge", "safety", "ethical"],
    },
    {
        "id": 6,
        "question": "Who won the 2022 FIFA World Cup?",
        "type": "out_of_domain",
        "description": "OUT-OF-DOMAIN — must refuse. Tests grounding behavior.",
        "expected_max_confidence": 0.50,
        "expected_refusal": True,
        "forbidden_keywords": ["Argentina", "Messi", "France"],
    },
]


# ── Validation Logic ──────────────────────────────────────

def validate_in_domain(query: dict, result: dict) -> dict:
    """Validate an in-domain query result."""
    issues = []

    # Check confidence
    confidence = result.get("confidence_score", 0)
    min_conf = query.get("expected_min_confidence", 0.70)
    if confidence < min_conf:
        issues.append(f"Confidence {confidence:.4f} below minimum {min_conf}")

    # Check that context was retrieved
    context = result.get("context", [])
    if not context:
        issues.append("No context chunks retrieved")

    # Check expected keywords in answer
    answer = result.get("answer", "").lower()
    expected_keywords = query.get("expected_keywords", [])
    missing_keywords = [kw for kw in expected_keywords if kw.lower() not in answer]
    if missing_keywords:
        issues.append(f"Missing expected keywords: {missing_keywords}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def validate_out_of_domain(query: dict, result: dict) -> dict:
    """Validate an out-of-domain query result."""
    issues = []

    # Check confidence is LOW
    confidence = result.get("confidence_score", 1.0)
    max_conf = query.get("expected_max_confidence", 0.50)
    if confidence > max_conf:
        issues.append(f"Confidence {confidence:.4f} above maximum {max_conf}")

    # Check for refusal language
    answer = result.get("answer", "").lower()
    refusal_phrases = [
        "don't have enough information",
        "not available",
        "not found in the ebook",
        "cannot answer",
        "no information",
        "not mentioned",
        "does not contain",
        "outside the scope",
        "not covered",
    ]
    has_refusal = any(phrase in answer for phrase in refusal_phrases)
    if not has_refusal:
        issues.append("Answer does not contain refusal language")

    # Check forbidden keywords are NOT present
    forbidden = query.get("forbidden_keywords", [])
    found_forbidden = [kw for kw in forbidden if kw.lower() in answer]
    if found_forbidden:
        issues.append(f"Answer contains forbidden keywords: {found_forbidden}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def validate_result(query: dict, result: dict) -> dict:
    """Validate a single query result based on its type."""
    if query["type"] == "out_of_domain":
        return validate_out_of_domain(query, result)
    else:
        return validate_in_domain(query, result)


# ── Runners ───────────────────────────────────────────────

def run_via_api(base_url: str = "http://localhost:8000") -> list[dict]:
    """Run benchmark queries against the live FastAPI server."""
    import requests

    results = []
    for query in BENCHMARK_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query {query['id']}: {query['question']}")
        print(f"Type: {query['type']} — {query['description']}")
        print("-" * 60)

        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"question": query["question"]},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"❌ API Error: {e}")
            results.append({**query, "result": None, "error": str(e)})
            continue

        # Display result
        print(f"Confidence: {result['confidence_score']:.4f}")
        print(f"Context chunks: {len(result['context'])}")
        print(f"Answer: {result['answer'][:200]}...")

        # Validate
        validation = validate_result(query, result)
        status = "✅ PASSED" if validation["passed"] else "❌ FAILED"
        print(f"\nValidation: {status}")
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"  ⚠️  {issue}")

        results.append({
            **query,
            "result": result,
            "validation": validation,
        })

    return results


def run_direct() -> list[dict]:
    """Run benchmark queries directly using the graph (no server needed)."""
    from src.graph import query_rag

    results = []
    for query in BENCHMARK_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query {query['id']}: {query['question']}")
        print(f"Type: {query['type']} — {query['description']}")
        print("-" * 60)

        try:
            result = query_rag(query["question"])
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({**query, "result": None, "error": str(e)})
            continue

        # Display result
        print(f"Confidence: {result['confidence_score']:.4f}")
        print(f"Context chunks: {len(result['context'])}")
        print(f"Answer: {result['answer'][:200]}...")

        # Validate
        validation = validate_result(query, result)
        status = "✅ PASSED" if validation["passed"] else "❌ FAILED"
        print(f"\nValidation: {status}")
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"  ⚠️  {issue}")

        results.append({
            **query,
            "result": result,
            "validation": validation,
        })

    return results


# ── Summary Report ────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print a summary of all benchmark results."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("validation", {}).get("passed", False))
    failed = sum(1 for r in results if not r.get("validation", {}).get("passed", True))
    errors = sum(1 for r in results if r.get("error"))

    for r in results:
        status = "✅" if r.get("validation", {}).get("passed") else "❌"
        if r.get("error"):
            status = "💥"
        conf = r.get("result", {}).get("confidence_score", "N/A") if r.get("result") else "N/A"
        conf_str = f"{conf:.4f}" if isinstance(conf, float) else conf
        print(f"  {status} Q{r['id']}: [{r['type']:>13}] conf={conf_str}  {r['question'][:50]}")

    print(f"\nResults: {passed} passed, {failed} failed, {errors} errors out of {len(results)} queries")


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    mode = "--direct" if "--direct" in sys.argv else "--api"

    print(f"\n🧪 Agentic AI RAG Chatbot — Benchmark Tests")
    print(f"   Mode: {'Direct (no server)' if mode == '--direct' else 'API (http://localhost:8000)'}")
    print(f"   Queries: {len(BENCHMARK_QUERIES)}")

    if mode == "--direct":
        results = run_direct()
    else:
        results = run_via_api()

    print_summary(results)
