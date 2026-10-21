
import re


def normalize_text(text):
    """Normalize text for basic answer comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def evaluate_retrieval(results, relevant_pages):
    """Evaluate retrieved chunks against labelled PDF pages."""
    relevant_pages = set(relevant_pages)

    retrieved_pages = [
        result["chunk"].get("page")
        for result in results
    ]

    relevant_hits = sum(
        page in relevant_pages
        for page in retrieved_pages
    )

    hit = any(
        page in relevant_pages
        for page in retrieved_pages
    )

    precision = (
        relevant_hits / len(retrieved_pages)
        if retrieved_pages else 0.0
    )

    recall = (
        len(set(retrieved_pages) & relevant_pages)
        / len(relevant_pages)
        if relevant_pages else None
    )

    return {
        "retrieved_pages": retrieved_pages,
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "retrieved_count": len(results),
    }


def evaluate_answer(answer, reference_answer, answerable):
    """Perform basic answer matching and refusal checks."""
    normalized_answer = normalize_text(answer or "")

    refusal_phrases = [
        "couldn't find sufficiently relevant information",
        "do not contain enough information",
        "does not contain enough information",
        "not enough evidence",
        "cannot answer this question",
        "can't answer this question",
    ]

    abstained = any(
        phrase in normalized_answer
        for phrase in refusal_phrases
    )

    if not answerable:
        return {
            "abstained": abstained,
            "expected_abstention": True,
            "abstention_correct": abstained,
            "exact_match": None,
        }

    normalized_reference = normalize_text(
        reference_answer or ""
    )

    exact_match = (
        normalized_reference in normalized_answer
        if normalized_reference else False
    )

    return {
        "abstained": abstained,
        "expected_abstention": False,
        "abstention_correct": not abstained,
        "exact_match": exact_match,
    }
