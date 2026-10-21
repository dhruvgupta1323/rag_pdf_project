
import json
import csv

from rag.pdf_handle import extract_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_chunks
from rag.vector_store import build_vector_store, search_chunks
from rag.custom_evaluator import (
    evaluate_retrieval,
    evaluate_answer,
)
from rag.generator import generate_answer
from rag.faithfulness import evaluate_faithfulness


# Configuration
PDF_PATH = "Finance_Bill.pdf"
DATASET_PATH = "evaluation_dataset.json"
REPORT_PATH = "rag_evaluation_report.csv"

TOP_K = 3
MIN_SCORE = 0.30


def main():
    # 1. Load the evaluation dataset.
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        print("Evaluation dataset is empty.")
        return

    # 2. Build the RAG retrieval pipeline.
    print("Loading PDF and building FAISS index...")

    pages = extract_pdf(PDF_PATH)
    chunks = chunk_pages(pages, chunk_size=400, overlap=50)
    embeddings = embed_chunks(chunks)
    index = build_vector_store(chunks, embeddings)

    report_rows = []

    print("\n========== CUSTOM RAG EVALUATION ==========\n")

    # 3. Evaluate each test question.
    for item in dataset:
        question = item["question"]
        answerable = item["answerable"]
        reference = item.get("reference_answer", "")
        relevant_pages = item.get("relevant_pages", [])

        print(f"\nQuestion ID: {item.get('id', 'Unknown')}")
        print(f"Question: {question}")

        # A. Retrieve relevant chunks.
        results = search_chunks(
            question,
            index,
            chunks,
            top_k=TOP_K,
            min_score=MIN_SCORE,
        )

        # B. Evaluate retrieval quality.
        retrieval_metrics = evaluate_retrieval(
            results,
            relevant_pages,
        )

        # C. Generate an answer.
        generation_error = ""

        if results:
            try:
                answer = generate_answer(question, results)
            except Exception as exc:
                answer = ""
                generation_error = str(exc)
        else:
            answer = (
                "I couldn't find sufficiently relevant information "
                "in the uploaded document to answer this question."
            )

        # D. Evaluate answer behavior.
        answer_metrics = evaluate_answer(
            answer,
            reference,
            answerable,
        )

        # E. Evaluate faithfulness only when evidence was retrieved.
        # An intentional refusal with no evidence is not scored as
        # an unfaithful answer.
        if results and answer and not generation_error:
            faithfulness_metrics = evaluate_faithfulness(
                answer,
                results,
            )
        else:
            faithfulness_metrics = {
                "score": None,
                "supported_claims": [],
                "unsupported_claims": [],
                "explanation": (
                    "Skipped because no usable answer and evidence "
                    "were available for evaluation."
                ),
                "error": None,
            }

        faithfulness_score = faithfulness_metrics["score"]

        unsupported_claims = json.dumps(
            faithfulness_metrics["unsupported_claims"],
            ensure_ascii=False,
        )

        supported_claims = json.dumps(
            faithfulness_metrics["supported_claims"],
            ensure_ascii=False,
        )

        faithfulness_error = (
            faithfulness_metrics.get("error") or ""
        )

        # F. Prepare one CSV row per question.
        row = {
            "id": item.get("id", ""),
            "question": question,
            "answerable": answerable,
            "reference_answer": reference,
            "retrieved_pages": json.dumps(
                retrieval_metrics["retrieved_pages"]
            ),
            "retrieved_chunks": retrieval_metrics["retrieved_count"],
            "retrieval_hit": retrieval_metrics["hit"],
            "retrieval_precision": retrieval_metrics["precision"],
            "retrieval_recall": retrieval_metrics["recall"],
            "generated_answer": answer,
            "answer_abstained": answer_metrics["abstained"],
            "abstention_correct": answer_metrics["abstention_correct"],
            "exact_match": answer_metrics["exact_match"],
            "faithfulness_score": faithfulness_score,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "faithfulness_explanation": faithfulness_metrics[
                "explanation"
            ],
            "faithfulness_error": faithfulness_error,
            "generation_error": generation_error,
        }

        report_rows.append(row)

        # G. Print detailed results.
        print(f"Retrieved pages: {retrieval_metrics['retrieved_pages']}")
        print(f"Retrieval hit: {retrieval_metrics['hit']}")
        print(
            "Retrieval precision: "
            f"{retrieval_metrics['precision']:.2f}"
        )
        print(f"Generated answer: {answer}")

        if faithfulness_score is not None:
            print(f"Faithfulness score: {faithfulness_score:.1%}")
        else:
            print("Faithfulness score: N/A")

        print("Unsupported claims:", unsupported_claims)

        if generation_error:
            print("Generation error:", generation_error)

        if faithfulness_error:
            print("Faithfulness evaluation error:", faithfulness_error)

        print("-" * 60)

    # 4. Calculate aggregate metrics.
    answerable_rows = [
        row for row in report_rows if row["answerable"]
    ]

    unanswerable_rows = [
        row for row in report_rows if not row["answerable"]
    ]

    hit_rate = (
        sum(row["retrieval_hit"] for row in answerable_rows)
        / len(answerable_rows)
        if answerable_rows else None
    )

    precision_values = [
        row["retrieval_precision"]
        for row in answerable_rows
        if row["retrieved_chunks"] > 0
    ]

    mean_precision = (
        sum(precision_values) / len(precision_values)
        if precision_values else None
    )

    rejection_rate = (
        sum(
            row["abstention_correct"]
            for row in unanswerable_rows
        ) / len(unanswerable_rows)
        if unanswerable_rows else None
    )

    exact_match_values = [
        row["exact_match"]
        for row in answerable_rows
        if row["exact_match"] is not None
    ]

    exact_match_rate = (
        sum(exact_match_values) / len(exact_match_values)
        if exact_match_values else None
    )

    faithfulness_values = [
        row["faithfulness_score"]
        for row in report_rows
        if row["faithfulness_score"] is not None
    ]

    mean_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values)
        if faithfulness_values else None
    )

    # 5. Save detailed results to CSV.
    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=report_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(report_rows)

    # 6. Print the summary.
    print("\n========== EVALUATION SUMMARY ==========")

    def show_metric(label, value):
        if value is None:
            print(f"{label}: N/A")
        else:
            print(f"{label}: {value:.1%}")

    show_metric("Answerable-question hit rate", hit_rate)
    show_metric("Mean retrieval precision", mean_precision)
    show_metric("Unanswerable rejection rate", rejection_rate)
    show_metric("Reference exact-match rate", exact_match_rate)
    show_metric("Mean faithfulness score", mean_faithfulness)

    print(f"\nDetailed report saved to: {REPORT_PATH}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
