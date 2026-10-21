
from rag.faithfulness import evaluate_faithfulness

results = [
    {
        "score": 0.62,
        "chunk": {
            "source": "Finance_Bill.pdf",
            "page": 9,
            "text": (
                "The Bill gives effect to the financial proposals "
                "of the Central Government for the financial year "
                "2026-2027."
            ),
        },
    }
]

# Test A: Supported answer
answer_a = (
    "The financial year is 2026-2027. [PDF page 9]"
)

# Test B: Unsupported answer
answer_b = (
    "The financial year is 2026-2027. "
    "The document says Japan has 150 million people."
)

for label, answer in [
    ("SUPPORTED ANSWER", answer_a),
    ("UNSUPPORTED ANSWER", answer_b),
]:
    print(f"\n===== {label} =====")

    evaluation = evaluate_faithfulness(answer, results)

    print(f"Faithfulness score: {evaluation['score']}")
    print("Supported claims:", evaluation["supported_claims"])
    print("Unsupported claims:", evaluation["unsupported_claims"])
    print("Explanation:", evaluation["explanation"])

    if evaluation["error"]:
        print("Evaluation error:", evaluation["error"])
