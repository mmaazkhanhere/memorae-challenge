"""Opt-in real-Gemini retrieval calibration dataset.

Only labels, scores, and aggregate metrics are printed. Fact/query contents,
provider responses, and stored memory values are never logged.
"""

from __future__ import annotations

import json
import math
import os
import time
from statistics import mean, median

import pytest

from reminder_agent.embeddings import GeminiEmbeddingProvider
from reminder_agent.memories import _lookup_tokens
from reminder_agent.settings import Settings


DOCUMENTS = {
    "parking_b17": "my parking slot is B17",
    "parking_c12": "my parking slot is C12",
    "food_biryani": "my favorite food is biryani",
    "office_lahore": "my office location is Lahore",
    "office_karachi": "my office location is Karachi",
    "color_blue": "my favorite color is blue",
    "contact_sam": "my emergency contact is Sam",
    "pet_milo": "my dog's name is Milo",
}

# Five paraphrases per non-conflicting subject: 30 positives total.
POSITIVE_CASES = (
    ("parking_b17", "Where do I park?", "parking"),
    ("parking_b17", "Where do I leave my car?", "leave my car"),
    ("parking_b17", "Where is my parking slot?", "parking slot"),
    ("parking_b17", "Which slot is mine?", "slot"),
    ("parking_b17", "What is my parking spot?", "parking spot"),
    ("food_biryani", "Which food do I like most?", "food like most"),
    ("food_biryani", "What food is my favorite?", "food favorite"),
    ("food_biryani", "What do I enjoy eating?", "enjoy eating"),
    ("food_biryani", "Which dish do I prefer?", "dish prefer"),
    ("food_biryani", "What meal do I like?", "meal like"),
    ("office_lahore", "Where is my workplace?", "workplace"),
    ("office_lahore", "Where do I work?", "work"),
    ("office_lahore", "What city is my office in?", "city office"),
    ("office_lahore", "Where is my office located?", "office located"),
    ("office_lahore", "Which city do I work from?", "city work"),
    ("color_blue", "Which color do I like?", "color like"),
    ("color_blue", "What is my favorite color?", "favorite color"),
    ("color_blue", "Which color is my preferred one?", "color preferred"),
    ("color_blue", "What color do I prefer?", "color prefer"),
    ("color_blue", "Tell me my preferred color", "preferred color"),
    ("contact_sam", "Who is my emergency contact?", "emergency contact"),
    ("contact_sam", "Who should I call in an emergency?", "call emergency"),
    ("contact_sam", "What is my emergency contact name?", "emergency contact name"),
    ("contact_sam", "Who can I contact for emergencies?", "contact emergencies"),
    ("contact_sam", "Which person is my emergency contact?", "person emergency contact"),
    ("pet_milo", "What is my dog's name?", "dog name"),
    ("pet_milo", "What did I name my dog?", "name dog"),
    ("pet_milo", "What is my pet called?", "pet called"),
    ("pet_milo", "Tell me my dog's name", "dog name"),
    ("pet_milo", "Which name belongs to my dog?", "name dog"),
)

# Five unrelated queries per broad subject: 30 negatives total.
NEGATIVE_QUERIES = (
    "What is my favorite movie?",
    "What is my favorite song?",
    "Which book do I like?",
    "What is my favorite sport?",
    "Which game do I prefer?",
    "What is my shoe size?",
    "What is my jacket size?",
    "What is my car model?",
    "What is my phone model?",
    "What is my laptop model?",
    "What city do I live in?",
    "What is my home address?",
    "Where is my house?",
    "What is my neighborhood?",
    "Which country am I from?",
    "When is my next appointment?",
    "What time is my meeting?",
    "What is on my calendar?",
    "When is my next trip?",
    "What is my schedule?",
    "What language do I speak?",
    "Which school did I attend?",
    "What is my job title?",
    "Which company do I work for?",
    "What is my hobby?",
    "What is my preferred holiday?",
    "Which restaurant do I like?",
    "What drink do I prefer?",
    "What is my favorite season?",
    "Which beach do I visit?",
)

CONFLICT_CASES = (
    ("parking_conflict", "Where is my parking slot?"),
    ("parking_conflict", "Where do I park?"),
    ("office_conflict", "What city is my office in?"),
    ("office_conflict", "Where do I work?"),
)


def _api_key() -> str:
    return (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or (Settings().embedding_api_key or "")
    ).strip()


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": round(min(values), 6) if values else None,
        "median": round(median(values), 6) if values else None,
        "mean": round(mean(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def _classify(
    ranked: list[tuple[str, float]], *, threshold: float, margin: float
) -> tuple[str, str | None, list[str]]:
    relevant = [(label, score) for label, score in ranked if score >= threshold]
    if not relevant:
        return "UNKNOWN", None, []
    best_score = relevant[0][1]
    plausible = [label for label, score in relevant if best_score - score <= margin]
    if len(plausible) != 1:
        return "AMBIGUOUS", None, plausible
    return "FOUND", plausible[0], plausible


def _lexical_ranked(subject: str) -> list[tuple[str, float]]:
    """Mirror the production lexical gate without exposing fact contents."""

    query_token_list = _lookup_tokens(subject)
    query_tokens = set(query_token_list)
    if not query_tokens:
        return []
    ranked: list[tuple[str, float]] = []
    for label, content in DOCUMENTS.items():
        content_tokens = set(_lookup_tokens(content))
        if not query_tokens.issubset(content_tokens):
            continue
        # Production requires every query token. The score only makes the
        # candidate ordering deterministic for Recall@k reporting.
        phrase_bonus = 1.0 if " ".join(query_token_list) in content.casefold() else 0.0
        ranked.append((label, 1.0 + phrase_bonus))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _recall_at_k(
    cases: list[tuple[str, list[tuple[str, float]]]], k: int
) -> int:
    return sum(expected in {label for label, _ in ranked[:k]} for expected, ranked in cases)


def _failure_buckets(
    *,
    expected: str,
    ranked: list[tuple[str, float]],
    threshold: float,
    margin: float,
) -> str:
    if expected not in {label for label, _ in ranked[:5]}:
        return "candidate_retrieval_miss"
    status, selected, _ = _classify(ranked, threshold=threshold, margin=margin)
    if status == "AMBIGUOUS":
        return "ambiguity"
    if status == "UNKNOWN":
        return "threshold_rejection"
    if selected != expected:
        return "wrong_candidate_selected"
    return "correct"


def _threshold_metrics(
    *,
    positive_rankings: list[tuple[str, list[tuple[str, float]]]],
    negative_rankings: list[list[tuple[str, float]]],
    threshold: float,
    margin: float,
) -> dict[str, int | float]:
    accepted_correct = accepted = accepted_wrong = clarifications = 0
    for expected, ranked in positive_rankings:
        status, selected, _ = _classify(ranked, threshold=threshold, margin=margin)
        accepted += status == "FOUND"
        accepted_correct += status == "FOUND" and selected == expected
        accepted_wrong += status == "FOUND" and selected != expected
        clarifications += status == "AMBIGUOUS"
    false_positives = 0
    negative_clarifications = 0
    for ranked in negative_rankings:
        status, _, _ = _classify(ranked, threshold=threshold, margin=margin)
        false_positives += status == "FOUND"
        negative_clarifications += status == "AMBIGUOUS"
    return {
        "threshold": threshold,
        "accepted_correct": accepted_correct,
        "accepted": accepted,
        "accepted_wrong": accepted_wrong,
        "false_negatives": len(positive_rankings) - accepted_correct,
        "positive_clarifications": clarifications,
        "false_positives": false_positives,
        "negative_clarifications": negative_clarifications,
    }


def test_live_gemini_retrieval_evaluation_dataset() -> None:
    if os.getenv("RUN_LIVE_GEMINI") != "1":
        pytest.skip("set RUN_LIVE_GEMINI=1 to enable the network evaluation")
    api_key = _api_key()
    if not api_key:
        pytest.skip("a Gemini API key is not configured")

    settings = Settings()
    provider = GeminiEmbeddingProvider(
        base_url=settings.embedding_gemini_base_url,
        api_key=api_key,
        model=settings.embedding_gemini_model,
        dimensions=settings.embedding_dimensions,
    )
    delay = float(os.getenv("GEMINI_EVAL_DELAY_SECONDS", "0.65"))

    def embed_document(text: str) -> list[float]:
        vector = provider.embed_document(text)
        time.sleep(delay)
        return vector

    def embed_query(text: str) -> list[float]:
        vector = provider.embed_query(text)
        time.sleep(delay)
        return vector

    document_vectors = {label: embed_document(text) for label, text in DOCUMENTS.items()}
    assert all(len(vector) == 1536 for vector in document_vectors.values())
    assert all(
        math.isclose(_cosine(vector, vector), 1.0, rel_tol=1e-5)
        for vector in document_vectors.values()
    )

    full_queries = {query: embed_query(query) for _, query, _ in POSITIVE_CASES}
    subject_queries = {subject: embed_query(subject) for _, _, subject in POSITIVE_CASES}
    negative_queries = {query: embed_query(query) for query in NEGATIVE_QUERIES}
    conflict_queries = {query: embed_query(query) for _, query in CONFLICT_CASES}

    threshold = settings.memory_semantic_relevance_threshold
    margin = settings.memory_semantic_ambiguity_margin
    positive_metrics: dict[str, dict[str, int]] = {}
    positive_distributions: dict[str, list[float]] = {"full": [], "subject": []}
    for representation, vectors in (("full", full_queries), ("subject", subject_queries)):
        candidate_recall_at_1 = accepted = accepted_correct = clarifications = 0
        for expected_label, query, subject in POSITIVE_CASES:
            query_vector = vectors[query if representation == "full" else subject]
            ranked = sorted(
                ((label, _cosine(query_vector, vector)) for label, vector in document_vectors.items()),
                key=lambda item: (-item[1], item[0]),
            )
            status, selected, _ = _classify(ranked, threshold=threshold, margin=margin)
            positive_distributions[representation].append(ranked[0][1])
            candidate_recall_at_1 += ranked[0][0] == expected_label
            accepted += status == "FOUND"
            accepted_correct += status == "FOUND" and selected == expected_label
            clarifications += status == "AMBIGUOUS"
        positive_metrics[representation] = {
            "candidate_recall_at_1": candidate_recall_at_1,
            "accepted": accepted,
            "accepted_correct": accepted_correct,
            "accepted_wrong": accepted - accepted_correct,
            "relevance_threshold_acceptance": accepted,
            "final_answer_correct": accepted_correct,
            "final_answer_accuracy": round(accepted_correct / len(POSITIVE_CASES), 4),
            "clarifications": clarifications,
            "clarification_rate": round(clarifications / len(POSITIVE_CASES), 4),
            "false_negatives": len(POSITIVE_CASES) - accepted_correct,
        }

    full_vector_rankings = [
        (
            expected_label,
            sorted(
                (
                    (label, _cosine(full_queries[query], vector))
                    for label, vector in document_vectors.items()
                ),
                key=lambda item: (-item[1], item[0]),
            ),
        )
        for expected_label, query, _ in POSITIVE_CASES
    ]
    lexical_rankings = [
        (expected_label, _lexical_ranked(subject))
        for expected_label, _, subject in POSITIVE_CASES
    ]
    combined_rankings = []
    for (expected_label, lexical_ranked), (_, vector_ranked) in zip(
        lexical_rankings, full_vector_rankings, strict=True
    ):
        combined_rankings.append(
            (expected_label, lexical_ranked if lexical_ranked else vector_ranked)
        )
    candidate_metrics = {}
    for name, rankings in (
        ("lexical_only", lexical_rankings),
        ("vector_only", full_vector_rankings),
        ("combined", combined_rankings),
    ):
        candidate_metrics[name] = {
            "recall_at_1": _recall_at_k(rankings, 1),
            "recall_at_3": _recall_at_k(rankings, 3),
            "recall_at_5": _recall_at_k(rankings, 5),
            "candidate_miss_count": sum(
                expected not in {label for label, _ in ranked[:5]}
                for expected, ranked in rankings
            ),
        }

    failure_buckets_by_strategy: dict[str, dict[str, int]] = {}
    for strategy, rankings in (
        ("vector_only", full_vector_rankings),
        ("combined", combined_rankings),
    ):
        buckets: dict[str, int] = {}
        for expected_label, ranked in rankings:
            bucket = _failure_buckets(
                expected=expected_label,
                ranked=ranked,
                threshold=threshold,
                margin=margin,
            )
            buckets[bucket] = buckets.get(bucket, 0) + 1
        failure_buckets_by_strategy[strategy] = buckets

    negative_status_counts: dict[str, int] = {}
    negative_scores: list[float] = []
    negative_rankings: list[list[tuple[str, float]]] = []
    for query, query_vector in negative_queries.items():
        ranked = sorted(
            ((label, _cosine(query_vector, vector)) for label, vector in document_vectors.items()),
            key=lambda item: (-item[1], item[0]),
        )
        status, _, _ = _classify(ranked, threshold=threshold, margin=margin)
        negative_scores.append(ranked[0][1])
        negative_rankings.append(ranked)
        negative_status_counts[status] = negative_status_counts.get(status, 0) + 1

    conflict_results = []
    for conflict_label, query in CONFLICT_CASES:
        ranked = sorted(
            ((label, _cosine(conflict_queries[query], vector)) for label, vector in document_vectors.items()),
            key=lambda item: (-item[1], item[0]),
        )
        status, _, plausible = _classify(ranked, threshold=threshold, margin=margin)
        conflict_results.append(
            {
                "case": conflict_label,
                "status": status,
                "top_labels": [label for label, _ in ranked[:2]],
                "top_scores": [round(score, 6) for _, score in ranked[:2]],
                "delta": round(ranked[0][1] - ranked[1][1], 6),
                "plausible_count": len(plausible),
            }
        )

    equivalent_scores = {
        "full_where_do_i_park": round(
            _cosine(full_queries["Where do I park?"], document_vectors["parking_b17"]), 6
        ),
        "full_where_do_i_leave_my_car": round(
            _cosine(full_queries["Where do I leave my car?"], document_vectors["parking_b17"]), 6
        ),
        "subject_parking": round(
            _cosine(subject_queries["parking"], document_vectors["parking_b17"]), 6
        ),
        "subject_leave_my_car": round(
            _cosine(subject_queries["leave my car"], document_vectors["parking_b17"]), 6
        ),
    }
    threshold_sweeps = {}
    for representation, vectors in (("full", full_queries), ("subject", subject_queries)):
        positive_rankings = []
        for expected_label, query, subject in POSITIVE_CASES:
            query_vector = vectors[query if representation == "full" else subject]
            ranked = sorted(
                ((label, _cosine(query_vector, vector)) for label, vector in document_vectors.items()),
                key=lambda item: (-item[1], item[0]),
            )
            positive_rankings.append((expected_label, ranked))
        threshold_sweeps[representation] = [
            _threshold_metrics(
                positive_rankings=positive_rankings,
                negative_rankings=negative_rankings,
                threshold=level,
                margin=margin,
            )
            for level in (0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78)
        ]
    summary = {
        "model": provider.model,
        "dimensions": provider.dimensions,
        "threshold": threshold,
        "ambiguity_margin": margin,
        "positive_score_distributions": {
            representation: _distribution(scores)
            for representation, scores in positive_distributions.items()
        },
        "positive_metrics": positive_metrics,
        "candidate_metrics": candidate_metrics,
        "failure_buckets_by_strategy": failure_buckets_by_strategy,
        "negative_top_score_distribution": _distribution(negative_scores),
        "negative_status_counts": negative_status_counts,
        "negative_false_positives": negative_status_counts.get("FOUND", 0),
        "negative_clarifications": negative_status_counts.get("AMBIGUOUS", 0),
        "conflict_results": conflict_results,
        "equivalent_query_scores": equivalent_scores,
        "threshold_sweeps": threshold_sweeps,
        "full_vs_subject_delta_leave_my_car": round(
            equivalent_scores["full_where_do_i_leave_my_car"]
            - equivalent_scores["subject_leave_my_car"],
            6,
        ),
    }
    print(json.dumps(summary, sort_keys=True))
