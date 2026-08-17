from __future__ import annotations

import json

import pytest

from ah_memory.benchmarks.challenge_evaluation import (
    InferenceBenchmarkItem,
    InferenceObservation,
    RoleBenchmarkItem,
    run_m1_benchmark,
    run_m2_benchmark,
)
from ah_memory.benchmarks.challenge_metrics import (
    InferenceEvaluation,
    RoleActant,
    RoleCorpusItem,
    comparison_score,
    evaluate_role_corpus,
    explain_score,
    gc_efficiency,
    hallucination_rate,
    robustness_gain,
    role_extraction_score,
    total_score,
)
from ah_memory.benchmarks.challenge_report import write_challenge_report
from ah_memory.gc import collect, orphan_uids
from ah_memory.hyperparams import HyperParams
from ah_memory.perception import JsonLLMPerception
from ah_memory.store import AHStore
from ah_memory.types import (
    AbstractSymbol,
    AssocLink,
    LinkId,
    SecondOrderSymbol,
    Section,
)


def test_m1_role_precision_recall_and_weighted_f1() -> None:
    expected = [
        RoleActant("first", "SUBJECT", "A"),
        RoleActant("first", "OBJECT", "B"),
        RoleActant("first", "LOCATION", "C"),
        RoleActant("second", "SUBJECT", "D"),
        RoleActant("second", "OBJECT", "E"),
        RoleActant("second", "LOCATION", "F"),
    ]
    predicted = [
        RoleActant("first", "SUBJECT", "A"),
        RoleActant("first", "OBJECT", "B"),
        RoleActant("first", "LOCATION", "C"),
        RoleActant("second", "SUBJECT", "WRONG"),
        RoleActant("second", "OBJECT", "E"),
        RoleActant("extra", "OBJECT", "X"),
    ]

    score = role_extraction_score(predicted, expected)

    assert score.by_role["SUBJECT"].precision == pytest.approx(0.5)
    assert score.by_role["SUBJECT"].recall == pytest.approx(0.5)
    assert score.by_role["SUBJECT"].f1 == pytest.approx(0.5)
    assert score.by_role["OBJECT"].precision == pytest.approx(2 / 3)
    assert score.by_role["OBJECT"].recall == pytest.approx(1.0)
    assert score.by_role["OBJECT"].f1 == pytest.approx(0.8)
    assert score.by_role["LOCATION"].precision == pytest.approx(1.0)
    assert score.by_role["LOCATION"].recall == pytest.approx(0.5)
    assert score.by_role["LOCATION"].f1 == pytest.approx(2 / 3)
    assert score.weighted_f1 == pytest.approx(2 * 0.5 + 2 * 0.8 + 2 / 3)
    assert score.normalized_weighted_f1 == pytest.approx(
        (2 * 0.5 + 2 * 0.8 + 2 / 3) / 5
    )


def test_m1_scores_surface_gold_against_slugged_predictions() -> None:
    score = role_extraction_score(
        [RoleActant("one", "SUBJECT", "DR_VALE")],
        [RoleActant("one", "SUBJECT", "Dr. Vale")],
    )
    assert score.by_role["SUBJECT"].f1 == 1.0


def test_m1_always_reports_all_mandatory_roles() -> None:
    score = role_extraction_score([], [])
    assert {"SUBJECT", "OBJECT", "LOCATION"}.issubset(score.by_role)
    assert score.weighted_f1 == 0.0


def test_m1_runs_noisy_corpus_through_perception_backend() -> None:
    outputs = {
        "Actro moved Item into Place.": ("Actro", "moved"),
        "Into Place Item moved Actor.": ("Actor", "moved"),
        "Actor: Item — Place.": ("Actor", ":"),
    }

    def call_fn(prompt: str) -> str:
        text = json.loads(prompt)["text"]
        subject, relation = outputs[text]
        return json.dumps(
            {
                "kind": "fact",
                "candidates": [
                    {
                        "raw_relation": relation,
                        "canonical_relation": "RELATE",
                        "predicate": "RELATE",
                        "roles": {
                            "SUBJECT": subject,
                            "OBJECT": "Item",
                            "LOCATION": "Place",
                        },
                        "raw_span": text,
                        "confidence": 1.0,
                        "statement_type": "assertion",
                    }
                ],
            }
        )

    corpus = [
        RoleCorpusItem(
            item_id=text,
            text=text,
            expected_roles=(
                {
                    "SUBJECT": subject.upper(),
                    "OBJECT": "ITEM",
                    "LOCATION": "PLACE",
                },
            ),
        )
        for text, (subject, _) in outputs.items()
    ]
    score = evaluate_role_corpus(JsonLLMPerception(call_fn), corpus)
    assert score.normalized_weighted_f1 == 1.0
    assert all(score.by_role[role].f1 == 1.0 for role in score.by_role)


def test_m2_twenty_questions_are_depth_weighted_and_trace_gated() -> None:
    evaluations = [
        InferenceEvaluation(
            correct=True,
            depth=(index % 6) + 1,
            expected_trace=(f"F_{index}_A", f"F_{index}_B"),
            actual_trace=(f"F_{index}_A", f"F_{index}_B"),
        )
        for index in range(20)
    ]
    evaluations[5] = InferenceEvaluation(
        correct=True,
        depth=6,
        expected_trace=("F_5_A", "F_5_B"),
        actual_trace=("F_5_A",),
    )

    expected = (
        sum(((index % 6) + 1) / 6 for index in range(20)) - 1.0
    ) / 20
    assert explain_score(evaluations, d_max=6) == pytest.approx(expected)


def test_m2_graph_bypass_never_scores_even_when_answer_is_correct() -> None:
    guessed = InferenceEvaluation(
        correct=True,
        depth=6,
        expected_trace=("EDGE_1", "EDGE_2"),
        actual_trace=(),
    )
    assert explain_score([guessed], d_max=6) == 0.0


def test_m2_complete_chain_allows_unrelated_trace_factors() -> None:
    evaluation = InferenceEvaluation(
        correct=True,
        depth=3,
        expected_trace=("EDGE_1", "EDGE_2", "EDGE_3"),
        actual_trace=("NOISE_A", "EDGE_1", "NOISE_B", "EDGE_2", "EDGE_3"),
    )
    assert evaluation.trace_complete
    assert explain_score([evaluation], d_max=6) == pytest.approx(0.5)


def test_m3_removes_200_orphans_within_50_ticks_and_keeps_live_nodes() -> None:
    store = AHStore()
    store.add_abstract_symbol(AbstractSymbol(uid="LIVE"))
    store.add_element(Section.C, SecondOrderSymbol(uid="M_LIVE"))
    store.add_link(
        AssocLink(
            uid="LIVE_LINK",
            id=LinkId.ASSOC.value,
            w=1.0,
            e1=store.s_ref("LIVE"),
            e2=store.m_ref("M_LIVE"),
        )
    )
    expected_orphans = {f"ORPHAN_{index:03d}" for index in range(200)}
    for uid in expected_orphans:
        store.add_element(Section.C, SecondOrderSymbol(uid=uid))

    before = orphan_uids(store)
    assert before == expected_orphans

    store.ah.tau = 50
    report = collect(store, HyperParams(ttl=50))
    after = orphan_uids(store)

    assert expected_orphans.issubset(report.removed_elements)
    assert gc_efficiency(len(before), len(after)) == 1.0
    assert not after
    assert "LIVE" in store.ah.S
    assert "M_LIVE" in store.ah.C
    assert "LIVE_LINK" in store.ah.L


def test_m3_removes_nonzero_weight_component_detached_from_s() -> None:
    store = AHStore()
    store.add_element(Section.C, SecondOrderSymbol(uid="DETACHED_A"))
    store.add_element(Section.C, SecondOrderSymbol(uid="DETACHED_B"))
    store.add_link(
        AssocLink(
            uid="DETACHED_LINK",
            id=LinkId.ASSOC.value,
            w=1.0,
            e1=store.m_ref("DETACHED_A"),
            e2=store.m_ref("DETACHED_B"),
        )
    )
    assert orphan_uids(store) == {"DETACHED_A", "DETACHED_B"}
    store.ah.tau = 50
    collect(store, HyperParams(ttl=50))
    assert not store.ah.all_hyper()
    assert not store.ah.L


def test_m3_rejects_inconsistent_orphan_counts() -> None:
    with pytest.raises(ValueError):
        gc_efficiency(10, 11)


def test_m4_computes_explainability_and_hallucination_deltas() -> None:
    ah_hallucination = hallucination_rate([True, True, False, True])
    rag_hallucination = hallucination_rate([True, False, False, False])
    score = comparison_score(
        ah_explain_score=0.72,
        rag_explain_score=0.18,
        ah_hallucination=ah_hallucination,
        rag_hallucination=rag_hallucination,
    )
    assert score.delta_explainability == pytest.approx(0.54)
    assert score.delta_hallucination == pytest.approx(0.5)


def test_m5_robustness_gain_uses_both_model_classes() -> None:
    gain = robustness_gain(
        ah_slm_f1=0.72,
        rag_slm_f1=0.48,
        ah_llm_f1=0.90,
        rag_llm_f1=0.75,
    )
    assert gain == pytest.approx(0.3)


def test_m5_requires_nonzero_vanilla_baselines() -> None:
    with pytest.raises(ValueError):
        robustness_gain(
            ah_slm_f1=0.5,
            rag_slm_f1=0.0,
            ah_llm_f1=0.8,
            rag_llm_f1=0.7,
        )


def test_total_score_uses_published_weights() -> None:
    assert total_score(m1=1, m2=2, m3=3, m4=4, m5=5) == pytest.approx(2.8)


def test_m1_runner_logs_validation_failures_without_aborting_corpus() -> None:
    class Backend:
        def parse(self, text: str, wm_context=None):
            if text == "broken":
                raise ValueError("invalid response")
            return JsonLLMPerception(
                lambda _: {
                    "kind": "fact",
                    "candidates": [
                        {
                            "predicate": "RELATE",
                            "roles": {"SUBJECT": "Actor", "OBJECT": "Item"},
                            "raw_span": text,
                        }
                    ],
                }
            ).parse(text)

    report = run_m1_benchmark(
        Backend(),
        [
            RoleBenchmarkItem(
                "ok",
                "Actor Item",
                ({"SUBJECT": "ACTOR", "OBJECT": "ITEM"},),
            ),
            RoleBenchmarkItem(
                "bad",
                "broken",
                ({"SUBJECT": "OTHER", "OBJECT": "VALUE"},),
            ),
        ],
        model="fake",
    )
    assert len(report.items) == 2
    assert report.items[1].error.startswith("ValueError:")
    assert report.score.by_role["SUBJECT"].recall == pytest.approx(0.5)


def test_m2_runner_uses_actual_answers_and_ordered_traces() -> None:
    corpus = [
        InferenceBenchmarkItem(
            "q1", "Question one?", "Answer", 2, ("L::A", "L::B")
        ),
        InferenceBenchmarkItem(
            "q2", "Question two?", "Other", 6, ("L::C", "L::D")
        ),
    ]

    def answer(item: InferenceBenchmarkItem) -> InferenceObservation:
        if item.item_id == "q1":
            return InferenceObservation(
                " answer ", ("NOISE", "L::A", "L::B"), {"source": "graph"}
            )
        return InferenceObservation("Other", ("L::D", "L::C"))

    report = run_m2_benchmark(corpus, answer, d_max=6)
    assert report.explain_score == pytest.approx((2 / 6) / 2)
    assert report.items[0].trace_complete
    assert not report.items[1].trace_complete


def test_challenge_report_writes_jsonl_and_redacts_secrets(tmp_path) -> None:
    output = write_challenge_report(
        {"score": 0.5, "credentials": "secret"},
        {
            "m1/items": [
                {"item": "one", "nested": {"access_token": "secret-token"}}
            ]
        },
        root=tmp_path,
        run_id="fixed-run",
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    row = json.loads(
        (output / "m1_items.jsonl").read_text(encoding="utf-8").strip()
    )
    assert summary["credentials"] == "[REDACTED]"
    assert row["nested"]["access_token"] == "[REDACTED]"
