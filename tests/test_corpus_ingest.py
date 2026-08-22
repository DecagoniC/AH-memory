from ah_memory.agent import Agent
from ah_memory.eval.corpus_ingest import (
    ingest_text_batches,
    load_fact_payload,
    records_to_payload,
    split_atomic_segments,
    split_batches,
)
from ah_memory.examples.closed_world import closed_world_facts_path
from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore


def test_split_batches_skips_source_header_and_short_titles() -> None:
    text = (
        "Источник: русская Википедия, статья «Тиманский кряж».\n\n"
        "География\n\n"
        "Тиманский кряж — крупная возвышенность на северо-востоке "
        "Восточно-Европейской равнины. Тянется на девятьсот километров.\n\n"
        "Высшая точка — Четласский Камень. Высота достигает четырёхсот "
        "семидесяти одного метра."
    )
    batches = split_batches(text, max_chars=200)
    assert batches
    blob = "\n".join(batches)
    assert "Источник:" not in blob
    assert "Тиманский кряж" in blob
    assert "Четласский" in blob


def test_split_atomic_segments_expands_top_level_enumeration() -> None:
    segments = split_atomic_segments(
        "Объект содержит: альфа, бета (в месте), гамма и дельта."
    )

    assert [segment.text for segment in segments] == [
        "альфа",
        "бета (в месте)",
        "гамма",
        "дельта",
    ]
    assert all("Объект содержит:" in segment.context for segment in segments)


def test_ingest_repairs_each_uncovered_list_item() -> None:
    batch = "Объект содержит: альфа, бета, гамма."

    class PartialPerception:
        def parse(self, text: str, wm_context=None) -> PerceptionResult:
            if text == batch:
                values = ["альфа"]
            else:
                values = [
                    value
                    for value in ("бета", "гамма")
                    if value in text
                ]
            candidates = [
                FactCandidate(
                    predicate="CONTAINS",
                    canonical_relation="CONTAINS",
                    raw_relation="содержит",
                    raw_span=value,
                    roles={"SUBJECT": "ОБЪЕКТ", "OBJECT": value},
                )
                for value in values
            ]
            return PerceptionResult(
                kind="fact",
                candidates=candidates,
                seed_tokens=[],
                meta={"backend": "partial"},
            )

    agent = Agent(store=AHStore(), perception=PartialPerception())
    records = ingest_text_batches(agent, [batch])
    payload = records_to_payload(records, source="test")

    assert records[0].repair_attempts == 2
    assert records[0].coverage_ratio == 1.0
    assert records[0].uncovered_segments == []
    assert payload["coverage"] == {
        "segments": 3,
        "covered": 3,
        "ratio": 1.0,
        "uncovered": [],
    }
    assert len(agent.store.list_semantic_factors()) == 3


def test_list_structure_infers_one_residual_item_after_repair() -> None:
    batch = "Объект содержит: альфа, бета, гамма."

    class StubbornPerception:
        def parse(self, text: str, wm_context=None) -> PerceptionResult:
            values = ["альфа", "бета"] if text == batch else []
            return PerceptionResult(
                kind="fact",
                candidates=[
                    FactCandidate(
                        predicate="CONTAINS",
                        canonical_relation="CONTAINS",
                        raw_relation="содержит",
                        raw_span=value,
                        roles={"SUBJECT": "ОБЪЕКТ", "OBJECT": value},
                    )
                    for value in values
                ],
            )

    agent = Agent(store=AHStore(), perception=StubbornPerception())
    record = ingest_text_batches(agent, [batch])[0]

    assert record.coverage_ratio == 1.0
    assert record.uncovered_segments == []
    objects = {
        factor.roles["OBJECT"]
        for factor in agent.store.list_semantic_factors()
    }
    assert objects == {"M_АЛЬФА", "M_БЕТА", "M_ГАММА"}


def test_closed_world_cache_has_full_audited_resource_coverage() -> None:
    payload = load_fact_payload(closed_world_facts_path())

    assert payload["coverage"] == {
        "segments": 23,
        "covered": 23,
        "ratio": 1.0,
        "uncovered": [],
    }
    resources = " ".join(
        str(candidate.get("raw_span") or "").lower()
        for candidate in payload["candidates"]
        if candidate.get("canonical_relation") == "HAS_MINERAL"
    )
    for expected in (
        "титановых минералов",
        "бокситы",
        "агаты",
        "нефти",
        "газа",
        "конденсата",
        "битумы",
        "горючие сланцы",
        "торф",
        "строительные камни",
    ):
        assert expected in resources
