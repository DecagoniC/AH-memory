"""Deterministic control dataset for entity resolution (v1)."""
from __future__ import annotations

from ah_memory.benchmarks.entity_resolution.cases import (
    CaseType,
    EntityResolutionCase,
    ExpectedRelation,
    SymbolSpec,
)
from ah_memory.benchmarks.entity_resolution.dataset_extended import (
    extended_cases,
    extended_facts,
    extended_symbols,
    surface_aliases_for_base,
)

DATASET_VERSION = "entity_resolution_v2"


def control_symbols() -> list[SymbolSpec]:
    base = [
        SymbolSpec("s_001", "Юлия", ("Юля",), "person"),
        SymbolSpec("s_002", "Антон", (), "person"),
        SymbolSpec("s_005", "Мария", ("Маша",), "person"),
        SymbolSpec("s_006", "Сергей", ("Серёжа", "Сережа"), "person"),
        SymbolSpec("s_003", "Москва", (), "place"),
        SymbolSpec("s_004", "Санкт-Петербург", ("Петербург", "Питер"), "place"),
        SymbolSpec("s_007", "Казань", (), "place"),
        SymbolSpec("s_010", "автомобиль", ("авто", "машина"), "concept"),
        SymbolSpec("s_011", "врач", ("доктор",), "concept"),
        SymbolSpec("s_012", "покупка", ("приобретение",), "concept"),
        SymbolSpec("s_013", "ноутбук", ("ноут", "лэптоп"), "concept"),
        SymbolSpec("s_014", "телефон", ("смартфон", "мобильник"), "concept"),
        SymbolSpec("s_020", "мотоцикл", ("байк",), "concept"),
        SymbolSpec("s_021", "велосипед", ("велик",), "concept"),
        SymbolSpec("s_022", "пациент", (), "concept"),
        SymbolSpec("s_023", "медсестра", (), "concept"),
        SymbolSpec("s_030", "BMW", (), "brand"),
        SymbolSpec("s_031", "Audi", (), "brand"),
        SymbolSpec("s_032", "Opel", (), "brand"),
        SymbolSpec("s_040", "транспорт", (), "concept"),
        SymbolSpec("s_041", "медицина", (), "concept"),
    ]
    extra = surface_aliases_for_base()
    merged: list[SymbolSpec] = []
    for spec in base:
        add = extra.get(spec.uid, ())
        aliases = tuple(dict.fromkeys((*spec.aliases, *add)))
        merged.append(
            SymbolSpec(
                spec.uid,
                spec.name,
                aliases,
                spec.kind,
                spec.context_cues,
            )
        )
    merged.extend(extended_symbols())
    return merged


def control_graph_facts() -> list[tuple[str, str, str]]:
    """(subject_uid, relation, object_uid) loaded into AH for contextual/activation tests."""
    return [
        ("s_001", "LIVES_IN", "s_003"),
        ("s_001", "WORKS_FOR", "s_011"),
        ("s_002", "PURCHASE", "s_030"),
        ("s_002", "SELL", "s_030"),
        ("s_002", "PURCHASE", "s_032"),
        ("s_005", "LIVES_IN", "s_007"),
        ("s_006", "PURCHASE", "s_013"),
        ("s_006", "PURCHASE", "s_014"),
        ("s_010", "IS_A", "s_040"),
        ("s_011", "PART_OF", "s_041"),
        *extended_facts(),
    ]


def control_cases() -> list[EntityResolutionCase]:
    yulia = "s_001"
    moscow = "s_003"
    car = "s_010"
    doctor = "s_011"
    purchase = "s_012"
    anton = "s_002"

    cases: list[EntityResolutionCase] = []

    # --- MORPHOLOGY ---
    for i, mention in enumerate(
        ("Юли", "Юлию", "Юлией", "Юлии", "Юле"),
        start=1,
    ):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_yulia_{i:02d}",
                mention=mention,
                target_uid=yulia,
                candidate_uids=(yulia, anton, moscow),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_ENTITY,
                difficulty="easy",
            )
        )
    for i, mention in enumerate(("Москве", "Москву", "Москвы", "Москвой"), start=1):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_moscow_{i:02d}",
                mention=mention,
                target_uid=moscow,
                candidate_uids=(moscow, "s_004", yulia),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_ENTITY,
            )
        )
    for i, mention in enumerate(
        ("автомобиля", "автомобилем", "автомобилю", "автомобиле"),
        start=1,
    ):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_car_{i:02d}",
                mention=mention,
                target_uid=car,
                candidate_uids=(car, "s_020", "s_021"),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_CONCEPT,
            )
        )
    for i, mention in enumerate(("Антона", "Антону", "Антоном", "Антоне"), start=1):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_anton_{i:02d}",
                mention=mention,
                target_uid=anton,
                candidate_uids=(anton, yulia, moscow),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_ENTITY,
            )
        )
    maria, sergey, kazan = "s_005", "s_006", "s_007"
    laptop, phone = "s_013", "s_014"
    for i, mention in enumerate(("Марии", "Марию", "Машей", "Маше"), start=1):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_maria_{i:02d}",
                mention=mention,
                target_uid=maria,
                candidate_uids=(maria, yulia, anton),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_ENTITY,
            )
        )
    for i, mention in enumerate(("Сергея", "Сергею", "Сергеем"), start=1):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_sergey_{i:02d}",
                mention=mention,
                target_uid=sergey,
                candidate_uids=(sergey, anton, maria),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_ENTITY,
            )
        )
    for i, mention in enumerate(("Казани", "Казань", "Казанью"), start=1):
        cases.append(
            EntityResolutionCase(
                case_id=f"morph_kazan_{i:02d}",
                mention=mention,
                target_uid=kazan,
                candidate_uids=(kazan, moscow, "s_004"),
                case_type=CaseType.MORPHOLOGY,
                expected_relation=ExpectedRelation.SAME_ENTITY,
            )
        )

    # --- SYNONYM ---
    for case_id, mention, target, cands in (
        ("syn_001", "машина", car, (car, "s_020", "s_021")),
        ("syn_002", "доктор", doctor, (doctor, "s_022", "s_023")),
        ("syn_003", "приобретение", purchase, (purchase, car, doctor)),
        ("syn_004", "автомашина", car, (car, "s_020", laptop)),
        ("syn_005", "ноут", laptop, (laptop, phone, car)),
        ("syn_006", "смартфон", phone, (phone, laptop, car)),
        ("syn_007", "Питер", "s_004", ("s_004", moscow, kazan)),
        ("syn_008", "Маша", maria, (maria, yulia, anton)),
    ):
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=target,
                candidate_uids=cands,
                case_type=CaseType.SYNONYM,
                expected_relation=ExpectedRelation.SAME_CONCEPT
                if target in {car, doctor, purchase, laptop, phone}
                else ExpectedRelation.SAME_ENTITY,
                difficulty="medium",
            )
        )

    # --- NEGATIVE / SEMANTIC_NEAR (must NOT merge) ---
    negatives = (
        ("neg_001", "мотоцикл", car, (car, "s_020", "s_021"), CaseType.NEGATIVE),
        ("neg_002", "пациент", doctor, (doctor, "s_022", "s_023"), CaseType.NEGATIVE),
        ("neg_003", "Санкт-Петербург", moscow, (moscow, "s_004"), CaseType.NEGATIVE),
        ("neg_004", "Audi", "s_030", ("s_030", "s_031", "s_032"), CaseType.NEGATIVE),
        ("neg_005", "велосипед", car, (car, "s_020", "s_021"), CaseType.NEGATIVE),
        ("neg_006", "медсестра", doctor, (doctor, "s_022", "s_023"), CaseType.NEGATIVE),
        (
            "near_001",
            "мотоцикл",
            car,
            (car, "s_020"),
            CaseType.SEMANTIC_NEAR,
        ),
        (
            "near_002",
            "пациент",
            doctor,
            (doctor, "s_022"),
            CaseType.SEMANTIC_NEAR,
        ),
    )
    for case_id, mention, distractor, cands, ctype in negatives:
        # mention refers to its OWN symbol if present in catalog.
        # Identity tests: resolve to own uid, never to the distractor.
        own = {
            "мотоцикл": "s_020",
            "пациент": "s_022",
            "Санкт-Петербург": "s_004",
            "Audi": "s_031",
            "велосипед": "s_021",
            "медсестра": "s_023",
        }.get(mention)
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=own,
                candidate_uids=cands,
                case_type=ctype,
                expected_relation=ExpectedRelation.DIFFERENT_ENTITY
                if ctype == CaseType.NEGATIVE
                else ExpectedRelation.DIFFERENT_CONCEPT,
                difficulty="hard",
                notes=f"must_not_merge_with={distractor}",
            )
        )

    # Explicit "reject merge" negatives where predicted must be None if forced
    # against only the wrong candidate pool (no own uid in candidates).
    for case_id, mention, wrong in (
        ("neg_force_001", "мотоцикл", car),
        ("neg_force_002", "пациент", doctor),
        ("neg_force_003", "Санкт-Петербург", moscow),
        ("neg_force_004", "Audi", "s_030"),
    ):
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=None,
                candidate_uids=(wrong,),
                case_type=CaseType.NEGATIVE,
                expected_relation=ExpectedRelation.DIFFERENT_ENTITY,
                difficulty="hard",
                notes=f"force_reject={wrong}",
            )
        )

    # --- CONTEXTUAL ---
    cases.append(
        EntityResolutionCase(
            case_id="ctx_001",
            mention="Юли",
            target_uid=yulia,
            candidate_uids=(yulia, anton, moscow),
            case_type=CaseType.CONTEXTUAL,
            expected_relation=ExpectedRelation.SAME_ENTITY,
            context="Юли позвонил Антон",
            difficulty="medium",
        )
    )
    cases.append(
        EntityResolutionCase(
            case_id="ctx_002",
            mention="машине",
            target_uid=None,  # experimental: not required to map to Opel
            candidate_uids=(car, "s_030", "s_032"),
            case_type=CaseType.CONTEXTUAL,
            expected_relation=ExpectedRelation.SAME_CONCEPT,
            context="Антон сейчас ездит на машине",
            difficulty="hard",
            notes="experimental_ownership_not_required",
        )
    )
    # Paraphrased question-style contexts (mention resolution only).
    for case_id, mention, target, cands, context in (
        (
            "ctx_003",
            "Юлию",
            yulia,
            (yulia, anton, moscow),
            "Где сейчас живёт Юлию?",
        ),
        (
            "ctx_004",
            "Маше",
            maria,
            (maria, yulia, kazan),
            "В каком городе живёт Маше?",
        ),
        (
            "ctx_005",
            "ноут",
            laptop,
            (laptop, phone, car),
            "Какой ноут купил Сергей?",
        ),
        (
            "ctx_006",
            "Питере",
            "s_004",
            ("s_004", moscow, kazan),
            "Кто работает в Питере?",
        ),
    ):
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=target,
                candidate_uids=cands,
                case_type=CaseType.CONTEXTUAL,
                expected_relation=ExpectedRelation.SAME_ENTITY
                if target in {yulia, maria, "s_004"}
                else ExpectedRelation.SAME_CONCEPT,
                context=context,
                difficulty="medium",
            )
        )

    cases.extend(extended_cases())
    return cases

