"""Extended ER corpus: ambiguous, surface, hard-negatives, hold-out."""
from __future__ import annotations

from ah_memory.benchmarks.entity_resolution.cases import (
    CaseType,
    EntityResolutionCase,
    ExpectedRelation,
    SymbolSpec,
)
from ah_memory.identity import DEFAULT_SYNONYMS


def extended_symbols() -> list[SymbolSpec]:
    return [
        # Ambiguous same display name, different context cues
        SymbolSpec(
            "s_050",
            "Иван",
            (),
            "person",
            context_cues=("москва", "москве", "москвы"),
        ),
        SymbolSpec(
            "s_051",
            "Иван",
            (),
            "person",
            context_cues=("казань", "казани", "казанью"),
        ),
        SymbolSpec(
            "s_052",
            "Анна",
            (),
            "person",
            context_cues=("берлин", "берлине"),
        ),
        SymbolSpec(
            "s_053",
            "Анна",
            (),
            "person",
            context_cues=("париж", "париже"),
        ),
        SymbolSpec("s_054", "Берлин", (), "place"),
        SymbolSpec("s_055", "Париж", (), "place"),
        SymbolSpec(
            "s_056",
            "Нью-Йорк",
            ("New York", "нью йорк", "NewYork"),
            "place",
        ),
        SymbolSpec("s_057", "Toyota", ("toyota", "TOYOTA"), "brand"),
        SymbolSpec("s_058", "Lada", ("lada", "ЛАДА"), "brand"),
        SymbolSpec("s_059", "инженер", (), "concept"),
        SymbolSpec("s_060", "программист", (), "concept"),
    ]


def extended_facts() -> list[tuple[str, str, str]]:
    return [
        ("s_050", "LIVES_IN", "s_003"),  # Иван→Москва
        ("s_051", "LIVES_IN", "s_007"),  # Иван→Казань
        ("s_052", "LIVES_IN", "s_054"),
        ("s_053", "LIVES_IN", "s_055"),
    ]


def surface_aliases_for_base() -> dict[str, tuple[str, ...]]:
    """Extra aliases merged into base control symbols (latin/typo/multiword)."""
    return {
        "s_001": ("Yuliya", "Yulia", "yuliya", "Юлия "),
        "s_003": ("Moskva", "moskva"),
        "s_004": ("Санкт Петербург", "санкт  петербург", "St Petersburg"),
        "s_030": ("bmw", "Bmw", " BMW "),
        "s_031": ("audi", "AUDI"),
    }


# Hold-out mentions must NOT appear as keys in DEFAULT_SYNONYMS.
HOLD_OUT_POSITIVES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("hold_001", "легковушка", "s_010", ("s_010", "s_020", "s_021")),
    ("hold_002", "тачка", "s_010", ("s_010", "s_020", "s_057")),
    ("hold_003", "эскулап", "s_011", ("s_011", "s_022", "s_059")),
    ("hold_004", "гаджет", "s_014", ("s_014", "s_013", "s_010")),
    ("hold_005", "ультрабук", "s_013", ("s_013", "s_014", "s_010")),
    ("hold_006", "автолюбительский транспорт", "s_010", ("s_010", "s_040", "s_020")),
    ("hold_007", "медик", "s_011", ("s_011", "s_023", "s_022")),
    ("hold_008", "сотовый", "s_014", ("s_014", "s_013")),
)


def assert_hold_out_not_in_lexicon() -> None:
    keys = {_norm_key(k) for k in DEFAULT_SYNONYMS}
    for _cid, mention, _t, _c in HOLD_OUT_POSITIVES:
        assert _norm_key(mention) not in keys, mention


def _norm_key(text: str) -> str:
    return text.strip().lower().replace("ё", "е")


def extended_cases() -> list[EntityResolutionCase]:
    assert_hold_out_not_in_lexicon()
    cases: list[EntityResolutionCase] = []
    ivan_msk, ivan_kzn = "s_050", "s_051"
    anna_ber, anna_par = "s_052", "s_053"
    nyc = "s_056"
    toyota, lada = "s_057", "s_058"
    engineer, programmer = "s_059", "s_060"
    car, doctor = "s_010", "s_011"
    moscow, spb = "s_003", "s_004"

    # --- AMBIGUOUS ---
    cases.append(
        EntityResolutionCase(
            case_id="amb_001",
            mention="Иван",
            target_uid=None,
            candidate_uids=(ivan_msk, ivan_kzn),
            case_type=CaseType.AMBIGUOUS,
            expected_relation=ExpectedRelation.DIFFERENT_ENTITY,
            context=None,
            difficulty="hard",
            notes="ambiguous_without_context",
        )
    )
    cases.append(
        EntityResolutionCase(
            case_id="amb_002",
            mention="Иван",
            target_uid=ivan_msk,
            candidate_uids=(ivan_msk, ivan_kzn),
            case_type=CaseType.AMBIGUOUS,
            expected_relation=ExpectedRelation.SAME_ENTITY,
            context="Иван живёт в Москве",
            difficulty="hard",
        )
    )
    cases.append(
        EntityResolutionCase(
            case_id="amb_003",
            mention="Иван",
            target_uid=ivan_kzn,
            candidate_uids=(ivan_msk, ivan_kzn),
            case_type=CaseType.AMBIGUOUS,
            expected_relation=ExpectedRelation.SAME_ENTITY,
            context="Иван работает в Казани",
            difficulty="hard",
        )
    )
    cases.append(
        EntityResolutionCase(
            case_id="amb_004",
            mention="Анна",
            target_uid=None,
            candidate_uids=(anna_ber, anna_par),
            case_type=CaseType.AMBIGUOUS,
            expected_relation=ExpectedRelation.DIFFERENT_ENTITY,
            difficulty="hard",
            notes="ambiguous_without_context",
        )
    )
    cases.append(
        EntityResolutionCase(
            case_id="amb_005",
            mention="Анна",
            target_uid=anna_ber,
            candidate_uids=(anna_ber, anna_par),
            case_type=CaseType.AMBIGUOUS,
            expected_relation=ExpectedRelation.SAME_ENTITY,
            context="Анна переехала в Берлин",
            difficulty="hard",
        )
    )
    cases.append(
        EntityResolutionCase(
            case_id="amb_006",
            mention="Анна",
            target_uid=anna_par,
            candidate_uids=(anna_ber, anna_par),
            case_type=CaseType.AMBIGUOUS,
            expected_relation=ExpectedRelation.SAME_ENTITY,
            context="Анна учится в Париже",
            difficulty="hard",
        )
    )

    # --- SURFACE: latin / multiword / spacing / case ---
    for case_id, mention, target, cands in (
        ("surf_001", "Yuliya", "s_001", ("s_001", "s_002", "s_005")),
        ("surf_002", "yulia", "s_001", ("s_001", "s_002")),
        ("surf_003", "bmw", "s_030", ("s_030", "s_031", "s_057")),
        ("surf_004", " BMW ", "s_030", ("s_030", "s_031")),
        ("surf_005", "Санкт Петербург", spb, (spb, moscow, nyc)),
        ("surf_006", "санкт  петербург", spb, (spb, moscow)),
        ("surf_007", "New York", nyc, (nyc, moscow, spb)),
        ("surf_008", "нью йорк", nyc, (nyc, moscow, spb)),
        ("surf_009", "Moskva", moscow, (moscow, spb, nyc)),
        ("surf_010", "audi", "s_031", ("s_031", "s_030", toyota)),
    ):
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=target,
                candidate_uids=cands,
                case_type=CaseType.SURFACE,
                expected_relation=ExpectedRelation.SAME_ENTITY,
                difficulty="medium",
            )
        )

    # --- HARD NEGATIVES (high semantic proximity, must not merge) ---
    hard_negs = (
        ("hneg_001", "Toyota", "s_030", (toyota, "s_030", "s_031"), toyota),
        ("hneg_002", "Lada", toyota, (lada, toyota, "s_030"), lada),
        ("hneg_003", "инженер", doctor, (engineer, doctor, programmer), engineer),
        ("hneg_004", "программист", engineer, (programmer, engineer, doctor), programmer),
        ("hneg_005", "Нью-Йорк", moscow, (nyc, moscow, spb), nyc),
        ("hneg_006", "Берлин", "s_055", ("s_054", "s_055", moscow), "s_054"),
    )
    for case_id, mention, distractor, cands, own in hard_negs:
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=own,
                candidate_uids=cands,
                case_type=CaseType.NEGATIVE,
                expected_relation=ExpectedRelation.DIFFERENT_ENTITY,
                difficulty="hard",
                notes=f"must_not_merge_with={distractor}",
            )
        )
    for case_id, mention, wrong in (
        ("hneg_force_001", "Toyota", "s_030"),
        ("hneg_force_002", "инженер", doctor),
        ("hneg_force_003", "Нью-Йорк", moscow),
        ("hneg_force_004", "программист", doctor),
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

    # --- HOLD_OUT (not in synonym lexicon; embeddings / future alias) ---
    for case_id, mention, target, cands in HOLD_OUT_POSITIVES:
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=target,
                candidate_uids=cands,
                case_type=CaseType.HOLD_OUT,
                expected_relation=ExpectedRelation.SAME_CONCEPT,
                difficulty="hard",
                notes="hold_out_not_in_lexicon",
            )
        )
    # Hold-out negatives: similar jargon must not collapse
    for case_id, mention, wrong in (
        ("hold_neg_001", "легковушка", "s_020"),
        ("hold_neg_002", "эскулап", "s_022"),
        ("hold_neg_003", "гаджет", "s_013"),
    ):
        cases.append(
            EntityResolutionCase(
                case_id=case_id,
                mention=mention,
                target_uid=None,
                candidate_uids=(wrong,),
                case_type=CaseType.HOLD_OUT,
                expected_relation=ExpectedRelation.DIFFERENT_CONCEPT,
                difficulty="hard",
                notes=f"force_reject={wrong}",
            )
        )

    return cases
