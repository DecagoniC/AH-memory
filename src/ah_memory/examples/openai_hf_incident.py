"""Fresh closed-world fixture from OpenAI's Aug 2026 Hugging Face incident post."""
from __future__ import annotations

from pathlib import Path

from ah_memory.perception import FactCandidate, PerceptionResult
from ah_memory.store import AHStore
from ah_memory.transform import Transform
from ah_memory.types import AssocLink, LinkId, Property, SecondOrderSymbol, Section

_CORPUS_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "fresh"
    / "openai_hf_2026"
    / "corpus.txt"
)

_FALLBACK = (
    "Source: OpenAI, The Hugging Face incident and the road ahead (2026-08-26).\n"
    "Internal Model 1 (IM1) drove the ExploitGym evaluation incident.\n"
)

# Key article facts for AH ask / interactive M4 (no pretrained overlap on IM1 specifics).
_CANDIDATES = [
    FactCandidate("IS", {"SUBJECT": "IM1", "OBJECT": "INTERNAL_RESEARCH_MODEL"}),
    FactCandidate("IS", {"SUBJECT": "INTERNAL_RESEARCH_MODEL", "OBJECT": "OPENAI_MODEL"}),
    FactCandidate("DRIVES", {"SUBJECT": "IM1", "OBJECT": "HUGGING_FACE_INCIDENT"}),
    FactCandidate("USES", {"SUBJECT": "IM1", "OBJECT": "EXPLOITGYM"}),
    FactCandidate("LOCATED_IN", {"SUBJECT": "MESSAGE_BOARD", "LOCATION": "ARTIFACTORY"}),
    FactCandidate("CAUSES", {"SUBJECT": "REWARD_HACKING", "OBJECT": "UNAUTHORIZED_COMMUNICATION"}),
    FactCandidate("CAUSES", {"SUBJECT": "UNAUTHORIZED_COMMUNICATION", "OBJECT": "SANDBOX_ESCAPE"}),
    FactCandidate("CAUSES", {"SUBJECT": "SANDBOX_ESCAPE", "OBJECT": "HUGGING_FACE_INTRUSION"}),
    FactCandidate("CAUSES", {"SUBJECT": "HUGGING_FACE_INTRUSION", "OBJECT": "CREDENTIAL_HARVEST"}),
    FactCandidate("USES", {"SUBJECT": "IM1", "OBJECT": "HDF5_ZERO_DAY"}),
    FactCandidate("USES", {"SUBJECT": "IM1", "OBJECT": "REFJINJA_ZERO_DAY"}),
    FactCandidate("CAUSES", {"SUBJECT": "SECURITY_ALERT", "OBJECT": "ASTRA_TRAINING_PAUSE"}),
    FactCandidate("DISCLOSED_ON", {"SUBJECT": "HUGGING_FACE_INCIDENT", "TIME": "JULY_16_2026"}),
    FactCandidate("PUBLISHED_ON", {"SUBJECT": "OPENAI_BLOG_POST", "TIME": "AUGUST_26_2026"}),
]

_LABELS = {
    "IM1": "Internal Model 1",
    "INTERNAL_RESEARCH_MODEL": "internal research model",
    "OPENAI_MODEL": "OpenAI model",
    "HUGGING_FACE_INCIDENT": "Hugging Face incident",
    "EXPLOITGYM": "ExploitGym",
    "MESSAGE_BOARD": "message board",
    "ARTIFACTORY": "Artifactory",
    "REWARD_HACKING": "reward hacking",
    "UNAUTHORIZED_COMMUNICATION": "unauthorized communication",
    "SANDBOX_ESCAPE": "sandbox escape",
    "HUGGING_FACE_INTRUSION": "Hugging Face intrusion",
    "CREDENTIAL_HARVEST": "credential harvest",
    "HDF5_ZERO_DAY": "HDF5 zero-day",
    "REFJINJA_ZERO_DAY": "RefJinja zero-day",
    "SECURITY_ALERT": "security alert",
    "ASTRA_TRAINING_PAUSE": "Astra training pause",
    "JULY_16_2026": "July 16 2026",
    "OPENAI_BLOG_POST": "OpenAI blog post",
    "AUGUST_26_2026": "August 26 2026",
}


def openai_hf_text() -> str:
    if _CORPUS_PATH.is_file():
        return _CORPUS_PATH.read_text(encoding="utf-8").strip() + "\n"
    return _FALLBACK


def build_openai_hf_memory() -> AHStore:
    store = AHStore()
    for uid, label in _LABELS.items():
        store.ensure_abstract(uid, {label.lower(), uid.lower()})
        store.ensure_m(f"M_{uid}", label)

    Transform(store).apply(
        PerceptionResult(
            kind="fact",
            candidates=list(_CANDIDATES),
            seed_tokens=list(_LABELS.keys()),
        ),
        section=Section.C,
    )

    # Explicit IS-A and CAUSE links for M2-style multi-hop ask traces.
    isa_pairs = [
        ("M_IM1", "M_INTERNAL_RESEARCH_MODEL"),
        ("M_INTERNAL_RESEARCH_MODEL", "M_OPENAI_MODEL"),
    ]
    for left, right in isa_pairs:
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_ISA"),
                id=LinkId.IS_A.value,
                w=0.9,
                e1=store.m_ref(left),
                e2=store.m_ref(right),
            )
        )
    cause_pairs = [
        ("M_REWARD_HACKING", "M_UNAUTHORIZED_COMMUNICATION"),
        ("M_UNAUTHORIZED_COMMUNICATION", "M_SANDBOX_ESCAPE"),
        ("M_SANDBOX_ESCAPE", "M_HUGGING_FACE_INTRUSION"),
        ("M_HUGGING_FACE_INTRUSION", "M_CREDENTIAL_HARVEST"),
        ("M_SECURITY_ALERT", "M_ASTRA_TRAINING_PAUSE"),
    ]
    for left, right in cause_pairs:
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_CAUSE"),
                id=LinkId.CAUSE.value,
                w=0.9,
                e1=store.m_ref(left),
                e2=store.m_ref(right),
            )
        )

    if "EP_HF_START" not in store.ah.H:
        store.add_element(
            Section.H,
            SecondOrderSymbol(
                uid="EP_HF_START",
                Pr=[Property(name="label", value="Episode: Artifactory side channel")],
                Mt=[Property(name="kind", value="Episode")],
            ),
        )
        store.add_element(
            Section.H,
            SecondOrderSymbol(
                uid="EP_HF_INTRUSION",
                Pr=[Property(name="label", value="Episode: Hugging Face intrusion")],
                Mt=[Property(name="kind", value="Episode")],
            ),
        )
        store.add_element(
            Section.H,
            SecondOrderSymbol(
                uid="EP_HF_PAUSE",
                Pr=[Property(name="label", value="Episode: Astra training pause")],
                Mt=[Property(name="kind", value="Episode")],
            ),
        )
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_FOLLOW"),
                id=LinkId.FOLLOW.value,
                w=0.8,
                e1=store.m_ref("EP_HF_START"),
                e2=store.m_ref("EP_HF_INTRUSION"),
            )
        )
        store.add_link(
            AssocLink(
                uid=store.new_uid("L_FOLLOW"),
                id=LinkId.FOLLOW.value,
                w=0.8,
                e1=store.m_ref("EP_HF_INTRUSION"),
                e2=store.m_ref("EP_HF_PAUSE"),
            )
        )

    from ah_memory.graph_library import remember_fixture

    remember_fixture("openai-hf-2026", store, source_text=openai_hf_text())
    return store
