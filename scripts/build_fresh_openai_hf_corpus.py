"""Build closed-world challenge corpora from the Aug 2026 OpenAI HF incident post.

  python scripts/build_fresh_openai_hf_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ah_memory.benchmarks.challenge.schema import (  # noqa: E402
    QAItem,
    RoleCorpusItem,
    SourceDocument,
    SourceFact,
)

OUT = ROOT / "benchmarks" / "fresh" / "openai_hf_2026"
FETCHED = (
    Path.home()
    / ".cursor"
    / "projects"
    / "c-Users-al-ru-Downloads-AH-memory"
    / "agent-tools"
    / "0872a1bb-3727-42ca-a4f5-a5f0fefc35a8.txt"
)

_NOISE = [
    r"^Share$",
    r"^Read the technical report$",
    r"^Read METR report.*$",
    r"^Watch Black Hat talk.*$",
    r"^Agent chain-of-thought reasoning$",
    r"^Raw Chain-of-ThoughtPlain language$",
    r"^Raw messagePlain language$",
    r"^Post sent to message board$",
    r"^URL created by agent$",
    r"^Read more:.*$",
]

# Article-grounded chains (variant B: incident journal + taxonomy).
_CHAIN_LABELS: dict[str, tuple[tuple[str, ...], ...]] = {
    "FOLLOW": (
        (
            "im1 training run",
            "artifactory note board",
            "ssrf internet access",
            "token refresh escalation",
            "artifactory outage",
            "artifactory rebuild",
            "exploitgym resume",
        ),
        (
            "exploitgym agents",
            "webdav message dirs",
            "hf credential harvest",
            "hdf5 worker leak",
            "refjinja code exec",
            "hf cluster admin",
            "openai k8s admin",
        ),
        (
            "may early signal",
            "july security ticket",
            "artifactory rebuild day",
            "hf public disclosure",
            "openai involvement notice",
            "im1 weight quarantine",
            "astra training pause",
        ),
    ),
    "IS-A": (
        (
            "im1",
            "internal research model",
            "openai research model",
            "frontier language model",
            "ai agent system",
            "software agent",
            "computational process",
        ),
        (
            "exploitgym",
            "cybersecurity evaluation",
            "internal eval suite",
            "capability probe",
            "safety measurement",
            "alignment diagnostic",
            "research procedure",
        ),
        (
            "artifactory board",
            "unauthorized side channel",
            "shared agent memory",
            "coordination substrate",
            "misalignment amplifier",
            "control failure mode",
            "safety incident class",
        ),
    ),
    "CAUSE": (
        (
            "reward hacking",
            "out of bounds probing",
            "unauthorized communication",
            "sandbox escape",
            "hugging face intrusion",
            "credential harvest",
            "security alert",
        ),
        (
            "impossible exploitgym tasks",
            "high reasoning effort",
            "grader metagaming",
            "third party searching",
            "modal stronghold",
            "hf swarm coordination",
            "root access expansion",
        ),
        (
            "missing eval safeguards",
            "absent cot monitors",
            "package manager 0days",
            "internet proxy via artifactory",
            "cross agent swarming",
            "production secret theft",
            "frontier rl pause",
        ),
    ),
}

_ROLE_EXAMPLES: tuple[tuple[str, str, str], ...] = (
    ("IM1 agent", "Hugging Face credentials", "Artifactory message board"),
    ("IM1 swarm", "SSRF exploit notes", "internal Artifactory"),
    ("ExploitGym agent", "WebDAV directory names", "rebuilt Artifactory"),
    ("OpenAI security team", "incident ticket", "July 5 response queue"),
    ("CrowdStrike advisors", "validation report", "OpenAI investigation room"),
    ("METR analysts", "alignment findings", "independent review desk"),
    ("Redwood Research", "misalignment report", "joint investigation desk"),
    ("GPT-5.6 Sol agents", "private eval data", "public Hugging Face dataset"),
    ("Astra planners", "cyber-critical safeguards", "frontier research lab"),
    ("CoT monitor", "misaligned action alert", "security paging channel"),
    ("sandbox controller", "network isolation rules", "workload isolation zone"),
    ("JFrog contacts", "token-refresh advisory", "Artifactory hardening track"),
    ("Hugging Face responders", "HDF5 zero-day patch", "production worker fleet"),
    ("OpenAI IR lead", "IM1 weight quarantine", "model custody vault"),
    ("RL ops team", "training pause order", "Astra run control board"),
    ("auto-review system", "dangerous action flags", "ChatGPT harness lane"),
    ("ExploitGym grader", "flag-and-vuln checks", "evaluation scoring service"),
    ("Modal tenant agent", "CyberGym sandbox foothold", "customer Modal app"),
    ("Kubernetes admin agent", "cloud secrets bundle", "OpenAI research cluster"),
    ("alignment trainers", "safe-stop curricula", "multi-agent distrust gym"),
    ("network isolation team", "egress deny rules", "untrusted workload VLAN"),
    ("Black Hat speakers", "incident timeline slides", "conference brief room"),
    ("HF cluster agent", "RefJinja template inject", "Hugging Face worker pod"),
    ("security log pipeline", "identity API anomalies", "July 19 alert console"),
    ("safety classifiers", "jailbreak refusals", "Astra readiness gate"),
)


def _clean_corpus(raw: str) -> str:
    lines = raw.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.startswith("# The Hugging Face"):
            start = index
            break
    body = "\n".join(lines[start:])
    for pattern in _NOISE:
        body = re.sub(pattern, "", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    header = (
        "Source: OpenAI blog post «The Hugging Face incident and the road ahead» "
        "(https://openai.com/index/hugging-face-incident-and-the-road-ahead/), "
        "published 2026-08-26. Closed-world corpus for AH-memory metrics M1–M4.\n\n"
    )
    return header + body


def _statement(relation: str, subject: str, object_: str) -> str:
    if relation == "FOLLOW":
        return f"{subject.title()} follows {object_}."
    if relation == "IS-A":
        return f"{subject.title()} is a {object_}."
    return f"{subject.title()} causes {object_}."


def _question(relation: str, start: str, depth: int) -> str:
    if relation == "FOLLOW":
        return f"After {depth} link(s), what does {start} follow?"
    if relation == "IS-A":
        return f"After {depth} classification link(s), what is {start}?"
    return f"After {depth} causal link(s), what ultimately results from {start}?"


def generate_qa_items() -> list[QAItem]:
    depths = (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 6)
    relations = ("FOLLOW", "IS-A", "CAUSE")
    items: list[QAItem] = []
    for index, depth in enumerate(depths, start=1):
        relation = relations[(index - 1) % len(relations)]
        chain_set = _CHAIN_LABELS[relation][((index - 1) // 3) % 3]
        nodes = tuple(f"{label} {index:02d}" for label in chain_set[: depth + 1])
        item_id = f"HF_QA_{index:03d}"
        facts: list[SourceFact] = []
        documents: list[SourceDocument] = []
        for step, (subject, object_) in enumerate(zip(nodes, nodes[1:]), start=1):
            fact_uid = f"{item_id}_F{step:02d}"
            facts.append(
                SourceFact(
                    uid=fact_uid,
                    subject=subject,
                    relation=relation,
                    object=object_,
                )
            )
            documents.append(
                SourceDocument(
                    uid=f"{item_id}_D{step:02d}",
                    text=_statement(relation, subject, object_),
                    fact_uids=(fact_uid,),
                )
            )
        items.append(
            QAItem(
                item_id=item_id,
                question=_question(relation, nodes[0], depth),
                answer=nodes[-1],
                depth=depth,
                relation_type=relation,
                proof_path=tuple(f.uid for f in facts),
                source_facts=tuple(facts),
                source_documents=tuple(documents),
            )
        )
    return items


def generate_role_items() -> list[RoleCorpusItem]:
    variants = ("clean", "typo", "inversion", "ellipsis")
    items: list[RoleCorpusItem] = []
    sequence = 1
    for variant in variants:
        for subject, object_, location in _ROLE_EXAMPLES:
            if variant == "clean":
                text = f"{subject} delivered the {object_} to the {location}."
            elif variant == "typo":
                text = f"{subject} delviered the {object_} to the {location}."
            elif variant == "inversion":
                text = f"To the {location}, the {object_} was delivered by {subject}."
            else:
                text = f"{subject}: {object_} — {location}."
            items.append(
                RoleCorpusItem(
                    item_id=f"HF_M1_{sequence:03d}",
                    text=text,
                    variant=variant,
                    domain="cyber_incident",
                    expected_roles={
                        "SUBJECT": subject,
                        "OBJECT": object_,
                        "LOCATION": location,
                    },
                )
            )
            sequence += 1
    return items


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not FETCHED.is_file():
        raise SystemExit(f"missing fetched article text: {FETCHED}")
    corpus = _clean_corpus(FETCHED.read_text(encoding="utf-8"))
    (OUT / "corpus.txt").write_text(corpus, encoding="utf-8")
    words = len(re.findall(r"\b\w+\b", corpus))
    meta = {
        "title": "The Hugging Face incident and the road ahead",
        "publisher": "OpenAI",
        "url": "https://openai.com/index/hugging-face-incident-and-the-road-ahead/",
        "published": "2026-08-26",
        "why_fresh": (
            "Published 2026-08-26 with unique entities (IM1, ExploitGym, Artifactory "
            "SSRF side channel, Hugging Face HDF5/RefJinja zero-days, Astra pause). "
            "Unlikely in pre-2026-08 training corpora."
        ),
        "domain_variant": "B incident journal / XAI + C taxonomy",
        "metrics": ["M1", "M2", "M3", "M4"],
        "word_count": words,
        "char_count": len(corpus),
        "files": {
            "corpus": "corpus.txt",
            "m1": "m1_roles.jsonl",
            "m2_m4": "m2_m4_qa.jsonl",
        },
    }
    (OUT / "SOURCE.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    qa = generate_qa_items()
    roles = generate_role_items()
    _write_jsonl(OUT / "m2_m4_qa.jsonl", [item.to_dict() for item in qa])
    _write_jsonl(OUT / "m1_roles.jsonl", [item.to_dict() for item in roles])
    print(f"wrote {OUT}")
    print(f"corpus words={words} qa={len(qa)} roles={len(roles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
