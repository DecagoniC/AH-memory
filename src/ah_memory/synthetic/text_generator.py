"""Deterministic natural-language templates for synthetic factors."""
from __future__ import annotations

from typing import Mapping, Sequence

from ah_memory.synthetic.entities import Entity
from ah_memory.synthetic.events import SyntheticFactor
from ah_memory.synthetic.ground_truth import SyntheticDocument


_TEMPLATES: Mapping[str, tuple[str, ...]] = {
    "PURCHASE": (
        "{buyer} купил {object}.",
        "{buyer} приобрёл {object}.",
        "{buyer} стал владельцем {object}.",
        "{object} был приобретён {buyer}.",
    ),
    "SELL": (
        "{buyer} продал {object}.",
        "{buyer} реализовал {object}.",
        "{object} был продан {buyer}.",
    ),
    "WORKS_FOR": (
        "{person} работает в {company}.",
        "{person} является сотрудником {company}.",
        "{person} трудится в компании {company}.",
    ),
    "LIVES_IN": (
        "{person} живёт в {location}.",
        "{person} проживает в {location}.",
        "Место жительства {person} — {location}.",
    ),
    "LOCATED_IN": (
        "{subject} находится в {location}.",
        "{subject} расположен в {location}.",
    ),
    "MOVE": (
        "{person} переехал из {from} в {to}.",
        "{person} переместился из {from} в {to}.",
    ),
    "VISITS": (
        "{person} посетил {location}.",
        "{person} был в {location}.",
    ),
    "USES": (
        "{person} использует {object}.",
        "{person} применяет {object}.",
    ),
    "CREATED": (
        "{person} создал {object}.",
        "{object} был создан {person}.",
    ),
    "CAUSE": (
        "{cause} вызвал {effect}.",
        "{effect} произошёл из-за {cause}.",
    ),
    "FOLLOW": (
        "После {previous_event} последовал {next_event}.",
        "{previous_event} предшествовал {next_event}.",
    ),
    "PART_OF": (
        "{subject} является частью {object}.",
        "{subject} входит в {object}.",
    ),
    "IS_A": (
        "{child} является разновидностью {parent}.",
        "{child} — это {parent}.",
    ),
    "KNOW": (
        "{person} знает {object}.",
        "{person} знаком с {object}.",
    ),
    "OWNS": (
        "{person} владеет {object}.",
        "{object} принадлежит {person}.",
    ),
}


class TextGenerator:
    """Template-based NL; optional llm callable is unused for ground truth."""

    def __init__(self, llm: object | None = None) -> None:
        self.llm = llm

    def render_factor(
        self,
        factor: SyntheticFactor,
        entities: Mapping[str, Entity],
    ) -> list[str]:
        templates = _TEMPLATES.get(factor.type)
        if not templates:
            args = ", ".join(
                f"{role}={entities[uid].name if uid in entities else uid}"
                for role, uid in factor.arguments.items()
            )
            return [f"{factor.type}({args})."]
        values = {
            role: entities[uid].name if uid in entities else uid
            for role, uid in factor.arguments.items()
        }
        lines: list[str] = []
        for template in templates:
            try:
                lines.append(template.format(**values))
            except KeyError:
                continue
        return lines or [f"{factor.type}({factor.uid})."]

    def build_documents(
        self,
        factors: Sequence[SyntheticFactor],
        entities: Mapping[str, Entity],
        *,
        max_documents: int | None = None,
    ) -> list[SyntheticDocument]:
        documents: list[SyntheticDocument] = []
        # Skip pure distractors for corpus cleanliness; still only GT factors.
        source = [
            factor
            for factor in factors
            if not factor.properties.get("distractor")
        ]
        if max_documents is not None:
            source = source[:max_documents]
        for index, factor in enumerate(source, start=1):
            variants = self.render_factor(factor, entities)
            documents.append(
                SyntheticDocument(
                    uid=f"document_{index:06d}",
                    text=variants[0],
                    factor_uids=(factor.uid,),
                    paraphrases=tuple(variants[1:]),
                )
            )
        return documents
