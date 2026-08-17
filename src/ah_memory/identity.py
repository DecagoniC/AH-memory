"""Identity: mention в тексте → существующий bare UID в AH.S (без ложных merge).

Цепочка: exact → morphology → alias-lexicon → embedding*только через IdentityGate*.
Cosine сам по себе никогда не мержит: нужны threshold, margin, type, anti-merge.
Используется из Transform._resolve_bare / _resolve_value.
Читать после perception/transform; детали резолверов — benchmarks/entity_resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ah_memory.benchmarks.entity_resolution.cases import (
    CandidateScore,
    ResolutionResult,
    SymbolSpec,
)
from ah_memory.benchmarks.entity_resolution.resolvers import (
    EmbeddingFn,
    EmbeddingResolver,
    ExactResolver,
    MorphologyResolver,
    _filter_symbols,
    _norm,
    upgrade_head_to_compound,
)
from ah_memory.morph import lemma
from ah_memory.store import AHStore

# ── Политика безопасности ────────────────────────────────────────────────────
# synonyms: жёсткий лексикон («машина»→«автомобиль»).
# anti_merge: пары, которые embeddings любят путать — merge запрещён.
# COMPATIBLE_KINDS: person≠place и т.п.

DEFAULT_SYNONYMS: dict[str, str] = {
    "машина": "автомобиль",
    "машины": "автомобиль",
    "машине": "автомобиль",
    "машину": "автомобиль",
    "машиной": "автомобиль",
    "авто": "автомобиль",
    "автомашина": "автомобиль",
    "доктор": "врач",
    "доктора": "врач",
    "доктору": "врач",
    "доктором": "врач",
    "приобретение": "покупка",
    "приобретения": "покупка",
    "приобретением": "покупка",
    "питер": "санкт-петербург",
    "питере": "санкт-петербург",
    "питера": "санкт-петербург",
    "петербург": "санкт-петербург",
    "петербурге": "санкт-петербург",
    "петербурга": "санкт-петербург",
    "петербургом": "санкт-петербург",
    "петербургу": "санкт-петербург",
    "ноут": "ноутбук",
    "лэптоп": "ноутбук",
    "смартфон": "телефон",
    "мобильник": "телефон",
    "байк": "мотоцикл",
    "велик": "велосипед",
    "маша": "мария",
    "маше": "мария",
    "машу": "мария",
    "машей": "мария",
    "маши": "мария",  # genitive of Маша (not «машин-»)
    "серёжа": "сергей",
    "сережа": "сергей",
    "серёже": "сергей",
    "сереже": "сергей",
    "yuliya": "юлия",
    "yulia": "юлия",
    "moskva": "москва",
    "new york": "нью-йорк",
    "newyork": "нью-йорк",
}

DEFAULT_ANTI_MERGE: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"автомобиль", "мотоцикл"}),
        frozenset({"автомобиль", "велосипед"}),
        frozenset({"мотоцикл", "велосипед"}),
        frozenset({"врач", "пациент"}),
        frozenset({"врач", "медсестра"}),
        frozenset({"пациент", "медсестра"}),
        frozenset({"москва", "санкт-петербург"}),
        frozenset({"москва", "нью-йорк"}),
        frozenset({"санкт-петербург", "нью-йорк"}),
        frozenset({"bmw", "audi"}),
        frozenset({"bmw", "opel"}),
        frozenset({"audi", "opel"}),
        frozenset({"bmw", "toyota"}),
        frozenset({"audi", "toyota"}),
        frozenset({"toyota", "lada"}),
        frozenset({"bmw", "lada"}),
        frozenset({"инженер", "программист"}),
        frozenset({"врач", "инженер"}),
        frozenset({"врач", "программист"}),
        frozenset({"ноутбук", "телефон"}),
    }
)

COMPATIBLE_KINDS: dict[str, frozenset[str]] = {
    "entity": frozenset({"entity", "person", "place"}),
    "person": frozenset({"person", "entity"}),
    "place": frozenset({"place", "entity"}),
    "brand": frozenset({"brand"}),
    "concept": frozenset({"concept"}),
}


@dataclass(frozen=True)
class IdentityPolicy:
    safety_threshold: float = 0.94
    margin: float = 0.05
    enforce_type: bool = True
    anti_merge: frozenset[frozenset[str]] = DEFAULT_ANTI_MERGE
    synonyms: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYNONYMS))


def are_anti_merged(a: str, b: str, policy: IdentityPolicy) -> bool:
    pair = frozenset({_norm(a), _norm(b)})
    if len(pair) < 2:
        return False
    return pair in policy.anti_merge


def kinds_compatible(a: str, b: str) -> bool:
    ka, kb = (a or "entity").lower(), (b or "entity").lower()
    if ka == kb:
        return True
    return kb in COMPATIBLE_KINDS.get(ka, frozenset({ka})) and ka in COMPATIBLE_KINDS.get(
        kb, frozenset({kb})
    )


def infer_kind(name: str, *, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    stripped = name.strip()
    n = _norm(stripped)
    if (
        stripped.isascii()
        and stripped.replace("-", "").replace("_", "").isalnum()
        and stripped.upper() == stripped
        and any(c.isalpha() for c in stripped)
    ):
        return "brand"
    if n in {"москва", "санкт-петербург", "петербург", "питер"}:
        return "place"
    if n in {"юлия", "антон", "юля"}:
        return "person"
    if n[:1].isalpha() and stripped[:1].isupper() and not stripped.isupper():
        return "entity"
    return "concept"


# ── Ступени резолва ──────────────────────────────────────────────────────────


class AliasResolver:
    """Синоним из лексикона → совпадение с name/aliases символа в каталоге."""

    name = "alias"

    def __init__(self, synonyms: dict[str, str] | None = None) -> None:
        self.synonyms = {_norm(k): _norm(v) for k, v in (synonyms or DEFAULT_SYNONYMS).items()}

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        needle = _norm(mention)
        target = self.synonyms.get(needle) or self.synonyms.get(_norm(lemma(mention) or ""))
        if target is None:
            return ResolutionResult(None, 0.0, self.name, [])
        from ah_memory.benchmarks.entity_resolution.resolvers import forms_match

        pool = _filter_symbols(symbols, candidate_uids)
        for symbol in pool:
            forms = (symbol.name, *symbol.aliases)
            if any(forms_match(target, form) or forms_match(needle, form) for form in forms):
                conf = 0.97
                if conf < threshold:
                    return ResolutionResult(None, conf, self.name, [])
                return ResolutionResult(
                    symbol.uid,
                    conf,
                    self.name,
                    [CandidateScore(symbol.uid, conf, self.name)],
                )
        return ResolutionResult(None, 0.0, self.name, [])


class IdentityGate:
    """Последний фильтр перед merge по embeddings: floor / margin / anti / type."""

    name = "identity_gate"

    def __init__(self, policy: IdentityPolicy | None = None) -> None:
        self.policy = policy or IdentityPolicy()

    def decide(
        self,
        mention: str,
        candidates: list[CandidateScore],
        symbols: Sequence[SymbolSpec],
        *,
        threshold: float,
        context: str | None = None,
    ) -> ResolutionResult:
        policy = self.policy
        floor = max(float(threshold), policy.safety_threshold)
        by_uid = {s.uid: s for s in symbols}
        scored = sorted(candidates, key=lambda c: (-c.similarity, c.uid))
        if not scored:
            return ResolutionResult(None, 0.0, self.name, [])
        mention_n = _norm(mention)

        from ah_memory.benchmarks.entity_resolution.resolvers import (
            disambiguate_by_context,
            forms_match,
        )

        exact_hits = [
            s
            for s in symbols
            if any(forms_match(mention, form) for form in (s.name, *s.aliases))
        ]
        if exact_hits:
            chosen = disambiguate_by_context(exact_hits, context)
            if chosen is not None:
                return ResolutionResult(
                    chosen.uid,
                    1.0,
                    "exact",
                    [CandidateScore(chosen.uid, 1.0, "exact")],
                )

        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        if best.similarity < floor:
            return ResolutionResult(None, best.similarity, self.name, scored)
        if second is not None and (best.similarity - second.similarity) < policy.margin:
            return ResolutionResult(None, best.similarity, self.name, scored)

        best_sym = by_uid.get(best.uid)
        if best_sym is None:
            return ResolutionResult(None, best.similarity, self.name, scored)

        if are_anti_merged(mention, best_sym.name, policy):
            return ResolutionResult(None, best.similarity, self.name, scored)
        for symbol in symbols:
            forms = {_norm(symbol.name), *(_norm(a) for a in symbol.aliases)}
            own_lemma = _norm(lemma(mention) or "")
            if mention_n in forms or own_lemma == _norm(symbol.name):
                if symbol.uid != best.uid and are_anti_merged(
                    symbol.name, best_sym.name, policy
                ):
                    return ResolutionResult(None, best.similarity, self.name, scored)

        if policy.enforce_type:
            mention_kind = infer_kind(mention)
            for symbol in symbols:
                forms = {_norm(symbol.name), *(_norm(a) for a in symbol.aliases)}
                if mention_n in forms:
                    mention_kind = symbol.kind
                    break
            if not kinds_compatible(mention_kind, best_sym.kind):
                return ResolutionResult(None, best.similarity, self.name, scored)

        return ResolutionResult(best.uid, best.similarity, "embedding", scored)


class RejectEmbeddingResolver:
    """Заглушка: embeddings выключены (offline / без API)."""

    name = "embedding"

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        return ResolutionResult(None, 0.0, self.name, [])


class GatedEmbeddingResolver:
    """Сырой EmbeddingResolver → кандидаты → IdentityGate.decide."""

    name = "embedding"

    def __init__(
        self,
        embed: EmbeddingFn | None = None,
        *,
        dimensions: int = 64,
        policy: IdentityPolicy | None = None,
    ) -> None:
        self.inner = EmbeddingResolver(embed, dimensions=dimensions)
        self.gate = IdentityGate(policy)

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        raw = self.inner.resolve(
            mention,
            context,
            symbols,
            candidate_uids=candidate_uids,
            threshold=0.0,
        )
        return self.gate.decide(
            mention,
            raw.candidates,
            _filter_symbols(symbols, candidate_uids),
            threshold=threshold,
            context=context,
        )


class SafeHybridResolver:
    """Основной пайплайн: exact → morph → alias → gated embedding."""

    name = "hybrid"

    def __init__(
        self,
        embed: EmbeddingFn | None = None,
        *,
        dimensions: int = 64,
        policy: IdentityPolicy | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self.policy = policy or IdentityPolicy()
        self.exact = ExactResolver()
        self.morphology = MorphologyResolver()
        self.alias = AliasResolver(self.policy.synonyms)
        if use_embeddings:
            self.embedding: ExactResolver | GatedEmbeddingResolver | RejectEmbeddingResolver = (
                GatedEmbeddingResolver(
                    embed, dimensions=dimensions, policy=self.policy
                )
            )
        else:
            self.embedding = RejectEmbeddingResolver()

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        for resolver in (self.exact, self.morphology, self.alias):
            result = resolver.resolve(
                mention,
                context,
                symbols,
                candidate_uids=candidate_uids,
                threshold=0.0,
            )
            if result.selected_uid is not None:
                upgraded = upgrade_head_to_compound(
                    mention, result.selected_uid, symbols
                )
                if upgraded != result.selected_uid:
                    return ResolutionResult(
                        upgraded,
                        result.confidence,
                        f"{result.method}+compound",
                        result.candidates,
                    )
                return result
        return self.embedding.resolve(
            mention,
            context,
            symbols,
            candidate_uids=candidate_uids,
            threshold=threshold,
        )


def catalog_from_store(store: AHStore) -> list[SymbolSpec]:
    # Зачем: резолверы работают на SymbolSpec, а не на сыром AH — единый каталог.
    out: list[SymbolSpec] = []
    for uid, abstract in store.ah.S.items():
        forms = tuple(sorted({_norm(f) for f in (abstract.R.get("TEXT") or set()) if f}))
        label = uid.replace("_", " ")
        m_uid = uid if uid.startswith("M_") else f"M_{uid}"
        try:
            el = store._find_anywhere(m_uid)
            for prop in getattr(el, "Pr", []) or []:
                if getattr(prop, "name", None) == "label" and prop.value:
                    label = str(prop.value)
                    break
        except Exception:
            pass
        if label.isupper() and "_" in label:
            label = label.replace("_", " ").title()
        elif label.isupper() and label.isalpha():
            label = label.title()
        kind = infer_kind(label)
        aliases = tuple(a for a in forms if a != _norm(label))
        out.append(SymbolSpec(uid=uid, name=label, aliases=aliases, kind=kind))
    return out


@dataclass
class SymbolIdentityService:
    """Фасад для Transform: resolve_bare_uid + attach_alias после успешного ensure."""

    store: AHStore
    resolver: SafeHybridResolver
    threshold: float = 0.94
    enabled: bool = True

    def resolve_bare_uid(self, mention: str, context: str | None = None) -> str | None:
        if not self.enabled:
            return None
        symbols = catalog_from_store(self.store)
        if not symbols:
            return None
        result = self.resolver.resolve(
            mention, context, symbols, threshold=self.threshold
        )
        return result.selected_uid

    def attach_alias(self, bare_uid: str, surface: str) -> None:
        s = self.store.ah.S.get(bare_uid)
        if s is None:
            return
        s.R.setdefault("TEXT", set()).add(_norm(surface))


def build_identity_service(
    store: AHStore,
    *,
    enabled: bool = True,
    use_embeddings: bool = False,
    embed: EmbeddingFn | None = None,
    safety_threshold: float = 0.94,
    margin: float = 0.05,
) -> SymbolIdentityService:
    # Зачем: единая сборка из config.yaml (web/Agent).
    policy = IdentityPolicy(safety_threshold=safety_threshold, margin=margin)
    resolver = SafeHybridResolver(
        embed,
        policy=policy,
        use_embeddings=bool(use_embeddings and embed is not None),
    )
    return SymbolIdentityService(
        store=store,
        resolver=resolver,
        threshold=safety_threshold,
        enabled=enabled,
    )
