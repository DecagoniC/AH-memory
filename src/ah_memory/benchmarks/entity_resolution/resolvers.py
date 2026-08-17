"""Symbol resolution strategies: Exact → Morphology → Embedding → Hybrid."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Protocol, Sequence

from ah_memory.benchmarks.entity_resolution.cases import (
    CandidateScore,
    ResolutionResult,
    SymbolSpec,
)
from ah_memory.morph import lemma
from ah_memory.perception import slug_uid
from ah_memory.relation_normalizer import deterministic_embedding
from ah_memory.relation_registry import cosine_similarity

EmbeddingFn = Callable[[str], Sequence[float]]

_CASE_TAGS = ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct")


def _norm(text: str) -> str:
    t = text.strip().lower().replace("ё", "е")
    t = re.sub(r"[\s_\-]+", " ", t).strip()
    return t


def _compact(text: str) -> str:
    return _norm(text).replace(" ", "").replace("-", "")


def form_keys(text: str) -> set[str]:
    n = _norm(text)
    if not n:
        return set()
    return {n, _compact(n)}


def forms_match(a: str, b: str) -> bool:
    return bool(form_keys(a) & form_keys(b))


def disambiguate_by_context(
    hits: Sequence[SymbolSpec],
    context: str | None,
) -> SymbolSpec | None:
    """Pick unique hit whose context_cues appear in context; else None if ambiguous."""
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    ctx = _norm(context or "")
    if not ctx:
        return None
    scored: list[tuple[int, SymbolSpec]] = []
    for symbol in hits:
        cues = [_norm(c) for c in symbol.context_cues if c]
        if not cues:
            scored.append((0, symbol))
            continue
        score = sum(1 for c in cues if c in ctx or _compact(c) in _compact(ctx))
        scored.append((score, symbol))
    scored.sort(key=lambda item: (-item[0], item[1].uid))
    best_score, best = scored[0]
    if best_score <= 0:
        return None
    if len(scored) > 1 and scored[1][0] == best_score:
        return None
    return best


def find_single_compound_extension(
    mention: str,
    symbols: Sequence[SymbolSpec],
) -> SymbolSpec | None:
    """
    «брат» / «брата» → «брат максим», если в каталоге есть единственное
    многословное имя с тем же корнем.
    """
    mention_n = _norm(mention)
    if not mention_n or " " in mention_n:
        return None
    mention_lemma = lemma(mention_n) or mention_n
    extensions: list[SymbolSpec] = []
    for sym in symbols:
        parts = _norm(sym.name).split()
        if len(parts) < 2:
            continue
        head = parts[0]
        head_lemma = lemma(head) or head
        if mention_lemma == head_lemma or mention_n == head:
            extensions.append(sym)
    if len(extensions) != 1:
        return None
    ext = extensions[0]
    ext_n = _norm(ext.name)
    if mention_lemma == ext_n or mention_n == ext_n or _compact(mention_n) == _compact(ext_n):
        return None
    return ext


def prefer_compound_extension(
    mention: str,
    hits: Sequence[SymbolSpec],
) -> SymbolSpec | None:
    return find_single_compound_extension(mention, hits)


def upgrade_head_to_compound(
    mention: str,
    selected_uid: str | None,
    symbols: Sequence[SymbolSpec],
) -> str | None:
    """Slug «БРАТ» → «БРАТ_МАКСИМ», только для автогенерированного head-slug."""
    if not selected_uid:
        return selected_uid
    compound = find_single_compound_extension(mention, symbols)
    if compound is None or compound.uid == selected_uid:
        return selected_uid
    mention_n = _norm(mention)
    if " " in mention_n:
        return selected_uid
    mention_lemma = lemma(mention_n) or mention_n
    head_slug = slug_uid(mention_lemma)
    if selected_uid != head_slug and selected_uid != slug_uid(mention_n):
        return selected_uid
    return compound.uid


@lru_cache(maxsize=4096)
def _surface_forms(word: str) -> frozenset[str]:
    """Canonical lemma + common Russian case surfaces (pymorphy)."""
    w = _norm(word)
    if not w:
        return frozenset()
    out: set[str] = {w, lemma(w) or w}
    try:
        from ah_memory.morph import _get_morph

        for parse in _get_morph().parse(w)[:4]:
            out.add(_norm(parse.normal_form))
            for tag in _CASE_TAGS:
                try:
                    inflected = parse.inflect({tag})
                except Exception:
                    continue
                if inflected is not None:
                    out.add(_norm(inflected.word))
    except Exception:
        pass
    return frozenset(out)


def _inflectional_proximity(a: str, b: str) -> float:
    """
    Soft match for truncated / near-lemma Russian forms (Юли↔Юлия).
    Requires a true stem prefix — not a short accidental overlap (машину≠маша).
    """
    left, right = _norm(a), _norm(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if min(len(left), len(right)) < 3:
        return 0.0
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    # Truncation / diminutive stem: shorter is a prefix of longer.
    if longer.startswith(shorter) and len(longer) - len(shorter) <= 3:
        return 0.9
    shared = 0
    for ca, cb in zip(left, right):
        if ca != cb:
            break
        shared += 1
    # Near-equal lemmas with tiny suffix drift (юлий/юлия), not unrelated stems.
    if (
        shared >= 4
        and abs(len(left) - len(right)) <= 2
        and shared / max(len(left), len(right)) >= 0.8
    ):
        return 0.85
    return 0.0


class SymbolResolver(Protocol):
    name: str

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult: ...


def _filter_symbols(
    symbols: Sequence[SymbolSpec],
    candidate_uids: Sequence[str] | None,
) -> list[SymbolSpec]:
    if not candidate_uids:
        return list(symbols)
    allowed = set(candidate_uids)
    return [s for s in symbols if s.uid in allowed]


class ExactResolver:
    name = "exact"

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        pool = _filter_symbols(symbols, candidate_uids)
        hits: list[SymbolSpec] = []
        for symbol in pool:
            forms = (symbol.name, *symbol.aliases)
            if any(forms_match(mention, form) for form in forms):
                hits.append(symbol)
        if not hits:
            return ResolutionResult(None, 0.0, self.name, [])
        compound = prefer_compound_extension(mention, hits)
        if compound is not None:
            cands = [CandidateScore(h.uid, 1.0, self.name) for h in hits]
            return ResolutionResult(compound.uid, 1.0, f"{self.name}+compound", cands)
        chosen = disambiguate_by_context(hits, context)
        cands = [CandidateScore(h.uid, 1.0, self.name) for h in hits]
        if chosen is None:
            return ResolutionResult(None, 1.0, "ambiguous", cands)
        return ResolutionResult(chosen.uid, 1.0, self.name, cands)


class MorphologyResolver:
    name = "morphology"

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        mention_n = _norm(mention)
        mention_lemma = lemma(mention) or mention_n
        mention_forms = _surface_forms(mention)
        pool = _filter_symbols(symbols, candidate_uids)
        by_uid = {s.uid: s for s in pool}
        scored: list[CandidateScore] = []
        for symbol in pool:
            forms = (symbol.name, *symbol.aliases)
            best = 0.0
            for form in forms:
                form_n = _norm(form)
                form_lemma = lemma(form) or form_n
                form_surfaces = _surface_forms(form)
                parts = form_n.split()
                if len(parts) > 1:
                    head = parts[0]
                    head_lemma = lemma(head) or head
                    if mention_lemma == head_lemma or mention_n == head:
                        best = max(best, 0.92)
                if forms_match(mention, form) or mention_n in form_surfaces:
                    best = max(best, 0.98)
                elif mention_lemma and mention_lemma == form_lemma:
                    best = max(best, 0.95)
                elif mention_lemma in form_surfaces or form_lemma in mention_forms:
                    best = max(best, 0.93)
                else:
                    prox = max(
                        _inflectional_proximity(mention_n, form_n),
                        _inflectional_proximity(mention_lemma, form_lemma),
                        _inflectional_proximity(mention_n, form_lemma),
                        _inflectional_proximity(mention_lemma, form_n),
                    )
                    best = max(best, prox)
            if best > 0.0:
                scored.append(CandidateScore(symbol.uid, best, self.name))
        scored.sort(key=lambda c: (-c.similarity, c.uid))
        if not scored:
            return ResolutionResult(None, 0.0, self.name, [])
        top = scored[0]
        if top.similarity < threshold:
            return ResolutionResult(None, top.similarity, self.name, scored)
        tied = [c for c in scored if abs(c.similarity - top.similarity) < 1e-9]
        hit_symbols = [by_uid[c.uid] for c in scored if c.uid in by_uid]
        compound = prefer_compound_extension(mention, hit_symbols)
        if compound is not None:
            return ResolutionResult(
                compound.uid, top.similarity, f"{self.name}+compound", scored
            )
        if len(tied) > 1:
            hits = [by_uid[c.uid] for c in tied if c.uid in by_uid]
            chosen = disambiguate_by_context(hits, context)
            if chosen is None:
                return ResolutionResult(None, top.similarity, "ambiguous", scored)
            return ResolutionResult(chosen.uid, top.similarity, self.name, scored)
        return ResolutionResult(top.uid, top.similarity, self.name, scored)


class EmbeddingResolver:
    name = "embedding"

    def __init__(
        self,
        embed: EmbeddingFn | None = None,
        *,
        dimensions: int = 64,
    ) -> None:
        self.dimensions = dimensions
        self.embed = embed or (
            lambda text: deterministic_embedding(text, dimensions)
        )

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        pool = _filter_symbols(symbols, candidate_uids)
        mention_vec = self.embed(mention)
        scored: list[CandidateScore] = []
        for symbol in pool:
            # max similarity against canonical name and aliases
            sims = [
                cosine_similarity(mention_vec, self.embed(form))
                for form in (symbol.name, *symbol.aliases)
            ]
            sim = max(sims) if sims else -1.0
            scored.append(CandidateScore(symbol.uid, float(sim), self.name))
        scored.sort(key=lambda c: (-c.similarity, c.uid))
        if not scored:
            return ResolutionResult(None, 0.0, self.name, [])
        best = scored[0]
        if best.similarity < threshold:
            return ResolutionResult(None, best.similarity, self.name, scored)
        return ResolutionResult(best.uid, best.similarity, self.name, scored)


class HybridResolver:
    """exact → morphology → alias → gated embedding (IdentityPolicy)."""

    name = "hybrid"

    def __init__(
        self,
        embed: EmbeddingFn | None = None,
        *,
        dimensions: int = 64,
        use_embeddings: bool = True,
        safety_threshold: float = 0.94,
        margin: float = 0.05,
    ) -> None:
        from ah_memory.identity import IdentityPolicy, SafeHybridResolver

        self._impl = SafeHybridResolver(
            embed,
            dimensions=dimensions,
            policy=IdentityPolicy(
                safety_threshold=safety_threshold,
                margin=margin,
            ),
            use_embeddings=use_embeddings,
        )

    def resolve(
        self,
        mention: str,
        context: str | None,
        symbols: Sequence[SymbolSpec],
        *,
        candidate_uids: Sequence[str] | None = None,
        threshold: float = 0.0,
    ) -> ResolutionResult:
        return self._impl.resolve(
            mention,
            context,
            symbols,
            candidate_uids=candidate_uids,
            threshold=threshold,
        )


def default_resolvers(
    embed: EmbeddingFn | None = None,
    *,
    dimensions: int = 64,
    safety_threshold: float = 0.94,
    margin: float = 0.05,
) -> dict[str, SymbolResolver]:
    from ah_memory.identity import AliasResolver, GatedEmbeddingResolver, IdentityPolicy

    policy = IdentityPolicy(
        safety_threshold=safety_threshold,
        margin=margin,
    )
    return {
        "exact": ExactResolver(),
        "morphology": MorphologyResolver(),
        "alias": AliasResolver(policy.synonyms),
        "embedding": GatedEmbeddingResolver(
            embed, dimensions=dimensions, policy=policy
        ),
        "hybrid": HybridResolver(
            embed,
            dimensions=dimensions,
            safety_threshold=safety_threshold,
            margin=margin,
        ),
    }


def make_embed_fn(
    model: str = "deterministic_ngram",
    *,
    dimensions: int = 64,
) -> tuple[str, EmbeddingFn, int]:
    """Build a pluggable embedding function from a config label."""
    raw = (model or "deterministic_ngram").strip()
    name = raw.lower()
    dims = dimensions

    gigachat_models = {
        "gigachat": "EmbeddingsGigaR",
        "gigachat_embeddings": "EmbeddingsGigaR",
        "embeddingsgigar": "EmbeddingsGigaR",
        "embeddings-gigar": "EmbeddingsGigaR",
        "embeddings_gigar": "EmbeddingsGigaR",
        "embeddings": "Embeddings",
        "embeddings-2": "Embeddings-2",
        "embeddings_2": "Embeddings-2",
    }
    if name in gigachat_models or raw in {
        "Embeddings",
        "Embeddings-2",
        "EmbeddingsGigaR",
    }:
        api_model = gigachat_models.get(name, raw)
        from ah_memory.config import load_config
        from ah_memory.gigachat_llm import GigaChatClient, GigaChatEmbedder

        cfg = load_config()
        if not cfg.gigachat.configured:
            raise RuntimeError(
                "GigaChat credentials are not configured "
                "(GIGACHAT_CREDENTIALS / config.yaml)"
            )
        embedder = GigaChatEmbedder(GigaChatClient(cfg.gigachat), model=api_model)
        probe = embedder("тест")
        return api_model, embedder, len(probe)

    if name in {"deterministic_ngram_128", "ngram_128"}:
        dims = 128
        name = "deterministic_ngram_128"
    elif name in {"deterministic_ngram_32", "ngram_32"}:
        dims = 32
        name = "deterministic_ngram_32"
    elif name in {"deterministic_ngram", "ngram", "deterministic"}:
        name = "deterministic_ngram"
    else:
        raise ValueError(
            f"Unknown embedding model {model!r}. "
            "Use deterministic_ngram | Embeddings | Embeddings-2 | EmbeddingsGigaR"
        )
    return name, (lambda text, d=dims: deterministic_embedding(text, d)), dims
