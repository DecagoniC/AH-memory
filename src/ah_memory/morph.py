"""Russian lemmatization + POS gates for AH symbols (pymorphy3)."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_morph: Any | None = None

# function words / discourse — never symbols or role fillers
STOP = {
    "и", "в", "на", "с", "со", "по", "к", "ко", "у", "о", "об", "обо", "а", "но", "да",
    "что", "это", "как", "или", "либо", "для", "из", "за", "от", "до", "не", "ни",
    "он", "она", "они", "оно", "вы", "его", "её", "ее", "их", "мне", "меня",
    "мой", "моя", "мое", "моё", "мои", "моего", "моей", "моём", "моем", "твои", "свой",
    "наш", "ваш", "ихний",
    "же", "ли", "бы", "б", "то", "все", "всё", "так", "уже", "ещё", "еще", "только",
    "также", "тоже", "если", "когда", "пока", "чтобы", "хотя", "потому", "поэтому",
    "ведь", "даже", "вот", "вон", "ну", "давай", "пусть", "разве", "неужели",
    "который", "которая", "которое", "которые", "какой", "какая", "какие",
    "этот", "эта", "эти", "тот", "та", "те", "там", "тут", "здесь", "куда", "откуда",
    "очень", "можно", "нужно", "надо", "есть", "будет", "быть", "был", "была", "были",
    "привет", "пока", "спасибо", "пожалуйста", "отлично", "хорошо", "ладно", "кстати",
    "младший", "старший",  # kinship adj — keep брат/имя as head, drop bare adj seeds
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "be", "and", "or",
    "who", "what", "when", "where", "why", "how", "can", "may", "if", "then",
}

# surface forms pymorphy mangles (acronyms / surnames)
LEMMA_OVERRIDE = {
    "мифи": "мифи",
    "нияу": "нияу",
    "душкин": "душкин",
    "душкина": "душкин",
    "душкину": "душкин",
    "душкиным": "душкин",
    "душкине": "душкин",
    "сергей": "сергей",
    "максим": "максим",
    "москве": "москва",
    "москвы": "москва",
    "москву": "москва",
}

# max tokens in a role UID (mega-subjects from whole-clause match)
MAX_UID_PARTS = 4

_NON_ENTITY_POS = frozenset({
    "VERB", "INFN", "GRND", "PRTF", "PRTS", "ADVB", "CONJ", "PREP", "PRCL",
    "INTJ", "PRED", "COMP",
})


def _get_morph() -> Any:
    global _morph
    if _morph is None:
        from pymorphy3 import MorphAnalyzer

        _morph = MorphAnalyzer()
    return _morph


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


@lru_cache(maxsize=8192)
def _parse(word: str) -> Any:
    return _get_morph().parse(_norm(word))[0]


def lemma(word: str) -> str:
    w = _norm(word)
    if not w:
        return ""
    if w in LEMMA_OVERRIDE:
        return LEMMA_OVERRIDE[w]
    if w.isdigit() or re.fullmatch(r"[a-z0-9_\-]+", w):
        # keep latin/digits stable (UIDs, years)
        return w
    return _parse(w).normal_form.replace("ё", "е")


def pos_of(word: str) -> str | None:
    w = _norm(word)
    if not w or w.isdigit():
        return "NUMR" if w.isdigit() else None
    if re.fullmatch(r"[a-z0-9_\-]+", w):
        return "LATN"
    return _parse(w).tag.POS


_PRONOUNS = frozenset({"я", "мы", "ты"})


def is_entity_token(word: str, *, allow_pronoun: bool = False) -> bool:
    """True if token can be SUBJECT/OBJECT/LOCATION / seed symbol."""
    w = _norm(word)
    if not w:
        return False
    if allow_pronoun and w in _PRONOUNS:
        return True
    if w in STOP or len(w) < 2:
        return False
    if w.isdigit():
        return True
    # latin tokens: keep for demo UIDs (HARE), reject long unknown latin junk lightly
    if re.fullmatch(r"[a-z0-9_\-]+", w):
        return True
    p = _parse(w)
    pos = p.tag.POS
    if pos in _NON_ENTITY_POS:
        return False
    if pos == "NPRO":
        return False
    if pos in {"NOUN", "ADJF", "ADJS", "NUMR"}:
        return True
    tag = str(p.tag)
    return any(g in tag for g in ("Name", "Surn", "Patr", "Geox", "Orgn", "Trad"))


def is_nounish(word: str, *, allow_pronoun: bool = False) -> bool:
    """Stricter: prefer nouns / names as role heads."""
    w = _norm(word)
    if not w or w in STOP:
        return False
    if w.isdigit() or re.fullmatch(r"[a-z0-9_\-]+", w):
        return True
    p = _parse(w)
    pos = p.tag.POS
    if pos == "NOUN":
        return True
    if pos == "NPRO":
        return allow_pronoun and w in _PRONOUNS
    tag = str(p.tag)
    return any(g in tag for g in ("Name", "Surn", "Patr", "Geox", "Orgn", "Trad"))


def slug_uid(token: str) -> str:
    """Surface / multiword → UPPER_SNAKE lemma UID."""
    t = _norm(token).replace("—", "-").replace("–", "-")
    parts = [p for p in re.split(r"[^a-zа-я0-9]+", t) if p]
    if not parts:
        return "UNK"
    lemmas: list[str] = []
    for p in parts:
        lem = lemma(p)
        if not lem or lem in STOP:
            continue
        lemmas.append(lem)
    if not lemmas:
        # fallback: lemma of whole token
        lem = lemma(parts[0])
        lemmas = [lem] if lem else ["unk"]
    # prefer last nounish head for overlong spans
    if len(lemmas) > MAX_UID_PARTS:
        nouns = [x for x in lemmas if is_nounish(x, allow_pronoun=True)]
        lemmas = (nouns or lemmas)[-MAX_UID_PARTS:]
    uid = "_".join(lemmas).upper()
    return uid[:48] if uid else "UNK"


def uid_too_wide(uid: str) -> bool:
    bare = uid[2:] if uid.startswith("M_") else uid
    parts = [p for p in bare.split("_") if p]
    if len(parts) > MAX_UID_PARTS:
        return True
    # clause-glue: repeated pronouns / duplicated stems
    low = [p.lower() for p in parts]
    if low.count("я") > 1 or low.count("мы") > 1:
        return True
    return False


def head_entity(tokens: list[str], *, allow_pronoun: bool = True) -> str | None:
    """Entity span: adj+noun compounds kept; must contain noun/name (or pronoun)."""
    kept: list[str] = []
    for t in tokens:
        if is_entity_token(t, allow_pronoun=allow_pronoun):
            kept.append(lemma(t))
    if not kept:
        return None
    if not any(is_nounish(k, allow_pronoun=allow_pronoun) for k in kept):
        return None
    if len(kept) > MAX_UID_PARTS:
        nouns = [k for k in kept if is_nounish(k, allow_pronoun=allow_pronoun)]
        kept = (nouns or kept)[-MAX_UID_PARTS:]
    uid = slug_uid(" ".join(kept))
    if uid_too_wide(uid):
        return None
    return uid


def filter_entity_uids(uids: list[str], *, allow_pronoun: bool = False) -> list[str]:
    out: list[str] = []
    for u in uids:
        bare = u[2:] if u.startswith("M_") else u
        if uid_too_wide(bare):
            continue
        parts = [p for p in re.split(r"[_\s]+", bare.lower()) if p]
        head = head_entity(parts, allow_pronoun=allow_pronoun)
        if head is None:
            continue
        if head not in out and head != "UNK" and not uid_too_wide(head):
            out.append(head)
    return out


def seeds_from_roles(candidates: list, *, extra: list[str] | None = None) -> list[str]:
    """Seeds = role fillers (+ optional short extras), not whole-utterance bag-of-nouns."""
    raw: list[str] = []
    for c in candidates:
        roles = getattr(c, "roles", None) or {}
        raw.extend(str(v) for v in roles.values())
    if extra:
        raw.extend(extra)
    return filter_entity_uids(raw, allow_pronoun=True)


def sanitize_roles(roles: dict[str, str]) -> dict[str, str] | None:
    """Drop / rewrite role fillers that are not entity-like. None = discard candidate."""
    out: dict[str, str] = {}
    for role, val in roles.items():
        bare = val[2:] if str(val).startswith("M_") else str(val)
        if uid_too_wide(bare):
            return None
        parts = [p for p in re.split(r"[_\s]+", bare.lower()) if p]
        if role in {"SUBJECT", "OBJECT", "LOCATION", "TOOL", "MATERIAL"}:
            allow_p = role == "SUBJECT"
            if bare.isdigit() or (re.fullmatch(r"[A-Za-z0-9_]+", bare) and "_" not in bare and len(bare) <= 8):
                # years / short latin codes
                if bare.isdigit():
                    out[role] = bare
                    continue
            head = head_entity(parts, allow_pronoun=allow_p)
            if head is None:
                return None
            # OBJECT/LOCATION must be nounish (drop pure adjectives like СЕРЬЁЗНЫЙ)
            if role in {"OBJECT", "LOCATION", "TOOL", "MATERIAL"}:
                if not any(is_nounish(p, allow_pronoun=False) for p in head.lower().split("_")):
                    return None
            out[role] = head
        elif role == "TIME":
            out[role] = slug_uid(bare)
        else:
            out[role] = slug_uid(bare)
    if "SUBJECT" not in out:
        return None
    # SUBJECT must not be a verb lemma
    subj_parts = out["SUBJECT"].lower().split("_")
    if not any(is_nounish(p, allow_pronoun=True) for p in subj_parts):
        return None
    if uid_too_wide(out["SUBJECT"]):
        return None
    return out
