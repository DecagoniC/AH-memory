"""Control-model templates T (≥8, CREATE with 7 actants)."""
from __future__ import annotations

from ah_memory.store import AHStore
from ah_memory.types import ActantSlot, Role, Section, Template

# CREATE etalon (fig. 6): 7 roles
CREATE_ROLES = [
    Role.SUBJECT,
    Role.OBJECT,
    Role.LOCATION,
    Role.TIME,
    Role.CAUSE,
    Role.TOOL,
    Role.MATERIAL,
]

TEMPLATES: list[tuple[str, str, list[Role]]] = [
    ("T_CREATE", "CREATE", CREATE_ROLES),
    ("T_IS", "IS", [Role.SUBJECT, Role.OBJECT]),
    ("T_LIVE_IN", "LIVE_IN", [Role.SUBJECT, Role.LOCATION]),
    ("T_BE_BORN", "BE_BORN", [Role.SUBJECT, Role.LOCATION]),
    ("T_HAVE", "HAVE", [Role.SUBJECT, Role.OBJECT]),
    ("T_RUN", "RUN", [Role.SUBJECT, Role.HOW_TO, Role.CAUSE]),
    ("T_COLOR", "BE_COLORED", [Role.SUBJECT, Role.OBJECT, Role.TIME]),
    ("T_CAUSE", "CAUSE_EVENT", [Role.SUBJECT, Role.OBJECT, Role.CAUSE]),
    ("T_USE", "USE", [Role.SUBJECT, Role.OBJECT, Role.TOOL, Role.PURPOSE]),
    ("T_MOVE", "MOVE", [Role.SUBJECT, Role.OBJECT, Role.LOCATION, Role.TIME]),
]


def ensure_template(store: AHStore, predicate: str) -> str:
    """Create one T for predicate on demand. Returns template UID."""
    pred = predicate.upper()
    for tpl_uid, p, roles in TEMPLATES:
        if p != pred:
            continue
        store.ensure_abstract(p, {p.lower(), p.replace("_", " ").lower()})
        if tpl_uid not in store.ah.all_hyper():
            store.add_element(
                Section.C,
                Template(
                    uid=tpl_uid,
                    predicate=store.s_ref(p),
                    actants=[ActantSlot(role=r) for r in roles],
                ),
            )
        return tpl_uid
    raise KeyError(f"unknown predicate template: {predicate}")


def seed_templates(store: AHStore) -> None:
    """Eager seed of all templates (tests / encyclopedia demos only)."""
    for _, pred, _ in TEMPLATES:
        ensure_template(store, pred)


def template_roles_coverage(store: AHStore) -> set[Role]:
    covered: set[Role] = set()
    for t in store.find_templates():
        covered.update(a.role for a in t.actants)
    return covered
