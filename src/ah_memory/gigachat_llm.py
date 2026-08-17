"""GigaChat client + HybridPerception (LLM → gate; offline seeds)."""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import httpx

from ah_memory.config import GigaChatConfig
from ah_memory.morph import seeds_from_roles, slug_uid
from ah_memory.perception import (
    PerceptionResult,
    SeedPerception,
    _is_question,
    _norm,
    candidates_from_llm_json,
    content_entity_uids,
    gate_candidates,
    llm_payload_errors,
)
from ah_memory.perception_prompt import SYSTEM_PROMPT, build_user_payload


class GigaChatClient:
    """OAuth (credentials → access_token) + /chat/completions."""

    def __init__(self, cfg: GigaChatConfig) -> None:
        self.cfg = cfg
        self._token: str | None = None
        self._expires_at_ms: int = 0

    def _http(self) -> httpx.Client:
        return httpx.Client(timeout=self.cfg.timeout_sec, verify=self.cfg.verify_ssl)

    def _normalize_credentials(self) -> str:
        raw = self.cfg.credentials.strip().strip('"').strip("'")
        if raw.lower().startswith("basic "):
            raw = raw[6:].strip()
        return raw

    def _ensure_token(self, client: httpx.Client) -> str:
        now = int(time.time() * 1000)
        if self._token and now < self._expires_at_ms - 60_000:
            return self._token

        credentials = self._normalize_credentials()
        scopes = [self.cfg.scope]
        for alt in ("GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"):
            if alt not in scopes:
                scopes.append(alt)

        last_err = ""
        for scope in scopes:
            r = client.post(
                self.cfg.auth_url,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": scope},
            )
            if r.status_code == 200:
                data = r.json()
                self._token = str(data["access_token"])
                self._expires_at_ms = int(data.get("expires_at") or (now + 25 * 60_000))
                return self._token
            last_err = (r.text or r.reason_phrase or "")[:400]
            # неверный scope — пробуем следующий; иначе сразу стоп
            if "scope" not in last_err.lower() and r.status_code != 400:
                break
            if r.status_code not in {400, 403}:
                break

        raise httpx.HTTPStatusError(
            f"GigaChat OAuth failed ({self.cfg.auth_url}): {last_err}. "
            f"Проверь GIGACHAT_CREDENTIALS и GIGACHAT_SCOPE "
            f"(PERS / B2B / CORP должен совпадать с типом ключа в Studio).",
            request=r.request,
            response=r,
        )

    @staticmethod
    def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """GigaChat: at most one system message, and it must be first."""
        systems: list[str] = []
        rest: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                if content:
                    systems.append(content)
                continue
            if role not in {"user", "assistant", "function"}:
                role = "user"
            rest.append({"role": role, "content": content})
        out: list[dict[str, str]] = []
        if systems:
            out.append({"role": "system", "content": "\n\n".join(systems)})
        out.extend(rest)
        if not out:
            out = [{"role": "user", "content": ""}]
        return out

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._normalize_messages(messages),
            "temperature": self.cfg.temperature if json_mode else min(0.7, self.cfg.temperature + 0.4),
        }
        # GigaChat не всегда принимает response_format; JSON просим промптом.
        with self._http() as client:
            token = self._ensure_token(client)
            url = self.cfg.base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            r = client.post(url, headers=headers, json=payload)
            if r.status_code == 401:
                self._token = None
                token = self._ensure_token(client)
                headers["Authorization"] = f"Bearer {token}"
                r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                detail = (r.text or r.reason_phrase or "")[:500]
                raise httpx.HTTPStatusError(
                    f"GigaChat chat failed ({r.status_code}): {detail}",
                    request=r.request,
                    response=r,
                )
            data = r.json()
        return data["choices"][0]["message"]["content"]

    def embeddings(
        self,
        texts: list[str],
        *,
        model: str = "EmbeddingsGigaR",
    ) -> list[list[float]]:
        """POST /embeddings — Sber vector representations for input texts."""
        if not texts:
            return []
        payload = {"model": model, "input": texts}
        with self._http() as client:
            token = self._ensure_token(client)
            url = self.cfg.base_url.rstrip("/") + "/embeddings"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            r = client.post(url, headers=headers, json=payload)
            if r.status_code == 401:
                self._token = None
                token = self._ensure_token(client)
                headers["Authorization"] = f"Bearer {token}"
                r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                detail = (r.text or r.reason_phrase or "")[:500]
                raise httpx.HTTPStatusError(
                    f"GigaChat embeddings failed ({r.status_code}): {detail}",
                    request=r.request,
                    response=r,
                )
            data = r.json()
        items = sorted(data.get("data") or [], key=lambda row: int(row.get("index", 0)))
        if len(items) != len(texts):
            raise RuntimeError(
                f"GigaChat embeddings size mismatch: got {len(items)} for {len(texts)} inputs"
            )
        return [list(map(float, row["embedding"])) for row in items]


# Symmetric instruction for EmbeddingsGigaR (entity / synonym matching).
DEFAULT_GIGAR_INSTRUCTION = (
    "Дано обозначение сущности или понятия, найди тождественное "
    "или синонимичное обозначение той же сущности\nтекст: {query}"
)


class GigaChatEmbedder:
    """Cached embed callable backed by GigaChat /embeddings."""

    def __init__(
        self,
        client: GigaChatClient,
        *,
        model: str = "EmbeddingsGigaR",
        instruction: str | None = DEFAULT_GIGAR_INSTRUCTION,
        batch_size: int = 16,
    ) -> None:
        self.client = client
        self.model = model
        self.instruction = (
            instruction
            if instruction and model.lower() == "embeddingsgigar"
            else None
        )
        self.batch_size = max(1, batch_size)
        self._cache: dict[str, tuple[float, ...]] = {}

    def _payload_text(self, text: str) -> str:
        raw = text.strip()
        if self.instruction:
            return self.instruction.format(query=raw)
        return raw

    def warm(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            vectors = self.client.embeddings(
                [self._payload_text(t) for t in chunk],
                model=self.model,
            )
            for text, vector in zip(chunk, vectors):
                self._cache[text] = tuple(vector)

    def __call__(self, text: str) -> tuple[float, ...]:
        key = text.strip()
        if key not in self._cache:
            self.warm([key])
        return self._cache[key]


class GigaChatPerception:
    def __init__(self, cfg: GigaChatConfig, *, require_grounding: bool = True) -> None:
        self.client = GigaChatClient(cfg)
        self.require_grounding = require_grounding

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        user = build_user_payload(text, wm_context)
        raw = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        data = _parse_json(raw)
        cands = candidates_from_llm_json(data)
        gated_result = gate_candidates(
            text,
            cands,
            require_grounding=self.require_grounding,
            allow_open_relations=True,
            report=True,
        )
        gated, gate_report = gated_result
        validation_errors = llm_payload_errors(data)
        validation_errors.extend(
            f"candidate rejected: {item.get('reason', 'validation')}"
            for item in gate_report.get("dropped", [])
        )
        kind = data.get("kind", "fact")
        if kind not in {"fact", "question", "message"}:
            kind = "fact"
        low = _norm(text.strip())
        if _is_question(text.strip(), low):
            kind = "question"
        elif gated:
            kind = "fact"
        seeds = seeds_from_roles(
            gated,
            extra=[slug_uid(str(s)) for s in data.get("seed_tokens", [])][:12],
        )
        if not seeds:
            seeds = content_entity_uids(text)[:8]
        if kind != "question" and not gated:
            kind = "message"
        return PerceptionResult(
            kind=kind,
            candidates=gated,
            seed_tokens=seeds,
            meta={
                "backend": "gigachat",
                "llm_raw": data,
                "gate_report": gate_report,
                "validation_errors": validation_errors,
                "system_prompt": SYSTEM_PROMPT,
            },
        )


class HybridPerception:
    """LLM → morph gate; offline / error → SeedPerception (seeds only, no facts)."""

    def __init__(self, cfg: GigaChatConfig, fallback: bool = True) -> None:
        self.cfg = cfg
        self.fallback = fallback
        self.seeds = SeedPerception()
        self.llm = GigaChatPerception(cfg) if cfg.configured else None

    def parse(self, text: str, wm_context: list[str] | None = None) -> PerceptionResult:
        offline = self.seeds.parse(text, wm_context)
        if self.llm is None:
            return offline
        try:
            llm = self.llm.parse(text, wm_context)
        except Exception as exc:  # noqa: BLE001
            if not self.fallback:
                raise
            return PerceptionResult(
                kind=offline.kind,
                candidates=[],
                seed_tokens=offline.seed_tokens,
                meta={**offline.meta, "backend": "seeds", "llm_error": str(exc)},
            )

        seeds = seeds_from_roles(llm.candidates, extra=list(llm.seed_tokens)[:8])
        if not seeds:
            seeds = list(offline.seed_tokens)[:8]
        return PerceptionResult(
            kind=llm.kind,
            candidates=list(llm.candidates),
            seed_tokens=seeds,
            meta={
                "backend": "hybrid-llm" if llm.candidates else "hybrid-seeds",
                "llm_raw": llm.meta.get("llm_raw"),
                "gate_report": llm.meta.get("gate_report"),
                "system_prompt": llm.meta.get("system_prompt", SYSTEM_PROMPT),
                "llm_candidates": len(llm.candidates),
            },
        )


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))
