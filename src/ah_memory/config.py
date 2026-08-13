"""Load config.yaml + env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from ah_memory.experiment import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"
LOCAL_CONFIG = ROOT / "config.local.yaml"

LlmProvider = Literal["gigachat", "deepseek"]


@dataclass(frozen=True)
class GigaChatConfig:
    """GigaChat Studio authorization key → OAuth access_token."""

    credentials: str
    scope: str = "GIGACHAT_API_B2B"
    base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    model: str = "GigaChat-2-Pro"
    timeout_sec: float = 60.0
    temperature: float = 0.1
    verify_ssl: bool = False

    @property
    def configured(self) -> bool:
        key = self.credentials.strip()
        return bool(key) and key not in {
            "PASTE_GIGACHAT_CREDENTIALS_HERE",
            "YOUR_KEY",
            "",
        }


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_sec: float = 60.0
    temperature: float = 0.1

    @property
    def configured(self) -> bool:
        key = self.api_key.strip()
        return bool(key) and key not in {
            "PASTE_DEEPSEEK_API_KEY_HERE",
            "YOUR_KEY",
            "",
        }


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class AgentConfig:
    use_llm: bool = True
    fallback_rules: bool = True
    ticks: int = 6
    preload: str = "empty"
    llm_provider: LlmProvider = "gigachat"


@dataclass(frozen=True)
class OpenSemanticsConfig:
    enabled: bool = True
    normalization_mode: str = "normalized"
    embedding_similarity_threshold: float = 0.55
    parameter_seed: int = 42
    trace_messages: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding backend label + size (entity resolution / open semantics)."""

    model: str = "deterministic_ngram"
    dimensions: int = 64


@dataclass(frozen=True)
class IdentityConfig:
    """Symbol identity at ingest: avoid duplicates via resolve-before-create."""

    enabled: bool = True
    use_embeddings: bool = True
    safety_threshold: float = 0.94
    margin: float = 0.05


@dataclass(frozen=True)
class AppConfig:
    gigachat: GigaChatConfig
    deepseek: DeepSeekConfig
    web: WebConfig
    agent: AgentConfig
    experiment: ExperimentConfig = ExperimentConfig()
    open_semantics: OpenSemanticsConfig = OpenSemanticsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    identity: IdentityConfig = IdentityConfig()


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _norm_provider(raw: str | None) -> LlmProvider:
    p = (raw or "gigachat").strip().lower()
    if p in {"deepseek", "ds"}:
        return "deepseek"
    return "gigachat"


def load_config(path: Path | None = None) -> AppConfig:
    _load_dotenv()
    raw: dict[str, Any] = {}
    cfg_path = path or DEFAULT_CONFIG
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if LOCAL_CONFIG.exists() and path is None:
        local = yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, local)

    gc = raw.get("gigachat") or {}
    ds = raw.get("deepseek") or {}
    web = raw.get("web", {})
    ag = raw.get("agent", {})
    semantics = raw.get("open_semantics", {})
    emb = raw.get("embedding", {}) or {}
    ident = raw.get("identity", {}) or {}

    credentials = (
        os.environ.get("GIGACHAT_CREDENTIALS")
        or os.environ.get("GIGACHAT_API_KEY")
        or gc.get("credentials")
        or gc.get("api_key")
        or ""
    )
    scope = os.environ.get("GIGACHAT_SCOPE") or gc.get("scope", "GIGACHAT_API_B2B")
    ds_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or ds.get("api_key")
        or ""
    )
    provider = _norm_provider(
        os.environ.get("LLM_PROVIDER") or ag.get("llm_provider") or raw.get("llm_provider")
    )

    return AppConfig(
        gigachat=GigaChatConfig(
            credentials=str(credentials),
            scope=str(scope),
            base_url=str(gc.get("base_url", "https://gigachat.devices.sberbank.ru/api/v1")),
            auth_url=str(gc.get("auth_url", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")),
            model=str(gc.get("model", "GigaChat-2-Pro")),
            timeout_sec=float(gc.get("timeout_sec", 60)),
            temperature=float(gc.get("temperature", 0.1)),
            verify_ssl=bool(gc.get("verify_ssl", False)),
        ),
        deepseek=DeepSeekConfig(
            api_key=str(ds_key),
            base_url=str(ds.get("base_url", "https://api.deepseek.com")),
            model=str(ds.get("model", "deepseek-chat")),
            timeout_sec=float(ds.get("timeout_sec", 60)),
            temperature=float(ds.get("temperature", 0.1)),
        ),
        web=WebConfig(
            host=web.get("host", "127.0.0.1"),
            port=int(web.get("port", 8000)),
        ),
        agent=AgentConfig(
            use_llm=bool(ag.get("use_llm", True)),
            fallback_rules=bool(ag.get("fallback_rules", True)),
            ticks=int(ag.get("ticks", 6)),
            preload=str(ag.get("preload", "empty")),
            llm_provider=provider,
        ),
        experiment=ExperimentConfig.from_mapping(raw.get("experiment") or {}),
        open_semantics=OpenSemanticsConfig(
            enabled=bool(semantics.get("enabled", True)),
            normalization_mode=str(
                semantics.get("normalization_mode", "normalized")
            ),
            embedding_similarity_threshold=float(
                semantics.get("embedding_similarity_threshold", 0.55)
            ),
            parameter_seed=int(semantics.get("parameter_seed", 42)),
            trace_messages=bool(semantics.get("trace_messages", False)),
        ),
        embedding=EmbeddingConfig(
            model=str(emb.get("model", "deterministic_ngram")),
            dimensions=int(emb.get("dimensions", 64)),
        ),
        identity=IdentityConfig(
            enabled=bool(ident.get("enabled", True)),
            use_embeddings=bool(ident.get("use_embeddings", True)),
            safety_threshold=float(ident.get("safety_threshold", 0.94)),
            margin=float(ident.get("margin", 0.05)),
        ),
    )
