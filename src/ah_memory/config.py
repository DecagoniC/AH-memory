"""Load config.yaml + env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"
LOCAL_CONFIG = ROOT / "config.local.yaml"


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
        return bool(key) and key not in {"PASTE_DEEPSEEK_API_KEY_HERE", "YOUR_KEY", ""}


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


@dataclass(frozen=True)
class AppConfig:
    deepseek: DeepSeekConfig
    web: WebConfig
    agent: AgentConfig


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


def load_config(path: Path | None = None) -> AppConfig:
    _load_dotenv()
    raw: dict[str, Any] = {}
    cfg_path = path or DEFAULT_CONFIG
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if LOCAL_CONFIG.exists() and path is None:
        local = yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, local)

    ds = raw.get("deepseek", {})
    web = raw.get("web", {})
    ag = raw.get("agent", {})

    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or ds.get("api_key")
        or ""
    )

    return AppConfig(
        deepseek=DeepSeekConfig(
            api_key=api_key,
            base_url=ds.get("base_url", "https://api.deepseek.com"),
            model=ds.get("model", "deepseek-chat"),
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
        ),
    )
