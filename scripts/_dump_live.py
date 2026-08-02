import json
import urllib.request
from pathlib import Path

dump = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/dump").read().decode("utf-8"))
graph = json.loads(
    urllib.request.urlopen("http://127.0.0.1:8000/api/graph?mode=hyper&limit=400").read().decode("utf-8")
)
out: list[str] = []
st = graph["stats"]
out.append("## Stats")
out.append(
    f"S={st['S']} C={st['C']} H={st['H']} L={st['L']} hyperedges={st['hyperedges']} tau={st['tau']}"
)
out.append("")
out.append("## Episodes (Q/A)")
for sec in ("H", "C", "P"):
    for uid, e in (dump.get(sec) or {}).items():
        pr = e.get("Pr") or []
        mt = e.get("Mt") or []
        label = next((x.get("value") for x in pr if isinstance(x, dict) and x.get("name") == "label"), None)
        kind = next((x.get("value") for x in mt if isinstance(x, dict) and x.get("name") == "kind"), None)
        if kind == "Episode" or (label and ("USER" in str(label) or "ASSISTANT" in str(label))):
            out.append(f"- {uid}: {label}")
            out.append(f"  items: {e.get('items') or []}")
out.append("")
out.append("## Hyperedges")
for h in graph.get("hyperedges", []):
    roles = ", ".join(f"{k}={v}" for k, v in (h.get("roles") or {}).items())
    out.append(f"- {h['id']} {h['predicate']}({roles}) w={float(h.get('w', 0)):.3f}")
out.append("")
out.append("## Symbols S / m")
for n in graph.get("nodes", []):
    g = str(n.get("group", ""))
    if g == "S" or g.endswith("_m"):
        out.append(f"- [{g}] {n.get('id')} :: {n.get('label')}")
out.append("")
out.append("## Episode nodes in viz")
for n in graph.get("nodes", []):
    if str(n.get("id", "")).startswith("EP_"):
        out.append(f"- {n.get('id')} :: {n.get('label')}")

path = Path(r"D:\Go-prog\AH-memory\docs\_live_qa_dump.md")
path.write_text("\n".join(out), encoding="utf-8")
print(path, "lines", len(out))
