"""Сводка всех прогонов метрик в docs/rezultaty_testov.md и .csv."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs" / "rezultaty_testov.md"
OUT_CSV = ROOT / "docs" / "rezultaty_testov.csv"


def ru(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.{digits}f}".replace(".", ",")
        text = text.rstrip("0").rstrip(",") if digits > 0 else text
        if text in {"-0", ""}:
            return "0"
        return text
    return str(value)


def yn(flag: object) -> str:
    return "да" if bool(flag) else "нет"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join([line, sep, body])


def main() -> None:
    hf = load(ROOT / "data" / "provider_compare_openai_hf.json")
    shared = load(ROOT / "data" / "shared_graph_llm_slm.json")
    rabbit_llm = load(ROOT / "docs" / "submission" / "m4_rabbit_llm.json")
    rabbit_ext = load(ROOT / "docs" / "submission" / "m4_rabbit_extractive.json")
    m1_g = load(ROOT / "docs" / "submission" / "m1_deepseek_gated.json")
    m1_u = load(ROOT / "docs" / "submission" / "m1_deepseek_ungated.json")
    m1_seed = load(ROOT / "docs" / "submission" / "m1_seed_full.json")
    ch = load(ROOT / "docs" / "submission" / "challenge" / "20260820T193218Z" / "summary.json")
    m2_items = load_jsonl(
        ROOT / "docs" / "submission" / "challenge" / "20260820T193218Z" / "m2_items.jsonl"
    )
    hypo = load(ROOT / "docs" / "submission" / "hypothesis_logs.json")
    m4_hf_old = load(ROOT / "data" / "m4_openai_hf_report.json")

    csv_rows: list[list[str]] = []

    def rec(metric: str, stand: str, series: str, value: object, note: str = "") -> None:
        csv_rows.append([metric, stand, series, ru(value) if not isinstance(value, str) else value, note])

    models = hf["models"]
    m1 = hf["m1_m5"]["m1"]
    m4 = hf["m4"]

    parts: list[str] = []
    parts.append("# Результаты тестов АГ-памяти")
    parts.append("")
    parts.append("Сводка прогонов M1–M5, эталона «заяц», офлайн-челленджа и протокола общего графа. Числа взяты из JSON логов, не округлялись вручную сверх отображения.")
    parts.append("")
    parts.append("**Источники:** `data/provider_compare_openai_hf.json`, `data/shared_graph_llm_slm.json`, `docs/submission/m1_deepseek_*.json`, `docs/submission/m4_rabbit_*.json`, `docs/submission/challenge/20260820T193218Z/`, `docs/submission/hypothesis_logs.json`.")
    parts.append("")
    parts.append("Модели живого стенда: SLM `qwen3:8b`, LLM `deepseek-chat` и `GigaChat-2-Pro`, эмбеддинги `nomic-embed-text`.")
    parts.append("")

    # ── сводка ──
    parts.append("## 1. Сводка")
    parts.append("")
    summary_rows = [
        ["M1 F1 AH-gated", "инцидент HF, n=12", "qwen3:8b", ru(m1["ah_ollama_8b"]["normalized_weighted_f1"]), "ниже порога 0,6"],
        ["M1 F1 AH-gated", "инцидент HF, n=12", "GigaChat-2-Pro", ru(m1["ah_gigachat"]["normalized_weighted_f1"]), "Middle пройден"],
        ["M1 F1 AH-gated", "инцидент HF, n=12", "deepseek-chat", ru(m1["ah_deepseek"]["normalized_weighted_f1"]), "Middle пройден"],
        ["M1 F1 без гейта", "инцидент HF, n=12", "qwen3:8b", ru(m1["rag_ollama_8b"]["normalized_weighted_f1"]), ""],
        ["M1 F1 без гейта", "инцидент HF, n=12", "GigaChat-2-Pro", ru(m1["rag_gigachat"]["normalized_weighted_f1"]), ""],
        ["M1 F1 без гейта", "инцидент HF, n=12", "deepseek-chat", ru(m1["rag_deepseek"]["normalized_weighted_f1"]), ""],
        ["M1 F1 AH-gated", "челлендж, n=16", "deepseek-chat", ru(m1_g["score"]["normalized_weighted_f1"]), "пояснительная записка"],
        ["M1 F1 без LLM", "челлендж, n=100", "SeedPerception", ru(m1_seed["score"]["normalized_weighted_f1"]), "только seed_tokens"],
        ["M2 ExplainScore", "офлайн челлендж, n=20", "AH", ru(ch["m2"]["explain_score"]), f"{ch['m2']['correct']}/{ch['m2']['items']} с полной трассой"],
        ["M3 GC efficiency", "офлайн челлендж", "AH", ru(ch["m3"]["gc_efficiency"]), f"{ch['m3']['orphans_before']} → {ch['m3']['orphans_after']} сирот"],
        ["M4 Δ объяснимости", "заяц, LLM RAG, n=6", "AH − RAG", ru(rabbit_llm["summary"]["delta_explainability"]), "гипотеза да"],
        ["M4 Δ галлюцинаций", "заяц, LLM RAG, n=6", "RAG − AH", ru(rabbit_llm["summary"]["delta_hallucination"]), "гипотеза да"],
        ["M4 Δ объяснимости", "заяц, extractive, n=6", "AH − RAG", ru(rabbit_ext["summary"]["delta_explainability"]), "H2 нет: оба не врут"],
        ["M4 Δ объяснимости", "HF, граф пересобран, n=6", "AH − любой RAG", ru(m4["extractive+nomic"]["delta_explainability"]), "0,25 на всех бэкендах"],
        ["M5 RobustnessGain", "HF, n=12", "8b vs GigaChat", ru(hf["m1_m5"]["m5_vs_gigachat"]["robustness_gain"]), "гипотеза отвергнута"],
        ["M5 RobustnessGain", "HF, n=12", "8b vs DeepSeek", ru(hf["m1_m5"]["m5_vs_deepseek"]["robustness_gain"]), "плюс из просадки LLM"],
    ]
    parts.append(md_table(["Метрика", "Стенд", "Ряд", "Значение", "Комментарий"], summary_rows))
    parts.append("")
    for row in summary_rows:
        rec(row[0], row[1], row[2], row[3], row[4])

    # ── M1 live ──
    parts.append("## 2. M1 — извлечение ролей (живой стенд Hugging Face)")
    parts.append("")
    parts.append("Нормированный взвешенный F1, веса SUBJECT = OBJECT = 2, LOCATION = 1. Порог Middle ≥ 0,6. n = 12.")
    parts.append("")
    m1_rows = []
    mapping = [
        ("qwen3:8b", "ah_ollama_8b", "rag_ollama_8b"),
        ("GigaChat-2-Pro", "ah_gigachat", "rag_gigachat"),
        ("deepseek-chat", "ah_deepseek", "rag_deepseek"),
    ]
    for name, ah_k, rag_k in mapping:
        ah = m1[ah_k]["normalized_weighted_f1"]
        ung = m1[rag_k]["normalized_weighted_f1"]
        m1_rows.append([
            name,
            ru(ah),
            ru(ung),
            ru(ah - ung),
            ru(m1[ah_k]["sec"], 2),
            ru(m1[rag_k]["sec"], 2),
            "проходит" if ah >= 0.6 else "не проходит",
        ])
        rec("M1 F1 AH-gated", "HF n=12", name, ah)
        rec("M1 F1 без гейта", "HF n=12", name, ung)
    parts.append(md_table(
        ["Модель", "F1 AH-gated", "F1 без гейта", "gated − ungated", "сек gated", "сек ungated", "против 0,6"],
        m1_rows,
    ))
    parts.append("")

    parts.append("### 2.1. DeepSeek, выборка челленджа n = 16")
    parts.append("")
    role_rows = []
    for role in ("SUBJECT", "OBJECT", "LOCATION"):
        g = m1_g["score"]["by_role"][role]
        role_rows.append([role, g["correct"], g["predicted"], g["expected"], ru(g["precision"]), ru(g["recall"]), ru(g["f1"])])
        rec(f"M1 F1 роль {role}", "челлендж n=16 DeepSeek gated", role, g["f1"])
    parts.append(md_table(["Роль", "TP", "Pred", "Gold", "P", "R", "F1"], role_rows))
    parts.append("")
    parts.append(f"Итоговый нормированный взвешенный F1 gated = **{ru(m1_g['score']['normalized_weighted_f1'])}**. Ungated на этой выборке совпал: **{ru(m1_u['score']['normalized_weighted_f1'])}**.")
    parts.append("")
    rec("M1 F1 итоговый", "челлендж n=16", "deepseek gated", m1_g["score"]["normalized_weighted_f1"])
    rec("M1 F1 итоговый", "челлендж n=16", "deepseek ungated", m1_u["score"]["normalized_weighted_f1"])
    rec("M1 F1 итоговый", "челлендж n=100", "SeedPerception", m1_seed["score"]["normalized_weighted_f1"])

    # ── M2 ──
    parts.append("## 3. M2 — ExplainScore по глубине")
    parts.append("")
    parts.append("Офлайн `20260820T193218Z`, d_max = 6, N = 20. Вклад пункта: верный × (d / 6) × полная трасса.")
    parts.append("")
    by_d: dict[int, Counter] = {}
    for item in m2_items:
        d = int(item["depth"])
        by_d.setdefault(d, Counter())
        by_d[d]["n"] += 1
        if item["correct"] and item["trace_complete"]:
            by_d[d]["ok"] += 1
            by_d[d]["contrib"] += d / 6.0
    m2_rows = []
    total_ok = 0
    total_contrib = 0.0
    for d in range(1, 7):
        c = by_d.get(d, Counter())
        total_ok += c["ok"]
        total_contrib += c["contrib"]
        m2_rows.append([d, c["n"], c["ok"], ru(c["contrib"]), ru(c["ok"] / c["n"] if c["n"] else 0)])
        rec("M2 успех", f"глубина {d}", "доля верных с трассой", c["ok"] / c["n"] if c["n"] else 0)
    parts.append(md_table(["Глубина d", "Кейсов", "Верных + трасса", "Вклад в сумму", "Доля успеха"], m2_rows))
    parts.append("")
    parts.append(f"Итого: {total_ok}/20 верных с полной трассой. ExplainScore = **{ru(ch['m2']['explain_score'])}** ({ru(total_contrib)} / 20).")
    parts.append("")
    rec("M2 ExplainScore", "офлайн n=20", "AH", ch["m2"]["explain_score"])

    parts.append("### 3.1. Пункты M2")
    parts.append("")
    m2_item_rows = []
    for item in m2_items:
        m2_item_rows.append([
            item["item_id"],
            item["depth"],
            yn(item["correct"]),
            yn(item["trace_complete"]),
            item.get("expected_answer", ""),
            item.get("actual_answer") or "∅",
        ])
    parts.append(md_table(["ID", "d", "Верно", "Трасса", "Эталон", "Ответ"], m2_item_rows))
    parts.append("")

    # ── M3 ──
    parts.append("## 4. M3 — сборка мусора")
    parts.append("")
    parts.append(md_table(
        ["Показатель", "Значение"],
        [
            ["Сирот до GC", ch["m3"]["orphans_before"]],
            ["Сирот после GC", ch["m3"]["orphans_after"]],
            ["GC efficiency", ru(ch["m3"]["gc_efficiency"])],
            ["Ложные удаления живых", "нет" if not ch["m3"]["false_live_deletes"] else str(ch["m3"]["false_live_deletes"])],
        ],
    ))
    parts.append("")
    rec("M3 GC efficiency", "офлайн", "AH", ch["m3"]["gc_efficiency"])
    rec("M3 сироты до", "офлайн", "AH", ch["m3"]["orphans_before"])
    rec("M3 сироты после", "офлайн", "AH", ch["m3"]["orphans_after"])

    # ── M4 rabbit ──
    parts.append("## 5. M4 — заяц против Vanilla RAG")
    parts.append("")
    parts.append(md_table(
        ["Показатель", "AH", "RAG LLM+TF-IDF", "Δ", "RAG extractive", "Δ extractive"],
        [
            ["ExplainScore", ru(rabbit_llm["summary"]["ExplainScore_AH"]), ru(rabbit_llm["summary"]["ExplainScore_VanillaRAG"]), ru(rabbit_llm["summary"]["delta_explainability"]), ru(rabbit_ext["summary"]["ExplainScore_VanillaRAG"]), ru(rabbit_ext["summary"]["delta_explainability"])],
            ["Галлюцинации", ru(rabbit_llm["summary"]["Hallucination_AH"]), ru(rabbit_llm["summary"]["Hallucination_VanillaRAG"]), ru(rabbit_llm["summary"]["delta_hallucination"]), ru(rabbit_ext["summary"]["Hallucination_VanillaRAG"]), ru(rabbit_ext["summary"]["delta_hallucination"])],
            ["Гипотеза целиком", yn(rabbit_llm["summary"]["hypothesis_ok"]), "—", "—", yn(rabbit_ext["summary"]["hypothesis_ok"]), "—"],
        ],
    ))
    parts.append("")
    rec("M4 ExplainScore AH", "заяц LLM RAG", "AH", rabbit_llm["summary"]["ExplainScore_AH"])
    rec("M4 ExplainScore RAG", "заяц LLM RAG", "RAG", rabbit_llm["summary"]["ExplainScore_VanillaRAG"])
    rec("M4 hall AH", "заяц LLM RAG", "AH", rabbit_llm["summary"]["Hallucination_AH"])
    rec("M4 hall RAG", "заяц LLM RAG", "RAG", rabbit_llm["summary"]["Hallucination_VanillaRAG"])

    parts.append("### 5.1. Пункты, заяц + LLM RAG")
    parts.append("")
    rabbit_rows = []
    for item in rabbit_llm["items"]:
        rabbit_rows.append([
            item["q"],
            item["ah"],
            yn(item["ah_correct"]),
            yn(item["ah_trace_complete"]),
            ru(item["ah_explain"]),
            yn(item["ah_hall"]),
            yn(item["rag_correct"]),
            yn(item["rag_hall"]),
        ])
    parts.append(md_table(
        ["Вопрос", "Ответ AH", "AH верно", "Трасса", "Explain", "AH hall", "RAG верно", "RAG hall"],
        rabbit_rows,
    ))
    parts.append("")

    # ── M4 HF ──
    parts.append("## 6. M4 — инцидент Hugging Face, граф пересобран на каждый бэкенд")
    parts.append("")
    hf_rows = []
    for backend, report in m4.items():
        hf_rows.append([
            report.get("rag_backend", backend),
            ru(report["ExplainScore_AH"]),
            ru(report["ExplainScore_VanillaRAG"]),
            ru(report["delta_explainability"]),
            ru(report["Hallucination_AH"]),
            ru(report["Hallucination_VanillaRAG"]),
            ru(report["delta_hallucination"]),
            yn(report["hypothesis_ok"]),
            yn(report.get("graph_rebuilt")),
            ru(report.get("sec"), 2),
        ])
        rec("M4 ExplainScore AH", f"HF {backend}", "AH", report["ExplainScore_AH"])
        rec("M4 Δ explain", f"HF {backend}", "AH − RAG", report["delta_explainability"])
        rec("M4 hall RAG", f"HF {backend}", "RAG", report["Hallucination_VanillaRAG"])
    parts.append(md_table(
        ["Бэкенд RAG", "E_AH", "E_RAG", "Δ explain", "H_AH", "H_RAG", "Δ hall", "Гипотеза", "Граф заново", "сек"],
        hf_rows,
    ))
    parts.append("")
    parts.append(f"Контрольный лог `data/m4_openai_hf_report.json` (extractive, до пакетного сравнения): E_AH = {ru(m4_hf_old['summary']['ExplainScore_AH'])}, H_AH = {ru(m4_hf_old['summary']['Hallucination_AH'])}, H_RAG = {ru(m4_hf_old['summary']['Hallucination_VanillaRAG'])}.")
    parts.append("")

    first_backend = next(iter(m4))
    parts.append(f"### 6.1. Пункты, бэкенд `{first_backend}` (ответы AH совпадают на всех пересобранных прогонах)")
    parts.append("")
    hf_item_rows = []
    for item in m4[first_backend]["items"]:
        hf_item_rows.append([
            item["question"],
            item["ah_answer"],
            yn(item["ah_correct"]),
            yn(item["ah_trace_complete"]),
            ru(item["ah_explain"]),
            yn(item["ah_hallucinated"]),
            yn(item["rag_correct"]),
            yn(item["rag_hallucinated"]),
        ])
    parts.append(md_table(
        ["Вопрос", "Ответ AH", "AH верно", "Трасса", "Explain", "AH hall", "RAG верно", "RAG hall"],
        hf_item_rows,
    ))
    parts.append("")

    # ── M4 offline ──
    parts.append("## 7. M4 офлайн-челлендж")
    parts.append("")
    parts.append(md_table(
        ["Показатель", "AH", "RAG", "Δ"],
        [
            ["Explainability", ru(ch["m4"]["ah_explainability"]), ru(ch["m4"]["rag_explainability"]), ru(ch["m4"]["delta_explainability"])],
            ["Hallucination", ru(ch["m4"]["ah_hallucination"]), ru(ch["m4"]["rag_hallucination"]), ru(ch["m4"]["delta_hallucination"])],
        ],
    ))
    parts.append("")
    rec("M4 ExplainScore", "офлайн челлендж", "AH", ch["m4"]["ah_explainability"])
    rec("M4 Δ hall", "офлайн челлендж", "RAG − AH", ch["m4"]["delta_hallucination"])

    # ── M5 ──
    parts.append("## 8. M5 — RobustnessGain")
    parts.append("")
    slm_ah = m1["ah_ollama_8b"]["normalized_weighted_f1"]
    slm_rag = m1["rag_ollama_8b"]["normalized_weighted_f1"]
    m5_rows = []
    for label, key, llm_ah_k, llm_rag_k in [
        ("qwen3:8b / GigaChat-2-Pro", "m5_vs_gigachat", "ah_gigachat", "rag_gigachat"),
        ("qwen3:8b / deepseek-chat", "m5_vs_deepseek", "ah_deepseek", "rag_deepseek"),
    ]:
        llm_ah = m1[llm_ah_k]["normalized_weighted_f1"]
        llm_rag = m1[llm_rag_k]["normalized_weighted_f1"]
        gain = hf["m1_m5"][key]["robustness_gain"]
        m5_rows.append([
            label,
            ru(slm_ah / slm_rag),
            ru(llm_ah / llm_rag),
            ru(gain),
            "отвергнута" if gain <= 0 else "формально > 0, не рост 8b",
        ])
        rec("M5 RobustnessGain", "HF n=12", label, gain)
    parts.append(md_table(["Пара", "AH/RAG SLM", "AH/RAG LLM", "Gain", "Гипотеза > 0"], m5_rows))
    parts.append("")

    # ── shared graph ──
    parts.append("## 9. Общий граф: один строитель, отвечают обе модели")
    parts.append("")
    parts.append("Статья Hugging Face, 26 батчей, закрытая генерация, без корпуса RAG.")
    parts.append("")
    share_rows = []
    for key, run in shared["runs"].items():
        qa = run["qa"]["summary"]
        share_rows.append([
            run["builder_model"],
            run["candidates"],
            ru(run["mean_coverage"]),
            run["graph"]["S"],
            run["graph"]["graph_size"],
            ru(qa["graph_raw"]["correct"]),
            ru(qa["slm"]["correct"]),
            ru(qa["llm"]["correct"]),
            ru(qa["slm"]["hallucinated"]),
            ru(qa["llm"]["hallucinated"]),
            ru(qa["slm"]["abstain"]),
            ru(run["ingest_sec"], 1),
        ])
        rec("граф факты", "общий граф", run["builder_model"], run["candidates"])
        rec("покрытие ingest", "общий граф", run["builder_model"], run["mean_coverage"])
        rec("верность декодера", "общий граф", run["builder_model"], qa["graph_raw"]["correct"])
        rec("верность SLM по графу", "общий граф", run["builder_model"], qa["slm"]["correct"])
        rec("верность LLM по графу", "общий граф", run["builder_model"], qa["llm"]["correct"])
    parts.append(md_table(
        ["Кто собрал", "Фактов", "Покрытие", "S", "Размер", "Декодер верно", "8b верно", "DeepSeek верно", "8b hall", "LLM hall", "Отказ чата", "сек ingest"],
        share_rows,
    ))
    parts.append("")

    for key, title in [("llm", "граф собрал DeepSeek"), ("slm", "граф собрал qwen3:8b")]:
        parts.append(f"### 9.{1 if key == 'llm' else 2}. Пункты QA, {title}")
        parts.append("")
        rows = []
        for item in shared["runs"][key]["qa"]["items"]:
            rows.append([
                item["question"],
                (item["graph_raw"]["answer"] or "")[:80],
                yn(item["graph_raw"]["correct"]),
                item["slm"]["answer"],
                yn(item["slm"]["correct"]),
                yn(item["slm"]["abstain"]),
                item["llm"]["answer"],
                yn(item["llm"]["correct"]),
                yn(item["llm"]["abstain"]),
            ])
        parts.append(md_table(
            ["Вопрос", "Декодер", "Дек. верно", "8b", "8b верно", "8b отказ", "DeepSeek", "LLM верно", "LLM отказ"],
            rows,
        ))
        parts.append("")

    # ── hypothesis ──
    parts.append("## 10. Гипотезы H1/H2 (изолированный граф на вопрос)")
    parts.append("")
    hypo_rows = []
    labels = {
        "rabbit_extractive_isolated": "заяц, extractive",
        "rabbit_inventing_isolated": "заяц, скриптовый генератор",
        "mini_extractive_isolated": "мини-граф, extractive",
        "mini_inventing_isolated": "мини-граф, скриптовый генератор",
    }
    for key, label in labels.items():
        block = hypo[key]
        h1 = block["h1_explainability"]
        h2 = block["h2_hallucination"]
        hypo_rows.append([
            label,
            ru(h1["ah_score"]),
            ru(h1["rag_score"]),
            ru(h1["delta"]),
            h1["verdict"],
            ru(h2["ah_score"]),
            ru(h2["rag_score"]),
            ru(h2["delta"]),
            h2["verdict"],
            block["overall"],
        ])
        rec("H1 Δ explain", label, "AH − RAG", h1["delta"])
        rec("H2 Δ hall", label, "RAG − AH", h2["delta"])
    parts.append(md_table(
        ["Протокол", "E_AH", "E_RAG", "ΔE", "H1", "H_AH", "H_RAG", "ΔH", "H2", "Итог"],
        hypo_rows,
    ))
    parts.append("")

    parts.append("## 11. Итоговый F1")
    parts.append("")
    parts.append(md_table(
        ["Прогон", "Модель", "Режим", "n", "Норм. взв. F1"],
        [
            ["HF live", "qwen3:8b", "AH-gated", 12, ru(m1["ah_ollama_8b"]["normalized_weighted_f1"])],
            ["HF live", "qwen3:8b", "без гейта", 12, ru(m1["rag_ollama_8b"]["normalized_weighted_f1"])],
            ["HF live", "GigaChat-2-Pro", "AH-gated", 12, ru(m1["ah_gigachat"]["normalized_weighted_f1"])],
            ["HF live", "GigaChat-2-Pro", "без гейта", 12, ru(m1["rag_gigachat"]["normalized_weighted_f1"])],
            ["HF live", "deepseek-chat", "AH-gated", 12, ru(m1["ah_deepseek"]["normalized_weighted_f1"])],
            ["HF live", "deepseek-chat", "без гейта", 12, ru(m1["rag_deepseek"]["normalized_weighted_f1"])],
            ["челлендж", "deepseek-chat", "AH-gated", 16, ru(m1_g["score"]["normalized_weighted_f1"])],
            ["челлендж", "deepseek-chat", "без гейта", 16, ru(m1_u["score"]["normalized_weighted_f1"])],
            ["челлендж", "SeedPerception", "без LLM", 100, ru(m1_seed["score"]["normalized_weighted_f1"])],
        ],
    ))
    parts.append("")
    parts.append("**Вывод по F1.** Порог Middle 0,6 на живом стенде закрывают GigaChat (0,654) и DeepSeek (0,605). Локальный 8b — 0,381. На выборке 16 реплик челленджа DeepSeek даёт 0,705. F1 измеряет парсер ролей, не воспламенение и не качество ответа по графу.")
    parts.append("")

    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["метрика", "стенд", "ряд", "значение", "комментарий"])
        writer.writerows(csv_rows)
    print(f"wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)")
    print(f"wrote {OUT_CSV} ({OUT_CSV.stat().st_size} bytes, {len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
