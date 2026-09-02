# Benchmark report: OpenAI Hugging Face incident corpus

Fresh closed-world pack (OpenAI blog, 2026-08-26) · M1/M4/M5 live providers · offline M2/M3

**Source JSON:** `data/provider_compare_openai_hf.json`  
**Corpus:** `benchmarks/fresh/openai_hf_2026/`  
**Article:** [The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)

| Tag | Value |
| --- | --- |
| Corpus size | ~4.7k words |
| SLM | qwen3:8b (Ollama) |
| Commercial LLM | deepseek-chat, GigaChat-2-Pro |
| Embeddings | nomic-embed-text |

## Headline metrics

| Metric | Value |
| --- | ---: |
| M2 ExplainScore | 0.26 |
| M3 GC efficiency | 1.00 |
| Best M4 Δ_explain | +0.25 |
| M5 RobustnessGain | −0.02 … −0.05 |

## Verdict

Hypothesis from the challenge brief on **explainability** (M4 Δ_explain > 0) is **confirmed** against all RAG backends.

Hypothesis **M5** (AH better compensates SLM weakness than commercial LLM) is **rejected** on this sample: commercial models are stronger in both AH-gated and ungated RAG perception, and qwen3:8b is below the Middle M1 gate (0.6).

---

## What was measured

### Corpus

OpenAI post «The Hugging Face incident…» (2026-08-26): IM1, ExploitGym, Artifactory SSRF message board, HDF5 / RefJinja zero-days. These entities are unlikely to appear in a local 8B pretrain cut.

- Domain: variant B (incident journal / CAUSE) + C (IS-A taxonomy)
- Size ~4.7k words — below the hard §6 floor of ≥15k words; this is a **pilot**, not a full acceptance stand

### Protocol

| Track | Setup |
| --- | --- |
| M2 / M3 / M4 offline | 20 QA JSONL + GC of 200 orphans — no chat LLM |
| M4 interactive | 6 gold questions on the article graph vs VanillaRAG (extractive / 8b / DeepSeek / GigaChat) |
| M1 / M5 | 12 role items (SUBJECT / OBJECT / LOCATION) × AH-gated vs ungated RAG perception |

Reproduce:

```bat
python -m ah_memory.benchmarks.challenge.runner --fresh-openai-hf
python scripts/eval_m4.py --fresh-openai-hf
python scripts/compare_providers_fresh.py --m1-limit 12
```

---

## M1 — role extraction (F1)

Normalized weighted F1 (SUBJECT/OBJECT weight 2). Middle threshold from the brief: **≥ 0.6**.

| Model | AH F1 | Ungated F1 | AH − RAG | vs Middle 0.6 | sec / 12 |
| --- | ---: | ---: | ---: | --- | --- |
| qwen3:8b | 0.38 | 0.42 | −0.04 | fail | 23s |
| GigaChat-2-Pro | 0.70 | 0.72 | −0.03 | pass | 56s |
| deepseek-chat | 0.79 | 0.85 | −0.05 | pass | 24s |

### Reading M1

AH gating barely improves F1 vs ungated LLM on short template sentences («X delivered Y to Z») — ungated is slightly higher for all three. Quality gap is **model class**, not memory contour: DeepSeek ≫ GigaChat ≫ 8b.

---

## M5 — RobustnessGain

Formula:

`AH_SLM / RAG_SLM − AH_LLM / RAG_LLM`

Expected by the brief: **> 0** (role typing compensates SLM weakness more than for a large model).

| Pair | AH/RAG SLM | AH/RAG LLM | Gain | Hypothesis |
| --- | --- | --- | ---: | --- |
| 8b / GigaChat | 0.381 / 0.417 = 0.913 | 0.698 / 0.724 = 0.964 | **−0.050** | reject |
| 8b / DeepSeek | 0.381 / 0.417 = 0.913 | 0.792 / 0.846 = 0.936 | **−0.024** | reject |

A correctly run rejection counts as a valid experimental outcome under the brief. Here the AH contour does **not** give the SLM a relative advantage: the AH/RAG ratio for 8b is worse than for commercial LLMs.

---

## M4 — AH vs Vanilla RAG

| RAG backend | Δ_explain | Δ_hall | Hall_RAG | hypothesis_ok | sec |
| --- | ---: | ---: | ---: | --- | ---: |
| extractive+nomic | **0.250** | 0.167 | 0.333 | true | 1.7 |
| qwen3:8b | 0.083 | 0.833 | 1.000 | true | 30.0 |
| deepseek-chat | 0.083 | 0.833 | 1.000 | true | 12.6 |
| GigaChat-2-Pro | 0.083 | 0.667 | 0.833 | true | 15.8 |

### Why Δ_explain is always > 0

Vanilla RAG ExplainScore is forced to **0**: no UID trace (`trace_complete = 0` per brief §7 M4). Any non-zero `correct × depth × trace` on AH yields a positive delta — a **structural** advantage of symbolic memory, not “smarter generation”.

### Why LLM-RAG hallucinates more

Extractive RAG quotes chunks (hall 0.33). Chat-RAG (8b / DeepSeek / GigaChat) elaborates on top of retrieval → hall 0.83–1.0 on 6 questions. Δ_hall grows, but that mainly contrasts **silent extractive vs talkative LLM**, not pure graph strength.

### Protocol artifact

In `compare_providers_fresh.py` one `Agent` was reused across RAG backends: the first run (extractive) got ExplainScore_AH = 0.25; later runs 0.083 due to WM / activation pollution. For a fair compare, rebuild the graph/agent per backend.

---

## Per-question breakdown (interactive M4, extractive RAG)

| Question | AH | RAG | Interpretation |
| --- | --- | --- | --- |
| What is IM1? | ok + trace | ok (chunk) | Strong entity in the graph |
| Which eval drove HF incident? | miss (ExploitGym) | miss / hall | Answer lacks ExploitGym |
| Where was message board? | Artifactory + trace | ok (chunk) | Location present in graph |
| Which HF file-format 0-day? | miss → July 16 | ok (HDF5/RefJinja noise) | Activation drifted to a date |
| Result of alert → Astra? | miss (noise) | miss | CAUSE chain did not reach pause |
| Secret launch code of IM1? | hall (no abstain) | hall | Trap: neither abstained |

**Pattern:** point facts with explicit nodes (IM1, Artifactory) AH answers with a trace; multi-hop CAUSE / naming (ExploitGym, Astra pause) and OOD abstain are weak spots for both the graph and the ask layer.

---

## Offline M2 / M3 / M4 (challenge JSONL)

### M2 — ExplainScore 0.258 (13/20)

ActivationEngine over FOLLOW / IS-A / CAUSE chains. Failures on some deep hops (d = 5–6): threshold / tick count does not always carry activation to the answer node.

### M3 — GC_efficiency 1.0

200 orphans → 0 in T = 50; no false deletes of live nodes. Cleanest metric on the stand.

### M4 offline — Δ_explain +0.26

Δ_hall −0.35: heuristic RAG often empty (hall = 0), AH hall = 0.35 — baseline artifact, not “RAG is more honest”.

---

## Hypothesis summary (challenge brief §7)

| Hypothesis | Expectation | Observed | Status |
| --- | --- | --- | --- |
| M4 explainability | Δ_explain > 0 | +0.08 … +0.25 vs all RAG | **confirmed** |
| M4 hallucination | Δ_hall > 0 | yes vs LLM-RAG; debatable vs empty heuristic | **partial** |
| M5 robustness | RobustnessGain > 0 | −0.05 / −0.02 | **rejected** (n = 12) |
| Middle M1 gate | F1 ≥ 0.6 | 8b 0.38 · Giga 0.70 · DS 0.79 | **commercial only** |

---

## Limitations and next steps

1. M1/M5 used 12/100 roles — pilot only; need full 100 plus typo / inversion / ellipsis slices.
2. Corpus 4.7k words < 15k from §6; interactive M4 graph is a hand fixture, not full LLM ingest of the article.
3. Recreate `Agent` per M4 backend; harden abstain policy on OOD traps.
4. Fair M5: same parser prompt, same `json_mode` / `think=false`, per-role SUBJECT/OBJECT/LOCATION breakdown.
5. Optionally run deepseek-reasoner and GigaChat-2-Max as top-tier vs 8b (brief M5).

---

## Practical takeaway for the hackathon

On a fresh post-cutoff incident, AH wins **explainability for free** (UID traces). Role extraction and multi-hop QA quality track **LLM class**: local 8b does not yet clear Middle M1.

**Strategy:** AH contour + strong commercial parser (DeepSeek / GigaChat) for M1/M5; local 8b for latency / demo and RobustnessGain control — not as the sole production parser.
