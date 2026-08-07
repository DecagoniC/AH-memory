# АГ-память как фактор-граф (общий вид)

Интерактивная схема — canvas
[ah-as-factor-graph](file:///C:/Users/decag/.cursor/projects/d-Go-prog-AH-memory/canvases/ah-as-factor-graph.canvas.tsx)
(открой рядом с чатом в Cursor).

Полная спецификация: [FACTOR_GRAPH_ACTIVATION.md](FACTOR_GRAPH_ACTIVATION.md), гиперпараметры: [HYPERPARAMS.md](HYPERPARAMS.md).

## Слои AH

| Слой | Роль в FG |
|------|-----------|
| S | переменные \(X_s\) |
| C / P / H | m → переменные; N → \(f_N\) |
| L | \(f_l\) |
| T | семейство \(\psi_\pi\), не узел |

## Факторы

| Фактор | Смысл |
|--------|--------|
| \(f_N\) | гиперребро-факт: совместная вспышка актантов |
| \(f_l\) | бинарная связь L (IS-A / ASSOC / FOLLOW) |
| \(f^{\mathrm{obs}}\) | evidence от Perception (seed → \(\lambda\)) |
| \(f^{\mathrm{prior}}\) | штраф за активность без поддержки (замена \(g\)) |

## Совместное

\[
P(X)\propto \prod_N f_N\cdot\prod_{l\in L}f_l\cdot\prod_v f_v^{\mathrm{obs}}\cdot\prod_v f_v^{\mathrm{prior}}
\]

Активация: \(x_v := b_v(1)\) после loopy BP. WM: \(\{v \mid b_v(1) > t\}\).
