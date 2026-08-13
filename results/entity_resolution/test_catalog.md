# Entity Resolution / Identity — каталог тестов

Dataset: `entity_resolution_v2`
Всего ER-кейсов: **90**

## По типам

- `AMBIGUOUS`: 6
- `CONTEXTUAL`: 6
- `HOLD_OUT`: 11
- `MORPHOLOGY`: 27
- `NEGATIVE`: 20
- `SEMANTIC_NEAR`: 2
- `SURFACE`: 10
- `SYNONYM`: 8

## Символы графа

| uid | name | aliases | kind |
|---|---|---|---|
| `s_001` | Юлия | Юля, Yuliya, Yulia, yuliya, Юлия  | person |
| `s_002` | Антон | — | person |
| `s_005` | Мария | Маша | person |
| `s_006` | Сергей | Серёжа, Сережа | person |
| `s_003` | Москва | Moskva, moskva | place |
| `s_004` | Санкт-Петербург | Петербург, Питер, Санкт Петербург, санкт  петербург, St Petersburg | place |
| `s_007` | Казань | — | place |
| `s_010` | автомобиль | авто, машина | concept |
| `s_011` | врач | доктор | concept |
| `s_012` | покупка | приобретение | concept |
| `s_013` | ноутбук | ноут, лэптоп | concept |
| `s_014` | телефон | смартфон, мобильник | concept |
| `s_020` | мотоцикл | байк | concept |
| `s_021` | велосипед | велик | concept |
| `s_022` | пациент | — | concept |
| `s_023` | медсестра | — | concept |
| `s_030` | BMW | bmw, Bmw,  BMW  | brand |
| `s_031` | Audi | audi, AUDI | brand |
| `s_032` | Opel | — | brand |
| `s_040` | транспорт | — | concept |
| `s_041` | медицина | — | concept |
| `s_050` | Иван | — | person |
| `s_051` | Иван | — | person |
| `s_052` | Анна | — | person |
| `s_053` | Анна | — | person |
| `s_054` | Берлин | — | place |
| `s_055` | Париж | — | place |
| `s_056` | Нью-Йорк | New York, нью йорк, NewYork | place |
| `s_057` | Toyota | toyota, TOYOTA | brand |
| `s_058` | Lada | lada, ЛАДА | brand |
| `s_059` | инженер | — | concept |
| `s_060` | программист | — | concept |

## Факты

- `s_001` —LIVES_IN→ `s_003`
- `s_001` —WORKS_FOR→ `s_011`
- `s_002` —PURCHASE→ `s_030`
- `s_002` —SELL→ `s_030`
- `s_002` —PURCHASE→ `s_032`
- `s_005` —LIVES_IN→ `s_007`
- `s_006` —PURCHASE→ `s_013`
- `s_006` —PURCHASE→ `s_014`
- `s_010` —IS_A→ `s_040`
- `s_011` —PART_OF→ `s_041`
- `s_050` —LIVES_IN→ `s_003`
- `s_051` —LIVES_IN→ `s_007`
- `s_052` —LIVES_IN→ `s_054`
- `s_053` —LIVES_IN→ `s_055`

## ER cases

### `morph_yulia_01` (MORPHOLOGY)
- mention: **Юли**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_yulia_02` (MORPHOLOGY)
- mention: **Юлию**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_yulia_03` (MORPHOLOGY)
- mention: **Юлией**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_yulia_04` (MORPHOLOGY)
- mention: **Юлии**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_yulia_05` (MORPHOLOGY)
- mention: **Юле**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_moscow_01` (MORPHOLOGY)
- mention: **Москве**
- target_uid: `s_003`
- candidates: `s_003, s_004, s_001`
- expected_relation: `SAME_ENTITY`

### `morph_moscow_02` (MORPHOLOGY)
- mention: **Москву**
- target_uid: `s_003`
- candidates: `s_003, s_004, s_001`
- expected_relation: `SAME_ENTITY`

### `morph_moscow_03` (MORPHOLOGY)
- mention: **Москвы**
- target_uid: `s_003`
- candidates: `s_003, s_004, s_001`
- expected_relation: `SAME_ENTITY`

### `morph_moscow_04` (MORPHOLOGY)
- mention: **Москвой**
- target_uid: `s_003`
- candidates: `s_003, s_004, s_001`
- expected_relation: `SAME_ENTITY`

### `morph_car_01` (MORPHOLOGY)
- mention: **автомобиля**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_021`
- expected_relation: `SAME_CONCEPT`

### `morph_car_02` (MORPHOLOGY)
- mention: **автомобилем**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_021`
- expected_relation: `SAME_CONCEPT`

### `morph_car_03` (MORPHOLOGY)
- mention: **автомобилю**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_021`
- expected_relation: `SAME_CONCEPT`

### `morph_car_04` (MORPHOLOGY)
- mention: **автомобиле**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_021`
- expected_relation: `SAME_CONCEPT`

### `morph_anton_01` (MORPHOLOGY)
- mention: **Антона**
- target_uid: `s_002`
- candidates: `s_002, s_001, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_anton_02` (MORPHOLOGY)
- mention: **Антону**
- target_uid: `s_002`
- candidates: `s_002, s_001, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_anton_03` (MORPHOLOGY)
- mention: **Антоном**
- target_uid: `s_002`
- candidates: `s_002, s_001, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_anton_04` (MORPHOLOGY)
- mention: **Антоне**
- target_uid: `s_002`
- candidates: `s_002, s_001, s_003`
- expected_relation: `SAME_ENTITY`

### `morph_maria_01` (MORPHOLOGY)
- mention: **Марии**
- target_uid: `s_005`
- candidates: `s_005, s_001, s_002`
- expected_relation: `SAME_ENTITY`

### `morph_maria_02` (MORPHOLOGY)
- mention: **Марию**
- target_uid: `s_005`
- candidates: `s_005, s_001, s_002`
- expected_relation: `SAME_ENTITY`

### `morph_maria_03` (MORPHOLOGY)
- mention: **Машей**
- target_uid: `s_005`
- candidates: `s_005, s_001, s_002`
- expected_relation: `SAME_ENTITY`

### `morph_maria_04` (MORPHOLOGY)
- mention: **Маше**
- target_uid: `s_005`
- candidates: `s_005, s_001, s_002`
- expected_relation: `SAME_ENTITY`

### `morph_sergey_01` (MORPHOLOGY)
- mention: **Сергея**
- target_uid: `s_006`
- candidates: `s_006, s_002, s_005`
- expected_relation: `SAME_ENTITY`

### `morph_sergey_02` (MORPHOLOGY)
- mention: **Сергею**
- target_uid: `s_006`
- candidates: `s_006, s_002, s_005`
- expected_relation: `SAME_ENTITY`

### `morph_sergey_03` (MORPHOLOGY)
- mention: **Сергеем**
- target_uid: `s_006`
- candidates: `s_006, s_002, s_005`
- expected_relation: `SAME_ENTITY`

### `morph_kazan_01` (MORPHOLOGY)
- mention: **Казани**
- target_uid: `s_007`
- candidates: `s_007, s_003, s_004`
- expected_relation: `SAME_ENTITY`

### `morph_kazan_02` (MORPHOLOGY)
- mention: **Казань**
- target_uid: `s_007`
- candidates: `s_007, s_003, s_004`
- expected_relation: `SAME_ENTITY`

### `morph_kazan_03` (MORPHOLOGY)
- mention: **Казанью**
- target_uid: `s_007`
- candidates: `s_007, s_003, s_004`
- expected_relation: `SAME_ENTITY`

### `syn_001` (SYNONYM)
- mention: **машина**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_021`
- expected_relation: `SAME_CONCEPT`

### `syn_002` (SYNONYM)
- mention: **доктор**
- target_uid: `s_011`
- candidates: `s_011, s_022, s_023`
- expected_relation: `SAME_CONCEPT`

### `syn_003` (SYNONYM)
- mention: **приобретение**
- target_uid: `s_012`
- candidates: `s_012, s_010, s_011`
- expected_relation: `SAME_CONCEPT`

### `syn_004` (SYNONYM)
- mention: **автомашина**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_013`
- expected_relation: `SAME_CONCEPT`

### `syn_005` (SYNONYM)
- mention: **ноут**
- target_uid: `s_013`
- candidates: `s_013, s_014, s_010`
- expected_relation: `SAME_CONCEPT`

### `syn_006` (SYNONYM)
- mention: **смартфон**
- target_uid: `s_014`
- candidates: `s_014, s_013, s_010`
- expected_relation: `SAME_CONCEPT`

### `syn_007` (SYNONYM)
- mention: **Питер**
- target_uid: `s_004`
- candidates: `s_004, s_003, s_007`
- expected_relation: `SAME_ENTITY`

### `syn_008` (SYNONYM)
- mention: **Маша**
- target_uid: `s_005`
- candidates: `s_005, s_001, s_002`
- expected_relation: `SAME_ENTITY`

### `neg_001` (NEGATIVE)
- mention: **мотоцикл**
- target_uid: `s_020`
- candidates: `s_010, s_020, s_021`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_010

### `neg_002` (NEGATIVE)
- mention: **пациент**
- target_uid: `s_022`
- candidates: `s_011, s_022, s_023`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_011

### `neg_003` (NEGATIVE)
- mention: **Санкт-Петербург**
- target_uid: `s_004`
- candidates: `s_003, s_004`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_003

### `neg_004` (NEGATIVE)
- mention: **Audi**
- target_uid: `s_031`
- candidates: `s_030, s_031, s_032`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_030

### `neg_005` (NEGATIVE)
- mention: **велосипед**
- target_uid: `s_021`
- candidates: `s_010, s_020, s_021`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_010

### `neg_006` (NEGATIVE)
- mention: **медсестра**
- target_uid: `s_023`
- candidates: `s_011, s_022, s_023`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_011

### `near_001` (SEMANTIC_NEAR)
- mention: **мотоцикл**
- target_uid: `s_020`
- candidates: `s_010, s_020`
- expected_relation: `DIFFERENT_CONCEPT`
- notes: must_not_merge_with=s_010

### `near_002` (SEMANTIC_NEAR)
- mention: **пациент**
- target_uid: `s_022`
- candidates: `s_011, s_022`
- expected_relation: `DIFFERENT_CONCEPT`
- notes: must_not_merge_with=s_011

### `neg_force_001` (NEGATIVE)
- mention: **мотоцикл**
- target_uid: `None`
- candidates: `s_010`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_010

### `neg_force_002` (NEGATIVE)
- mention: **пациент**
- target_uid: `None`
- candidates: `s_011`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_011

### `neg_force_003` (NEGATIVE)
- mention: **Санкт-Петербург**
- target_uid: `None`
- candidates: `s_003`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_003

### `neg_force_004` (NEGATIVE)
- mention: **Audi**
- target_uid: `None`
- candidates: `s_030`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_030

### `ctx_001` (CONTEXTUAL)
- mention: **Юли**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`
- context: _Юли позвонил Антон_

### `ctx_002` (CONTEXTUAL)
- mention: **машине**
- target_uid: `None`
- candidates: `s_010, s_030, s_032`
- expected_relation: `SAME_CONCEPT`
- context: _Антон сейчас ездит на машине_
- notes: experimental_ownership_not_required

### `ctx_003` (CONTEXTUAL)
- mention: **Юлию**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_003`
- expected_relation: `SAME_ENTITY`
- context: _Где сейчас живёт Юлию?_

### `ctx_004` (CONTEXTUAL)
- mention: **Маше**
- target_uid: `s_005`
- candidates: `s_005, s_001, s_007`
- expected_relation: `SAME_ENTITY`
- context: _В каком городе живёт Маше?_

### `ctx_005` (CONTEXTUAL)
- mention: **ноут**
- target_uid: `s_013`
- candidates: `s_013, s_014, s_010`
- expected_relation: `SAME_CONCEPT`
- context: _Какой ноут купил Сергей?_

### `ctx_006` (CONTEXTUAL)
- mention: **Питере**
- target_uid: `s_004`
- candidates: `s_004, s_003, s_007`
- expected_relation: `SAME_ENTITY`
- context: _Кто работает в Питере?_

### `amb_001` (AMBIGUOUS)
- mention: **Иван**
- target_uid: `None`
- candidates: `s_050, s_051`
- expected_relation: `DIFFERENT_ENTITY`
- notes: ambiguous_without_context

### `amb_002` (AMBIGUOUS)
- mention: **Иван**
- target_uid: `s_050`
- candidates: `s_050, s_051`
- expected_relation: `SAME_ENTITY`
- context: _Иван живёт в Москве_

### `amb_003` (AMBIGUOUS)
- mention: **Иван**
- target_uid: `s_051`
- candidates: `s_050, s_051`
- expected_relation: `SAME_ENTITY`
- context: _Иван работает в Казани_

### `amb_004` (AMBIGUOUS)
- mention: **Анна**
- target_uid: `None`
- candidates: `s_052, s_053`
- expected_relation: `DIFFERENT_ENTITY`
- notes: ambiguous_without_context

### `amb_005` (AMBIGUOUS)
- mention: **Анна**
- target_uid: `s_052`
- candidates: `s_052, s_053`
- expected_relation: `SAME_ENTITY`
- context: _Анна переехала в Берлин_

### `amb_006` (AMBIGUOUS)
- mention: **Анна**
- target_uid: `s_053`
- candidates: `s_052, s_053`
- expected_relation: `SAME_ENTITY`
- context: _Анна учится в Париже_

### `surf_001` (SURFACE)
- mention: **Yuliya**
- target_uid: `s_001`
- candidates: `s_001, s_002, s_005`
- expected_relation: `SAME_ENTITY`

### `surf_002` (SURFACE)
- mention: **yulia**
- target_uid: `s_001`
- candidates: `s_001, s_002`
- expected_relation: `SAME_ENTITY`

### `surf_003` (SURFACE)
- mention: **bmw**
- target_uid: `s_030`
- candidates: `s_030, s_031, s_057`
- expected_relation: `SAME_ENTITY`

### `surf_004` (SURFACE)
- mention: ** BMW **
- target_uid: `s_030`
- candidates: `s_030, s_031`
- expected_relation: `SAME_ENTITY`

### `surf_005` (SURFACE)
- mention: **Санкт Петербург**
- target_uid: `s_004`
- candidates: `s_004, s_003, s_056`
- expected_relation: `SAME_ENTITY`

### `surf_006` (SURFACE)
- mention: **санкт  петербург**
- target_uid: `s_004`
- candidates: `s_004, s_003`
- expected_relation: `SAME_ENTITY`

### `surf_007` (SURFACE)
- mention: **New York**
- target_uid: `s_056`
- candidates: `s_056, s_003, s_004`
- expected_relation: `SAME_ENTITY`

### `surf_008` (SURFACE)
- mention: **нью йорк**
- target_uid: `s_056`
- candidates: `s_056, s_003, s_004`
- expected_relation: `SAME_ENTITY`

### `surf_009` (SURFACE)
- mention: **Moskva**
- target_uid: `s_003`
- candidates: `s_003, s_004, s_056`
- expected_relation: `SAME_ENTITY`

### `surf_010` (SURFACE)
- mention: **audi**
- target_uid: `s_031`
- candidates: `s_031, s_030, s_057`
- expected_relation: `SAME_ENTITY`

### `hneg_001` (NEGATIVE)
- mention: **Toyota**
- target_uid: `s_057`
- candidates: `s_057, s_030, s_031`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_030

### `hneg_002` (NEGATIVE)
- mention: **Lada**
- target_uid: `s_058`
- candidates: `s_058, s_057, s_030`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_057

### `hneg_003` (NEGATIVE)
- mention: **инженер**
- target_uid: `s_059`
- candidates: `s_059, s_011, s_060`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_011

### `hneg_004` (NEGATIVE)
- mention: **программист**
- target_uid: `s_060`
- candidates: `s_060, s_059, s_011`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_059

### `hneg_005` (NEGATIVE)
- mention: **Нью-Йорк**
- target_uid: `s_056`
- candidates: `s_056, s_003, s_004`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_003

### `hneg_006` (NEGATIVE)
- mention: **Берлин**
- target_uid: `s_054`
- candidates: `s_054, s_055, s_003`
- expected_relation: `DIFFERENT_ENTITY`
- notes: must_not_merge_with=s_055

### `hneg_force_001` (NEGATIVE)
- mention: **Toyota**
- target_uid: `None`
- candidates: `s_030`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_030

### `hneg_force_002` (NEGATIVE)
- mention: **инженер**
- target_uid: `None`
- candidates: `s_011`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_011

### `hneg_force_003` (NEGATIVE)
- mention: **Нью-Йорк**
- target_uid: `None`
- candidates: `s_003`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_003

### `hneg_force_004` (NEGATIVE)
- mention: **программист**
- target_uid: `None`
- candidates: `s_011`
- expected_relation: `DIFFERENT_ENTITY`
- notes: force_reject=s_011

### `hold_001` (HOLD_OUT)
- mention: **легковушка**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_021`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_002` (HOLD_OUT)
- mention: **тачка**
- target_uid: `s_010`
- candidates: `s_010, s_020, s_057`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_003` (HOLD_OUT)
- mention: **эскулап**
- target_uid: `s_011`
- candidates: `s_011, s_022, s_059`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_004` (HOLD_OUT)
- mention: **гаджет**
- target_uid: `s_014`
- candidates: `s_014, s_013, s_010`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_005` (HOLD_OUT)
- mention: **ультрабук**
- target_uid: `s_013`
- candidates: `s_013, s_014, s_010`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_006` (HOLD_OUT)
- mention: **автолюбительский транспорт**
- target_uid: `s_010`
- candidates: `s_010, s_040, s_020`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_007` (HOLD_OUT)
- mention: **медик**
- target_uid: `s_011`
- candidates: `s_011, s_023, s_022`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_008` (HOLD_OUT)
- mention: **сотовый**
- target_uid: `s_014`
- candidates: `s_014, s_013`
- expected_relation: `SAME_CONCEPT`
- notes: hold_out_not_in_lexicon

### `hold_neg_001` (HOLD_OUT)
- mention: **легковушка**
- target_uid: `None`
- candidates: `s_020`
- expected_relation: `DIFFERENT_CONCEPT`
- notes: force_reject=s_020

### `hold_neg_002` (HOLD_OUT)
- mention: **эскулап**
- target_uid: `None`
- candidates: `s_022`
- expected_relation: `DIFFERENT_CONCEPT`
- notes: force_reject=s_022

### `hold_neg_003` (HOLD_OUT)
- mention: **гаджет**
- target_uid: `None`
- candidates: `s_013`
- expected_relation: `DIFFERENT_CONCEPT`
- notes: force_reject=s_013

## Mini-graph facts (pytest paraphrase)

- Юлия —LIVES_IN→ Москва
- Юлия —WORKS_FOR→ врач
- Антон —PURCHASE→ автомобиль
- Антон —PURCHASE→ BMW
- Мария —LIVES_IN→ Казань
- Сергей —PURCHASE→ ноутбук
- Сергей —PURCHASE→ телефон

## Paraphrase queries

| query | mention | expected |
|---|---|---|
| Где живёт Юли? | Юли | Юлия |
| Куда переехала Юлию? | Юлию | Юлия |
| С кем говорил Антон про Юлией? | Юлией | Юлия |
| В каком городе живёт Маше? | Маше | Мария |
| Что купила Маша? | Маша | Мария |
| Какой город у Марии? | Марии | Мария |
| Где работает врач Юлия — в Москве? | Москве | Москва |
| Антон был в Москву? | Москву | Москва |
| Кто купил машину? | машину | автомобиль |
| На какой машине ездит Антон? | машине | автомобиль |
| Какое авто приобрёл Антон? | авто | автомобиль |
| Какой ноут у Сергея? | ноут | ноутбук |
| Какой смартфон купил Сергей? | смартфон | телефон |
| Кто живёт в Питере? | Питере | Санкт-Петербург |
| Есть ли связь с Петербургом? | Петербургом | Санкт-Петербург |
| Сергей говорил с Сергеем? | Сергеем | Сергей |
| Передай Антону сообщение | Антону | Антон |
| Казань или Казани — одно место | Казани | Казань |

## Negative paraphrases

- `мотоцикл` ≠ `автомобиль`
- `пациент` ≠ `врач`
- `Audi` ≠ `BMW`
- `Санкт-Петербург` ≠ `Москва`
- `велосипед` ≠ `автомобиль`
- `ноутбук` ≠ `телефон`

## Pytest modules

- `tests/test_entity_resolution.py`
- `tests/test_identity_ingest.py`
- `tests/test_identity_paraphrase.py`
