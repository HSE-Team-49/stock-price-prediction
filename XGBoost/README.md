# Сравнение XGBoost-моделей прогнозирования цен акций в rolling/direct и fixed-origin постановках

## Аннотация

В данном эксперименте исследуется качество прогнозирования цен акций с помощью моделей XGBoost в двух постановках оценки:

1. **Rolling/direct forecast** — оценка обновляемого прогноза, при которой признаки для каждого горизонта формируются на дату, отстоящую от целевой даты на величину горизонта.
2. **Fixed-origin forecast** — оценка прогноза из одной начальной точки, при которой весь будущий период прогнозируется на основе признаков, доступных только на момент начала прогноза.

Цель сравнения — отделить качество модели в режиме регулярного обновления прогноза от качества модели в более строгом сценарии, когда требуется построить весь прогнозируемый период из одной даты без использования фактических цен внутри будущего интервала.

В работе сравниваются глобальные XGBoost-модели и модели, обученные отдельно внутри кластеров акций:

- `global` — одна модель обучается на всех тикерах;
- `per-cluster` — отдельные модели обучаются для каждого кластера акций.

Рассматриваются три частоты прогнозирования:

- дневная (`daily`);
- недельная (`weekly`);
- месячная (`monthly`).

Основная метрика качества — **WAPE**.

---

## 1. Постановка задачи

Пусть для акции доступен временной ряд цен:

```text
P_t
```

Для горизонта `h` целевая переменная задаётся через логарифмическую доходность:

```text
target_h = log(P_{t+h}) - log(P_t)
```

Модель прогнозирует `target_h`, после чего прогноз цены восстанавливается как:

```text
P_hat_{t+h} = P_t * exp(target_hat_h)
```

Оценка качества производится по восстановленным ценам.

---

## 2. Две постановки оценки качества

### 2.1 Rolling/direct forecast

В rolling/direct постановке для каждого горизонта `h` признаки формируются на дату:

```text
feature_date = target_date - h
```

Это означает, что для каждой целевой даты используется максимально близкая к ней доступная точка истории.

Такой режим соответствует сценарию **регулярно обновляемого прогноза**:

```text
каждый новый период появляются новые фактические цены;
признаки пересобираются;
модель строит прогноз на следующий горизонт.
```

Эта постановка полезна, если модель предполагается использовать как постоянно обновляемую систему прогнозирования.

---

### 2.2 Fixed-origin forecast

В fixed-origin постановке выбирается одна начальная дата прогноза:

```text
origin_date = последняя доступная дата до начала прогнозируемого периода
```

Для всех горизонтов используются признаки только на этой дате:

```text
feature_date = origin_date
```

Далее модель строит прогнозы:

```text
origin_date + 1 шаг
origin_date + 2 шага
...
origin_date + H шагов
```

Фактические цены внутри прогнозируемого периода используются только для расчёта метрик, но не используются как признаки.

Такой режим соответствует более строгой задаче:

```text
построить весь будущий период из одной начальной точки.
```

---

## 3. Используемые данные

В эксперименте используются дневные OHLCV-данные по акциям.

Основные входные файлы:

```text
data/prices_all.csv
data/prices_all_2025_12_20_today.csv
cluster_fullstart_assignments.csv
results_stocks/prices_all/
data/
```

Назначение файлов:

| Файл / директория | Назначение |
|---|---|
| `data/prices_all.csv` | Основной исторический файл с ценами |
| `data/prices_all_2025_12_20_today.csv` | Дополнительные фактические цены для периода проверки |
| `cluster_fullstart_assignments.csv` | Разметка тикеров по кластерам |
| `results_stocks/prices_all/` | Сезонные FFT-признаки |
| `data/` | Макроэкономические признаки |

Макроэкономические признаки ограничиваются датой:

```text
macro_known_until = 2025-12-20
```

Это означает, что значения макропоказателей после этой даты не используются как известные факты. Для последующих дат применяются последние доступные значения и признаки возраста макроэкономической информации.

---

## 4. Модели

Были построены шесть основных вариантов моделей:

| Папка / архив | Модель | Частота | Тип модели | Горизонты |
|---|---|---|---|---|
| `daily_global/` | `daily_global` | Daily | Global XGBoost | h=1..21 торговый день |
| `daily_per_cluster/` | `daily_per_cluster` | Daily | Per-cluster XGBoost | h=1..21 торговый день |
| `weekly_global/` | `weekly_global` | Weekly | Global XGBoost | h=1..52 недели |
| `weekly_per_cluster/` | `weekly_per_cluster` | Weekly | Per-cluster XGBoost | h=1..52 недели |
| `monthly_global/` | `monthly_global` | Monthly | Global XGBoost | h=1..12 месяцев |
| `monthly_per_cluster/` | `monthly_per_cluster` | Monthly | Per-cluster XGBoost | h=1..12 месяцев |

---

## 5. Разделение горизонтов

Для каждой частоты горизонты делятся на две группы:

1. `short_macro` — короткие горизонты, где используются макроэкономические признаки;
2. `long_no_macro` — длинные горизонты, где макроэкономические признаки исключаются.

| Частота | Общий горизонт | `short_macro` | `long_no_macro` |
|---|---:|---:|---:|
| Daily | 21 торговый день | h=1..5 | h=6..21 |
| Weekly | 52 недели | h=1..13 | h=14..52 |
| Monthly | 12 месяцев | h=1..3 | h=4..12 |

Такое разделение используется для снижения риска чрезмерной зависимости длинного прогноза от макроэкономических признаков, которые в реальной постановке могут быть неизвестны или устаревать.

---

## 6. Признаки

Во всех вариантах используются одинаковые группы признаков, соответствующие исходным XGBoost-моделям.

Основные группы признаков:

```text
price / log_price
returns
return lags
price lags
rolling mean / std / min / max / median / skew / kurt
momentum
realized volatility
downside volatility
upside volatility
ATR
RSI
SMA / EMA
MACD
Bollinger Bands
distance from rolling high / low
volume features
market-wide features
cluster features
rank features
season / FFT features
macro features
macro age / freeze features
calendar features
ticker encoded features
cluster encoded features
horizon
```

Для недельной и месячной частот дневные данные агрегируются.

### 6.1 Weekly-признаки

Для weekly-моделей формируются недельные агрегаты:

```text
week_price
week_open
week_high
week_low
week_close
week_volume
trading_days_in_week
daily_ret_mean_in_week
daily_ret_std_in_week
daily_ret_min_in_week
daily_ret_max_in_week
daily_ret_sum_in_week
daily_volume_mean_in_week
daily_volume_std_in_week
daily_volume_max_in_week
```

### 6.2 Monthly-признаки

Для monthly-моделей формируются месячные агрегаты:

```text
month_price
month_open
month_high
month_low
month_close
month_volume
trading_days_in_month
daily_ret_mean_in_month
daily_ret_std_in_month
daily_ret_min_in_month
daily_ret_max_in_month
daily_ret_sum_in_month
daily_volume_mean_in_month
daily_volume_std_in_month
daily_volume_max_in_month
```

---

## 7. Защита от утечки информации

Для fixed-origin постановки используются следующие ограничения.

### 7.1 Единая дата признаков

Для всех горизонтов используется одна дата признаков:

```text
feature_date = origin_date
```

Цена внутри прогнозируемого периода не используется как feature.

---

### 7.2 Ограничение target-даты при обучении

В обучающей выборке используются только те строки, у которых целевая дата находится до начала периода оценки:

```text
target_date < eval_start
```

Это исключает попадание фактических значений из evaluation-периода в обучение.

---

### 7.3 Ограничение макроэкономических данных

Макроэкономические ряды обрезаются по дате:

```text
macro_known_until = 2025-12-20
```

После этой даты модель получает не будущие макроэкономические значения, а последние доступные значения и признаки возраста информации:

```text
*_age_available_days
*_age_observation_days
*_after_macro_known_until
```

---

## 8. Метрики

Основная метрика — **WAPE**:

```text
WAPE = 100 * sum(abs(actual_price - predicted_price)) / sum(abs(actual_price))
```

Дополнительно считаются:

```text
MAPE
MSE
MAE
RMSE
```

Метрики считаются по цене, а не по логарифмической доходности.

---

## 9. Итоговые fixed-origin результаты

### 9.1 Общие метрики

| Модель | n | WAPE | MAPE | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| **daily_global** | 4179 | **7.43%** | 6.84% | 20.69 | **40.62** |
| daily_per_cluster | 4179 | 7.49% | **6.79%** | 20.86 | 45.55 |
| **weekly_global** | 10329 | **19.36%** | **22.50%** | **53.81** | **263.45** |
| weekly_per_cluster | 10329 | 20.85% | 23.50% | 57.94 | 293.14 |
| monthly_global | 2384 | 21.85% | 25.18% | 60.79 | 249.22 |
| **monthly_per_cluster** | 2384 | **21.83%** | **24.68%** | **60.74** | **248.08** |

### 9.2 Основные выводы по fixed-origin

По WAPE лучшие модели:

| Частота | Лучшая модель | WAPE |
|---|---|---:|
| Daily | `daily_global` | 7.43% |
| Weekly | `weekly_global` | 19.36% |
| Monthly | `monthly_per_cluster` | 21.83% |

При этом разница между `monthly_global` и `monthly_per_cluster` минимальна:

```text
monthly_global:      WAPE 21.85%
monthly_per_cluster: WAPE 21.83%
```

То есть для месячного прогноза global и per-cluster модели практически эквивалентны по WAPE.

---

## 10. Анализ по группам горизонтов

| Модель | short_macro WAPE | long_no_macro WAPE |
|---|---:|---:|
| daily_global | 5.03% | **8.17%** |
| daily_per_cluster | **4.46%** | 8.42% |
| weekly_global | 7.69% | **23.16%** |
| weekly_per_cluster | **7.49%** | 25.20% |
| monthly_global | 12.83% | **24.83%** |
| monthly_per_cluster | **10.28%** | 25.65% |

По всем трём частотам наблюдается один и тот же эффект:

```text
per-cluster модели лучше на коротких горизонтах;
global модели лучше на длинных горизонтах.
```

Интерпретация:

- на коротком горизонте кластеры помогают учитывать локальную структуру рынка;
- на длинном горизонте внутри кластеров становится меньше обучающих примеров;
- global-модель лучше обобщает долгосрочную динамику за счёт большего объёма данных.

---

## 11. Гибридная схема

На основе анализа по группам горизонтов можно построить гибридный вариант:

```text
short_macro   -> per-cluster
long_no_macro -> global
```

### 11.1 Гибрид по частотам

| Частота | short model | long model | WAPE |
|---|---|---|---:|
| Daily | per-cluster | global | **7.30%** |
| Weekly | per-cluster | global | **19.31%** |
| Monthly | per-cluster | global | **21.22%** |

### 11.2 Сравнение гибрида с чистыми моделями

| Частота | Global WAPE | Per-cluster WAPE | Hybrid WAPE |
|---|---:|---:|---:|
| Daily | 7.43% | 7.49% | **7.30%** |
| Weekly | 19.36% | 20.85% | **19.31%** |
| Monthly | 21.85% | 21.83% | **21.22%** |

Гибридная схема даёт лучший результат среди рассмотренных вариантов.

---

## 12. Сравнение rolling/direct и fixed-origin

Rolling/direct оценка измеряет качество обновляемого прогноза. Fixed-origin оценка измеряет качество прогноза всего будущего периода из одной начальной точки.

| Частота | Модель | Rolling/direct WAPE | Fixed-origin WAPE | Разница |
|---|---|---:|---:|---:|
| Daily | global | **6.15%** | 7.43% | +1.28 п.п. |
| Daily | per-cluster | **6.31%** | 7.49% | +1.18 п.п. |
| Weekly | global | **17.32%** | 19.36% | +2.04 п.п. |
| Weekly | per-cluster | **17.90%** | 20.85% | +2.95 п.п. |
| Monthly | global | **19.47%** | 21.85% | +2.38 п.п. |
| Monthly | per-cluster | **19.19%** | 21.83% | +2.64 п.п. |

### 12.1 Интерпретация различий

Rolling/direct постановка даёт более низкую ошибку, поскольку признаки обновляются по мере движения по тестовому периоду.

Fixed-origin постановка сложнее, так как весь прогноз строится из одной точки. Поэтому рост ошибки на `1.2–3.0 п.п. WAPE` является ожидаемым.

### 12.2 Практический смысл

Обе постановки имеют практическую интерпретацию:

| Постановка | Что измеряет | Когда уместна |
|---|---|---|
| Rolling/direct | Качество регулярно обновляемого прогноза | Если модель будет переоцениваться/обновляться по мере появления новых цен |
| Fixed-origin | Качество прогноза всего периода из одной даты | Если требуется построить долгосрочный прогноз без знания будущей ценовой траектории |

---

## 13. Анализ по кластерам

### 13.1 Наиболее сложные кластеры

| Частота | Наиболее сложные кластеры |
|---|---|
| Daily | `cluster_3` — WAPE 12.93% |
| Weekly | `cluster_3` — WAPE 30.24%, `cluster_4` — WAPE 29.86% |
| Monthly | `cluster_3` — WAPE 38.69%, `cluster_4` — WAPE 25.06% |

### 13.2 Наиболее устойчивые кластеры

| Частота | Лучший кластер |
|---|---|
| Daily | `cluster_6` — WAPE 4.24% |
| Weekly | `cluster_6` — WAPE 10.50% |
| Monthly | `cluster_6` — WAPE 12.29% |

Вывод:

```text
cluster_3 и cluster_4 требуют дополнительного анализа;
cluster_6 прогнозируется наиболее стабильно.
```

---

## 14. Анализ по тикерам

Среди наиболее сложных тикеров регулярно встречаются:

```text
WDC
MU
BKNG
WBD
AMD
CVNA
LRCX
GLW
VRT
NEM
```

Для дневного прогноза особенно сложными являются:

```text
INTC
AMD
MRVL
MU
WDC
```

Возможные причины высокой ошибки:

- высокая волатильность;
- резкие движения на новостях;
- зависимость от секторальных факторов;
- индивидуальные корпоративные события;
- недостаточность только технических, макроэкономических и кластерных признаков.

---

## 15. Соответствие папок, моделей и файлов

### 15.1 Общая структура результатов

После запуска формируется директория:

```text
results_xgb_old_features_fixed_origin/run_YYYYMMDD_HHMMSS/
```

Внутри:

```text
daily_global/
daily_per_cluster/
weekly_global/
weekly_per_cluster/
monthly_global/
monthly_per_cluster/
_status/
_embedded_scripts/
fixed_origin_metrics_all_models.csv
single_runner_config.json
```

---

### 15.2 `daily_global`

Назначение:

```text
Глобальная дневная модель XGBoost по всем тикерам.
```

Горизонты:

```text
h=1..21 торговый день
```

Группы:

```text
short_macro:   h=1..5
long_no_macro: h=6..21
```

Основные файлы метрик:

```text
fixed_origin_daily_predictions.csv
fixed_origin_daily_metrics_overall.json
fixed_origin_daily_metrics_by_horizon.csv
fixed_origin_daily_metrics_by_model_group.csv
fixed_origin_daily_metrics_by_date.csv
fixed_origin_daily_metrics_by_ticker.csv
```

Файлы моделей:

```text
global_xgb_daily_model_short_macro.json
global_xgb_daily_model_long_no_macro.json
```

---

### 15.3 `daily_per_cluster`

Назначение:

```text
Дневные XGBoost-модели, обученные отдельно внутри каждого кластера.
```

Горизонты:

```text
h=1..21 торговый день
```

Группы:

```text
short_macro:   h=1..5
long_no_macro: h=6..21
```

Основные файлы метрик:

```text
fixed_origin_daily_predictions.csv
fixed_origin_daily_metrics_overall.json
fixed_origin_daily_metrics_by_horizon.csv
fixed_origin_daily_metrics_by_model_group.csv
fixed_origin_daily_metrics_by_date.csv
fixed_origin_daily_metrics_by_ticker.csv
fixed_origin_daily_metrics_by_cluster.csv
```

Примеры файлов моделей:

```text
cluster_0_xgb_daily_model_short_macro.json
cluster_0_xgb_daily_model_long_no_macro.json
cluster_1_xgb_daily_model_short_macro.json
cluster_1_xgb_daily_model_long_no_macro.json
```

---

### 15.4 `weekly_global`

Назначение:

```text
Глобальная недельная модель XGBoost по всем тикерам.
```

Горизонты:

```text
h=1..52 недели
```

Группы:

```text
short_macro:   h=1..13
long_no_macro: h=14..52
```

Основные файлы метрик:

```text
fixed_origin_weekly_predictions.csv
fixed_origin_weekly_metrics_overall.json
fixed_origin_weekly_metrics_by_horizon.csv
fixed_origin_weekly_metrics_by_model_group.csv
fixed_origin_weekly_metrics_by_date.csv
fixed_origin_weekly_metrics_by_ticker.csv
```

Файлы моделей:

```text
global_xgb_weekly_model_short_macro.json
global_xgb_weekly_model_long_no_macro.json
```

---

### 15.5 `weekly_per_cluster`

Назначение:

```text
Недельные XGBoost-модели, обученные отдельно внутри каждого кластера.
```

Горизонты:

```text
h=1..52 недели
```

Группы:

```text
short_macro:   h=1..13
long_no_macro: h=14..52
```

Основные файлы метрик:

```text
fixed_origin_weekly_predictions.csv
fixed_origin_weekly_metrics_overall.json
fixed_origin_weekly_metrics_by_horizon.csv
fixed_origin_weekly_metrics_by_model_group.csv
fixed_origin_weekly_metrics_by_date.csv
fixed_origin_weekly_metrics_by_ticker.csv
fixed_origin_weekly_metrics_by_cluster.csv
```

Примеры файлов моделей:

```text
cluster_0_xgb_weekly_model_short_macro.json
cluster_0_xgb_weekly_model_long_no_macro.json
cluster_1_xgb_weekly_model_short_macro.json
cluster_1_xgb_weekly_model_long_no_macro.json
```

---

### 15.6 `monthly_global`

Назначение:

```text
Глобальная месячная модель XGBoost по всем тикерам.
```

Горизонты:

```text
h=1..12 месяцев
```

Группы:

```text
short_macro:   h=1..3
long_no_macro: h=4..12
```

Основные файлы метрик:

```text
fixed_origin_monthly_predictions.csv
fixed_origin_monthly_metrics_overall.json
fixed_origin_monthly_metrics_by_horizon.csv
fixed_origin_monthly_metrics_by_model_group.csv
fixed_origin_monthly_metrics_by_date.csv
fixed_origin_monthly_metrics_by_ticker.csv
```

Файлы моделей:

```text
global_xgb_monthly_model_short_macro.json
global_xgb_monthly_model_long_no_macro.json
```

---

### 15.7 `monthly_per_cluster`

Назначение:

```text
Месячные XGBoost-модели, обученные отдельно внутри каждого кластера.
```

Горизонты:

```text
h=1..12 месяцев
```

Группы:

```text
short_macro:   h=1..3
long_no_macro: h=4..12
```

Основные файлы метрик:

```text
fixed_origin_monthly_predictions.csv
fixed_origin_monthly_metrics_overall.json
fixed_origin_monthly_metrics_by_horizon.csv
fixed_origin_monthly_metrics_by_model_group.csv
fixed_origin_monthly_metrics_by_date.csv
fixed_origin_monthly_metrics_by_ticker.csv
fixed_origin_monthly_metrics_by_cluster.csv
```

Примеры файлов моделей:

```text
cluster_0_xgb_monthly_model_short_macro.json
cluster_0_xgb_monthly_model_long_no_macro.json
cluster_1_xgb_monthly_model_short_macro.json
cluster_1_xgb_monthly_model_long_no_macro.json
```

---

---

## 16. Python-файлы проекта и их назначение

В проекте использовалось несколько групп Python-файлов. Они относятся к разным этапам работы: подготовка признаков, исходные rolling/direct эксперименты, отдельные fixed-origin эксперименты и итоговый единый runner.

### 16.1 Базовые и вспомогательные файлы

| Файл | Назначение |
|---|---|
| `season.py` | Расчёт сезонных признаков и признаков на основе рядов Фурье / FFT для временных рядов акций |
| `clusster.py` | Кластеризация акций на основе признаков временных рядов |
---

### 16.2 Файлы rolling/direct XGBoost-экспериментов

До fixed-origin постановки использовались отдельные XGBoost-скрипты, в которых качество оценивалось в rolling/direct-режиме. В этой постановке для каждого горизонта признаки формируются на дату:

```text
feature_date = target_date - h
```

Эти скрипты нужны для оценки качества регулярно обновляемого прогноза.

| Файл | Частота | Тип модели | Назначение |
|---|---|---|---|
| `global_xgb_daily_last_month_h100_forecast_clean.py` | Daily | Global | Глобальная дневная XGBoost-модель |
| `global_xgb_daily_last_month_h100_forecast_per_cluster.py` | Daily | Per-cluster | Дневные XGBoost-модели отдельно по кластерам |
| `global_xgb_weekly_h100_forecast_two_models.py` | Weekly | Global | Глобальная недельная XGBoost-модель |
| `global_xgb_weekly_h100_forecast_per_cluster.py` | Weekly | Per-cluster | Недельные XGBoost-модели отдельно по кластерам |
| `global_xgb_monthly_h100_forecast_two_models.py` | Monthly | Global | Глобальная месячная XGBoost-модель |
| `global_xgb_monthly_h100_forecast_per_cluster.py` | Monthly | Per-cluster | Месячные XGBoost-модели отдельно по кластерам |

Именно по этим файлам получены rolling/direct результаты, которые используются для сравнения с fixed-origin постановкой:

```text
daily_global:          WAPE 6.15%
daily_per_cluster:     WAPE 6.31%
weekly_global:         WAPE 17.32%
weekly_per_cluster:    WAPE 17.90%
monthly_global:        WAPE 19.47%
monthly_per_cluster:   WAPE 19.19%
```

### 16.3 Итоговый единый fixed-origin runner

Финальный файл:

```text
run_xgb_fixed_origin_old_features_single_full_data_patched_macro_age_v2.py
```

Это единый runner, который последовательно запускает все шесть fixed-origin моделей:

```text
daily_global
daily_per_cluster
weekly_global
weekly_per_cluster
monthly_global
monthly_per_cluster
```

## 17. Запуск fixed-origin эксперимента

Для воспроизведения fixed-origin эксперимента используется единый исполняемый файл:

```text
run_xgb_fixed_origin_old_features_single_full_data_patched_macro_age_v2.py
```

Пример запуска:

```bash
python3 run_xgb_fixed_origin_old_features_single_full_data_patched_macro_age_v2.py \
  --out-root results_xgb_old_features_fixed_origin \
  --prices-csv data/prices_all.csv \
  --future-prices-csv data/prices_all_2025_12_20_today.csv \
  --eval-end-date 2026-05-08 \
  --macro-known-until 2025-12-20 \
  --macro-dir data \
  --cluster-csv cluster_fullstart_assignments.csv \
  --season-dir results_stocks/prices_all \
  --price-col Close \
  --n-trials 300 \
  --n-jobs 8 \
  --numba-threads 8 \
  --use-gpu 1 \
  --gpu-id 0 \
  2>&1 | tee run_xgb_fixed_origin_old_features_$(date +%Y%m%d_%H%M%S).log
```

---

## 18. Resume после остановки

Каждый шаг после успешного завершения создаёт marker-файл:

```text
run_dir/_status/<step>.done
```

Если выполнение остановилось на некоторой модели, повторный запуск с тем же `--run-dir` пропускает уже завершённые шаги и продолжает с первого незавершённого.

Пример:

```bash
python3 run_xgb_fixed_origin_old_features_single_full_data_patched_macro_age_v2.py \
  --run-dir results_xgb_old_features_fixed_origin/run_YYYYMMDD_HHMMSS \
  --prices-csv data/prices_all.csv \
  --future-prices-csv data/prices_all_2025_12_20_today.csv \
  --eval-end-date 2026-05-08 \
  --macro-known-until 2025-12-20 \
  --macro-dir data \
  --cluster-csv cluster_fullstart_assignments.csv \
  --season-dir results_stocks/prices_all \
  --price-col Close \
  --n-trials 300 \
  --n-jobs 8 \
  --numba-threads 8 \
  --use-gpu 1 \
  --gpu-id 0 \
  2>&1 | tee resume_xgb_fixed_origin_old_features_$(date +%Y%m%d_%H%M%S).log
```

Можно явно выбрать стартовый шаг:

```bash
--start-from weekly_per_cluster
```

Можно запустить только часть моделей:

```bash
--only monthly_global,monthly_per_cluster
```

---

## 19. Оптимизация памяти

Пайплайн построен так, чтобы не держать все модели и все панели признаков в памяти одновременно.

Основные элементы оптимизации:

```text
1. Модели выполняются строго последовательно.
2. Каждый шаг запускается отдельным Python-процессом.
3. После завершения процесса память освобождается операционной системой.
4. Daily, weekly и monthly модели не находятся в памяти одновременно.
5. Global и per-cluster модели не находятся в памяти одновременно.
6. Resume не запускает заново уже готовые шаги.
7. joblib и numba используются для ускорения CPU-части.
8. XGBoost использует GPU при use_gpu=1.
```

По умолчанию используется полный объём обучающих данных:

```text
--max-optuna-rows 0
--max-train-rows 0
```

Значение `0` означает:

```text
без ограничения строк
```

---

## 20. Итоговая интерпретация

В эксперименте рассмотрены две постановки оценки качества прогнозных моделей:

```text
rolling/direct forecast
fixed-origin forecast
```

Rolling/direct постановка показывает качество обновляемого прогноза. Fixed-origin постановка показывает качество прогноза всего будущего периода из одной начальной точки.

В fixed-origin постановке итоговые результаты составляют:

```text
daily_global:          WAPE ≈ 7.43%
weekly_global:         WAPE ≈ 19.36%
monthly_per_cluster:   WAPE ≈ 21.83%
```

Сравнение global и per-cluster моделей показывает, что кластеризация полезна преимущественно на коротких горизонтах. На длинных горизонтах global-модель оказывается устойчивее.

Наиболее перспективным вариантом по итогам сравнения является гибридная схема:

```text
short horizon -> per-cluster
long horizon  -> global
```

Она даёт лучшие значения WAPE:

```text
daily hybrid:   7.30%
weekly hybrid:  19.31%
monthly hybrid: 21.22%
```

---

## 21. Выводы

1. Rolling/direct и fixed-origin постановки измеряют разные сценарии использования модели.
2. Rolling/direct даёт более низкую ошибку, так как прогноз регулярно обновляется.
3. Fixed-origin является более строгой постановкой для долгосрочного прогноза из одной даты.
4. Global-модели устойчивее на длинных горизонтах.
5. Per-cluster модели эффективнее на коротких горизонтах.
6. Гибридная схема `short -> per-cluster`, `long -> global` показывает лучшие результаты.
7. Наиболее проблемными являются `cluster_3` и `cluster_4`.
8. Для дальнейшего улучшения качества следует отдельно исследовать волатильные тикеры и проблемные кластеры.

---


