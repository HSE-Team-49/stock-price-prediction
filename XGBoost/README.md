# README: сравнение моделей прогноза акций

## 1. Общая постановка задачи

В проекте исследуется прогнозирование цен акций на разных временных масштабах:

```text
1. Недельный прогноз — последний год.
2. Месячный прогноз — последний год.
3. Дневной прогноз — последний месяц.
```

Для всех вариантов используется одна общая идея:

```text
XGBoost + Optuna + признаки ценовой динамики + объём торгов + макро + сезонность + кластеры
```

Целевая переменная во всех моделях строится через логарифмическую доходность:

```text
target_logret_h = log(price[t+h]) - log(price[t])
```

После прогноза лог-доходности цена восстанавливается так:

```text
pred_price = price[t] * exp(pred_logret_h)
```

Основные метрики качества:

```text
WAPE
MAPE
MSE
MAE
RMSE
```

В качестве основной метрики для сравнения чаще использовалась `WAPE`, потому что она устойчивее обычного MAPE при разных масштабах цен акций.

---

## 2. Исходные данные

### 2.1. Ценовые данные

Используются два файла:

```text
data/prices_all.csv
data/prices_all_2025_12_20_today.csv
```

Первый файл содержит исторические данные до конца 2025 года, второй — фактические данные после `2025-12-20` до `2026-05-08`.

Основные колонки:

```text
date
Ticker
Open
High
Low
Close
Volume
```

В качестве основной цены использовалась колонка:

```text
Close
```

---

### 2.2. Макропоказатели

Используются макропоказатели из папки:

```text
data/
```

Список макро-файлов:

```text
CPIAUCSL.csv      — CPI, индекс потребительских цен
DCOILBRENTEU.csv  — Brent spot
DEXUSEU.csv       — USD/EUR
DGS2.csv          — доходность 2-летних UST
DGS10.csv         — доходность 10-летних UST
DGS30.csv         — доходность 30-летних UST
EFFR.csv          — Effective Federal Funds Rate
M2SL.csv          — денежная масса M2
UNRATE.csv        — уровень безработицы
```

Для макро учитывались лаги публикации:

```text
месячные макро: 21 бизнес-день
дневные макро: 1 бизнес-день
```
---

### 2.3. Кластеры

Кластеры загружаются из файла:

```text
cluster_fullstart_assignments.csv
```

Используются признаки:

```text
cluster_id
cluster_size
cluster_share
cluster_is_noise
```

В отдельных экспериментах модели обучались отдельно для каждого `cluster_id`.

---

### 2.4. Сезонность

Результаты поиска сезонности загружаются из:

```text
results_stocks/prices_all
```

Используются FFT-признаки:

```text
period_days
peak_share
prominence
signal_share
noise_share
norm_peak_share
period_band
```

Также добавляются фазовые признаки сезонности:

```text
season_*_phase_sin
season_*_phase_cos
```

---

## 3. Общая архитектура моделей

Все модели построены по схеме разделения горизонта на две части:

```text
short_macro
long_no_macro
```

### 3.1. Short-модель

Short-модель использует все признаки, включая макро:

```text
short_macro = price features + volume features + market features + cluster features + season features + macro features
```

Идея: на коротких горизонтах последнее известное макросостояние может быть полезным.

---

### 3.2. Long-модель

Long-модель не использует макро и производные макро-признаки:

```text
long_no_macro = все признаки, кроме macro features
```

Идея: на дальних горизонтах будущие макро-показатели неизвестны, а замороженные старые значения могут ухудшать переносимость модели.

---

## 4. Основные группы признаков

Во всех версиях использовались близкие группы признаков.

### 4.1. Ценовые признаки

```text
price
log_price
price_ratio
price_momentum
distance_from_high
distance_from_low
```

### 4.2. Доходности

```text
ret_lag
ret_abs_lag
ret_sign
ret_roll_mean
ret_roll_std
ret_roll_min
ret_roll_max
ret_roll_median
ret_roll_skew
ret_roll_kurt
realized_vol
downside_vol
upside_vol
```

### 4.3. Технические индикаторы

```text
ATR
RSI
SMA
EMA
MACD
Bollinger Bands
```

### 4.4. Объём торгов

```text
Volume
volume_lag
volume_change
volume_roll_mean
volume_roll_std
volume_zscore
ret_volume_corr
```

### 4.5. Рыночные признаки

```text
market_return
market_return_lags
market_rolling_mean
market_rolling_std
market_regime_bull
market_regime_bear
excess_return
beta_to_market
correlation_with_market
```


### 4.6. Кластерные признаки

```text
cluster_id
cluster_size
cluster_share
cluster_is_noise
cluster_return_mean
cluster_return_std
ret_minus_cluster_mean
ret_zscore_in_cluster
```

### 4.7. Сезонные признаки

```text
season_period_days
season_peak_share
season_prominence
season_signal_share
season_noise_share
season_phase_sin
season_phase_cos
```

### 4.8. Макро-признаки

```text
macro value
macro lag
macro diff
macro pct_change
macro rolling mean
macro rolling std
macro z-score
macro age features
yield spreads
interest-rate spreads
inflation growth
money supply growth
unemployment changes
```

---

# 5. Недельные модели

## 5.1. Лучшая недельная модель

Файл:

```text
global_xgb_weekly_h100_forecast_two_models.py
```

### Подход

Данные агрегируются по неделям.

Горизонты:

```text
1–13 недель   -> short_macro
14–52 недели  -> long_no_macro
```

Модель глобальная:

```text
одна short_macro модель на все тикеры
одна long_no_macro модель на все тикеры
```

Кластеры используются как признаки, но не как отдельные модели.

### Результат

```text
WAPE ≈ 17.32%
MAPE ≈ 19.97%
RMSE ≈ 233.61
```

### Вывод

Это лучшая недельная модель. Она стабильнее кластерной версии и остаётся основной недельной моделью.

---

## 5.2. Недельная модель по кластерам

Файл:

```text
global_xgb_weekly_h100_forecast_per_cluster.py
```

### Подход

Для каждого `cluster_id` обучаются отдельные модели:

```text
cluster_{id}_short_macro
cluster_{id}_long_no_macro
```

Горизонты такие же:

```text
1–13 недель   -> short_macro
14–52 недели  -> long_no_macro
```

### Результат

```text
WAPE ≈ 17.90%
MAPE ≈ 20.47%
RMSE ≈ 261.78
```

### Вывод

per-cluster версия оказалась хуже глобальной недельной модели.  
Кластеризация помогала отдельным группам акций, но ухудшала результат на других кластерах.

---

## 5.3. Недельная гибридная схема

Отдельного финального `.py` файла пока нет.

### Идея

Использовать кластерные модели только там, где они работают лучше, а для проблемного кластера использовать fallback.

По результатам эксперимента лучшая гибридная схема:

```text
clusters 0,1,2,3,5,6 -> per-cluster model
cluster 4             -> fallback
```

### Потенциальный результат

```text
WAPE ≈ 16.95%
MAPE ≈ 19.21%
RMSE ≈ 216.31
```

### Вывод

Гибридная недельная схема потенциально лучше чистой глобальной модели, но для неё нужен отдельный финальный код.

---

# 6. Месячные модели

## 6.1. Месячная глобальная модель

Файл:

```text
global_xgb_monthly_h100_forecast_two_models.py
```

### Подход

Данные агрегируются по месяцам.

Горизонты:

```text
1–3 месяца    -> short_macro
4–12 месяцев  -> long_no_macro
```

Модель глобальная:

```text
одна short_macro модель на все тикеры
одна long_no_macro модель на все тикеры
```

### Результат

```text
WAPE ≈ 19.47%
MAPE ≈ 22.08%
RMSE ≈ 233.84
```

### Вывод

Месячная глобальная модель рабочая, но хуже недельной. Основная причина — потеря части информации при месячной агрегации и меньшее число обучающих наблюдений.

---

## 6.2. Месячная модель по кластерам

Файл:

```text
global_xgb_monthly_h100_forecast_per_cluster.py
```

### Подход

Для каждого кластера обучаются отдельные модели:

```text
cluster_{id}_short_macro
cluster_{id}_long_no_macro
```

Горизонты:

```text
1–3 месяца    -> short_macro
4–12 месяцев  -> long_no_macro
```

### Результат

```text
WAPE ≈ 19.19%
MAPE ≈ 21.75%
RMSE ≈ 241.05
```

### Вывод

Месячная per-cluster модель лучше месячной глобальной по WAPE/MAPE/MAE, но хуже по MSE/RMSE.  
Если нужна именно месячная постановка, лучшая текущая месячная версия — per-cluster.

---

# 7. Дневные модели

## 7.1. Дневная глобальная модель

Файл:

```text
global_xgb_daily_last_month_h100_forecast_clean.py
```

### Подход

Данные используются в дневном виде.

Backtest:

```text
2026-04-08 — 2026-05-08
```

Горизонты:

```text
1–5 торговых дней   -> short_macro
6–21 торговый день  -> long_no_macro
```

### Особенности реализации

Дневная версия оказалась самой тяжёлой по RAM. Для стабильного запуска были внесены технические оптимизации:

```text
1. Сохранение признаков батчами в parquet.
2. Использование threading вместо loky.
3. Сжатие float64/int64 до float32/int32.
4. Облегчение тяжёлых rolling-блоков.
5. Resume-режим для продолжения после сохранения short-модели.
6. Использование более компактного формата матрицы для финального обучения.
```

Важно: эти изменения в основном касаются памяти и не меняют постановку прогноза.

### Результат

```text
WAPE ≈ 6.1516%
MAPE ≈ 5.9135%
MSE ≈ 1189.18
MAE ≈ 17.15
RMSE ≈ 34.48
n = 96 117
```

### Вывод

Это лучшая дневная модель.  
На последнем месяце она показала очень хорошее качество, но её нельзя напрямую сравнивать с недельными/месячными моделями, потому что проверочный период и горизонт другие.

---

## 7.2. Дневная модель по кластерам

Файл:

```text
global_xgb_daily_last_month_h100_forecast_per_cluster.py
```

### Подход

Для каждого кластера обучаются отдельные дневные модели:

```text
cluster_{id}_short_macro
cluster_{id}_long_no_macro
```

Горизонты:

```text
1–5 дней   -> short_macro
6–21 день  -> long_no_macro
```

### Результат

```text
WAPE ≈ 6.3146%
MAPE ≈ 5.9787%
MSE ≈ 1560.56
MAE ≈ 17.60
RMSE ≈ 39.50
n = 96 117
```

### Вывод

дневная per-cluster модель хуже глобальной дневной модели по всем основным метрикам.

При этом per-cluster улучшила результат для части тикеров и кластеров:

```text
clusters 0, 2, 3
```

Но ухудшение на других кластерах, особенно на cluster 4, перекрыло этот выигрыш.

---

## 7.3. Дневная гибридная схема

Отдельного финального `.py` файла пока нет.

### Идея

Использовать:

```text
clusters 0, 2, 3 -> per-cluster daily
clusters 1, 4, 5, 6 -> global daily
```

### Потенциальный результат

```text
WAPE ≈ 6.0798%
MAPE ≈ 5.9277%
MSE ≈ 1145.16
MAE ≈ 16.95
RMSE ≈ 33.84
```

### Вывод

Гибридная дневная схема даёт лучший WAPE/RMSE, но MAPE чуть хуже, чем у чистой global daily.

---

# 8. Сводная таблица результатов

| Масштаб | Модель | `.py` файл | WAPE | MAPE | RMSE | Статус |
|---|---|---|---:|---:|---:|---|
| Weekly | Global two_models | `global_xgb_weekly_h100_forecast_two_models.py` | **17.32%** | **19.97%** | **233.61** | Лучшая weekly |
| Weekly | Per-cluster | `global_xgb_weekly_h100_forecast_per_cluster.py` | 17.90% | 20.47% | 261.78 | Исследовательская |
| Weekly | Hybrid | отдельного файла нет | **16.95%** | **19.21%** | **216.31** | Потенциально лучшая weekly |
| Monthly | Global two_models | `global_xgb_monthly_h100_forecast_two_models.py` | 19.47% | 22.08% | **233.84** | Baseline monthly |
| Monthly | Per-cluster | `global_xgb_monthly_h100_forecast_per_cluster.py` | **19.19%** | **21.75%** | 241.05 | Лучшая monthly |
| Daily | Global clean | `global_xgb_daily_last_month_h100_forecast_clean.py` | **6.1516%** | **5.9135%** | **34.48** | Лучшая daily |
| Daily | Per-cluster | `global_xgb_daily_last_month_h100_forecast_per_cluster.py` | 6.3146% | 5.9787% | 39.50 | Исследовательская |
| Daily | Hybrid | отдельного файла нет | **6.0798%** | 5.9277% | **33.84** | Потенциально лучшая daily |

---

# 9. Сравнение подходов

## 9.1. Global модели

### Преимущества

```text
1. Больше обучающих данных на одну модель.
2. Лучше устойчивость.
3. Меньше риск переобучения на малых группах.
4. Проще запуск и анализ.
5. Хорошо работают на дневном и недельном масштабе.
```

### Недостатки

```text
1. Сглаживают различия между типами акций.
2. Кластеры используются только как признаки.
3. Могут недоучитывать особенности отдельных групп.
```

---

## 9.2. Per-cluster модели

### Преимущества

```text
1. Более специализированные модели.
2. Лучше учитывают поведение отдельных групп акций.
3. Могут улучшать отдельные кластеры и тикеры.
4. Для monthly постановки дали лучший WAPE/MAPE.
```

### Недостатки

```text
1. Меньше данных на каждую модель.
2. Выше риск переобучения.
3. Больше моделей и дольше запуск.
4. На weekly и daily в чистом виде хуже global.
5. Некоторые кластеры сильно портят общий результат.
```

---

## 9.3. Hybrid схемы

### Преимущества

```text
1. Используют сильные стороны global и per-cluster.
2. Могут улучшить WAPE/RMSE.
3. Позволяют не применять кластерные модели к проблемным кластерам.
```

### Недостатки

```text
1. Нужен отдельный селектор моделей.
2. Надо выбирать правила честно по validation, а не по test.
3. Сложнее объяснять и поддерживать.
```

---

# 10. Главные выводы

## 10.1. Лучшая недельная модель

```text
global_xgb_weekly_h100_forecast_two_models.py
```

Она лучше чистой недельной per-cluster версии и остаётся основной weekly-моделью.

---

## 10.2. Лучшая месячная модель

```text
global_xgb_monthly_h100_forecast_per_cluster.py
```

Месячная per-cluster версия лучше месячной global по WAPE/MAPE.

---

## 10.3. Лучшая дневная модель

```text
global_xgb_daily_last_month_h100_forecast_clean.py
```

Она лучше дневной per-cluster версии по всем основным метрикам.

---

## 10.4. Лучшие потенциальные модели

Для weekly и daily лучшие результаты даёт гибридный подход, но для него пока нет отдельного финального `.py` файла:

```text
weekly hybrid:
    per-cluster для хороших кластеров
    fallback для проблемного кластера 4

daily hybrid:
    clusters 0,2,3 -> per-cluster
    clusters 1,4,5,6 -> global
```


---

## 11. Следующий шаг

Сделать две финальные гибридные версии:

```text
global_xgb_weekly_h100_forecast_hybrid_by_cluster.py
global_xgb_daily_last_month_h100_forecast_hybrid_by_cluster.py
```

Они должны выбирать модель по кластеру на основании validation-качества, а не test-результатов.

---

# 12. Краткое резюме

```text
1. Недельная global-модель — лучшая  модель для прогноза на год по неделям.
2. Месячная per-cluster модель — лучшая месячная модель.
3. Дневная global модель — лучшая дневная модель на последний месяц.
4. Per-cluster подход полезен, но не всегда лучше global.
5. Самое перспективное развитие — hybrid by cluster.
```
