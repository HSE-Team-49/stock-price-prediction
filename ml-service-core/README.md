# README — ML Core (SQLite Feature Store + MLflow Retrain) 

## 1) Что это за модуль и зачем он нужен

Этот репозиторий — **ядро ML-части** проекта, которое решает ровно две задачи:

1. **Хранение фичей и логов инференса в SQLite** (feature storage / request logs).
2. **Переобучение моделей с трекингом экспериментов в MLflow** и **простым “деплоем”** через локальный registry в SQLite (назначение активной модели).

Важно:

* Модуль сделан так, чтобы его можно было подключить:

  * как **библиотеку** (`import mlcore...`),
  * как **CLI** (`python scripts/...`),
  * как **docker runner** через `docker compose run runner ...`.

---

## 2) Главная идея архитектуры

### Данные и потоки

**A. Extract → Feature Store**

* Берём `prices_all.csv`,
* строим месячную панель признаков (`make_monthly_panel`),
* сохраняем всё в SQLite таблицу `feature_store`.

**B. Retrain → MLflow + SQLite experiments**

* Читаем `feature_store` из SQLite,
* строим **отдельную модель на каждый тикер** (per-ticker),
* тюним гиперпараметры Optuna на train-only (до `test_year`),
* оцениваем walk-forward на `test_year`,
* логируем:

  * метрики и артефакты (csv/json/bundle) → **MLflow**,
  * запись об эксперименте → SQLite `experiments`,
  * путь к бандлу модели сохраняется в `experiments.model_path`.

**C. Deploy → SQLite registry**

* “Деплой” — это назначить **active model**:

  * записать в SQLite `model_registry` активный `experiment_id` и `active_model_path`.

**D. Inference → SQLite logs**

* Инференс читает активный бандл (joblib),
* достаёт фичи из `feature_store` по ключу `(ticker, target_month)`,
* считает предикт,
* пишет лог запроса в `request_logs` (в том числе ошибки/latency).

---

## 3) Быстрый старт (самое короткое)

### 3.1 Поднять MLflow (Docker)

```bash
docker compose up -d --build mlflow
```

MLflow UI:

* [http://localhost:5000](http://localhost:5000)

### 3.2 Заполнить feature_store из CSV

(предполагается, что `data/prices_all.csv` уже есть)

```bash
docker compose run --rm runner python scripts/extract_data.py \
  --csv data/prices_all.csv \
  --db storage/app.db
```

### 3.3 Переобучить модель (пример: XGB)

```bash
docker compose run --rm runner python scripts/retrain.py \
  --db storage/app.db \
  --model xgb \
  --test-year 2024 \
  --holdout 12 \
  --trials 25 \
  --seed 13 \
  --use-gpu-xgb \
  --mlflow-exp stocks-monthly \
  --out-dir storage/models
```

Скрипт выведет `experiment_id`.

### 3.4 Назначить активную модель

```bash
docker compose run --rm runner python scripts/set_active.py \
  --db storage/app.db \
  --experiment-id 1
```

### 3.5 Сделать прогноз

```bash
docker compose run --rm runner python scripts/predict.py \
  --db storage/app.db \
  --ticker AAPL \
  --target-month 2024-06
```

---

## 4) Список файлов проекта и что делает каждый


### 4.1 Корень репозитория

#### `docker-compose.yml`

Поднимает:

* `mlflow` — сервер MLflow с SQLite backend внутри volume
* `runner` — контейнер, в котором запускаются наши CLI-скрипты (extract/retrain/predict)

`runner` не стартует как сервис — он запускается командой:

```bash
docker compose run --rm runner <команда>
```

#### `requirements.txt`

Питон-зависимости:

* `pandas/numpy/sklearn` — работа с данными и метрики
* `optuna` — подбор гиперпараметров
* `xgboost`, `catboost` — модели
* `mlflow` — трекинг экспериментов
* `joblib` — сериализация model bundle
* `python-dotenv` — можно добавить `.env` и читать переменные (опционально)

#### `.flake8`

Правила форматирования/линтинга.

#### `README.md`

Этот документ.

#### `storage/.gitkeep`

Папка `storage/` хранит:

* `storage/app.db` — SQLite база проекта (feature_store + logs + registry + experiments)
* `storage/models/` — сохранённые model bundles
* `storage/mlflow/` — MLflow DB + artifacts (volume для сервиса mlflow)

#### `data/.gitkeep`

Папка `data/` — входные файлы, минимум:

* `data/prices_all.csv`

---

### 4.2 Папка `docker/`

#### `docker/mlflow/Dockerfile`

Собирает контейнер MLflow:

* стартует `mlflow server`
* использует:

  * backend store: `sqlite:////mlflow/mlflow.db`
  * artifact root: `/mlflow/artifacts`
* обе директории мапятся в volume `./storage/mlflow:/mlflow`

#### `docker/runner/Dockerfile`

Собирает контейнер “исполнителя”:

* ставит зависимости из `requirements.txt`
* копирует `mlcore/` и `scripts/`
* задаёт `PYTHONPATH=/app`, чтобы можно было `import mlcore`

---

### 4.3 Папка `mlcore/` — библиотека (то, что импортируют другие части проекта)

#### `mlcore/__init__.py`

Экспортирует основные entrypoints:

* `SQLiteStore`
* `predict_one`, `predict_batch`
* `retrain_and_log`

Это позволяет делать:

```python
from mlcore import SQLiteStore, predict_one, retrain_and_log
```

#### `mlcore/utils_git.py`

Функция:

* `get_commit_hash()` — пытается взять текущий git hash через `git rev-parse HEAD`.

Зачем:

* мы логируем commit hash в MLflow и в SQLite `experiments` как метадату эксперимента.

#### `mlcore/features.py`

Здесь находится весь feature engineering.

Основные функции:

* `load_prices_csv(csv_path)`
  Загружает `prices_all.csv`, нормализует имена колонок (`date`, `Ticker`, `Open`, ...), приводит `date` к datetime.

* `pick_price_col(df, pref="auto")`
  Выбирает колонку цены:

  * `Adj Close` если есть
  * иначе `Close`

* `filter_tickers_starting_at_global_min(df)`
  Оставляет только тикеры, которые начинаются **с самой ранней общей даты** (чтобы выровнять историю).

* `make_monthly_panel(df_daily, price_col)`
  Ключевая функция: строит **месячную панель** и целевую переменную.

  * `feature_month = t`
  * `target_month = t+1`
  * `target_next = ret(t+1)` (лог-доходность следующего месяца)

* `get_feature_columns(panel)`
  Возвращает список фичей (все колонки, кроме служебных).

Фичи, которые реально рассчитываются:

* Лаги доходности: `ret_lag{1,2,3,6,9,12}`
* Скользящие по доходности (на окнах 3/6/12):

  * `ret_roll_mean_W`
  * `ret_roll_std_W`
  * `ret_mom_W` (сумма доходностей на окне = моментум)
* ATR (на окнах 6/12): `atr_W`
* RSI (периоды 6/12): `rsi_P`
* По объёму (если есть `Volume`):

  * `vol_dln_1` (разность лог-объёма)
  * `vol_roll_std_W`
* Рыночный фактор (простой): `mkt_ret` = средняя доходность по тикерам

  * плюс лаги `mkt_ret_lag{1,3,6,12}`
* Календарная сезонность месяца:

  * `month_sin`, `month_cos`

#### `mlcore/store_sqlite.py`

Класс `SQLiteStore` — **единственная точка доступа** к SQLite.

Таблицы:

1. `feature_store` — feature storage
2. `experiments` — история retrain
3. `model_registry` — активная модель (deploy)
4. `request_logs` — логирование инференса

Методы:

* `ensure_feature_store_table(feature_cols)`
  Создаёт `feature_store` с динамическими колонками под ваши фичи.

* `replace_features(panel, feature_cols)`
  Делает полную перезаливку `feature_store` (DELETE + INSERT).

* `fetch_features_for_requests(requests)`
  Достаёт строки фичей по списку ключей `(ticker, target_month)`.

* `load_panel_for_training()`
  Загружает все данные из `feature_store` для обучения.

* `insert_experiment(...)` / `get_experiment(id)`
  Пишет/читает эксперименты.

* `set_active_model(experiment_id, model_path)` / `get_active_model()`
  Управляет active model в `model_registry`.

* `insert_request_log(...)`
  Пишет лог инференса (включая ошибки).

#### `mlcore/model_bundle.py`

`ModelBundle` — сериализуемый объект, который является вашим “деплой-артефактом”.

Содержит:

* `model_kind` (xgb/cat)
* `created_at`, `commit_hash`, `experiment_name`
* `feature_columns` — список фичей (контроль дрейфа схемы)
* `per_ticker_params` — лучшие гиперпараметры per ticker
* `models` — **обученные модели per ticker** (объекты XGB/Cat)

Методы:

* `save(path)` и `load(path)` через `joblib`.

Важно:

* После “deploy” сервисы инференса должны использовать именно bundle.

#### `mlcore/train.py`

Содержит весь pipeline retrain.

Ключевые части:

* `optuna_xgb_one_ticker(...)`

* `optuna_cat_one_ticker(...)`
  Подбор гиперпараметров на трейне, где валид = последние `holdout_months` по `target_month`.

* `walk_forward_predict_one_ticker_xgb(...)`

* `walk_forward_predict_one_ticker_cat(...)`
  Walk-forward expanding: для каждого тестового месяца обучаемся на всей истории до него.

* `retrain_and_log(...)`
  Главный entrypoint retrain:

  1. читает `feature_store` из SQLite
  2. тюнит и учит per-ticker
  3. собирает preds + per_ticker metrics + overall metrics
  4. логирует в MLflow:

     * params
     * overall metrics
     * artifacts: `preds.csv`, `per_ticker_metrics.csv`, `summary.json`
     * model bundle как artifact в `models/`
  5. пишет запись в SQLite `experiments`
  6. возвращает `RetrainOutputs` (experiment_id/run_id/model_path/metrics)

#### `mlcore/infer.py`

Функции инференса:

* `predict_one(store, ticker, target_month)`

  1. читает active model из `model_registry`
  2. загружает `ModelBundle`
  3. достаёт фичи из `feature_store`
  4. берёт модель нужного тикера из `bundle.models[ticker]`
  5. считает `y_pred`
  6. пишет лог в `request_logs` (latency, payload, errors)
  7. возвращает JSON-like dict результата

* `predict_batch(store, requests)`
  Просто циклом вызывает `predict_one` (в этом каркасе без оптимизаций).

Ошибки:

* `ModelNotDeployedError` — активная модель не назначена
* `FeatureNotFoundError` — нет фичей для пары (ticker, target_month) или нет модели тикера в бандле

---

### 4.4 Папка `scripts/` — CLI-интерфейс (что дергают другие компоненты)

#### `scripts/extract_data.py`

Задача:

* заполнить `feature_store` из CSV.

Аргументы:

* `--csv` путь к `prices_all.csv`
* `--db` путь к SQLite базе
* `--price-col` `auto|Close|Adj Close`

Результат:

* полностью перезаписывает `feature_store`.

#### `scripts/retrain.py`

Задача:

* переобучить модель + залогировать в MLflow + записать experiment в SQLite.

Аргументы:

* `--db` SQLite
* `--model` `xgb|cat`
* `--test-year` например 2024
* `--holdout` например 12
* `--trials` например 25
* `--seed` например 13
* `--use-gpu-xgb`, `--use-gpu-cat`
* `--mlflow-uri` (по умолчанию берётся из env `MLFLOW_TRACKING_URI`)
* `--mlflow-exp` имя эксперимента MLflow
* `--out-dir` куда сохранять `bundle/joblib` и csv/json артефакты

Вывод:

* печатает `experiment_id` и `model_path`.

#### `scripts/set_active.py`

Задача:

* назначить активную модель (deploy).

Аргументы:

* `--db`
* `--experiment-id`

Что делает:

* читает experiment из SQLite
* берёт `model_path`
* пишет в `model_registry` активную модель

#### `scripts/predict.py`

Задача:

* сделать прогноз(ы) по активной модели.

Режимы:

1. одиночный:

* `--ticker`, `--target-month`

2. batch:

* `--csv` файл со столбцами `ticker,target_month`

Всегда:

* `--db` путь к SQLite

---

## 5) Контракты для подключения к остальным частям проекта

Здесь главное: **что должны передавать другие модули** и **что они могут ожидать**.

### 5.1 Контракт на feature storage

Другие модули НЕ должны пересчитывать фичи на лету.
Они должны:

* (a) один раз заполнить `feature_store` через `scripts/extract_data.py`
  или
* (b) в будущем — если захотите — можно сделать incremental-upsert, но сейчас у нас полная перезаливка.

**Ключ поиска фичей:**

* `ticker` (строка, например `"AAPL"`)
* `target_month` (строка `"YYYY-MM"`)

То есть внешний сервис инференса должен понимать, что запрос “дай прогноз на 2024-06” означает:

* в базе должна быть строка с `target_month="2024-06"`

### 5.2 Контракт на retrain

Внешний orchestrator (например, backend или cron) должен:

1. убедиться, что `feature_store` заполнен
2. вызвать retrain CLI:

```bash
python scripts/retrain.py --db storage/app.db --model xgb ...
```

3. получить `experiment_id`
4. по бизнес-логике решить: деплоить или нет
5. при деплое вызвать:

```bash
python scripts/set_active.py --db storage/app.db --experiment-id N
```

### 5.3 Контракт на inference (как подключить “к остальному проекту”)


Пример (внутри вашего backend):

```python
from mlcore.store_sqlite import SQLiteStore
from mlcore.infer import predict_one

store = SQLiteStore("storage/app.db")

def handle_request(ticker: str, target_month: str):
    return predict_one(store, ticker, target_month)
```

Что возвращается:

```json
{
  "ticker": "AAPL",
  "target_month": "2024-06",
  "y_pred": 0.0123,
  "experiment_id": 1,
  "model_kind": "xgb",
  "latency_ms": 3.21
}
```

Логи:

* запись попадёт в `request_logs` автоматически.

### 5.4 Где хранится “active model” и как это использовать

Внешние модули НЕ должны сами угадывать, какая модель активная.
Они должны:

* либо вызывать `predict_one` (он сам читает registry),
* либо (если очень надо) читать `model_registry` через SQLite.

Таблица:

* `model_registry` содержит `active_model_path` и `active_experiment_id`.

---

## 6) Схема SQLite базы данных (подробно)

Файл базы по умолчанию:

* `storage/app.db`

### 6.1 `feature_store`

Назначение:

* хранит фичи и таргет для обучения/оценки

Ключ:

* `PRIMARY KEY (ticker, target_month)`

Колонки:

* `ticker` TEXT
* `feature_month` TEXT (`YYYY-MM`) — месяц признаков t
* `target_month` TEXT (`YYYY-MM`) — месяц прогноза t+1
* `target_next` REAL — истинное значение (лог-доходность t+1)
* далее динамически: все фичи из `features.py`

### 6.2 `experiments`

Назначение:

* история retrain + метаданные + путь к модели

Колонки:

* `id` INTEGER PK
* `created_at` TEXT (UTC ISO)
* `model_kind` TEXT (`xgb`/`cat`)
* `test_year` INTEGER
* `holdout_months` INTEGER
* `n_trials` INTEGER
* `run_id` TEXT (MLflow run id)
* `experiment_name` TEXT (MLflow experiment name)
* `commit_hash` TEXT
* `model_path` TEXT (путь к bundle .joblib)
* `metrics_json` TEXT (summary, JSON строкой)

### 6.3 `model_registry`

Назначение:

* хранит, какая модель активна (deploy)

Строка всегда одна (`id=1`).
Колонки:

* `active_experiment_id`
* `active_model_path`
* `updated_at`

### 6.4 `request_logs`

Назначение:

* журнал прогнозов (включая ошибки)

Колонки:

* `ts` — время
* `experiment_id` — какой эксперимент был активен
* `model_kind` — тип модели
* `ticker`, `target_month`
* `payload_json` — вход
* `y_pred`
* `latency_ms`
* `error` — строка ошибки, если была

---

## 7) MLflow: где смотреть результаты и какие артефакты создаются

MLflow поднимается сервисом `mlflow`:

* UI: [http://localhost:5000](http://localhost:5000)

В каждом retrain-run логируются:

1. Params:

* `model_kind`, `test_year`, `holdout_months`, `n_trials`, `seed`, `use_gpu_*`, `commit_hash`

2. Metrics:

* `overall_rmse`, `overall_mae`, `overall_r2`
* `n_rows`, `n_tickers`
* `train_total_seconds`

3. Artifacts:

* `artifacts/<model_kind>_preds_test_<year>.csv`
* `artifacts/<model_kind>_per_ticker_metrics_test_<year>.csv`
* `artifacts/<model_kind>_summary_test_<year>.json`
* `models/<model_kind>_bundle_<year>_<timestamp>.joblib`

---

## 8) Типовые сценарии использования (для команды)

### Сценарий 1: Первичная инициализация

1. положить CSV в `data/prices_all.csv`
2. поднять MLflow
3. extract_data
4. retrain
5. deploy active model
6. predict test

### Сценарий 2: Периодическое переобучение (например, раз в неделю)

1. обновить `data/prices_all.csv`
2. `extract_data.py` (пересоздать feature_store)
3. `retrain.py`
4. (опционально) сравнить метрики в MLflow
5. `set_active.py` если нужно

### Сценарий 3: Инференс в прод-части

Ваш API/GUI/Backend сервис:

* хранит доступ к `storage/app.db` (volume/shared storage)
* вызывает `predict_one(...)`
* получает ответ и возвращает пользователю
* логи автоматически пишутся в `request_logs`

---

## 9) Как подключить к “остальным частям проекта” — практическая инструкция

### Вариант A: Подключение как Python-библиотеки (рекомендуется)

1. Добавьте этот репозиторий как git submodule или как отдельную папку.
2. В вашем backend-проекте добавьте зависимость:

   * либо через `pip install -e .` (editable),
   * либо копированием `mlcore/` как пакет.

Минимальный код в вашем backend:

```python
from mlcore.store_sqlite import SQLiteStore
from mlcore.infer import predict_one

store = SQLiteStore("storage/app.db")

def forward_handler(ticker: str, target_month: str):
    return predict_one(store, ticker, target_month)
```

Требования к окружению backend:

* иметь доступ к файлу `storage/app.db`
* иметь доступ к `storage/models/...` (если `model_registry` указывает на локальный путь)
* зависимости `joblib`, `pandas`, и библиотека модели (xgb/cat) должны быть установлены

### Вариант B: Подключение как CLI (через subprocess)

Если ваш основной сервис на другом языке или вы хотите “изолировать” ML:

* Вызывайте `scripts/predict.py` из основного backend через subprocess,
* Парсите stdout (JSON или CSV).

Пример одиночного вызова:

```bash
python scripts/predict.py --db storage/app.db --ticker AAPL --target-month 2024-06
```

### Вариант C: Подключение через Docker runner

Если хотите стандартизировать окружение:

* основной проект вызывает:

```bash
docker compose run --rm runner python scripts/predict.py ...
```

---

## 10) Частые ошибки и как их быстро диагностировать

1. **"Active model is not set"**

* не был вызван `set_active.py`
* проверьте `model_registry` в `storage/app.db`

2. **"Features not found for (ticker, target_month)"**

* в `feature_store` нет строки с таким ключом
* убедитесь, что:

  * вы запускали `extract_data.py`
  * `target_month` формата `"YYYY-MM"`

3. **MLflow не пишет артефакты**

* проверьте, что `MLFLOW_TRACKING_URI` у runner указан
* проверьте доступность `http://mlflow:5000` внутри docker network

4. **CatBoost GPU падает/медленный**

* запускайте retrain без `--use-gpu-cat` (перейдёт на CPU)
* или используйте XGBoost для GPU

---
