# macro_parsing.ipynb - скрипты для выгрузки макропоказателей с помощью FRED_API
- Описание всех макропоказателей находится в файле макро_объянение.pdf
- Пропуски в данных практически отсутвуют
## Выходные файлы  
- all_long.csv — “длинный” формат для всех серий. Колонки: date, series, value
- all_wide.csv — “широкий” формат, одна строка на дату, по столбцу на серию. Колонки: date, CPIAUCSL, DCOILBRENTEU, DEXUSEU, DGS2, DGS10, DGS30, EFFR, M2SL, UNRATE
- CPIAUCSL.csv — CPI (индекс потребительских цен), мес.
- DCOILBRENTEU.csv — Brent spot (долл./барр.), дн.
- DEXUSEU.csv — USD/EUR (долларов за евро), дн.
- DGS2.csv — Доходность 2-летних UST, %, дн.
- DGS10.csv — Доходность 10-летних UST, %, дн.
- DGS30.csv — Доходность 30-летних UST, %, дн.
- EFFR.csv — Effective Federal Funds Rate, %, дн.
- M2SL.csv — Денежная масса M2, мес.
- UNRATE.csv — Уровень безработицы, %, мес.
- Так же строятся интерактивные графики 
# парсинг с yahoo.ipynb - скрипты для выгрузки рыночных данных с помощью FRED_API
- Сначала скачивается список тикеров топ 200 по капитализации компаний рынка США
- После для этих топ 200 выгружаются подневные данные по каждому тикеру с 1 января 2008 года
- Пропуски и выбросы в данных практически отсутвуют
1. Цена открытия
2. Топ цена за день
3. Минимальная цена за день
4. Цена закрытия
5. Объём торгов
## Выходные файлы 
- data/prices_all.csv — все тикеры одним файлом; колонки: date, Ticker, Open, High, Low, Close, Adj Close, Volume
- data/{TICKER}.csv — по одному файлу на тикер; те же колонки, что в prices_all.csv
- data/top200_tickers.csv — список запрошенных тикеров (по одному в строке), колонка: Ticker
- data/qa/duplicates.csv — дубликаты строк по ключу (Ticker, date); колонки как в prices_all.csv
- data/qa/missing_total.csv — суммарные пропуски по каждому столбцу; индекс = имя колонки, поля: missing_count, missing_pct
- data/qa/missing_by_ticker.csv — пропуски по каждому тикеру и полю; колонки: Ticker, rows, Open, High, Low, Close, Adj Close, Volume, Open_pct, … (только присутствующие поля + доли _pct).
- data/qa/anomalies_ohlc_rules.csv — нарушения правил OHLC и базовые несоответствия. Примеры reason: High < Low, Open > High, Close <= 0, Volume < 0, и т.п.
- data/qa/anomalies_returns.csv — всплески доходности
- data/qa/anomalies_volume_spikes.csv
- data/qa/zero_volume.csv
- data/qa/all_price_fields_missing.csv — строки, где все доступные ценовые поля одновременно
- Так же строятся интерактивные графики  
## Проверка данных 
- Общие настройки допусков:
1. Числовые поля приводим к float64 (объём — к float64 или int64).
2. Малый допуск на округления:
- eps = max(1e-8, 1e-6 * Close) (используется в сравнениях «≤/≥»).
## Правила OHLC
1. Базовый порядок цен
   - High + eps >= Low
   - High + eps >= max(Open, Close)
   - Low - eps <= min(Open, Close)
   - Если нарушено — reason = "OHLC_order_violation".
2. Положительность цен
   - Open > 0, High > 0, Low > 0, Close > 0 (и Adj Close > 0, если есть).
3. Высота свечи неотрицательна
   - High - Low >= -eps.
4. Экстремальная доходность по Close
   - Лог-доходность: r_t = ln(Close_t / Close_{t-1}). Если |r_t| > 0.3 (30%) — помечаем день как событие
   
