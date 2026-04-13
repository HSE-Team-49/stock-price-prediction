import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna


@dataclass
class Config:
    INPUT_PATH: Path = Path("stocks_long.csv")
    OUT_ROOT: Path = Path("results_stocks")
    OUT_DIR: Path = field(init=False)

    DATE_COL: str = "Date"
    TICKER_COL: str = "Ticker"
    PRICE_COL: str = "Close"

    WINDOW_SIZE: int = 90
    VAL_FRACTION: float = 0.2
    N_TRIALS: int = 30
    OPTUNA_N_JOBS: int = 1
    SEED: int = 42

    MIN_PERIOD: int = 2
    MAX_PERIOD: int = 400
    TOP_K: int = 3

    EXAMPLE_TICKERS: List[str] = field(default_factory=list)

    RUN_PRICE_BRANCH: bool = True
    RUN_RETURNS_BRANCH: bool = True

    RUN_AE_TRAIN: bool = True
    RUN_AE_FFT: bool = True

    PRICE_BEST_MODEL_PATH: Path = field(init=False)
    PRICE_FFT_SUMMARY_PATH: Path = field(init=False)
    PRICE_SPECTRA_CSV_PATH: Path = field(init=False)
    PRICE_SPECTRA_PDF_PATH: Path = field(init=False)
    PRICE_TS_CSV_PATH: Path = field(init=False)
    PRICE_TS_PDF_PATH: Path = field(init=False)
    PRICE_BEST_PARAMS_PATH: Path = field(init=False)

    RETURNS_BEST_MODEL_PATH: Path = field(init=False)
    RETURNS_FFT_SUMMARY_PATH: Path = field(init=False)
    RETURNS_SPECTRA_CSV_PATH: Path = field(init=False)
    RETURNS_SPECTRA_PDF_PATH: Path = field(init=False)
    RETURNS_TS_CSV_PATH: Path = field(init=False)
    RETURNS_TS_PDF_PATH: Path = field(init=False)
    RETURNS_BEST_PARAMS_PATH: Path = field(init=False)

    def __post_init__(self):
        stem = Path(self.INPUT_PATH).stem
        safe = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in stem).strip()
        if not safe:
            safe = "run"

        self.OUT_DIR = self.OUT_ROOT / safe
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

        self.PRICE_BEST_MODEL_PATH = self.OUT_DIR / "price_autoencoder_best.pt"
        self.PRICE_FFT_SUMMARY_PATH = self.OUT_DIR / "price_fft_seasonality_ae.xlsx"
        self.PRICE_SPECTRA_CSV_PATH = self.OUT_DIR / "price_fft_spectra_reconstructed_ae.csv"
        self.PRICE_SPECTRA_PDF_PATH = self.OUT_DIR / "price_fft_spectra_reconstructed_ae.pdf"
        self.PRICE_TS_CSV_PATH = self.OUT_DIR / "price_ae_original_vs_recon_timeseries.csv"
        self.PRICE_TS_PDF_PATH = self.OUT_DIR / "price_ae_original_vs_recon_timeseries.pdf"
        self.PRICE_BEST_PARAMS_PATH = self.OUT_DIR / "price_best_params.json"

        self.RETURNS_BEST_MODEL_PATH = self.OUT_DIR / "returns_autoencoder_best.pt"
        self.RETURNS_FFT_SUMMARY_PATH = self.OUT_DIR / "returns_fft_seasonality_ae.xlsx"
        self.RETURNS_SPECTRA_CSV_PATH = self.OUT_DIR / "returns_fft_spectra_reconstructed_ae.csv"
        self.RETURNS_SPECTRA_PDF_PATH = self.OUT_DIR / "returns_fft_spectra_reconstructed_ae.pdf"
        self.RETURNS_TS_CSV_PATH = self.OUT_DIR / "returns_ae_original_vs_recon_timeseries.csv"
        self.RETURNS_TS_PDF_PATH = self.OUT_DIR / "returns_ae_original_vs_recon_timeseries.pdf"
        self.RETURNS_BEST_PARAMS_PATH = self.OUT_DIR / "returns_best_params.json"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stocks_long(cfg: Config) -> pd.DataFrame:
    print(f"[LOAD] Читаю файл: {cfg.INPUT_PATH}")

    df = pd.read_csv(
        cfg.INPUT_PATH,
        sep=None,
        engine="python",
        encoding="utf-8",
    )

    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce")
    df[cfg.PRICE_COL] = pd.to_numeric(df[cfg.PRICE_COL], errors="coerce")

    df = df.dropna(subset=[cfg.DATE_COL, cfg.TICKER_COL, cfg.PRICE_COL]).copy()
    df = df.sort_values([cfg.TICKER_COL, cfg.DATE_COL]).reset_index(drop=True)

    print(f"[LOAD] Строк: {len(df)}, тикеров: {df[cfg.TICKER_COL].nunique()}")
    return df


def build_price_series(df_all: pd.DataFrame, cfg: Config, ticker: str) -> Optional[pd.Series]:
    sub = (
        df_all[df_all[cfg.TICKER_COL] == ticker]
        .set_index(cfg.DATE_COL)
        .sort_index()
    )

    if sub.empty:
        return None

    price = pd.to_numeric(sub[cfg.PRICE_COL], errors="coerce")
    price = price[~price.index.duplicated(keep="last")]
    price = price.dropna()
    price = price[price > 0]

    if price.empty:
        return None

    price.name = "Close"
    price.index.name = "date"
    return price


def transform_price_series(price_series: pd.Series) -> pd.Series:
    x = np.log(price_series.astype(float))
    mu = x.mean()
    sigma = x.std(ddof=0)

    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)

    x_norm = (x - mu) / sigma
    return x_norm


def transform_returns_series(price_series: pd.Series) -> pd.Series:
    x = np.log(price_series.astype(float)).diff().dropna()

    if x.empty:
        return x

    mu = x.mean()
    sigma = x.std(ddof=0)

    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)

    x_norm = (x - mu) / sigma
    return x_norm


def build_windows_from_series_list(
    series_list: List[np.ndarray],
    window_size: int,
) -> np.ndarray:
    all_windows: List[np.ndarray] = []

    for x in series_list:
        if len(x) < window_size:
            continue

        for j in range(len(x) - window_size + 1):
            all_windows.append(x[j:j + window_size])

    if not all_windows:
        raise RuntimeError("[AE] Не удалось собрать ни одного окна.")

    X = np.stack(all_windows).astype(np.float32)
    return X


def train_val_split(X: np.ndarray, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    n_val = int(n * val_fraction)
    n_train = n - n_val

    if n_train <= 0 or n_val <= 0:
        raise RuntimeError(f"[AE] Некорректное разбиение train/val: n={n}, val_fraction={val_fraction}")

    return X[:n_train], X[n_train:]


def compute_periodogram(
    series: np.ndarray,
    min_period: Optional[float],
    max_period: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    x = series.astype(float)
    x = x - np.mean(x)
    n = len(x)

    if n < 10:
        return None, None

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = (np.abs(X) ** 2) / n

    mask = freqs > 0
    if max_period is not None:
        mask &= freqs >= 1.0 / max_period
    if min_period is not None:
        mask &= freqs <= 1.0 / min_period

    freqs = freqs[mask]
    power = power[mask]

    if len(freqs) == 0:
        return None, None

    periods = 1.0 / freqs
    return periods, power


def find_dominant_periods(periods: np.ndarray, power: np.ndarray, top_k: int) -> pd.DataFrame:
    peaks, _ = find_peaks(power)

    if len(peaks) == 0:
        return pd.DataFrame(columns=["period", "power", "rank_by_power", "fft_index"])

    peak_powers = power[peaks]
    order = np.argsort(peak_powers)[::-1]
    top_idx = peaks[order][:top_k]

    return pd.DataFrame({
        "period": periods[top_idx],
        "power": power[top_idx],
        "rank_by_power": np.arange(1, len(top_idx) + 1, dtype=int),
        "fft_index": top_idx,
    })



class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, x


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()

        enc_layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            enc_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            dec_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


def eval_one_epoch(model, loader, criterion) -> float:
    model.eval()
    total = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)

            total += float(loss.item())
            n += 1

    return total / max(1, n)


def optuna_objective(trial, X_train: np.ndarray, X_val: np.ndarray) -> float:
    input_dim = X_train.shape[1]

    latent_dim = trial.suggest_int("latent_dim", 4, 32)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 10, 30)

    train_loader = DataLoader(
        WindowDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        WindowDataset(X_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(n_epochs):
        _ = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_one_epoch(model, val_loader, criterion)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss

    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val


def train_autoencoder_with_optuna(
    X: np.ndarray,
    model_path: Path,
    params_path: Path,
    cfg: Config,
    branch_name: str,
) -> Dict[str, object]:
    print(f"\n=== [AE:{branch_name}] Optuna обучение автоэнкодера ===")
    set_seed(cfg.SEED)

    X_train, X_val = train_val_split(X, cfg.VAL_FRACTION)
    print(f"[AE:{branch_name}] Train windows: {X_train.shape[0]}, Val windows: {X_val.shape[0]}")

    sampler = optuna.samplers.TPESampler(seed=cfg.SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val),
        n_trials=cfg.N_TRIALS,
        n_jobs=cfg.OPTUNA_N_JOBS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"[AE:{branch_name}] Best trial:")
    print(f"  val MSE: {best.value}")
    print(f"  params: {best.params}")

    bp = best.params
    input_dim = X_train.shape[1]

    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=bp["latent_dim"],
        hidden_dim=bp["hidden_dim"],
        n_layers=bp["n_layers"],
        dropout=bp["dropout"],
    ).to(DEVICE)

    train_loader = DataLoader(
        WindowDataset(X_train),
        batch_size=bp["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        WindowDataset(X_val),
        batch_size=bp["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=bp["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(bp["n_epochs"]):
        tr = train_one_epoch(model, train_loader, optimizer, criterion)
        vl = eval_one_epoch(model, val_loader, criterion)
        print(f"[AE:{branch_name} FINAL] epoch {epoch+1}/{bp['n_epochs']} train={tr:.6f} val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), model_path)

    pd.Series({
        "best_value": best.value,
        **best.params,
    }).to_json(params_path, force_ascii=False, indent=2)

    print(f"[AE:{branch_name}] Лучшая модель сохранена: {model_path}")
    print(f"[AE:{branch_name}] Лучшие параметры сохранены: {params_path}")

    return {"best_value": best.value, "best_params": dict(best.params)}


def load_trained_model(model_path: Path, input_dim: int, params: Dict[str, float]) -> Autoencoder:
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=int(params["latent_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        n_layers=int(params["n_layers"]),
        dropout=float(params["dropout"]),
    ).to(DEVICE)

    print(f"[LOAD MODEL] Гружу веса: {model_path}")
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def reconstruct_series_with_ae(model: Autoencoder, x: np.ndarray, window_size: int) -> np.ndarray:
    x = x.astype(np.float32)
    n = len(x)

    if n < window_size:
        return x.copy()

    recon = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(n - window_size + 1):
            window = x[i:i + window_size]
            inp = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            out = model(inp).detach().cpu().numpy().reshape(-1)

            recon[i:i + window_size] += out
            counts[i:i + window_size] += 1.0

    counts[counts == 0] = 1.0
    return recon / counts



def prepare_branch_windows(
    df: pd.DataFrame,
    cfg: Config,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, pd.Series]]:
    tickers = df[cfg.TICKER_COL].unique()
    series_map: Dict[str, pd.Series] = {}

    for i, ticker in enumerate(tickers, start=1):
        price = build_price_series(df, cfg, ticker)
        if price is None:
            continue

        if mode == "price":
            s = transform_price_series(price)
        elif mode == "returns":
            s = transform_returns_series(price)
        else:
            raise ValueError(f"Неизвестный mode={mode}")

        if s is None or s.empty:
            continue

        series_map[ticker] = s

        if i % 25 == 0:
            print(f"[PREP:{mode}] Тикеров обработано: {i}/{len(tickers)}")

    series_list = [s.values.astype(np.float32) for s in series_map.values()]
    X = build_windows_from_series_list(series_list, cfg.WINDOW_SIZE)

    print(f"[PREP:{mode}] Всего рядов: {len(series_map)}")
    print(f"[PREP:{mode}] Всего окон: {X.shape[0]}, длина окна: {cfg.WINDOW_SIZE}")

    return X, series_map



def run_fft_pipeline_for_branch(
    series_map: Dict[str, pd.Series],
    cfg: Config,
    best_params: Dict[str, float],
    model_path: Path,
    summary_path: Path,
    spectra_csv_path: Path,
    spectra_pdf_path: Path,
    ts_csv_path: Path,
    ts_pdf_path: Path,
    branch_name: str,
) -> None:
    print(f"\n=== [AE+FFT:{branch_name}] Реконструкция рядов + FFT ===")

    model = load_trained_model(model_path, input_dim=cfg.WINDOW_SIZE, params=best_params)

    spectra_for_plots: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ts_for_plots: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows: List[Dict[str, object]] = []

    for ticker, s in series_map.items():
        x_vals = s.values.astype(np.float32)
        x_recon = reconstruct_series_with_ae(model, x_vals, cfg.WINDOW_SIZE)

        if (not cfg.EXAMPLE_TICKERS) or (ticker in cfg.EXAMPLE_TICKERS):
            ts_for_plots[ticker] = (s.index.to_pydatetime(), x_vals, x_recon)

        periods, power = compute_periodogram(x_recon, cfg.MIN_PERIOD, cfg.MAX_PERIOD)
        if periods is None or power is None:
            continue

        if (not cfg.EXAMPLE_TICKERS) or (ticker in cfg.EXAMPLE_TICKERS):
            spectra_for_plots[ticker] = (periods, power)

        dom = find_dominant_periods(periods, power, cfg.TOP_K)

        total_power = float(np.nansum(power))
        if not np.isfinite(total_power) or total_power <= 0:
            total_power = np.nan

        if (not dom.empty) and np.isfinite(total_power) and total_power > 0:
            signal_power = float(np.nansum(dom["power"].to_numpy(dtype=float)))
            noise_power = float(total_power - signal_power)
            signal_share = signal_power / total_power
            noise_share = noise_power / total_power
        else:
            signal_power = np.nan
            noise_power = np.nan
            signal_share = np.nan
            noise_share = np.nan

        if dom.empty:
            summary_rows.append({
                "ticker": ticker,
                "mode": branch_name,
                "rank": np.nan,
                "period_days": np.nan,
                "peak_power": np.nan,
                "n_points": int(len(x_recon)),
                "total_power_band": float(total_power) if np.isfinite(total_power) else np.nan,
                "signal_power_topk": float(signal_power) if np.isfinite(signal_power) else np.nan,
                "noise_power_band": float(noise_power) if np.isfinite(noise_power) else np.nan,
                "signal_share": float(signal_share) if np.isfinite(signal_share) else np.nan,
                "noise_share": float(noise_share) if np.isfinite(noise_share) else np.nan,
            })
        else:
            for _, r in dom.iterrows():
                summary_rows.append({
                    "ticker": ticker,
                    "mode": branch_name,
                    "rank": int(r["rank_by_power"]),
                    "period_days": float(r["period"]),
                    "peak_power": float(r["power"]),
                    "n_points": int(len(x_recon)),
                    "total_power_band": float(total_power) if np.isfinite(total_power) else np.nan,
                    "signal_power_topk": float(signal_power) if np.isfinite(signal_power) else np.nan,
                    "noise_power_band": float(noise_power) if np.isfinite(noise_power) else np.nan,
                    "signal_share": float(signal_share) if np.isfinite(signal_share) else np.nan,
                    "noise_share": float(noise_share) if np.isfinite(noise_share) else np.nan,
                })

    fft_summary = pd.DataFrame(summary_rows)

    if not fft_summary.empty:
        fft_summary["peak_share"] = (
            pd.to_numeric(fft_summary["peak_power"], errors="coerce")
            / pd.to_numeric(fft_summary["total_power_band"], errors="coerce")
        )
        fft_summary.loc[~np.isfinite(fft_summary["peak_share"]), "peak_share"] = np.nan
    else:
        fft_summary["peak_share"] = pd.Series(dtype=float)

    if not fft_summary.empty:
        ps = pd.to_numeric(fft_summary["peak_share"], errors="coerce")
        ps_min = ps.min(skipna=True)
        ps_max = ps.max(skipna=True)
        denom = ps_max - ps_min

        if np.isfinite(denom) and denom > 0:
            fft_summary["norm_signal_share"] = (ps - ps_min) / denom
        else:
            fft_summary["norm_signal_share"] = np.nan
    else:
        fft_summary["norm_signal_share"] = pd.Series(dtype=float)

    with pd.ExcelWriter(summary_path, engine="xlsxwriter") as writer:
        fft_summary.to_excel(writer, sheet_name=f"fft_{branch_name}_ae", index=False)
    print(f"[AE+FFT:{branch_name}] Summary сохранён: {summary_path}")

    if spectra_for_plots:
        rows = []
        for ticker, (periods, power) in spectra_for_plots.items():
            power_norm = power / power.max() if power.max() > 0 else power
            for p, pw, pwn in zip(periods, power, power_norm):
                rows.append({
                    "ticker": ticker,
                    "mode": branch_name,
                    "period_days": float(p),
                    "power": float(pw),
                    "power_norm": float(pwn),
                })

        pd.DataFrame(rows).to_csv(spectra_csv_path, index=False)
        print(f"[AE+FFT:{branch_name}] Спектры CSV: {spectra_csv_path}")

        with PdfPages(spectra_pdf_path) as pdf:
            for ticker, (periods, power) in spectra_for_plots.items():
                plt.figure(figsize=(8, 4))
                power_plot = power / power.max() if power.max() > 0 else power
                plt.plot(periods, power_plot)
                plt.xlabel("Период, наблюдений")
                plt.ylabel("Нормированная мощность (0–1)")
                plt.title(f"Нормированный спектр реконструкции (AE), {branch_name}, тикер: {ticker}")
                plt.xscale("log")
                plt.grid(True, which="both", alpha=0.3)
                pdf.savefig()
                plt.close()

        print(f"[AE+FFT:{branch_name}] Спектры PDF: {spectra_pdf_path}")

    if ts_for_plots:
        rows = []
        value_col_orig = f"orig_{branch_name}"
        value_col_recon = f"recon_{branch_name}"

        for ticker, (dates, orig, recon) in ts_for_plots.items():
            for d, o, r in zip(dates, orig, recon):
                rows.append({
                    "ticker": ticker,
                    "mode": branch_name,
                    "date": d,
                    value_col_orig: float(o),
                    value_col_recon: float(r),
                })

        pd.DataFrame(rows).to_csv(ts_csv_path, index=False)
        print(f"[AE+FFT:{branch_name}] Временные ряды CSV: {ts_csv_path}")

        with PdfPages(ts_pdf_path) as pdf:
            for ticker, (dates, orig, recon) in ts_for_plots.items():
                plt.figure(figsize=(10, 4))
                plt.plot(dates, orig, label=f"Оригинал ({branch_name})", alpha=0.7)
                plt.plot(dates, recon, label=f"Реконструкция AE ({branch_name})", alpha=0.7)
                plt.xlabel("Дата")
                plt.ylabel("Нормированное значение")
                plt.title(f"Оригинал vs AE, {branch_name}, тикер: {ticker}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig()
                plt.close()

        print(f"[AE+FFT:{branch_name}] Временные ряды PDF: {ts_pdf_path}")



def run_price_branch(df: pd.DataFrame, cfg: Config) -> None:
    print("\n" + "=" * 70)
    print("[BRANCH] PRICE")
    print("=" * 70)

    X_price, series_map_price = prepare_branch_windows(df, cfg, mode="price")

    best_params = None
    if cfg.RUN_AE_TRAIN:
        result = train_autoencoder_with_optuna(
            X=X_price,
            model_path=cfg.PRICE_BEST_MODEL_PATH,
            params_path=cfg.PRICE_BEST_PARAMS_PATH,
            cfg=cfg,
            branch_name="price",
        )
        best_params = result["best_params"]

    if cfg.RUN_AE_FFT:
        if best_params is None:
            raise RuntimeError("[AE+FFT:price] Не найдены best_params.")
        run_fft_pipeline_for_branch(
            series_map=series_map_price,
            cfg=cfg,
            best_params=best_params,
            model_path=cfg.PRICE_BEST_MODEL_PATH,
            summary_path=cfg.PRICE_FFT_SUMMARY_PATH,
            spectra_csv_path=cfg.PRICE_SPECTRA_CSV_PATH,
            spectra_pdf_path=cfg.PRICE_SPECTRA_PDF_PATH,
            ts_csv_path=cfg.PRICE_TS_CSV_PATH,
            ts_pdf_path=cfg.PRICE_TS_PDF_PATH,
            branch_name="price",
        )


def run_returns_branch(df: pd.DataFrame, cfg: Config) -> None:
    print("\n" + "=" * 70)
    print("[BRANCH] RETURNS")
    print("=" * 70)

    X_returns, series_map_returns = prepare_branch_windows(df, cfg, mode="returns")

    best_params = None
    if cfg.RUN_AE_TRAIN:
        result = train_autoencoder_with_optuna(
            X=X_returns,
            model_path=cfg.RETURNS_BEST_MODEL_PATH,
            params_path=cfg.RETURNS_BEST_PARAMS_PATH,
            cfg=cfg,
            branch_name="returns",
        )
        best_params = result["best_params"]

    if cfg.RUN_AE_FFT:
        if best_params is None:
            raise RuntimeError("[AE+FFT:returns] Не найдены best_params.")
        run_fft_pipeline_for_branch(
            series_map=series_map_returns,
            cfg=cfg,
            best_params=best_params,
            model_path=cfg.RETURNS_BEST_MODEL_PATH,
            summary_path=cfg.RETURNS_FFT_SUMMARY_PATH,
            spectra_csv_path=cfg.RETURNS_SPECTRA_CSV_PATH,
            spectra_pdf_path=cfg.RETURNS_SPECTRA_PDF_PATH,
            ts_csv_path=cfg.RETURNS_TS_CSV_PATH,
            ts_pdf_path=cfg.RETURNS_TS_PDF_PATH,
            branch_name="returns",
        )



def run_one_file(input_path: Path):
    cfg = Config(INPUT_PATH=input_path, EXAMPLE_TICKERS=[])

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print(f"[RUN] FILE: {input_path}")
    print(f"[RUN] OUT : {cfg.OUT_DIR}")
    print(f"[RUN] DEVICE: {DEVICE}")
    print("==============================\n")

    set_seed(cfg.SEED)
    df = load_stocks_long(cfg)

    if cfg.RUN_PRICE_BRANCH:
        run_price_branch(df, cfg)

    if cfg.RUN_RETURNS_BRANCH:
        run_returns_branch(df, cfg)

    print(f"\n[DONE] {input_path} -> {cfg.OUT_DIR}")


def main():
    run_one_file(Path("stocks_long.csv"))


if __name__ == "__main__":
    main()
