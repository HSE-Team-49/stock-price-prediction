from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd


SCRIPT_ORDER: List[Tuple[str, str]] = [
    ("daily_global", "daily_global_fixed_origin_old_features.py"),
    ("daily_per_cluster", "daily_per_cluster_fixed_origin_old_features.py"),
    ("weekly_global", "weekly_global_fixed_origin_old_features.py"),
    ("weekly_per_cluster", "weekly_per_cluster_fixed_origin_old_features.py"),
    ("monthly_global", "monthly_global_fixed_origin_old_features.py"),
    ("monthly_per_cluster", "monthly_per_cluster_fixed_origin_old_features.py"),
]


@dataclass
class Config:
    prices_csv: str = "data/prices_all.csv"
    future_prices_csv: Optional[str] = None
    macro_dir: str = "data"
    cluster_csv: Optional[str] = None
    season_dir: Optional[str] = None
    out_root: str = "results_xgb_old_features_fixed_origin"
    run_dir: Optional[str] = None
    scripts_dir: Optional[str] = None
    price_col: str = "Close"

    eval_end_date: str = "2026-05-08"
    eval_years: int = 1
    macro_known_until: str = "2025-12-20"

    n_trials: int = 300
    n_jobs: int = 8
    numba_threads: int = 8
    use_gpu: int = 1
    gpu_id: int = 0
    random_state: int = 42

    max_optuna_rows: int = 0
    max_train_rows: int = 0

    start_from: Optional[str] = None
    only: Optional[str] = None
    force_rerun_step: Optional[str] = None

    daily_forecast_horizon_days: int = 21
    daily_short_horizon_days: int = 5
    weekly_forecast_horizon_weeks: int = 52
    weekly_short_horizon_weeks: int = 13
    monthly_forecast_horizon_months: int = 12
    monthly_short_horizon_months: int = 3

    include_noise_cluster: int = 1
    min_train_pairs_per_cluster: int = 3000
    min_valid_pairs_per_cluster: int = 300
    min_test_pairs_per_cluster: int = 300


def run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> Config:
    ap = argparse.ArgumentParser("Plain-code sequential runner for XGB fixed-origin models")

    ap.add_argument("--prices-csv", default="data/prices_all.csv")
    ap.add_argument("--future-prices-csv", default=None)
    ap.add_argument("--macro-dir", default="data")
    ap.add_argument("--cluster-csv", default=None)
    ap.add_argument("--season-dir", default=None)
    ap.add_argument("--out-root", default="results_xgb_old_features_fixed_origin")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--scripts-dir", default=None, help="Папка с 6 model scripts. По умолчанию ./scripts рядом с runner.")
    ap.add_argument("--price-col", default="Close")

    ap.add_argument("--eval-end-date", default="2026-05-08")
    ap.add_argument("--eval-years", type=int, default=1)
    ap.add_argument("--macro-known-until", default="2025-12-20")

    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--numba-threads", type=int, default=8)
    ap.add_argument("--use-gpu", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--max-optuna-rows", type=int, default=0, help="0 means no row limit")
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 means no row limit")

    ap.add_argument("--start-from", default=None, choices=[x[0] for x in SCRIPT_ORDER])
    ap.add_argument("--only", default=None, help="Например: monthly_global,monthly_per_cluster")
    ap.add_argument("--force-rerun-step", default=None, choices=[x[0] for x in SCRIPT_ORDER])

    ap.add_argument("--daily-forecast-horizon-days", type=int, default=21)
    ap.add_argument("--daily-short-horizon-days", type=int, default=5)
    ap.add_argument("--weekly-forecast-horizon-weeks", type=int, default=52)
    ap.add_argument("--weekly-short-horizon-weeks", type=int, default=13)
    ap.add_argument("--monthly-forecast-horizon-months", type=int, default=12)
    ap.add_argument("--monthly-short-horizon-months", type=int, default=3)

    ap.add_argument("--include-noise-cluster", type=int, default=1)
    ap.add_argument("--min-train-pairs-per-cluster", type=int, default=3000)
    ap.add_argument("--min-valid-pairs-per-cluster", type=int, default=300)
    ap.add_argument("--min-test-pairs-per-cluster", type=int, default=300)

    return Config(**vars(ap.parse_args()))


def filter_args_for_existing_parser(script_path: Path, args: List[str]) -> List[str]:
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    supported = set()

    for marker in ['ap.add_argument("', "ap.add_argument('"]:
        pos = 0
        while True:
            i = text.find(marker, pos)
            if i < 0:
                break
            quote = marker[-1]
            j = text.find(quote, i + len(marker))
            if j < 0:
                break
            opt = text[i + len(marker):j]
            if opt.startswith("--"):
                supported.add(opt)
            pos = j + 1

    filtered: List[str] = []
    i = 0
    while i < len(args):
        x = args[i]
        if x.startswith("--") and x not in supported:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        filtered.append(x)
        i += 1

    return filtered


def get_steps(cfg: Config) -> List[Tuple[str, str]]:
    steps = SCRIPT_ORDER[:]

    if cfg.only:
        allowed = {x.strip() for x in cfg.only.split(",") if x.strip()}
        bad = allowed - {x[0] for x in SCRIPT_ORDER}
        if bad:
            raise ValueError(f"Unknown --only steps: {sorted(bad)}")
        steps = [x for x in steps if x[0] in allowed]

    if cfg.start_from:
        names = [x[0] for x in steps]
        steps = steps[names.index(cfg.start_from):] if cfg.start_from in names else []

    return steps


def make_common_args(cfg: Config) -> List[str]:
    args = [
        "--prices-csv", cfg.prices_csv,
        "--eval-end-date", cfg.eval_end_date,
        "--macro-known-until", cfg.macro_known_until,
        "--macro-dir", cfg.macro_dir,
        "--price-col", cfg.price_col,
        "--n-trials", str(cfg.n_trials),
        "--n-jobs", str(cfg.n_jobs),
        "--numba-threads", str(cfg.numba_threads),
        "--use-gpu", str(cfg.use_gpu),
        "--gpu-id", str(cfg.gpu_id),
        "--max-optuna-rows", str(cfg.max_optuna_rows),
        "--max-train-rows", str(cfg.max_train_rows),
        "--eval-years", str(cfg.eval_years),
    ]

    if cfg.future_prices_csv:
        args += ["--future-prices-csv", cfg.future_prices_csv]
    if cfg.cluster_csv:
        args += ["--cluster-csv", cfg.cluster_csv]
    if cfg.season_dir:
        args += ["--season-dir", cfg.season_dir]

    return args


def make_step_args(step: str, cfg: Config) -> List[str]:
    args: List[str] = []

    if step.startswith("daily"):
        args += [
            "--forecast-horizon-days", str(cfg.daily_forecast_horizon_days),
            "--short-horizon-days", str(cfg.daily_short_horizon_days),
        ]
    elif step.startswith("weekly"):
        args += [
            "--forecast-horizon-weeks", str(cfg.weekly_forecast_horizon_weeks),
            "--short-horizon-weeks", str(cfg.weekly_short_horizon_weeks),
        ]
    elif step.startswith("monthly"):
        args += [
            "--forecast-horizon-months", str(cfg.monthly_forecast_horizon_months),
            "--short-horizon-months", str(cfg.monthly_short_horizon_months),
        ]

    if "per_cluster" in step:
        args += [
            "--include-noise-cluster", str(cfg.include_noise_cluster),
            "--min-train-pairs-per-cluster", str(cfg.min_train_pairs_per_cluster),
            "--min-valid-pairs-per-cluster", str(cfg.min_valid_pairs_per_cluster),
            "--min-test-pairs-per-cluster", str(cfg.min_test_pairs_per_cluster),
        ]

    return args


def find_overall_metrics_file(step: str, step_dir: Path) -> Optional[Path]:
    candidates = []

    if step.startswith("daily"):
        candidates.append(step_dir / "fixed_origin_daily_metrics_overall.json")
    elif step.startswith("weekly"):
        candidates.append(step_dir / "fixed_origin_weekly_metrics_overall.json")
    elif step.startswith("monthly"):
        candidates.append(step_dir / "fixed_origin_monthly_metrics_overall.json")

    candidates.extend(sorted(step_dir.glob("*metrics_overall.json")))

    for p in candidates:
        if p.exists():
            return p

    return None


def collect_metrics(run_dir: Path) -> None:
    rows = []

    for step, _script in SCRIPT_ORDER:
        step_dir = run_dir / step
        p = find_overall_metrics_file(step, step_dir)

        if not p:
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        row = {
            "step": step,
            "dir": str(step_dir),
            "metrics_file": str(p),
        }
        row.update(data)
        rows.append(row)

    if not rows:
        log("[METRICS] no overall metrics found yet")
        return

    out = run_dir / "fixed_origin_metrics_all_models.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"[METRICS] saved: {out}")


def main() -> None:
    cfg = parse_args()

    if cfg.run_dir:
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"[RESUME] run_dir: {run_dir.resolve()}")
    else:
        run_dir = Path(cfg.out_root) / run_id()
        run_dir.mkdir(parents=True, exist_ok=True)

    status_dir = run_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(cfg.scripts_dir) if cfg.scripts_dir else Path(__file__).resolve().parent / "scripts"
    if not scripts_dir.exists():
        raise FileNotFoundError(
            f"scripts_dir not found: {scripts_dir}. "
            "Use --scripts-dir or put 6 scripts next to this runner in ./scripts"
        )

    save_json(asdict(cfg), run_dir / "single_runner_config.json")

    log("=" * 100)
    log("[RUN] PLAIN-CODE XGB FIXED-ORIGIN OLD FEATURES")
    log(f"[RUN_DIR] {run_dir.resolve()}")
    log(f"[SCRIPTS_DIR] {scripts_dir.resolve()}")
    log("[STEPS] " + ", ".join([x[0] for x in SCRIPT_ORDER]))
    log(
        f"[ROW LIMITS] max_optuna_rows={cfg.max_optuna_rows}, "
        f"max_train_rows={cfg.max_train_rows}; 0 means full data"
    )
    log("=" * 100)

    common_args = make_common_args(cfg)

    for step, script_name in get_steps(cfg):
        script_path = scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(script_path)

        step_dir = run_dir / step
        step_dir.mkdir(parents=True, exist_ok=True)

        done_marker = status_dir / f"{step}.done"
        failed_marker = status_dir / f"{step}.failed"

        if cfg.force_rerun_step == step:
            done_marker.unlink(missing_ok=True)
            failed_marker.unlink(missing_ok=True)

        if done_marker.exists():
            log(f"[SKIP] {step} already done")
            continue

        failed_marker.unlink(missing_ok=True)

        args = common_args + ["--out-root", str(step_dir)] + make_step_args(step, cfg)
        args += ["--resume-run-dir", str(step_dir)]
        args = filter_args_for_existing_parser(script_path, args)

        cmd = [sys.executable, str(script_path)] + args

        log("=" * 100)
        log(f"[RUN STEP] {step}")
        log(f"[SCRIPT] {script_path}")
        log(f"[OUT] {step_dir}")
        log("[CMD] " + " ".join(cmd))
        log("=" * 100)

        started = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - started

        if proc.returncode != 0:
            failed_marker.write_text(
                json.dumps(
                    {
                        "step": step,
                        "script": str(script_path),
                        "returncode": proc.returncode,
                        "elapsed_sec": elapsed,
                        "cmd": cmd,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            log(f"[FAILED] step={step}, returncode={proc.returncode}. Resume with: --run-dir {run_dir}")
            sys.exit(proc.returncode)

        done_marker.write_text(
            json.dumps(
                {
                    "step": step,
                    "script": str(script_path),
                    "elapsed_sec": elapsed,
                    "cmd": cmd,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        log(f"[DONE] step={step}, elapsed={elapsed / 60:.2f} min")
        collect_metrics(run_dir)

    collect_metrics(run_dir)

    log("=" * 100)
    log("[ALL DONE]")
    log(f"Run dir: {run_dir.resolve()}")
    log(f"Summary metrics: {(run_dir / 'fixed_origin_metrics_all_models.csv').resolve()}")
    log("=" * 100)


if __name__ == "__main__":
    main()
