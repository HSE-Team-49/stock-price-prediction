
from datetime import datetime, timedelta, timezone
import json
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
import numpy as np
import pandas as pd

from mlcore.features import filter_tickers_starting_at_global_min, get_feature_columns, load_prices_csv, make_monthly_panel, pick_price_col
from src.models.model import ForwardRequestJson, PredictionResult, HistoryReq, RetrainOut, StatsResponse

from mlcore.train import RetrainOutputs
from mlcore.store_sqlite import SQLiteStore
from mlcore.infer import predict_one, predict_batch
from mlcore.train import retrain_and_log
from fastapi.security import APIKeyHeader

router = APIRouter()
store = SQLiteStore("storage/app.db")
add_date = 0

def calculate_quantiles(data: List[float]) -> Dict[str, float]:
    if not data:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    
    arr = np.array(data)
    
    return {
        "mean": float(np.mean(arr)), 
        "p50": float(np.percentile(arr, 50)), 
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }

@router.post("/forward", response_model=PredictionResult)
async def forward(
    request: Request,
    payload: Optional[ForwardRequestJson] = None
):
        
    if request.headers.get("content-type", "").startswith("application/json"):
        if payload is None:
            raise HTTPException(400, "bad request")
        try:
            res = predict_one(store, payload.name, payload.date)
            
            return PredictionResult(ticker=res['ticker'],target_month=res['target_month'], y_pred=res['y_pred'])
        except Exception as e:
            raise HTTPException(403, f"Модель не смогла обработать данные: {e}") 
    raise HTTPException(400, "bad request")

@router.post("/forward_batch", response_model=List[PredictionResult])
def forward_batch(
    request: Request,
    payload: Optional[List[ForwardRequestJson]] = None):
    
    if request.headers.get("content-type", "").startswith("application/json"):
        if payload is None:
            raise HTTPException(400, "bad request")
        try:
            
            mas = []
            for i, x in enumerate(payload):
                try:
                    mas.append((x.name, x.date))
                except AttributeError as e:
                    print(f"3.{i} ОШИБКА: {e}")
                    print(f"3.{i} Тип x: {type(x)}")
                    print(f"3.{i} Содержимое x: {x}")
                    raise
            
            preds = predict_batch(store, mas)
            results = []
            
            for res in preds:
                prediction_result = PredictionResult(ticker=res['ticker'],target_month=res['target_month'],y_pred=res['y_pred'])
                results.append(prediction_result)
            return results
        except Exception as e:
            raise HTTPException(403, f"модель не смогла обработать данные: {type(e).__name__}")

@router.put("/add_data")
def add_data(file: UploadFile = File(...)):
    
    if not file.filename.lower().endswith('.csv'): # type: ignore
            raise HTTPException(400, "Файл должен быть в формате CSV")
    contents = file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))  # type: ignore
    
    if df.empty:
        raise HTTPException(400, "CSV файл пустой")
    
    df = pd.read_csv(file.file)
    ref_df = pd.read_csv("data/prices_all.csv")
    ref_columns = set(ref_df.columns)
    new_columns = set(df.columns)
    if ref_columns != new_columns:
            # Ищем различия
        missing_in_new = ref_columns - new_columns
        extra_in_new = new_columns - ref_columns
            
        error_details = []
            
        if missing_in_new:
            error_details.append(f"Отсутствуют колонки: {list(missing_in_new)}")
            
        if extra_in_new:
            error_details.append(f"Лишние колонки: {list(extra_in_new)}")
            
        error_msg = "Несоответствие столбцов. "
        if error_details:
            error_msg += " ".join(error_details)
        return {
                "success": False,
                "error": error_msg,
                "expected_columns": sorted(list(ref_columns)),
                "received_columns": sorted(list(new_columns)),
                "expected_sample": ref_df.head(3).to_dict('records') if not ref_df.empty else []
            }
    combined_df = pd.concat([ref_df, df], ignore_index=True)
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    duplicates_removed = before_dedup - len(combined_df)
    combined_df.to_csv("data/prices_all.csv", index=False)
    add_date = 1
    
    return {"rows_added"}    


@router.put("/retrain", response_model=RetrainOut)
def retrain():
    global add_date
    if add_date == 0:
        raise HTTPException(403, f"Новые данные не отправлены!!")
    
    try:
        df = load_prices_csv("data/prices_all.csv")
        price_col = pick_price_col(df)
        df = filter_tickers_starting_at_global_min(df)

        panel = make_monthly_panel(df, price_col=price_col)
        feat_cols = get_feature_columns(panel)

        store.replace_features(panel, feat_cols)
        
        res = retrain_and_log(store,"xgb",2024, 12, 25, True,False,13, os.getenv("MLFLOW_TRACKING_URI", ""),"stocks-monthly","storage/models")
        out = RetrainOut(experiment_id=res.experiment_id, run_id=res.run_id, model_path=res.model_path,
                         overall=res.overall)
        add_date = 0
        
        return out
    except Exception as e:
        raise HTTPException(403, f"Не смог переобучить: {type(e).__name__}")


@router.post("/deploy/{experiment_id}")
def deploy(experiment_id: int):
    try:
        exp = store.get_experiment(experiment_id)
        if not exp:
            raise HTTPException(404, f"Эксперимент с ID {experiment_id} не найден")
        
        model_path = exp["model_path"]
        if not model_path:
            raise HTTPException(400, f"У эксперимента {experiment_id} не указан путь к модели")
        
        ts = datetime.now(timezone.utc).isoformat()
        store.set_active_model(experiment_id, model_path, updated_at=ts)
        return {"status": "deployed", "experiment_id": experiment_id}
    except Exception as e:
        raise HTTPException(403, f"Не смог сделатб деплой: {type(e).__name__}")


@router.get("/metrics/{experiment_id}")
def get_metrics(experiment_id: int):
    try:
        res = store.get_experiment(experiment_id)["metrics"]
        return res
    except Exception as e:
        raise HTTPException(403, f"Не смог получить метрики: {type(e).__name__}")

    
# @router.get("/history")
# async def get_history():
#     history = 0 #bd.../..................................................................
#     rows =[] # ...
#     items = []
#     for row in rows:
#         items.append(HistoryReq(
#             id = row["id"],
#             request_id = row["request_id"],
#             method =  row["method"],
#             in_data = json.loads(row["input_data"]),
#             out_data = json.loads(row["output_data"]),
#             status_code = row["status_code"],
#             response_time_ms = row["response_time_ms"]
#             ))


# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# @router.delete("/history")
# async def delete_history(
#     admin_auth: bool = Depends(api_key_header),
#     older_than_days: Optional[int] = None
# ):
#     pass

# @router.get("/stats", response_model=StatsResponse)
# async def get_statistics():
#     query = 0 #db//////////////////////////////////////////
#     time_all = []#////
#     return StatsResponse(
#         processing_time_stats=calculate_quantiles(time_all),    
#     )
    
# @router.get("/metadata")
# async def get_meta():
#     return #predictor.metadata()

    


# ---- /evaluate (CSV с target_next) ----
# @router.post("/evaluate", response_model=EvaluateResponse)
# def evaluate(file: UploadFile = File(...)):    
#     df = pd.read_csv(file.file)
#     if "target_next" not in df.columns:
#         raise HTTPException(400, "CSV must contain 'target_next'")
#     y = df["target_next"].values
#     X = df.drop(columns=["target_next"])
#     p = predictor.predict_df(X)
#     rmse = float(np.sqrt(mean_squared_error(y, p)))
#     mae = float(mean_absolute_error(y, p))
#     r2 = float(r2_score(y, p))
#     return EvaluateResponse(RMSE=rmse, MAE=mae, R2=r2)






