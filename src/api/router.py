
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
import numpy as np
import pandas as pd

from src.models.model import ForwardRequestJson, PredictionResult, HistoryReq, StatsResponse

from mlcore.store_sqlite import SQLiteStore
from mlcore.infer import predict_one
from fastapi.security import APIKeyHeader

router = APIRouter()
store = SQLiteStore("storage/app.db")

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


@router.get("/history")
async def get_history():
    history = 0 #bd.../..................................................................
    rows =[] # ...
    items = []
    for row in rows:
        items.append(HistoryReq(
            id = row["id"],
            request_id = row["request_id"],
            method =  row["method"],
            in_data = json.loads(row["input_data"]),
            out_data = json.loads(row["output_data"]),
            status_code = row["status_code"],
            response_time_ms = row["response_time_ms"]
            ))

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
@router.delete("/history")
async def delete_history(
    admin_auth: bool = Depends(api_key_header),
    older_than_days: Optional[int] = None
):
    pass

@router.get("/stats", response_model=StatsResponse)
async def get_statistics():
    query = 0 #db//////////////////////////////////////////
    time_all = []#////
    return StatsResponse(
        processing_time_stats=calculate_quantiles(time_all),    
    )
    
@router.get("/metadata")
async def get_meta():
    return #predictor.metadata()

    
# @router.post("/forward_batch")
# def forward_batch(file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)
#         preds = predictor.predict_df(df)
#         out = io.StringIO()
#         w = csv.writer(out)
#         w.writerow(["pred"]) # одна колонка
#         for v in preds:
#             w.writerow([float(v)])
#         out.seek(0)
#         return StreamingResponse(out, media_type="text/csv",
#             headers={"Content-Disposition": "attachment; filename=preds.csv"})
#     except Exception as e:
#         raise HTTPException(403, f"модель не смогла обработать данные: {type(e).__name__}")

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

# @router.put("/add_data")
# def add_data(file: UploadFile = File(...), store: Store = Depends(get_store)):
#     df = pd.read_csv(file.file)
#     n = store.append_data(df)
#     return {"rows_added": n}

# @router.put("/retrain")
# def retrain(store: Store = Depends(get_store)):
#     from pipelines.train_model import run_train
#     exp = run_train(store)
#     return {"experiment_id": exp["id"], "artifacts": exp["artifacts"]}

# @router.get("/metrics/{experiment_id}")
# def get_metrics(experiment_id: int, store: Store = Depends(get_store)):
#     return store.get_experiment_metrics(experiment_id)

# @router.post("/deploy/{experiment_id}")
# def deploy(experiment_id: int, store: Store = Depends(get_store)):
#     path = store.get_experiment_artifacts(experiment_id)
#     predictor.load_from(path)
#     return {"status": "deployed", "experiment_id": experiment_id}
