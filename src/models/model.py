
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, validator

# Модель для входных данных
class ForwardRequestJson(BaseModel):
    name : str = Field(..., description="Название компании")
    date : str = Field(..., description="Месяц предсказания YYYY-MM")

    @validator('date')
    def validate_date_format(cls, v):
        try:
            # Проверяем, что строка соответствует формату YYYY-MM
            datetime.strptime(v, "%Y-%m")
            return v
        except ValueError:
            raise ValueError('Дата должна быть в формате: YYYY-MM')


class PredictionResult(BaseModel):
    ticker: str
    target_month: str
    y_pred: float



class HistoryReq(BaseModel):
    id : int
    request_id : str
    method: str
    in_data: Dict[str, Any]
    out_data: Dict[str, Any]
    status_code : int
    response_time_ms : float

class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryReq]
    
class StatsResponse(BaseModel):
    processing_time_stats: Dict[str, float]
    
    

class RetrainOut(BaseModel):
    experiment_id: int
    run_id: Optional[str]
    model_path: str
    overall: Dict[str, float]
