from fastapi import FastAPI
from typing import  Any, Dict
from src.api.router import router

app = FastAPI()
app.include_router(router)

