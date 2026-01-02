FROM python:3.11-slim  

WORKDIR /app  

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY storage /app/storage
COPY src /app/src
COPY mlcore/ /app/mlcore/


EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]  
# "По умолчанию запускай вот эту команду"