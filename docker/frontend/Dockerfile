FROM python:3.11-slim  

WORKDIR /app  

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirementsFront.txt /app/requirementsFront.txt
RUN pip install --no-cache-dir -r requirementsFront.txt

COPY frontend/ /app/frontend/
EXPOSE 8501

CMD ["streamlit", "run", "frontend/front3.py", "--server.port=8501", "--server.address=0.0.0.0"]