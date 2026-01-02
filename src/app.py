from fastapi import FastAPI
from typing import  Any, Dict
from src.api.router import router



app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    #init_database()
    print("Service started successfully")

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    




def save_request_to_history(
    request_id: str,
    endpoint: str,
    method: str,
    in_data: Dict[str, Any],
    out_data: Dict[str, Any],
    status_code = int
):
    '''
    try:
        with get_db_connection() as conn:
            #cursor = conn.cursor()
            
            cursor.execute(
            INSERT INTO request_history 
            (request_id, endpoint, method, input_data, output_data, 
             status_code, response_time_ms, content_length, token_count,
             client_ip, user_agent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            , (
                request_id,
                endpoint,
                method,
                json.dumps(input_data),
                json.dumps(output_data),
                status_code,
                response_time_ms,
                content_length,
                token_count,
                client_ip,
                user_agent,
                datetime.now().isoformat()
            ))
            
            conn.commit()
    except Exception as e:
        print(f"Error saving request to history: {e}")
'''



