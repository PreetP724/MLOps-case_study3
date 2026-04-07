from __future__ import annotations
from typing import Any, Optional
from src import model
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import os
import prometheus_client 
from time import perf_counter

HEALTH_REQUESTS_TOTAL = prometheus_client.Counter(
    'backend_health_requests_total',
    'Total number of /health requests received by the backend.'
)
CHAT_REQUESTS_TOTAL = prometheus_client.Counter(
    'backend_chat_requests_total',
    'Total number of /chat requests received by the backend.'
)

CHAT_REQUEST_ERRORS_TOTAL = prometheus_client.Counter(
    'backend_chat_request_errors_total',
    'Total number of failed /chat requests handled by the backend.'
)

CHAT_REQUEST_DURATION_SECONDS = prometheus_client.Histogram(
    'backend_chat_request_duration_seconds',
    'Time spent generating chat responses in the backend.'
)

app = FastAPI()

conversations: dict[str, list] = {}



class ChatInput(BaseModel):
    message: str
    history: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    top_p: float
    use_local_model: bool


@app.get('/health')
def health() -> dict[str, str]:
    HEALTH_REQUESTS_TOTAL.inc()
    return {'status': 'ok'}


@app.post("/chat")
def chat(input: ChatInput):
    hf_token = os.getenv("HF_TOKEN")
    CHAT_REQUESTS_TOTAL.inc()
    started = perf_counter()
    try:
        response = ""
        for chunk in model.respond(input.message, input.history, input.max_tokens, input.temperature, input.top_p, hf_token, input.use_local_model):
            response = chunk

        return {"response": response}
    except Exception as e:
        CHAT_REQUEST_ERRORS_TOTAL.inc()
        raise
    finally:
        CHAT_REQUEST_DURATION_SECONDS.observe(perf_counter() - started)

@app.get('/metrics')
def metrics() -> Response:
    return Response(
        prometheus_client.generate_latest(),
        media_type=prometheus_client.CONTENT_TYPE_LATEST
    )







    