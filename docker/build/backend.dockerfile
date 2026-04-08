FROM python:3.13.5-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn transformers accelerate bitsandbytes prometheus-client

WORKDIR /app

#COPY .model /app/.model
COPY src/api.py /app/src/
COPY src/model.py /app/src/

EXPOSE 22076

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "22076"]
