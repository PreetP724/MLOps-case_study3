FROM python:3.13.5-slim

RUN pip install --no-cache-dir gradio requests prometheus-client

WORKDIR /app

COPY src/app.py /app/src/

EXPOSE 22075
EXPOSE 9090

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=22075


CMD ["python", "src/app.py"]
