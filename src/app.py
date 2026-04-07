import gradio as gr
import requests
import os
import prometheus_client 
from time import perf_counter

# Fancy styling
fancy_css = """
#main-container {
    background-color: #4CAF50; 
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""
print("starting prometheous")
prometheus_client.start_http_server(9090)
endpoint_url= "http://app_backend:22076"

FRONTEND_CHAT_REQUESTS_TOTAL = prometheus_client.Counter(
    'frontend_chat_requests_total',
    'Total number of chat generation requests made by the frontend.'
)
FRONTEND_CHAT_REQUEST_ERRORS_TOTAL = prometheus_client.Counter(
    'frontend_chat_request_errors_total',
    'Total number of frontend chat requests that failed.'
)
FRONTEND_CHAT_REQUEST_DURATION_SECONDS = prometheus_client.Histogram(
    'frontend_chat_request_duration_seconds',
    'Total time spent processing frontend chate requests.'
)

def get_response(message, history, max_tokens, temperature, top_p, use_local_model = False):
    FRONTEND_CHAT_REQUESTS_TOTAL.inc()
    
    started  = perf_counter()

    payload = {
        "message": message,
        "history": history,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "use_local_model": use_local_model,
    }
    try: 
        response = requests.post(f"{endpoint_url}/chat", json=payload)
        print(response.json())
        yield response.json()["response"]
    except Exception as e:
        FRONTEND_CHAT_REQUEST_ERRORS_TOTAL.inc()
        yield f'<p>Backend request failed: {e}</p>'
    finally:
        FRONTEND_CHAT_REQUEST_DURATION_SECONDS.observe(perf_counter() - started)




chatbot = gr.ChatInterface(
    fn=get_response,
    additional_inputs=[
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
   
    
)

with gr.Blocks(css=fancy_css, theme = gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center; color:#4CAF50'>🌟 Learning Assistant 🌟</h1>")
    chatbot.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=22075)
