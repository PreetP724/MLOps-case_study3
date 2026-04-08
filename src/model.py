from huggingface_hub import InferenceClient
from typing import Optional
import prometheus_client


pipe = None
stop_inference = False
tokenizer = None

def respond(
    message,
    history,
    max_tokens,
    temperature,
    top_p,
    hf_token: Optional[str] = None,
    use_local_model: bool = False
    ):
    
    global pipe, tokenizer

    # Build messages from history

    few_shot_examples = [
    {'role': 'user', 'content': 'I do not understand my homework.'},
    {'role': 'assistant', 'content': "That's okay! This is all part of the learning process. Let's take a look and figure it out together!"},
     {'role': 'user', 'content': 'What does the word responsible mean?'},
    {'role': 'assistant', 'content': "That's an amazing question and certainly a tough word! Responsible means doing things that you have to or should do. Let's see what it looks like in a sentence: you are responsible for doing your homework. This makes sense because your homework is something you have to do." },
     {'role': 'user', 'content': "What is 2 + 3?"},
    {'role': 'assistant', 'content': "Let's use our fingers to count this out! First, start with 2 by putting up two fingers. Now, put up three more fingers to add 3. Count how many figures you are holding up. The answer is 5!"}
]   
    system_message = "You are a helpful learning assistant for a children in elementary school in the age range of 6-10. You talk nicely and use simple words like you would to a kid an elementary school. Answer questions so a kid in elementary school would understand and enjoy. Take complex topics and turn them into simple ones. Be relatable to a kid in elementary school. Be encouraging."
    messages = [{"role": "system", "content": system_message}]
    messages.extend(few_shot_examples)
    
    for chat in history:
        content = chat["content"]
        if isinstance(content, list):
            content = content[0]["text"]
        messages.append({
            "role": chat["role"], 
            "content": str(content)
        })
    
    messages.append({"role": "user", "content": str(message)})

    if use_local_model:
        print("[MODE] local")
        
        if pipe is None:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

            model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True)

            pipe = pipeline("text-generation", model=model, tokenizer = tokenizer, device =-1)

        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()

    else:
        print("[MODE] api")

        if  not hf_token:
            yield "⚠️ Please log in with your Hugging Face account first."
            return

        client = InferenceClient(token=hf_token, model="openai/gpt-oss-20b")

        response = ""

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p= top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response