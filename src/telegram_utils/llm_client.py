import json
from parser import SESSION


def call_gemini_advice(api_key, model_url, prompt_template, question_text, history_text, context_text):
    if not api_key or not model_url:
        return None
    try:
        prompt = prompt_template.format(
            history=history_text,
            context=context_text,
            question=question_text
        )
    except KeyError:
        prompt = f"{prompt_template}\nHistory:\n{history_text}\nContext:\n{context_text}\nQuestion:\n{question_text}"

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    
    try:
        r = SESSION.post(model_url, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None