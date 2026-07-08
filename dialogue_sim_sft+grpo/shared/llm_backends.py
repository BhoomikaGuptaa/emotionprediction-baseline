"""
shared/llm_backends.py
======================
Unified, deterministic text generation for the zero-shot (B2) and few-shot (B3)
LLM baselines. Three backends:

  hf      : local HuggingFace transformers. Use this for the LLaMA rows that
            line up with the DPM paper:
              meta-llama/Llama-3.2-1B-Instruct
              meta-llama/Llama-3.1-8B-Instruct
  ollama  : local Ollama server (e.g. qwen2.5:3b-instruct, llama3.2:1b).
  openai  : OpenAI API (e.g. gpt-4o-mini).

All backends decode greedily (deterministic) so results are reproducible.
"""
import os
import time

_HF_CACHE = {}   # model_name -> (tokenizer, model)


def _hf_load(model_name):
    if model_name in _HF_CACHE:
        return _HF_CACHE[model_name]
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
    )
    model.eval()
    _HF_CACHE[model_name] = (tok, model)
    return tok, model


def _hf_generate_batch(prompts, model_name, system=None, max_new_tokens=64):
    import torch
    tok, model = _hf_load(model_name)
    chats = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": p}]
        chats.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def _ollama_generate(prompt, model_name, system=None, max_new_tokens=64):
    import requests
    full = (system + "\n\n" + prompt) if system else prompt
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model_name, "prompt": full, "stream": False,
                            "options": {"temperature": 0.0, "num_predict": max_new_tokens}},
                      timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


def _openai_generate(prompt, model_name, system=None, max_new_tokens=64):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=model_name, messages=msgs, temperature=0.0, max_tokens=max_new_tokens)
    return resp.choices[0].message.content


def generate_batch(prompts, backend, model_name, system=None,
                   max_new_tokens=64, sleep=0.0):
    """Return a list of raw completions, one per prompt. HF runs a true batch;
    ollama/openai loop one-by-one with an optional sleep between calls."""
    if backend == "hf":
        return _hf_generate_batch(prompts, model_name, system, max_new_tokens)
    outs = []
    for p in prompts:
        if backend == "ollama":
            outs.append(_ollama_generate(p, model_name, system, max_new_tokens))
        elif backend == "openai":
            outs.append(_openai_generate(p, model_name, system, max_new_tokens))
        else:
            raise ValueError(f"Unknown backend: {backend}")
        if sleep:
            time.sleep(sleep)
    return outs
