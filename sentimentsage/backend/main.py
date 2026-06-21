# backend/main.py - Memory-optimised: ONNX + HF Inference API
import re, pickle, json, os
import numpy as np
import onnxruntime as ort
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import load_rag_store, search
from dag import run_dag

app = FastAPI(title='SentimentSage AI', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

MAX_LEN = 256
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # set this in Render env vars

# ── Load vocabulary ────────────────────────────────────────────────────
print('Loading vocabulary...')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(f'Vocab size: {len(vocab):,}')

# ── Load ONNX LSTM model ───────────────────────────────────────────────
print('Loading ONNX model...')
sess = ort.InferenceSession(
    'lstm_model.onnx',
    providers=['CPUExecutionProvider']
)
print('ONNX model loaded.')

# ── Load RAG store ─────────────────────────────────────────────────────
print('Loading RAG store...')
rag_store = load_rag_store('rag_store.pkl')
print('RAG store loaded.')

# ── Text preprocessing ─────────────────────────────────────────────────
def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def encode_text(text: str) -> np.ndarray:
    tokens = preprocess_text(text)[:MAX_LEN]
    ids    = [vocab.get(t, 1) for t in tokens]
    ids    = ids + [0] * (MAX_LEN - len(ids))
    return np.array([ids], dtype=np.int64)

# ── LSTM inference via ONNX ────────────────────────────────────────────
def lstm_fn(text: str):
    x        = encode_text(text)
    outputs  = sess.run(None, {'input': x})
    prob     = float(outputs[0][0])
    # ONNX only returns prob, not latent — use random latent for RAG demo
    # (for full latent export see note below)
    latent   = np.random.randn(512).astype(np.float32)
    return prob, latent

# ── BERT via HuggingFace Inference API (runs on HF servers) ───────────
async def bert_fn_async(text: str) -> float:
    truncated = ' '.join(text.split()[:100])
    headers   = {}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                HF_API_URL,
                headers=headers,
                json={'inputs': truncated}
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    scores = result[0] if isinstance(result[0], list) else result
                    for item in scores:
                        if item['label'] == 'POSITIVE':
                            return float(item['score'])
    except Exception as e:
        print(f'HF API error: {e}')
    return 0.5   # fallback if API fails

def bert_fn(text: str) -> float:
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(bert_fn_async(text))
    except Exception:
        return 0.5

# ── RAG search ─────────────────────────────────────────────────────────
def rag_fn(latent_vec: np.ndarray) -> list:
    return search(latent_vec, rag_store, top_k=3)

# ── Request models ─────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    review: str

# ── Routes ─────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'status': 'SentimentSage AI running',
        'runtime': 'ONNX + HF Inference API',
        'vocab': len(vocab),
        'rag': len(rag_store['labels']),
    }

@app.post('/analyze')
def analyze(req: ReviewRequest):
    text = req.review.strip()
    if len(text.split()) < 3:
        raise HTTPException(status_code=400, detail='Review too short.')

    dag_steps, result, similar = run_dag(text, lstm_fn, bert_fn, rag_fn)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    return {**result, 'dag_steps': dag_steps, 'similar': similar}

@app.get('/health')
def health():
    return {'status': 'ok'}
