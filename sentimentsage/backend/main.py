# backend/main.py
# SentimentSage AI — FastAPI Server
# Run: uvicorn main:app --reload --port 8000

import re, pickle, json
import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

from lstm_model import SentimentLSTM
from rag import load_rag_store, search
from dag import run_dag

# ── App setup ──────────────────────────────────────────────────────────
app = FastAPI(title='SentimentSage AI', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 256

# ── Load vocabulary ────────────────────────────────────────────────────
print('Loading vocabulary...')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(f'Vocab size: {len(vocab):,}')

# ── Load LSTM model ────────────────────────────────────────────────────
print('Loading LSTM model...')
lstm = SentimentLSTM(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=256,
    n_layers=2,
    dropout=0.3,
)
lstm.load_state_dict(torch.load('lstm_model.pt', map_location=DEVICE))
lstm.eval().to(DEVICE)
print('LSTM loaded.')

# ── Load BERT via HuggingFace ──────────────────────────────────────────
# Week 13: Transformers — self-attention, word embeddings, positional encoding
# We use distilbert fine-tuned on SST-2 (movie reviews) — no training needed
# Lightweight BERT alternative — fits in 512MB
from transformers import pipeline
print('Loading lightweight model...')
bert_pipeline = pipeline(
    'text-classification',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=-1,
    truncation=True,
    max_length=128,
    model_kwargs={'low_cpu_mem_usage': True}
)
print('Model loaded.')
print('BERT loaded.')

# ── Load RAG store ─────────────────────────────────────────────────────
print('Loading RAG store...')
rag_store = load_rag_store('rag_store.pkl')
print('RAG store loaded.')

# ── Helper: Text preprocessing ────────────────────────────────────────
def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def encode_text(text: str) -> torch.Tensor:
    tokens = preprocess_text(text)[:MAX_LEN]
    ids    = [vocab.get(t, 1) for t in tokens]
    ids    = ids + [0] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)

# ── LSTM inference function ────────────────────────────────────────────
def lstm_fn(text: str):
    x = encode_text(text)
    with torch.no_grad():
        prob, latent = lstm(x, return_hidden=True)
    return float(prob.item()), latent.cpu().numpy()[0]

# ── BERT inference function ────────────────────────────────────────────
def bert_fn(text: str) -> float:
    # Truncate to 512 tokens for BERT
    truncated = ' '.join(text.split()[:400])
    result    = bert_pipeline(truncated)[0]
    score     = result['score']
    # HuggingFace returns label POSITIVE/NEGATIVE
    if result['label'] == 'NEGATIVE':
        score = 1.0 - score
    return float(score)

# ── RAG search function ────────────────────────────────────────────────
def rag_fn(latent_vec: np.ndarray) -> list:
    return search(latent_vec, rag_store, top_k=3)

# ── Request / Response models ──────────────────────────────────────────
class ReviewRequest(BaseModel):
    review: str

class AnalyzeResponse(BaseModel):
    sentiment:       str
    confidence:      float
    lstm_sentiment:  str
    lstm_confidence: float
    bert_sentiment:  str
    bert_confidence: float
    models_agree:    bool
    word_count:      int
    dag_steps:       list
    similar:         list
    total_ms:        int

# ── Routes ─────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'status':  'SentimentSage AI running',
        'device':  DEVICE,
        'vocab':   len(vocab),
        'rag':     len(rag_store['labels']),
    }

@app.post('/analyze', response_model=AnalyzeResponse)
def analyze(req: ReviewRequest):
    text = req.review.strip()
    if len(text.split()) < 3:
        raise HTTPException(status_code=400, detail='Review must be at least 3 words.')

    dag_steps, result, similar = run_dag(text, lstm_fn, bert_fn, rag_fn)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    return {
        'sentiment':       result['sentiment'],
        'confidence':      result['confidence'],
        'lstm_sentiment':  result['lstm_sentiment'],
        'lstm_confidence': result['lstm_confidence'],
        'bert_sentiment':  result['bert_sentiment'],
        'bert_confidence': result['bert_confidence'],
        'models_agree':    result['models_agree'],
        'word_count':      result['word_count'],
        'dag_steps':       dag_steps,
        'similar':         similar,
        'total_ms':        result['total_ms'],
    }

@app.get('/health')
def health():
    return {'status': 'ok', 'device': DEVICE}
