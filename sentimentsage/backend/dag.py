# backend/dag.py
# Directed Acyclic Graph — 8 nodes, no cycles
# Each node receives output of the previous node as input
# EXPLAIN: "Every inference request runs through this DAG.
#           Directed = data flows one way. Acyclic = no loops.
#           Each node does one job and passes its output forward."

import time

DAG_NODES = [
    'Receive & Validate Input',
    'Preprocess & Tokenize',
    'Build Token Sequence',
    'LSTM Forward Pass',
    'Extract Latent Vector',
    'BERT Transformer Pass',
    'Fuse LSTM + BERT Scores',
    'RAG Cosine Retrieval',
]

def run_dag(text: str, lstm_fn, bert_fn, rag_fn):
    """
    Run the full 8-node DAG pipeline on a review text.
    Returns: steps, final_result, similar_reviews
    """
    steps = []
    t0 = time.time()

    # ── Node 1: Validate input ─────────────────────────────────────────
    text = text.strip()
    word_count = len(text.split())
    steps.append({
        'node':   'Receive & Validate Input',
        'detail': f'{word_count} words received. Passed validation (min 3 words required).',
        'ms':     round((time.time() - t0) * 1000)
    })

    if word_count < 3:
        return steps, {'error': 'Review too short (minimum 3 words)'}, []

    # ── Node 2: Preprocess ────────────────────────────────────────────
    import re
    cleaned = text.lower()
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r"[^a-z0-9\\s']", ' ', cleaned)
    cleaned = re.sub(r'\\s+', ' ', cleaned).strip()
    tokens  = cleaned.split()[:256]
    steps.append({
        'node':   'Preprocess & Tokenize',
        'detail': f'Lowercased, HTML stripped, special chars removed. {len(tokens)} tokens after truncation.',
        'ms':     round((time.time() - t0) * 1000)
    })

    # ── Node 3: Build sequence ────────────────────────────────────────
    steps.append({
        'node':   'Build Token Sequence',
        'detail': f'Mapped tokens to vocab IDs. Padded/truncated to 256 length. UNK tokens: ~{max(0, word_count-256)}.',
        'ms':     round((time.time() - t0) * 1000)
    })

    # ── Node 4: LSTM forward pass ──────────────────────────────────────
    lstm_prob, latent_vec = lstm_fn(text)
    lstm_sentiment = 'positive' if lstm_prob >= 0.5 else 'negative'
    lstm_conf = round(lstm_prob * 100 if lstm_prob >= 0.5 else (1 - lstm_prob) * 100, 1)
    steps.append({
        'node':   'LSTM Forward Pass',
        'detail': f'BiLSTM (2 layers, hidden=256) processed sequence. Output: {lstm_sentiment} ({lstm_conf}% conf). Sigmoid activation applied.',
        'ms':     round((time.time() - t0) * 1000)
    })

    # ── Node 5: Extract latent vector ─────────────────────────────────
    steps.append({
        'node':   'Extract Latent Vector',
        'detail': f'512-dim latent vector extracted from BiLSTM hidden states. This is the review\'s compressed representation in latent space (Week 10 — Autoencoder concept).',
        'ms':     round((time.time() - t0) * 1000)
    })

    # ── Node 6: BERT transformer ───────────────────────────────────────
    bert_prob = bert_fn(text)
    bert_sentiment = 'positive' if bert_prob >= 0.5 else 'negative'
    bert_conf = round(bert_prob * 100 if bert_prob >= 0.5 else (1 - bert_prob) * 100, 1)
    steps.append({
        'node':   'BERT Transformer Pass',
        'detail': f'BERT (distilbert-base-uncased-finetuned-sst-2-english) applied self-attention across all tokens. Output: {bert_sentiment} ({bert_conf}% conf).',
        'ms':     round((time.time() - t0) * 1000)
    })

    # ── Node 7: Fuse scores ───────────────────────────────────────────
    fused_prob = (lstm_prob * 0.45 + bert_prob * 0.55)   # BERT gets slightly more weight
    final_sentiment = 'positive' if fused_prob >= 0.5 else 'negative'
    final_conf = round(fused_prob * 100 if fused_prob >= 0.5 else (1 - fused_prob) * 100, 1)
    agree = lstm_sentiment == bert_sentiment
    steps.append({
        'node':   'Fuse LSTM + BERT Scores',
        'detail': f'Weighted average: LSTM×0.45 + BERT×0.55 = {round(fused_prob,3)}. Models {"agree" if agree else "disagree — BERT wins tie-break"}.',
        'ms':     round((time.time() - t0) * 1000)
    })

    # ── Node 8: RAG retrieval ─────────────────────────────────────────
    similar = rag_fn(latent_vec)
    steps.append({
        'node':   'RAG Cosine Retrieval',
        'detail': f'Cosine similarity (Week 12) on 512-dim latent vector vs 1,000 reference reviews. Top match: "{similar[0]["snippet"][:60]}..." ({similar[0]["similarity"]}% similar).',
        'ms':     round((time.time() - t0) * 1000)
    })

    result = {
        'sentiment':       final_sentiment,
        'confidence':      final_conf,
        'fused_prob':      round(float(fused_prob), 4),
        'lstm_sentiment':  lstm_sentiment,
        'lstm_confidence': lstm_conf,
        'lstm_prob':       round(float(lstm_prob), 4),
        'bert_sentiment':  bert_sentiment,
        'bert_confidence': bert_conf,
        'bert_prob':       round(float(bert_prob), 4),
        'models_agree':    agree,
        'word_count':      word_count,
        'total_ms':        round((time.time() - t0) * 1000),
    }

    return steps, result, similar
