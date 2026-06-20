# SentimentSage AI — Full Deployment Guide

---

## Project Structure

```
sentimentsage/
├── colab/
│   └── SentimentSage_Train.ipynb     ← Run this FIRST on Google Colab
├── backend/
│   ├── 1_clean_data.py               ← DAG Node 1: data cleaning
│   ├── lstm_model.py                 ← LSTM architecture
│   ├── dag.py                        ← 8-node DAG pipeline
│   ├── rag.py                        ← Cosine similarity RAG
│   ├── main.py                       ← FastAPI server
│   ├── requirements.txt
│   ├── lstm_model.pt                 ← Downloaded from Colab
│   ├── vocab.pkl                     ← Downloaded from Colab
│   └── rag_store.pkl                 ← Downloaded from Colab
├── frontend/
│   ├── src/App.jsx                   ← Full React UI
│   ├── .env
│   └── .env.production
└── server/
    ├── index.js                      ← Express + MongoDB
    └── package.json
```

---

## Step 1 — Google Colab Training (~20 min)

1. Open `colab/SentimentSage_Train.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Upload your `kaggle.json` when prompted
4. Run all cells top to bottom
5. Final cell downloads `sentimentsage_models.zip`
6. Unzip and place these files in `backend/`:
   - `lstm_model.pt`
   - `vocab.pkl`
   - `rag_store.pkl`
   - `training_history.json`
   - `eval_results.json`

**Expected results:** ~87-89% test accuracy in ~15-20 minutes.

---

## Step 2 — Local Setup & Test

### Python backend

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt

# Optional: run data cleaning locally too
python 1_clean_data.py

# Start FastAPI
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs to test the `/analyze` endpoint.

**Note:** First startup downloads the BERT model (~250MB). Subsequent startups are fast.

### Express + MongoDB

```bash
cd server
npm install

# Create server/.env file:
echo "MONGO_URI=your_atlas_connection_string" > .env
echo "PORT=5000" >> .env

node index.js
```

### React frontend

```bash
cd frontend
npx create-react-app . --template cra-template
# When asked to overwrite — say YES
# Then replace src/App.js with src/App.jsx content

npm install axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Edit `tailwind.config.js`:
```js
module.exports = {
  content: ['./src/**/*.{js,jsx}'],
  theme: { extend: {} },
  plugins: [],
}
```

Edit `src/index.css` (top of file):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

```bash
npm start
```

Open http://localhost:3000 — paste a movie review and click Analyze!

---

## Step 3 — MongoDB Atlas (Free)

1. Go to https://cloud.mongodb.com
2. Create a free M0 cluster (any region)
3. Database Access → Add user with username + password
4. Network Access → Add IP → Allow from anywhere (0.0.0.0/0)
5. Connect → Drivers → Copy connection string
6. Replace `<password>` with your password
7. Paste into `server/.env` as `MONGO_URI=...`

---

## Step 4 — Deploy FastAPI to Render

1. Push your entire project to GitHub
   - **Include** `lstm_model.pt`, `vocab.pkl`, `rag_store.pkl` in `backend/`
   - These are small enough for GitHub (~30MB total)

2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Root Directory:** `backend`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. Deploy → wait ~5 min for first build (BERT downloads on first startup)

Copy the URL: `https://sentimentsage-api.onrender.com`

---

## Step 5 — Deploy Express to Render

1. Render → New → Web Service → Same repo
2. Settings:
   - **Root Directory:** `server`
   - **Environment:** Node
   - **Build Command:** `npm install`
   - **Start Command:** `node index.js`
   - **Plan:** Free
3. Environment Variables → Add:
   - `MONGO_URI` = your Atlas connection string
   - `PORT` = 5000
4. Deploy

Copy the URL: `https://sentimentsage-express.onrender.com`

---

## Step 6 — Deploy React to Vercel

1. Update `frontend/.env.production` with your Render URLs:
```
REACT_APP_FASTAPI=https://sentimentsage-api.onrender.com
REACT_APP_EXPRESS=https://sentimentsage-express.onrender.com
```

2. Push to GitHub

3. Go to https://vercel.com → New Project → Import your repo
4. Settings:
   - **Root Directory:** `frontend`
   - **Framework:** Create React App
5. Environment Variables → Add both `REACT_APP_FASTAPI` and `REACT_APP_EXPRESS`
6. Deploy

Your live app will be at: `https://sentimentsage.vercel.app`

---

## What to Say to Your Teacher (Viva Script)

### Opening
*"SentimentSage AI is a movie review sentiment analyzer that processes any review through an 8-node DAG pipeline. What makes it different from a simple classifier is that we combined two models — a BiLSTM we trained from scratch on 50,000 IMDB reviews, and a pre-trained BERT Transformer — and fused their outputs. We also added a RAG system that retrieves the 3 most similar reviews from our dataset using cosine similarity on the latent vectors. Let me walk you through the full pipeline."*

### Week-by-week explanation

| Week | Question teacher might ask | Your answer |
|---|---|---|
| **Week 2** | What loss function did you use? | Binary Cross-Entropy — standard for binary classification. Penalizes confident wrong predictions more. |
| **Week 3** | How does backpropagation work here? | Gradients flow backward through the LSTM gates via BPTT — Backpropagation Through Time. |
| **Week 4** | What activation functions did you use? | ReLU in the hidden FC layer to avoid vanishing gradients. Sigmoid in the output for probability 0-1. |
| **Week 7-9** | Explain the LSTM gates | Forget gate decides what cell state to erase. Input gate decides what new info to write. Output gate controls what the next token sees. |
| **Week 10** | Where is the latent space? | The BiLSTM's final hidden state — a 512-dimensional vector — is the latent representation of the entire review. Exactly like an autoencoder's bottleneck. |
| **Week 11** | What NLP preprocessing did you do? | Lowercased text, stripped HTML artifacts, removed special characters, tokenized on whitespace, mapped tokens to vocabulary IDs. |
| **Week 12** | How does the RAG similarity work? | We compute cosine similarity between the query review's 512-dim vector and 1,000 reference vectors. Cosine similarity measures the angle — not magnitude — so semantically similar reviews point in the same direction. |
| **Week 13** | What is BERT doing? | BERT applies self-attention across all tokens simultaneously. It uses positional encoding and learns relationships between distant tokens — which pure LSTMs struggle with on long reviews. |

### DAG explanation
*"Directed Acyclic Graph — each of the 8 nodes does exactly one job and passes its output to the next. Directed means data flows forward only. Acyclic means no loops. Node 5 is especially interesting — it extracts the 512-dim latent vector which then feeds BOTH Node 7 (fused score) AND Node 8 (RAG retrieval) — that's a DAG branch, not a cycle."*

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `lstm_model.pt not found` | Run the Colab notebook first, download the zip, extract to `backend/` |
| BERT download stuck | First run downloads ~250MB. Wait 3-5 min. Check internet. |
| CORS error | CORSMiddleware is already in main.py. Restart uvicorn. |
| MongoDB not saving | Check Atlas Network Access allows 0.0.0.0/0. Restart Express. |
| Render cold start slow | Free tier sleeps after inactivity. Open Render URL 5 min before demo. |
| Low accuracy | Ensure you ran all 10 epochs. Expected: 87-89% on test set. |
| `vocab.pkl not found` | Same as model — needs to be downloaded from Colab zip. |
