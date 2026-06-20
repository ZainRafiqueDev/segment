// server/index.js
// SentimentSage — Express + MongoDB history server
// Run: node index.js

require('dotenv').config();
const express    = require('express');
const mongoose   = require('mongoose');
const cors       = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// ── MongoDB connection ─────────────────────────────────────────────────
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB error:', err));

// ── Schema ─────────────────────────────────────────────────────────────
const AnalysisSchema = new mongoose.Schema({
  review:          { type: String, required: true },
  review_snippet:  String,
  sentiment:       String,
  confidence:      Number,
  lstm_sentiment:  String,
  lstm_confidence: Number,
  bert_sentiment:  String,
  bert_confidence: Number,
  models_agree:    Boolean,
  word_count:      Number,
  total_ms:        Number,
  createdAt:       { type: Date, default: Date.now },
});

const Analysis = mongoose.model('Analysis', AnalysisSchema);

// ── Routes ─────────────────────────────────────────────────────────────
// Save a new analysis
app.post('/api/analyses', async (req, res) => {
  try {
    const doc = new Analysis({
      ...req.body,
      review_snippet: req.body.review
        ? req.body.review.substring(0, 120) + '...'
        : '',
    });
    await doc.save();
    res.json(doc);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get recent analyses
app.get('/api/analyses', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const docs  = await Analysis.find()
      .sort({ createdAt: -1 })
      .limit(limit)
      .select('-review');   // don't return full review text in list
    res.json(docs);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Stats endpoint
app.get('/api/stats', async (req, res) => {
  try {
    const total     = await Analysis.countDocuments();
    const positive  = await Analysis.countDocuments({ sentiment: 'positive' });
    const negative  = await Analysis.countDocuments({ sentiment: 'negative' });
    const agree     = await Analysis.countDocuments({ models_agree: true });
    const avgConf   = await Analysis.aggregate([
      { $group: { _id: null, avg: { $avg: '$confidence' } } }
    ]);
    const avgMs     = await Analysis.aggregate([
      { $group: { _id: null, avg: { $avg: '$total_ms' } } }
    ]);
    res.json({
      total,
      positive,
      negative,
      models_agree_pct: total > 0 ? Math.round(agree / total * 100) : 0,
      avg_confidence:   avgConf[0] ? Math.round(avgConf[0].avg * 10) / 10 : 0,
      avg_ms:           avgMs[0]   ? Math.round(avgMs[0].avg) : 0,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Express server running on :${PORT}`));
