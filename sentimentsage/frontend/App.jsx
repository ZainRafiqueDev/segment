// SentimentSage AI — Premium Frontend
// Three.js neural-network background · Framer Motion · GSAP · Glassmorphism

import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import gsap from 'gsap';
import * as THREE from 'three';

const FAPI = process.env.REACT_APP_FASTAPI || 'http://localhost:8001';
const EXPR = process.env.REACT_APP_EXPRESS || 'http://localhost:5000';

/* ─────────────────────────────────────────────────────────────────────
   THREE.JS  Neural-Network Background
───────────────────────────────────────────────────────────────────── */
function NeuralBackground({ pulse }) {
  const mountRef = useRef();
  const pulseRef = useRef(pulse);
  useEffect(() => { pulseRef.current = pulse; }, [pulse]);

  useEffect(() => {
    const el = mountRef.current;
    const W = window.innerWidth, H = window.innerHeight;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 100);
    camera.position.z = 18;

    // ── particles ──
    const N   = 70;
    const pos = new Float32Array(N * 3);
    const vel = [];
    for (let i = 0; i < N; i++) {
      pos[i*3]   = (Math.random() - 0.5) * 28;
      pos[i*3+1] = (Math.random() - 0.5) * 20;
      pos[i*3+2] = (Math.random() - 0.5) * 8;
      vel.push([(Math.random()-0.5)*0.013, (Math.random()-0.5)*0.013, (Math.random()-0.5)*0.005]);
    }
    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const ptMat = new THREE.PointsMaterial({ color: 0x6366f1, size: 0.14, transparent: true, opacity: 0.8 });
    scene.add(new THREE.Points(ptGeo, ptMat));

    // ── connection lines ──
    const lnGeo = new THREE.BufferGeometry();
    const lnMat = new THREE.LineBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.13 });
    scene.add(new THREE.LineSegments(lnGeo, lnMat));

    const THRESH_SQ = 4.5 * 4.5;
    let raf;

    const tick = () => {
      raf = requestAnimationFrame(tick);
      const speed = pulseRef.current ? 3 : 1;

      for (let i = 0; i < N; i++) {
        pos[i*3]   += vel[i][0] * speed;
        pos[i*3+1] += vel[i][1] * speed;
        pos[i*3+2] += vel[i][2] * speed;
        if (Math.abs(pos[i*3])   > 14) vel[i][0] *= -1;
        if (Math.abs(pos[i*3+1]) > 10) vel[i][1] *= -1;
        if (Math.abs(pos[i*3+2]) > 4)  vel[i][2] *= -1;
      }
      ptGeo.attributes.position.needsUpdate = true;

      const lp = [];
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = pos[i*3]-pos[j*3], dy = pos[i*3+1]-pos[j*3+1], dz = pos[i*3+2]-pos[j*3+2];
          if (dx*dx + dy*dy + dz*dz < THRESH_SQ)
            lp.push(pos[i*3], pos[i*3+1], pos[i*3+2], pos[j*3], pos[j*3+1], pos[j*3+2]);
        }
      }
      lnGeo.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));

      const t = Date.now() * 0.001;
      camera.position.x = Math.sin(t * 0.18) * 0.7;
      camera.position.y = Math.cos(t * 0.13) * 0.4;
      renderer.render(scene, camera);
    };
    tick();

    const onResize = () => {
      const w = window.innerWidth, h = window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
      renderer.dispose();
      ptGeo.dispose(); ptMat.dispose(); lnGeo.dispose(); lnMat.dispose();
    };
  }, []);

  return <div ref={mountRef} className="fixed inset-0 pointer-events-none" style={{ zIndex: 0 }} />;
}

/* ─────────────────────────────────────────────────────────────────────
   GLASS CARD  (Framer Motion hover lift)
───────────────────────────────────────────────────────────────────── */
const glowColors = {
  indigo:  'hover:shadow-indigo-500/20',
  violet:  'hover:shadow-violet-500/20',
  emerald: 'hover:shadow-emerald-500/20',
  rose:    'hover:shadow-rose-500/20',
  amber:   'hover:shadow-amber-500/20',
  teal:    'hover:shadow-teal-500/20',
  none:    '',
};

function Card({ children, className = '', glow = 'none' }) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ type: 'spring', stiffness: 380, damping: 26 }}
      className={`relative bg-white/[0.032] backdrop-blur-md border border-white/[0.07]
                  rounded-2xl shadow-2xl transition-shadow duration-300
                  ${glowColors[glow]} ${className}`}
    >
      {children}
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────────────
   ANIMATED CONFIDENCE BAR  (Framer Motion width tween)
───────────────────────────────────────────────────────────────────── */
function Bar({ label, value, gradient, delay = 0 }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-xs text-slate-400">{label}</span>
        <motion.span
          className="text-xs font-bold text-white tabular-nums"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.45 }}
        >
          {value}%
        </motion.span>
      </div>
      <div className="h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${gradient}`}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
        />
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────
   DAG NODE  (GSAP stagger-in)
───────────────────────────────────────────────────────────────────── */
function DagNode({ step, index, total, ready }) {
  const ref = useRef();
  useEffect(() => {
    if (!ready || !ref.current) return;
    gsap.fromTo(
      ref.current,
      { opacity: 0, x: -18 },
      { opacity: 1, x: 0, duration: 0.38, delay: index * 0.09, ease: 'power2.out' }
    );
  }, [ready, index]);

  return (
    <div ref={ref} style={{ opacity: 0 }} className="flex items-start gap-3">
      <div className="flex flex-col items-center shrink-0">
        <div className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold
                        bg-gradient-to-br from-teal-400 to-cyan-600 shadow-md shadow-teal-500/30">
          {index + 1}
        </div>
        {index < total - 1 && (
          <div className="w-px bg-gradient-to-b from-teal-500/40 to-transparent mt-1" style={{ height: 22 }} />
        )}
      </div>
      <div className="pb-3 min-w-0">
        <p className="font-semibold text-teal-300 text-xs leading-tight">{step.node}</p>
        <p className="text-slate-500 text-xs mt-0.5 leading-relaxed">{step.detail}</p>
        <span className="text-xs text-slate-700">{step.ms}ms</span>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────
   SIMILAR REVIEW CARD  (Framer Motion stagger)
───────────────────────────────────────────────────────────────────── */
function SimilarCard({ item, index }) {
  const isPos = item.label === 'positive';
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.12 + 0.2 }}
      className={`rounded-xl p-3 border ${
        isPos ? 'bg-emerald-950/25 border-emerald-800/30' : 'bg-rose-950/25 border-rose-800/30'
      }`}
    >
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-xs font-bold uppercase tracking-wider ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}>
          {item.label}
        </span>
        <span className="text-xs text-slate-600">{item.similarity}% match</span>
      </div>
      <p className="text-slate-400 text-xs leading-relaxed line-clamp-3">"{item.snippet}"</p>
      <div className="mt-2 h-0.5 bg-white/[0.05] rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${isPos ? 'bg-emerald-500' : 'bg-rose-500'}`}
          initial={{ width: 0 }}
          animate={{ width: `${item.similarity}%` }}
          transition={{ duration: 0.7, delay: index * 0.12 + 0.4 }}
        />
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────────────
   MAIN APP
───────────────────────────────────────────────────────────────────── */
export default function App() {
  const [review,  setReview]  = useState('');
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');
  const [history, setHistory] = useState([]);
  const [stats,   setStats]   = useState(null);

  const headerRef  = useRef();
  const badgeRefs  = useRef([]);
  const titleRef   = useRef();

  const wordCount = review.trim().split(/\s+/).filter(Boolean).length;

  /* ── mount ── */
  useEffect(() => {
    fetchHistory();
    fetchStats();

    // GSAP entrance
    gsap.fromTo(headerRef.current,
      { y: -55, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.85, ease: 'power3.out' }
    );
    gsap.fromTo(titleRef.current,
      { opacity: 0, scale: 0.9 },
      { opacity: 1, scale: 1, duration: 0.6, delay: 0.3, ease: 'back.out(1.5)' }
    );
    const validBadges = badgeRefs.current.filter(Boolean);
    gsap.fromTo(validBadges,
      { y: -18, opacity: 0, scale: 0.75 },
      { y: 0, opacity: 1, scale: 1, duration: 0.45, stagger: 0.09, delay: 0.45, ease: 'back.out(2)' }
    );
  }, []);

  const fetchHistory = async () => {
    try { const { data } = await axios.get(`${EXPR}/api/analyses?limit=8`); setHistory(data); } catch {}
  };
  const fetchStats = async () => {
    try { const { data } = await axios.get(`${EXPR}/api/stats`); setStats(data); } catch {}
  };

  const analyze = async () => {
    if (!review.trim() || wordCount < 3) { setError('Please write at least a few words.'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const { data } = await axios.post(`${FAPI}/analyze`, { review: review.trim() });
      setResult(data);
      await axios.post(`${EXPR}/api/analyses`, {
        review: review.trim(),
        sentiment: data.sentiment, confidence: data.confidence,
        lstm_sentiment: data.lstm_sentiment, lstm_confidence: data.lstm_confidence,
        bert_sentiment: data.bert_sentiment, bert_confidence: data.bert_confidence,
        models_agree: data.models_agree, word_count: data.word_count, total_ms: data.total_ms,
      }).catch(() => {});
      fetchHistory(); fetchStats();
    } catch (err) {
      setError(err.response?.data?.detail || 'Cannot reach FastAPI. Is it running on port 8001?');
    }
    setLoading(false);
  };

  const isPos = result?.sentiment === 'positive';

  const samples = [
    "A masterpiece of modern cinema. Every frame is crafted with purpose, and the performances are nothing short of extraordinary.",
    "Complete waste of two hours. The plot was incoherent, the acting wooden, and the ending left me feeling cheated.",
    "It had its moments but overall felt rushed. The first act was promising but it never quite delivered on that potential.",
  ];

  const badges = [
    { label: 'BiLSTM', cls: 'from-indigo-500/15 border-indigo-500/25 text-indigo-300' },
    { label: 'BERT',   cls: 'from-violet-500/15 border-violet-500/25 text-violet-300' },
    { label: 'RAG',    cls: 'from-amber-500/15  border-amber-500/25  text-amber-300'  },
    { label: 'DAG',    cls: 'from-teal-500/15   border-teal-500/25   text-teal-300'   },
  ];

  /* ══════════════════════════════════════════════════════════════════ */
  return (
    <div className="min-h-screen bg-[#05050f] text-white relative overflow-x-hidden">

      {/* ── Three.js canvas ── */}
      <NeuralBackground pulse={loading} />

      {/* ── Ambient overlay ── */}
      <div className="fixed inset-0 bg-gradient-to-tr from-indigo-950/10 via-transparent to-violet-950/10 pointer-events-none" style={{ zIndex: 1 }} />

      <div className="relative" style={{ zIndex: 2 }}>

        {/* ════════════ HEADER ════════════ */}
        <header
          ref={headerRef}
          className="sticky top-0 border-b border-white/[0.05] bg-[#05050f]/75 backdrop-blur-xl px-6 py-4"
          style={{ zIndex: 10 }}
        >
          <div className="max-w-7xl mx-auto flex items-center gap-4">

            {/* Logo */}
            <motion.div
              animate={{ rotate: [0, 7, -7, 0], scale: [1, 1.06, 1] }}
              transition={{ duration: 5, repeat: Infinity, repeatDelay: 5 }}
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600
                         flex items-center justify-center text-xl shrink-0 shadow-lg shadow-indigo-500/30"
            >
              🎬
            </motion.div>

            {/* Title */}
            <div ref={titleRef}>
              <h1 className="text-base font-black tracking-tight bg-gradient-to-r
                             from-indigo-300 via-violet-200 to-indigo-300 bg-[length:200%_auto]
                             animate-shimmer bg-clip-text text-transparent">
                SentimentSage AI
              </h1>
              <p className="text-xs text-slate-600 mt-0.5">LSTM · BERT · RAG · DAG · IMDB 50K</p>
            </div>

            {/* Badges */}
            <div className="ml-auto flex flex-wrap gap-2 items-center">
              {badges.map((b, i) => (
                <span
                  key={b.label}
                  ref={el => { badgeRefs.current[i] = el; }}
                  className={`text-xs border rounded-full px-3 py-1 bg-gradient-to-r to-transparent ${b.cls}`}
                >
                  {b.label}
                </span>
              ))}
              {stats?.total > 0 && (
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-xs border border-white/10 text-slate-500 rounded-full px-3 py-1 bg-white/5"
                >
                  {stats.total} analyses
                </motion.span>
              )}
            </div>
          </div>
        </header>

        {/* ════════════ MAIN ════════════ */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* ──────────────────────────────────────────────
                COL 1 — Input + History
            ────────────────────────────────────────────── */}
            <div className="space-y-4">

              {/* Textarea card */}
              <Card className="p-5" glow="indigo">
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
                  Movie Review
                </label>
                <textarea
                  value={review}
                  onChange={e => setReview(e.target.value)}
                  placeholder="Paste any movie review here…"
                  rows={7}
                  className="w-full bg-transparent text-slate-200 text-sm resize-none
                             focus:outline-none placeholder-slate-700 leading-relaxed"
                />
                <div className="flex justify-between mt-2 pt-2 border-t border-white/[0.04] text-xs text-slate-700">
                  <span>{wordCount} words</span>
                  <span>{review.length} chars</span>
                </div>
              </Card>

              {/* Sample buttons */}
              <div>
                <p className="text-xs text-slate-700 uppercase tracking-widest mb-2">Try a sample</p>
                <div className="space-y-2">
                  {samples.map((s, i) => (
                    <motion.button
                      key={i}
                      whileHover={{ x: 5 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => { setReview(s); setResult(null); setError(''); }}
                      className="w-full text-left text-xs text-slate-600 hover:text-slate-300
                                 bg-white/[0.02] hover:bg-white/[0.05] border border-white/[0.05]
                                 rounded-xl px-3 py-2.5 transition-all line-clamp-2"
                    >
                      {s.slice(0, 90)}…
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Analyze button */}
              <motion.button
                onClick={analyze}
                disabled={!review.trim() || loading}
                whileHover={review.trim() && !loading ? { scale: 1.02 } : {}}
                whileTap={{ scale: 0.97 }}
                className="relative w-full py-4 rounded-xl font-bold text-sm overflow-hidden
                           disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {/* base gradient */}
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-violet-600" />
                {/* pulse overlay while loading */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-indigo-400 to-violet-400"
                  animate={{ opacity: loading ? [0, 0.5, 0] : 0 }}
                  transition={{ duration: 1.1, repeat: loading ? Infinity : 0 }}
                />
                <span className="relative flex items-center justify-center gap-2">
                  {loading ? (
                    <>
                      <motion.span
                        animate={{ rotate: 360 }}
                        transition={{ duration: 0.75, repeat: Infinity, ease: 'linear' }}
                        className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full inline-block"
                      />
                      Analyzing…
                    </>
                  ) : 'Analyze Sentiment'}
                </span>
              </motion.button>

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="text-xs text-rose-400 bg-rose-950/30 border border-rose-800/30 rounded-xl px-3 py-2 text-center"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Stats grid */}
              <AnimatePresence>
                {stats?.total > 0 && (
                  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                    <Card className="p-4" glow="violet">
                      <p className="text-xs text-slate-600 uppercase tracking-widest mb-3">Session Stats</p>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { label: 'Total',     value: stats.total,                  color: 'text-indigo-400'  },
                          { label: 'Positive',  value: stats.positive,               color: 'text-emerald-400' },
                          { label: 'Agreed',    value: `${stats.models_agree_pct}%`, color: 'text-violet-400'  },
                          { label: 'Avg Speed', value: `${stats.avg_ms}ms`,          color: 'text-amber-400'   },
                        ].map(s => (
                          <div key={s.label} className="bg-white/[0.04] rounded-xl p-3 text-center border border-white/[0.05]">
                            <p className={`text-xl font-black ${s.color}`}>{s.value}</p>
                            <p className="text-xs text-slate-600 mt-0.5">{s.label}</p>
                          </div>
                        ))}
                      </div>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* History */}
              <AnimatePresence>
                {history.length > 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <Card className="p-4">
                      <p className="text-xs text-slate-600 uppercase tracking-widest mb-3">Recent</p>
                      <div className="space-y-1.5">
                        {history.map((h, i) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.04 }}
                            className="flex items-center justify-between text-xs
                                       bg-white/[0.03] border border-white/[0.04] rounded-lg px-3 py-2 gap-2"
                          >
                            <span className="text-slate-500 truncate">{h.review_snippet}</span>
                            <span className={`shrink-0 font-bold ${h.sentiment === 'positive' ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {h.confidence}%
                            </span>
                          </motion.div>
                        ))}
                      </div>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* ──────────────────────────────────────────────
                COL 2 — Results
            ────────────────────────────────────────────── */}
            <div className="space-y-4">
              <AnimatePresence mode="wait">
                {!result ? (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-64
                               bg-white/[0.02] border border-white/[0.05] rounded-2xl gap-3"
                  >
                    <motion.span
                      animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
                      transition={{ duration: 4, repeat: Infinity }}
                      className="text-5xl"
                    >
                      🎭
                    </motion.span>
                    <p className="text-sm text-slate-600">Results appear here</p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-4"
                  >
                    {/* Verdict card */}
                    <Card className="p-6 overflow-hidden" glow={isPos ? 'emerald' : 'rose'}>
                      {/* Glow tint */}
                      <div className={`absolute inset-0 opacity-[0.08] bg-gradient-to-br pointer-events-none
                                       rounded-2xl ${isPos ? 'from-emerald-400' : 'from-rose-500'} to-transparent`} />

                      <p className="text-xs uppercase tracking-widest text-slate-600 mb-3">Sentiment Verdict</p>
                      <div className="flex items-center gap-4">
                        <motion.span
                          initial={{ scale: 0, rotate: -25 }}
                          animate={{ scale: 1, rotate: 0 }}
                          transition={{ type: 'spring', stiffness: 270, delay: 0.1 }}
                          className="text-5xl shrink-0"
                        >
                          {isPos ? '😊' : '😞'}
                        </motion.span>
                        <div>
                          <motion.p
                            initial={{ opacity: 0, x: -18 }} animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.15 }}
                            className={`text-4xl font-black capitalize ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}
                          >
                            {result.sentiment}
                          </motion.p>
                          <motion.p
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                            transition={{ delay: 0.28 }}
                            className="text-xs text-slate-500 mt-1"
                          >
                            {result.confidence}% confidence · {result.word_count} words · {result.total_ms}ms
                          </motion.p>
                        </div>
                      </div>

                      {!result.models_agree && (
                        <motion.div
                          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.4 }}
                          className="mt-4 text-xs bg-amber-950/30 border border-amber-800/30 rounded-xl px-3 py-2 text-amber-400"
                        >
                          ⚠️ LSTM and BERT disagreed — BERT weighted higher in fusion
                        </motion.div>
                      )}
                    </Card>

                    {/* Model comparison */}
                    <Card className="p-5" glow="violet">
                      <p className="text-xs text-slate-500 uppercase tracking-widest mb-4">Model Comparison</p>
                      <Bar
                        label={`BiLSTM → ${result.lstm_sentiment}`}
                        value={result.lstm_confidence}
                        gradient="bg-gradient-to-r from-indigo-600 to-indigo-400"
                        delay={0}
                      />
                      <Bar
                        label={`BERT → ${result.bert_sentiment}`}
                        value={result.bert_confidence}
                        gradient="bg-gradient-to-r from-violet-600 to-violet-400"
                        delay={0.18}
                      />
                      <Bar
                        label="Fused (LSTM×0.45 + BERT×0.55)"
                        value={result.confidence}
                        gradient={isPos
                          ? 'bg-gradient-to-r from-emerald-600 to-emerald-400'
                          : 'bg-gradient-to-r from-rose-600 to-rose-400'}
                        delay={0.36}
                      />
                      <motion.p
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
                        className="text-xs text-slate-600 mt-2 text-center"
                      >
                        {result.models_agree
                          ? '✅ Both models agree'
                          : '⚡ Models disagreed — BERT cast the deciding vote'}
                      </motion.p>
                    </Card>

                    {/* Similar reviews (RAG) */}
                    {result.similar?.length > 0 && (
                      <Card className="p-5" glow="amber">
                        <p className="text-sm font-bold text-amber-400 mb-0.5">Similar Reviews</p>
                        <p className="text-xs text-slate-600 mb-3">RAG · cosine similarity · 512-dim latent space</p>
                        <div className="space-y-2">
                          {result.similar.map((s, i) => <SimilarCard key={i} item={s} index={i} />)}
                        </div>
                      </Card>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* ──────────────────────────────────────────────
                COL 3 — DAG Pipeline
            ────────────────────────────────────────────── */}
            <div>
              <AnimatePresence mode="wait">
                {!result ? (
                  <motion.div
                    key="empty-dag"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="bg-white/[0.02] border border-white/[0.05] rounded-2xl p-5
                               h-48 flex items-center justify-center"
                  >
                    <p className="text-xs text-slate-600">DAG pipeline trace appears here</p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="dag"
                    initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }}
                  >
                    <Card className="p-5" glow="teal">
                      <p className="text-sm font-bold text-teal-400 mb-0.5">⚙ DAG Pipeline</p>
                      <p className="text-xs text-slate-600 mb-5">
                        8 nodes · no cycles · {result.total_ms}ms total
                      </p>
                      {result.dag_steps.map((step, i) => (
                        <DagNode
                          key={i}
                          step={step}
                          index={i}
                          total={result.dag_steps.length}
                          ready={!!result}
                        />
                      ))}
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

          </div>
        </main>

        <footer className="text-center py-5 text-xs text-slate-700 border-t border-white/[0.04] mt-4">
          SentimentSage AI · BiLSTM + BERT + RAG + DAG · IMDB 50K · MERN + FastAPI
        </footer>
      </div>
    </div>
  );
}
