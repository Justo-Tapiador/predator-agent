/**
 * PREDATOR Web Dashboard Server
 * Express + WebSocket real-time monitoring interface
 * Visit: http://localhost:3000
 */

import express    from 'express';
import { WebSocketServer } from 'ws';
import { createServer }    from 'http';
import path       from 'path';
import { fileURLToPath }   from 'url';
import { Predator }        from '../src/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app    = express();
const server = createServer(app);
const wss    = new WebSocketServer({ server });

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Shared PREDATOR instance ──────────────────────────────────────────────────
const agent = new Predator();
let agentReady = false;

// Bootstrap training asynchronously
(async () => {
  await agent.train({ epochsI: 4, epochsII_T1: 3, epochsII_T2: 3, epochsII_T3: 3,
                      epochsIII: 4, epochsIV: 3 });
  agentReady = true;
  broadcast({ type: 'ready' });
})();

// ── WebSocket broadcast ───────────────────────────────────────────────────────
function broadcast(msg) {
  const payload = JSON.stringify(msg);
  wss.clients.forEach(c => { if (c.readyState === 1) c.send(payload); });
}

// Forward PREDATOR events to all WebSocket clients
const events = ['tpsStep','tpsStart','tpsEnd','taskComplete','trainingProgress',
                 'phaseStart','extinction','cascadeIntervention','ownerEscalation',
                 'directiveReceived','budgetExhausted'];
events.forEach(ev => agent.on(ev, data => broadcast({ type: ev, data })));

// ── REST API ──────────────────────────────────────────────────────────────────
app.get('/api/status', (req, res) => {
  res.json({ ready: agentReady, ...agent.status() });
});

app.post('/api/task', async (req, res) => {
  const { directive, priority, tokens, energy } = req.body;
  if (!directive) return res.status(400).json({ error: 'directive required' });
  if (!agentReady) return res.status(503).json({ error: 'Agent still training' });

  try {
    const result = await agent.execute(directive, {
      priority: priority ?? 'routine',
      budget: {
        tokens: tokens   ?? 50_000,
        energy: energy   ?? 1.0,
      },
    });
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/resume', (req, res) => {
  agent.resume(req.body?.feedback ?? '');
  res.json({ resumed: true });
});

app.get('/api/history', (req, res) => {
  res.json(agent.history());
});

app.get('/api/audit', (req, res) => {
  res.json(agent.pse.getAuditLog());
});

app.get('/api/extinctions', (req, res) => {
  res.json(agent.cascadeMonitor.getExtinctionLog());
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT ?? 3000;
server.listen(PORT, () => {
  console.log(`\n  PREDATOR Web Dashboard → http://localhost:${PORT}`);
  console.log('  API endpoints: /api/status  /api/task  /api/history\n');
});
