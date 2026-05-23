# PREDATOR

> **P**raxic **R**einforcement and **E**xtinction-**D**riven **A**gentic **T**ask **O**rchestrator and **R**ealizer

A deep agentic AI system built on the **Artificial Junky Neuron (AJN)** framework, fully implemented in Node.js.

---

**Based on the Agentic Theory by Justo Tapiador García**  
Universidad de Alicante (UA)  
Preprints: WALLERMAX-AI 2604.00012 · 2604.00013 · 2604.00014

---

## Architecture

```
Owner Directive (natural language)
        │
        ▼
┌─────────────────────────────────────────┐
│  Hierarchical Command Interpreter (HCI) │  ← Parses directives into addiction targets
└─────────────────────────────────────────┘
        │  addiction target injection
        ▼
┌─────────────────────────────────────────┐
│  ANN-Ψ Backbone (12 layers)             │
│                                         │
│  L1–L2  Hybrid AJN  (sensory encoding)  │
│  L3     Hetero AJN  K=8  (features)     │
│  L4–L5  Transformer (contextual attn)   │
│  L6     Hetero AJN  K=16 (concepts)     │
│  L7     Hybrid AJN  (modulation)        │
│  L8–L9  Transformer (high-level reason) │
│  L10    Hetero AJN  K=32 (high-order)   │
│  L11    Hybrid AJN  (praxic assembly)   │
│  L12    Output AJN  (TPS emission)      │
└─────────────────────────────────────────┘
        │  praxis tensors
        ▼
┌─────────────────────────────────────────┐
│  Token-Energy Arbitrator (TEA)          │  ← Adaptive compute budgeting
└─────────────────────────────────────────┘
        │  filtered praxis stream
        ▼
┌─────────────────────────────────────────┐
│  Praxic Stream Executor (PSE)           │  ← Tool dispatch + constraint validation
└─────────────────────────────────────────┘
        │
        ▼
   Environment (tools, APIs, file system)
        │  feedback
        └──────────────────────────────────► ANN-Ψ (next step)
```

## The AJN Six-Phase Lifecycle

Each neuron autonomously cycles through:

| Phase | Name | Behaviour |
|-------|------|-----------|
| 1 | **Random** | High-entropy exploration |
| 2 | **Reinforce** | Bias developing toward stimulus |
| 3 | **Saturation** | Craving satisfied; praxes suppressed |
| 4 | **Withdrawal** | Threshold decaying; craving returns |
| 5 | **Frustration** | Failure; covariance expanding chaotically |
| 6 | **Extinction** | Addiction dissolved; full reset |

## Installation

```bash
git clone https://github.com/your-org/predator.git
cd predator
npm install
```

Requires **Node.js ≥ 18**.

## Quick Start

```bash
# Run a task directly
npm run cli -- task "Debug all failing unit tests in the auth module" --priority critical

# Interactive demo (3 tasks, full training)
npm run cli -- demo

# Web dashboard (http://localhost:3000)
npm run web

# Full benchmark suite
npm run benchmark
```

## Programmatic API

```js
import { Predator } from './src/index.js';

const agent = new Predator();

// Train all 4 phases
await agent.train();

// Execute an Owner directive
const result = await agent.execute(
  'Implement and test a rate-limiting middleware',
  { priority: 'expedited', budget: { tokens: 30000, energy: 1.0 } }
);

console.log(`Quality: ${(result.quality * 100).toFixed(1)}%`);
console.log(`Tokens used: ${result.tokenUsage.tokensUsed}`);
console.log(`Steps: ${result.steps} (${result.stepRecords.filter(s=>s.phase==='SATURATED').length} saturated)`);

// Register a custom tool
agent.registerTool('my_api', async (args) => {
  // call your API here
  return { status: 'ok', data: '…' };
});

// Listen to real-time events
agent.on('tpsStep',        s => console.log('step', s.step, 'craving', s.craving));
agent.on('extinction',     e => console.log('extinction on unit', e.id));
agent.on('ownerEscalation', e => { console.warn(e.message); agent.resume('continue'); });
```

## Training Pipeline

```js
await agent.train({
  epochsI:     10,   // Phase I:  Large-scale pre-training
  epochsII_T1:  5,   // Phase II: Addiction seeding
  epochsII_T2:  5,   //           Tolerance building
  epochsII_T3:  5,   //           Frustration hardening
  epochsIII:    8,   // Phase III: Hierarchical fine-tuning (HIFT)
  epochsIV:     6,   // Phase IV:  Adversarial frustration hardening
});
```

## Project Structure

```
predator/
├── src/
│   ├── core/
│   │   ├── ArtificialJunkyNeuron.js   # AJN unit (5-tuple definition)
│   │   ├── ANNPsi.js                  # 12-layer ANN-Ψ backbone
│   │   └── Predator.js                # Main agent orchestrator
│   ├── layers/
│   │   └── AJNLayer.js                # 3 integration paradigms
│   ├── modules/
│   │   ├── HierarchicalCommandInterpreter.js  # HCI
│   │   ├── TokenEnergyArbitrator.js           # TEA + PSE
│   │   └── CascadeMonitor.js                  # Runtime cascade prevention
│   ├── training/
│   │   └── TrainingPipeline.js        # 4-phase pipeline
│   └── index.js                       # Public exports
├── web/
│   ├── server.js                      # Express + WebSocket server
│   └── public/index.html             # Real-time dashboard
├── scripts/
│   ├── cli.js                         # Commander CLI
│   ├── benchmark.js                   # Efficiency benchmarks
│   └── lib/
│       ├── demo.js
│       └── format.js
├── tests/
│   └── predator.test.js
└── package.json
```

## Tests

```bash
node --experimental-vm-modules tests/predator.test.js
```

## Web Dashboard

```bash
npm run web
# → http://localhost:3000
```

Real-time monitoring of:
- Praxic stream (craving + cascade risk live chart)
- All 12 layer states (cascade risk per layer)
- Budget consumption (tokens + energy bars)
- Task history table
- Extinction event log

## Theoretical Basis

PREDATOR's efficiency advantages derive directly from AJN saturation dynamics:

- **Token suppression in saturation** — During Phase 3, `‖P_t‖_F → 0`, so near-zero tokens are emitted when the task is progressing well.
- **Chaotic intensification in frustration** — During Phase 5, covariance expands, generating high-diversity praxes that break stuck states.
- **Cascade prevention** — Lateral inhibition coupling reduces group extinction probability below the independent bound.
- **Adaptive compute** — The TEA combines energy-aware suppression with craving-driven override, allocating compute where it matters.

## License

MIT

---

*Agentic Theory original concepts © Justo Tapiador García, Universidad de Alicante (UA), 2024–2026.*
