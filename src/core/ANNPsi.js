/**
 * ANNPsi.js  –  Agentic Neural Network (ANN-Ψ)
 * ─────────────────────────────────────────────────────────────────────────────
 * The 12-layer PREDATOR backbone as specified in the technical report,
 * combining classical transformer-like layers with AJN layers in the three
 * compositional paradigms:
 *
 *   Tapiador García, J. (2024). Agentic Theory II: ANN-Ψ.
 *   Preprint WALLERMAX-AI 2604.00013. Universidad de Alicante (UA).
 *
 * Layer stack:
 *   1–2  : Hybrid (Conv + AJN homogeneous) – sensory encoding
 *   3    : Heterogeneous AJN (K=8) – low-level feature specialization
 *   4–5  : Classical transformer blocks – contextual attention
 *   6    : Heterogeneous AJN (K=16) – mid-level concept specialization
 *   7    : Hybrid AJN – contextual modulation
 *   8–9  : Classical transformer blocks – high-level reasoning
 *   10   : Heterogeneous AJN (K=32) – high-order addiction layer
 *   11   : Hybrid AJN – praxic assembly
 *   12   : Output AJN – TPS emitter
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import {
  HomogeneousAJNLayer,
  HeterogeneousAJNLayer,
  HybridAJNLayer,
} from '../layers/AJNLayer.js';

// ── Stimulus class factory helpers ────────────────────────────────────────────
const mkIntensityFn = (key) => (s) => {
  if (!s) return 0;
  if (typeof s[key] === 'number') return Math.min(1, Math.max(0, s[key]));
  return Math.min(1, Math.max(0, s.intensity ?? 0));
};

const STIMULUS_CLASSES_L3 = [
  'syntax',      'semantics',  'structure',  'novelty',
  'relevance',   'coherence',  'completion', 'correctness',
].map(name => ({ name, intensityFn: mkIntensityFn(name) }));

const STIMULUS_CLASSES_L6 = [
  'task_progress',  'context_depth',   'tool_success',   'information_gain',
  'plan_adherence', 'constraint_ok',   'output_quality', 'resource_efficiency',
  'owner_alignment','error_absence',   'exploration_gain','goal_proximity',
  'feedback_richness','action_impact', 'knowledge_use',  'completion_signal',
].map(name => ({ name, intensityFn: mkIntensityFn(name) }));

const STIMULUS_CLASSES_L10 = Array.from({ length: 32 }, (_, i) => ({
  name: `high_order_${i}`,
  intensityFn: (s) => {
    if (!s) return 0;
    // High-order addictions: products of lower-level signals
    const base = (s.task_progress ?? 0) * (s.goal_proximity ?? 0);
    const novelty = s.novelty ?? 0;
    return Math.min(1, base * 0.7 + novelty * 0.3 + Math.random() * 0.01);
  },
}));

// ── Simulated classical layer (transformer-like feature transformation) ────────
function classicalTransformerBlock(id) {
  return (stimulus) => {
    if (!stimulus) return { intensity: 0 };
    // Simulated attention: blend and normalize features
    const keys = Object.keys(stimulus).filter(k => typeof stimulus[k] === 'number');
    if (keys.length === 0) return stimulus;
    const attended = {};
    for (const k of keys) {
      // Self-attention proxy: weighted sum with softmax-like normalization
      attended[k] = Math.tanh(stimulus[k] * 1.2 + (Math.random() - 0.5) * 0.05);
    }
    attended.intensity = keys.reduce((s, k) => s + (attended[k] ?? 0), 0) / keys.length;
    attended._layer = id;
    return attended;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
export class ANNPsi extends EventEmitter {
  /**
   * @param {object} opts
   * @param {object} [opts.ajnParams]  – Global AJN hyperparameter overrides
   */
  constructor(opts = {}) {
    super();
    this.id       = opts.id ?? 'annpsi-predator';
    this.ajnParams = opts.ajnParams ?? {};
    this._buildLayers();
  }

  _buildLayers() {
    const p = this.ajnParams;

    // Layers 1–2: Hybrid (Conv + AJN homogeneous) – sensory encoding
    this.l1 = new HybridAJNLayer({
      id: 'l1-sensory-hybrid',
      classicalFn: classicalTransformerBlock('l1-conv'),
      ajnLayer: new HomogeneousAJNLayer({
        id: 'l1-ajn', N: 64, stimulusClass: 'raw_input',
        intensityFn: mkIntensityFn('intensity'), params: p,
      }),
      alpha: 0.3,
    });

    this.l2 = new HybridAJNLayer({
      id: 'l2-encoding-hybrid',
      classicalFn: classicalTransformerBlock('l2-conv'),
      ajnLayer: new HomogeneousAJNLayer({
        id: 'l2-ajn', N: 64, stimulusClass: 'encoded_features',
        intensityFn: mkIntensityFn('intensity'), params: p,
      }),
      alpha: 0.35,
    });

    // Layer 3: Heterogeneous AJN (K=8) – feature specialization
    this.l3 = new HeterogeneousAJNLayer({
      id: 'l3-hetero-8',
      K: 8, unitsPerClass: 8,
      classes: STIMULUS_CLASSES_L3, params: p,
    });

    // Layers 4–5: Classical transformer blocks
    this.l4fn = classicalTransformerBlock('l4-transformer');
    this.l5fn = classicalTransformerBlock('l5-transformer');

    // Layer 6: Heterogeneous AJN (K=16) – concept specialization
    this.l6 = new HeterogeneousAJNLayer({
      id: 'l6-hetero-16',
      K: 16, unitsPerClass: 8,
      classes: STIMULUS_CLASSES_L6, params: p,
    });

    // Layer 7: Hybrid AJN – contextual modulation
    this.l7 = new HybridAJNLayer({
      id: 'l7-context-hybrid',
      classicalFn: classicalTransformerBlock('l7-fc'),
      ajnLayer: new HomogeneousAJNLayer({
        id: 'l7-ajn', N: 128, stimulusClass: 'context_modulation',
        intensityFn: mkIntensityFn('task_progress'), params: p,
      }),
      alpha: 0.5,
    });

    // Layers 8–9: Classical transformer blocks
    this.l8fn = classicalTransformerBlock('l8-transformer');
    this.l9fn = classicalTransformerBlock('l9-transformer');

    // Layer 10: Heterogeneous AJN (K=32) – high-order addictions
    this.l10 = new HeterogeneousAJNLayer({
      id: 'l10-hetero-32',
      K: 32, unitsPerClass: 4,
      classes: STIMULUS_CLASSES_L10, params: p,
    });

    // Layer 11: Hybrid AJN – praxic assembly
    this.l11 = new HybridAJNLayer({
      id: 'l11-praxic-hybrid',
      classicalFn: classicalTransformerBlock('l11-fc'),
      ajnLayer: new HomogeneousAJNLayer({
        id: 'l11-ajn', N: 256, stimulusClass: 'praxic_assembly',
        intensityFn: mkIntensityFn('completion_signal'), params: p,
      }),
      alpha: 0.6,
    });

    // Layer 12: Output AJN (N=1) – TPS emitter
    this.l12 = new HomogeneousAJNLayer({
      id: 'l12-output-ajn',
      N: 1, stimulusClass: 'task_completion',
      intensityFn: (s) => Math.min(1, Math.max(0,
        (s?.task_progress ?? 0) * 0.5 + (s?.completion_signal ?? 0) * 0.5
      )),
      params: { ...p, thetaSat: 0.80 },
    });

    // Wire extinction events up
    const ajnLayers = [this.l1, this.l2, this.l3, this.l6, this.l7,
                       this.l10, this.l11, this.l12];
    ajnLayers.forEach(l => l.on('extinction', e => {
      this.emit('extinction', e);
    }));
    ajnLayers.forEach(l => l.on('layerStep', e => {
      this.emit('layerStep', { layer: l.id, ...e });
    }));
  }

  /**
   * Full forward pass through all 12 layers.
   * @param {object} stimulus  – Normalized stimulus object with feature scalars
   * @returns {{ outputPraxis: Float64Array, cascadeRisk: number, saturated: boolean, layerTrace: object[] }}
   */
  forward(stimulus) {
    const trace = [];
    let s = stimulus;

    // L1 – Hybrid sensory
    const r1 = this.l1.process(s);
    s = { ...s, ...(r1.modulated ?? {}), intensity: r1.modulated?.intensity ?? s.intensity };
    trace.push({ layer: 'L1', cascadeRisk: r1.cascadeRisk });

    // L2 – Hybrid encoding
    const r2 = this.l2.process(s);
    s = { ...s, ...(r2.modulated ?? {}) };
    trace.push({ layer: 'L2', cascadeRisk: r2.cascadeRisk });

    // L3 – Heterogeneous AJN (K=8)
    const r3 = this.l3.process(s);
    s = { ...s, _winnerL3: r3.winnerClass };
    trace.push({ layer: 'L3', winner: r3.winnerClass, cascadeRisk: r3.cascadeRisk });

    // L4–L5 – Classical
    s = this.l4fn(s);
    s = this.l5fn(s);
    trace.push({ layer: 'L4-L5' });

    // L6 – Heterogeneous AJN (K=16)
    const r6 = this.l6.process(s);
    s = { ...s, _winnerL6: r6.winnerClass };
    trace.push({ layer: 'L6', winner: r6.winnerClass, cascadeRisk: r6.cascadeRisk });

    // L7 – Hybrid contextual modulation
    const r7 = this.l7.process(s);
    s = { ...s, ...(r7.modulated ?? {}) };
    trace.push({ layer: 'L7', cascadeRisk: r7.cascadeRisk });

    // L8–L9 – Classical
    s = this.l8fn(s);
    s = this.l9fn(s);
    trace.push({ layer: 'L8-L9' });

    // L10 – Heterogeneous AJN (K=32) – high-order addictions
    const r10 = this.l10.process(s);
    s = { ...s, _winnerL10: r10.winnerClass };
    trace.push({ layer: 'L10', winner: r10.winnerClass, cascadeRisk: r10.cascadeRisk });

    // L11 – Hybrid praxic assembly
    const r11 = this.l11.process(s);
    s = { ...s, ...(r11.modulated ?? {}) };
    trace.push({ layer: 'L11', cascadeRisk: r11.cascadeRisk });

    // L12 – Output AJN: TPS emission
    const r12 = this.l12.process(s);
    trace.push({ layer: 'L12', cascadeRisk: r12.cascadeRisk,
                 saturated: r12.collectiveSaturated });

    const maxCascadeRisk = Math.max(...trace.map(t => t.cascadeRisk ?? 0));

    return {
      outputPraxis: r12.layerPraxis,
      praxisNorm: this._norm(r12.layerPraxis),
      saturated: r12.collectiveSaturated,
      cascadeRisk: maxCascadeRisk,
      layerTrace: trace,
      outputCraving: this.l12.units[0]?.M ?? 0,
    };
  }

  /** Inject HCI targets into all relevant AJN layers */
  injectHCITargets(targetsByLayer) {
    for (const [layerId, prototype] of Object.entries(targetsByLayer)) {
      switch (layerId) {
        case 'l1': this.l1.injectTarget(prototype); break;
        case 'l2': this.l2.injectTarget(prototype); break;
        case 'l7': this.l7.injectTarget(prototype); break;
        case 'l11': this.l11.injectTarget(prototype); break;
        case 'l12': this.l12.injectTarget(prototype); break;
        case 'l3': this.l3.injectTarget(STIMULUS_CLASSES_L3[0].name, prototype); break;
        case 'l6': this.l6.injectTarget(STIMULUS_CLASSES_L6[0].name, prototype); break;
        case 'l10': this.l10.injectTarget(STIMULUS_CLASSES_L10[0].name, prototype); break;
      }
    }
  }

  /** Full system snapshot */
  snapshot() {
    return {
      id: this.id,
      layers: {
        L1: this.l1.snapshot(),
        L2: this.l2.snapshot(),
        L3: this.l3.snapshot(),
        L6: this.l6.snapshot(),
        L7: this.l7.snapshot(),
        L10: this.l10.snapshot(),
        L11: this.l11.snapshot(),
        L12: this.l12.snapshot(),
      },
    };
  }

  _norm(arr) {
    let s = 0;
    for (const v of arr) s += v * v;
    return Math.sqrt(s);
  }
}
