/**
 * AJNLayer.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Three AJN layer integration paradigms as defined in:
 *
 *   Tapiador García, J. (2024). Agentic Theory II: The AJN and Its Integration
 *   into Multi-Layer Neural Architectures (ANN-Ψ).
 *   Preprint WALLERMAX-AI 2604.00013. Universidad de Alicante (UA).
 *
 * Paradigm I   – HomogeneousAJNLayer  (all units share the same stimulus class)
 * Paradigm II  – HeterogeneousAJNLayer (K competing stimulus classes)
 * Paradigm III – HybridAJNLayer        (classical layer + AJN context modulation)
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { ArtificialJunkyNeuron, AJNPhase } from '../core/ArtificialJunkyNeuron.js';

const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

// ─────────────────────────────────────────────────────────────────────────────
// Paradigm I – HOMOGENEOUS AJN LAYER
// All N units share a stimulus class; aggregate via mean praxis.
// Inter-unit coupling inhibition prevents collective saturation lock.
// ─────────────────────────────────────────────────────────────────────────────
export class HomogeneousAJNLayer extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number}   opts.N            – Number of AJN units
   * @param {string}   opts.stimulusClass
   * @param {Function} opts.intensityFn  – Shared I(S) for all units
   * @param {number}   [opts.kappa=0.2]  – Lateral inhibition coupling coefficient
   * @param {number}   [opts.rhoSat=0.7] – Collective saturation threshold (fraction)
   * @param {object}   [opts.params]     – AJN hyperparameter overrides
   */
  constructor(opts) {
    super();
    this.id           = opts.id ?? `homo-layer-${Math.random().toString(36).slice(2,7)}`;
    this.N            = opts.N ?? 32;
    this.kappa        = opts.kappa ?? 0.2;
    this.rhoSat       = opts.rhoSat ?? 0.7;
    this.stimulusClass = opts.stimulusClass;

    this.units = Array.from({ length: this.N }, (_, i) =>
      new ArtificialJunkyNeuron({
        id:           `${this.id}-u${i}`,
        stimulusClass: opts.stimulusClass,
        intensityFn:  opts.intensityFn,
        params:       opts.params,
      })
    );

    // Forward extinction events up
    this.units.forEach(u => u.on('extinction', e => this.emit('extinction', e)));
  }

  /**
   * Process one time step.
   * @param {*}      stimulus
   * @returns {{ layerPraxis: Float64Array, collectiveSaturated: boolean, cascadeRisk: number, unitResults: object[] }}
   */
  process(stimulus) {
    // Run each unit; apply lateral inhibition (Eq. coupling inhibition)
    const meanM = this._meanCraving();
    const results = this.units.map(u => {
      // Effective stimulus intensity reduced by mean craving of peers
      const inhibitedStimulus = this._inhibit(stimulus, u, meanM);
      return u.process(inhibitedStimulus);
    });

    // Layer praxis = average over all unit praxes (Eq. layer_praxis)
    const d = this.units[0].p.praximDim;
    const layerPraxis = new Float64Array(d);
    for (const r of results) {
      for (let i = 0; i < d; i++) layerPraxis[i] += r.praxis[i] / this.N;
    }

    // Collective saturation check (Eq. coll_sat)
    const satCount = results.filter(r => r.alpha > this.units[0].p.thetaSat).length;
    const collectiveSaturated = (satCount / this.N) > this.rhoSat;

    // Cascade risk = fraction of units approaching extinction
    const cascadeRisk = results.filter(r => r.nFail > this.units[0].p.tau / 2).length / this.N;

    const out = { layerPraxis, collectiveSaturated, cascadeRisk, unitResults: results };
    this.emit('layerStep', out);
    return out;
  }

  snapshot() {
    return {
      id: this.id,
      type: 'homogeneous',
      N: this.N,
      units: this.units.map(u => u.snapshot()),
      cascadeRisk: this._cascadeRisk(),
      meanCraving: this._meanCraving(),
    };
  }

  _inhibit(stimulus, unit, meanM) {
    // Clone stimulus and reduce effective intensity by peer craving
    const peerM = (meanM * this.N - unit.M) / Math.max(this.N - 1, 1);
    const inhibition = this.kappa * peerM;
    if (typeof stimulus === 'object' && stimulus !== null) {
      return { ...stimulus, intensity: clamp((stimulus.intensity ?? 0) - inhibition) };
    }
    return stimulus;
  }

  _meanCraving() {
    return this.units.reduce((s, u) => s + u.M, 0) / this.N;
  }

  _cascadeRisk() {
    const tau = this.units[0].p.tau;
    return this.units.filter(u => u.nFail > tau / 2).length / this.N;
  }

  /** Inject addiction target from HCI into all units */
  injectTarget(prototype) {
    this.units.forEach(u => u.injectAddictionTarget(prototype));
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Paradigm II – HETEROGENEOUS AJN LAYER
// K competing stimulus classes; winner-takes-all class selection.
// Spontaneous task specialization without explicit supervision.
// ─────────────────────────────────────────────────────────────────────────────
export class HeterogeneousAJNLayer extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number}   opts.K              – Number of stimulus classes
   * @param {number}   opts.unitsPerClass  – Units per class (n_k)
   * @param {Array<{ name:string, intensityFn:Function }>} opts.classes
   * @param {object}   [opts.params]
   */
  constructor(opts) {
    super();
    this.id   = opts.id ?? `hetero-layer-${Math.random().toString(36).slice(2,7)}`;
    this.K    = opts.K;
    this.unitsPerClass = opts.unitsPerClass ?? 16;

    // Build K groups of units
    this.groups = opts.classes.map((cls, k) => ({
      name:  cls.name,
      units: Array.from({ length: this.unitsPerClass }, (_, i) =>
        new ArtificialJunkyNeuron({
          id:           `${this.id}-cls${k}-u${i}`,
          stimulusClass: cls.name,
          intensityFn:  cls.intensityFn,
          params:       opts.params,
        })
      ),
      meanM: 0,
    }));

    this.winnerClass = 0;
    this.groups.forEach(g =>
      g.units.forEach(u => u.on('extinction', e => this.emit('extinction', e)))
    );
  }

  /**
   * Process one time step.
   * @returns {{ winnerClass: string, winnerPraxis: Float64Array, classResults: object[], cascadeRisk: number }}
   */
  process(stimulus) {
    // Run all groups
    const classResults = this.groups.map(g => {
      const results = g.units.map(u => u.process(stimulus));
      const meanM = results.reduce((s, r) => s + r.craving, 0) / g.units.length;

      // Group praxis = mean of winning group's praxes
      const d = g.units[0].p.praximDim;
      const praxis = new Float64Array(d);
      for (const r of results) {
        for (let i = 0; i < d; i++) praxis[i] += r.praxis[i] / g.units.length;
      }

      return { name: g.name, meanM, praxis, unitResults: results };
    });

    // Winner-takes-all selection (Eq. winner_class)
    let winnerIdx = 0, maxM = -Infinity;
    for (let k = 0; k < classResults.length; k++) {
      if (classResults[k].meanM > maxM) { maxM = classResults[k].meanM; winnerIdx = k; }
    }
    this.winnerClass = winnerIdx;

    // Cascade risk across all groups
    const allUnits = this.groups.flatMap(g => g.units);
    const tau = allUnits[0]?.p.tau ?? 20;
    const cascadeRisk = allUnits.filter(u => u.nFail > tau / 2).length / allUnits.length;

    const out = {
      winnerClass: classResults[winnerIdx].name,
      winnerPraxis: classResults[winnerIdx].praxis,
      classResults,
      cascadeRisk,
    };
    this.emit('layerStep', out);
    return out;
  }

  snapshot() {
    return {
      id: this.id,
      type: 'heterogeneous',
      K: this.K,
      winnerClass: this.groups[this.winnerClass]?.name,
      groups: this.groups.map((g, k) => ({
        name: g.name,
        isWinner: k === this.winnerClass,
        meanCraving: g.units.reduce((s, u) => s + u.M, 0) / g.units.length,
        units: g.units.map(u => u.snapshot()),
      })),
    };
  }

  injectTarget(className, prototype) {
    const g = this.groups.find(g => g.name === className);
    if (g) g.units.forEach(u => u.injectAddictionTarget(prototype));
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Paradigm III – HYBRID AGENTIC-CLASSICAL LAYER
// Classical transformation f(S) combined with AJN context modulation.
// Preserves gradient flow while injecting intrinsic motivation bias.
// ─────────────────────────────────────────────────────────────────────────────
export class HybridAJNLayer extends EventEmitter {
  /**
   * @param {object} opts
   * @param {Function} opts.classicalFn   – f(stimulus) → feature vector
   * @param {object}   opts.ajnLayer      – A HomogeneousAJNLayer instance
   * @param {number}   [opts.alpha=0.5]   – Blend weight for AJN modulation
   */
  constructor(opts) {
    super();
    this.id           = opts.id ?? `hybrid-layer-${Math.random().toString(36).slice(2,7)}`;
    this.classicalFn  = opts.classicalFn;
    this.ajnLayer     = opts.ajnLayer;
    this.blendAlpha   = opts.alpha ?? 0.5;

    this.ajnLayer.on('extinction', e => this.emit('extinction', e));
  }

  /**
   * Forward pass (Eq. hybrid):
   *   S̃^(ℓ+1) = f^(ℓ)(S^(ℓ)) ⊕ P̄^(ℓ)   (channel-wise concat / addition)
   */
  process(stimulus) {
    const classical = this.classicalFn(stimulus);
    const ajnOut    = this.ajnLayer.process(stimulus);

    // If not collectively saturated, apply AJN modulation
    const modulation = ajnOut.collectiveSaturated
      ? { intensity: 0, context: 'saturated' }
      : this._blend(classical, ajnOut.layerPraxis);

    const out = {
      classical,
      ajnOut,
      modulated: modulation,
      cascadeRisk: ajnOut.cascadeRisk,
    };
    this.emit('layerStep', out);
    return out;
  }

  snapshot() {
    return {
      id: this.id,
      type: 'hybrid',
      blendAlpha: this.blendAlpha,
      ajnLayer: this.ajnLayer.snapshot(),
    };
  }

  injectTarget(prototype) {
    this.ajnLayer.injectTarget(prototype);
  }

  _blend(classical, praxis) {
    const intensity = typeof classical?.intensity === 'number'
      ? classical.intensity * (1 - this.blendAlpha) + (this._norm(praxis) / (praxis.length || 1)) * this.blendAlpha
      : this._norm(praxis) / (praxis.length || 1);
    return { ...classical, intensity: clamp(intensity), praxisModulation: praxis };
  }

  _norm(arr) {
    let s = 0;
    for (const v of arr) s += v * v;
    return Math.sqrt(s);
  }
}
