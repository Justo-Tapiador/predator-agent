/**
 * ArtificialJunkyNeuron.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Core implementation of the Artificial Junky Neuron (AJN) as defined in:
 *
 *   Tapiador García, J. (2024). Agentic Theory: Definition of the
 *   Artificial Junky Neuron (AJN). Preprint WALLERMAX-AI 2604.00012.
 *   Universidad de Alicante (UA).
 *
 * The AJN is a computational unit defined by the five-element tuple:
 *   AJN = (M, θ, Ω, δ, τ)
 *
 * where:
 *   M(t)  – craving level ∈ [0,1]
 *   θ(t)  – activation threshold ∈ [0,1]
 *   Ω     – policy generator (praxic distribution parameters μ, Σ)
 *   δ     – metabolic decay rate
 *   τ     – extinction horizon
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';

// ── AJN Phase constants ────────────────────────────────────────────────────
export const AJNPhase = Object.freeze({
  RANDOM:       1,   // High-entropy exploration
  REINFORCE:    2,   // Bias developing toward stimulus source
  SATURATION:   3,   // Craving satisfied; praxes suppressed
  WITHDRAWAL:   4,   // Threshold decaying; craving returns
  FRUSTRATION:  5,   // Failure state; variance expanding
  EXTINCTION:   6,   // Addiction dissolved; reset to random
});

// ── Default hyperparameters ────────────────────────────────────────────────
const DEFAULTS = {
  betaM:        0.85,   // Exponential smoothing for craving (Eq. 1)
  lambdaUp:     0.30,   // Saturation ascent rate for threshold
  delta:        0.02,   // Metabolic decay rate (withdrawal speed)
  thetaSat:     0.75,   // Saturation threshold
  tau:          20,     // Extinction horizon (failure steps)
  eta:          0.05,   // Praxic learning rate
  lambdaSigma:  0.10,   // Entropy reduction on success
  gamma:        0.15,   // Chaotic expansion rate on failure
  sigmaMax:     2.0,    // Maximum covariance (extinction reset)
  praximDim:    64,     // Dimensionality of praxis tensor
};

// ── Utility: clamp ─────────────────────────────────────────────────────────
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

// ── Utility: Gaussian sample (Box-Muller) ──────────────────────────────────
function gaussianSample(mu, sigma) {
  const u1 = Math.random(), u2 = Math.random();
  const z  = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

// ─────────────────────────────────────────────────────────────────────────────
export class ArtificialJunkyNeuron extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string}   [opts.id]           – Unique neuron ID
   * @param {string}   [opts.stimulusClass] – Name of the stimulus class this AJN craves
   * @param {Function} [opts.intensityFn]  – I(S) → [0,1]: stimulus intensity function
   * @param {object}   [opts.params]       – Hyperparameter overrides
   */
  constructor(opts = {}) {
    super();
    this.id           = opts.id ?? uuidv4();
    this.stimulusClass = opts.stimulusClass ?? 'default';
    this.intensityFn  = opts.intensityFn ?? ((s) => clamp(s?.intensity ?? 0));
    this.p            = { ...DEFAULTS, ...(opts.params ?? {}) };

    // ── State variables ──────────────────────────────────────────────────
    this.M        = 0;          // Craving level M(t)
    this.theta    = 0.5;        // Activation threshold θ(t)
    this.phase    = AJNPhase.RANDOM;

    // ── Praxic policy: Gaussian (μ, σ) per dimension ─────────────────────
    const d       = this.p.praximDim;
    this.mu       = new Float64Array(d);            // Mean praxis
    this.sigma    = new Float64Array(d).fill(1.0);  // Std dev (diagonal Σ)

    // ── Counters ─────────────────────────────────────────────────────────
    this.nFail    = 0;       // Consecutive failure steps
    this.step     = 0;       // Total steps taken
    this.alphaPrev = 0;      // Previous stimulus intensity
    this.extinctions = 0;    // Total extinction events
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Main processing cycle.  Call once per time step.
   * @param {*} stimulus  – Raw stimulus object; passed to intensityFn
   * @returns {{ praxis: Float64Array, phase: number, alpha: number, craving: number }}
   */
  process(stimulus) {
    const alpha = clamp(this.intensityFn(stimulus));
    const delta_alpha = alpha - this.alphaPrev;

    // 1. Update craving M(t+1) = β·M(t) + (1-β)·α_t   [Eq. 1]
    this.M = clamp(this.p.betaM * this.M + (1 - this.p.betaM) * alpha);

    // 2. Determine phase and update threshold
    let praxis;

    if (alpha > this.p.thetaSat) {
      // ── Phase 3: SATURATION ─────────────────────────────────────────────
      this._setPhase(AJNPhase.SATURATION);
      this.theta = clamp(this.theta + this.p.lambdaUp * alpha);
      this.nFail = 0;
      praxis = new Float64Array(this.p.praximDim); // P_t → 0

    } else {
      // Withdrawal decay on threshold   θ(t+1) = max(0, θ(t) - δ)
      this.theta = clamp(this.theta - this.p.delta);

      if (this.theta < this.M && this.phase === AJNPhase.SATURATION) {
        this._setPhase(AJNPhase.WITHDRAWAL);
      }

      // Sample praxis from current policy
      praxis = this._samplePraxis();

      if (delta_alpha > 0) {
        // ── Phase 2: REINFORCEMENT ──────────────────────────────────────
        this._setPhase(AJNPhase.REINFORCE);
        this._onSuccess(delta_alpha);
        this.nFail = 0;
      } else {
        // ── Phase 5: FRUSTRATION ────────────────────────────────────────
        this._setPhase(AJNPhase.FRUSTRATION);
        this._onFailure(delta_alpha);
        this.nFail++;

        if (this.nFail >= this.p.tau) {
          // ── Phase 6: EXTINCTION ───────────────────────────────────────
          this._extinct();
          praxis = this._samplePraxis(); // High-entropy random after reset
        }
      }
    }

    this.alphaPrev = alpha;
    this.step++;

    const result = {
      id:      this.id,
      step:    this.step,
      phase:   this.phase,
      alpha,
      craving: this.M,
      theta:   this.theta,
      nFail:   this.nFail,
      praxis,
      praxisNorm: this._norm(praxis),
    };

    this.emit('step', result);
    return result;
  }

  /** Forcibly inject a stimulus target (used by HCI) */
  injectAddictionTarget(prototype) {
    const d = Math.min(prototype.length, this.p.praximDim);
    for (let i = 0; i < d; i++) this.mu[i] = prototype[i];
    this.M = Math.max(this.M, 0.3); // Seed craving
    this.nFail = 0;
    this._setPhase(AJNPhase.REINFORCE);
  }

  /** Get a snapshot of the neuron's internal state */
  snapshot() {
    return {
      id: this.id,
      stimulusClass: this.stimulusClass,
      phase: this.phase,
      phaseName: this._phaseName(),
      M: this.M,
      theta: this.theta,
      nFail: this.nFail,
      step: this.step,
      extinctions: this.extinctions,
      muNorm: this._norm(this.mu),
      sigmaMean: this._mean(this.sigma),
    };
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  _samplePraxis() {
    const praxis = new Float64Array(this.p.praximDim);
    for (let i = 0; i < this.p.praximDim; i++) {
      praxis[i] = gaussianSample(this.mu[i], this.sigma[i]);
    }
    return praxis;
  }

  /** On success: pull μ toward gradient and shrink Σ */
  _onSuccess(deltaAlpha) {
    for (let i = 0; i < this.p.praximDim; i++) {
      // μ_{t+1} = μ_t + η·Δα·sign(gradient proxy)
      const grad = (Math.random() - 0.5) * 2 * deltaAlpha; // approximated
      this.mu[i]    += this.p.eta * deltaAlpha * grad;
      // Σ_{t+1} = Σ_t · exp(-λ_Σ · Δα)
      this.sigma[i] *= Math.exp(-this.p.lambdaSigma * deltaAlpha);
      this.sigma[i]  = Math.max(this.sigma[i], 1e-4);
    }
  }

  /** On failure: expand Σ (chaotic intensification) */
  _onFailure(deltaAlpha) {
    for (let i = 0; i < this.p.praximDim; i++) {
      // Σ_{t+1} = Σ_t · exp(+γ · |Δα|)
      this.sigma[i] *= Math.exp(this.p.gamma * Math.abs(deltaAlpha));
      this.sigma[i]  = Math.min(this.sigma[i], this.p.sigmaMax);
    }
  }

  /** Extinction reset (Eq. 3 in ANN-Ψ paper) */
  _extinct() {
    this.mu.fill(0);
    this.sigma.fill(this.p.sigmaMax);
    this.M     = 0;
    this.nFail = 0;
    this.extinctions++;
    this._setPhase(AJNPhase.EXTINCTION);
    this.emit('extinction', { id: this.id, extinctions: this.extinctions });
    // Immediately revert to random exploration
    setTimeout(() => {
      if (this.phase === AJNPhase.EXTINCTION) this._setPhase(AJNPhase.RANDOM);
    }, 0);
  }

  _setPhase(p) {
    if (this.phase !== p) {
      const prev = this.phase;
      this.phase = p;
      this.emit('phaseChange', { id: this.id, from: prev, to: p });
    }
  }

  _norm(arr) {
    let s = 0;
    for (const v of arr) s += v * v;
    return Math.sqrt(s);
  }

  _mean(arr) {
    let s = 0;
    for (const v of arr) s += v;
    return s / arr.length;
  }

  _phaseName() {
    return Object.keys(AJNPhase).find(k => AJNPhase[k] === this.phase) ?? 'UNKNOWN';
  }
}
