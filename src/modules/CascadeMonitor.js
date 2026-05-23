/**
 * CascadeMonitor.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Background monitor that evaluates ρ_ext^(ℓ) for each AJN layer at every
 * time step and triggers interventions when cascade risk exceeds thresholds.
 *
 * Three interventions (in order of severity):
 *   1. Stimulus injection  – prime starving units back into Phase 2
 *   2. Task decomposition  – request finer sub-task granularity from owner
 *   3. Owner escalation    – pause TPS and notify Owner
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';

export class CascadeMonitor extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number} [opts.rhoWarn=0.35]     – Warning threshold for ρ_ext
   * @param {number} [opts.rhoCritical=0.65] – Critical threshold
   * @param {number} [opts.pollMs=500]       – Polling interval (ms)
   */
  constructor(opts = {}) {
    super();
    this.rhoWarn     = opts.rhoWarn     ?? 0.35;
    this.rhoCritical = opts.rhoCritical ?? 0.65;
    this.pollMs      = opts.pollMs      ?? 500;

    this.extinctionLog = [];   // { layer, unitId, t, cause }
    this.interventions = [];   // history of interventions taken
    this._timer        = null;
    this._backbone     = null; // ANNPsi reference
    this._paused       = false;
  }

  /** Attach to a running ANNPsi instance */
  attach(backbone) {
    this._backbone = backbone;

    // Listen to extinction events from backbone
    backbone.on('extinction', (e) => {
      this.extinctionLog.push({
        unitId: e.id,
        t:      Date.now(),
        cause:  'stimulus_starvation',
        extinctions: e.extinctions,
      });
      this.emit('extinctionLogged', e);
    });
  }

  /** Start the periodic cascade risk poll */
  start() {
    if (this._timer) return;
    this._timer = setInterval(() => this._poll(), this.pollMs);
    this.emit('started');
  }

  /** Stop monitoring */
  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    this.emit('stopped');
  }

  /** Evaluate cascade risk from a forward-pass result */
  evaluate(forwardResult, directive) {
    const { cascadeRisk, layerTrace } = forwardResult;

    if (cascadeRisk >= this.rhoCritical) {
      return this._intervene('owner_escalation', cascadeRisk, directive);
    }
    if (cascadeRisk >= this.rhoWarn) {
      return this._intervene('stimulus_injection', cascadeRisk, directive);
    }
    return { action: 'none', cascadeRisk };
  }

  getExtinctionLog() { return [...this.extinctionLog]; }
  getInterventions()  { return [...this.interventions]; }

  // ── Private ────────────────────────────────────────────────────────────────

  _poll() {
    if (!this._backbone) return;
    try {
      const snap = this._backbone.snapshot();
      // Check L12 (output layer) cascade risk directly
      const l12 = snap.layers?.L12;
      if (l12) {
        const rho = l12.units
          ? l12.units.filter(u => u.nFail > (u.tau ?? 20) / 2).length / (l12.units.length || 1)
          : 0;
        if (rho > this.rhoWarn) {
          this.emit('cascadeWarning', { layer: 'L12', rho });
        }
      }
    } catch (_) { /* backbone not ready */ }
  }

  _intervene(type, risk, directive) {
    const entry = { type, risk, directiveId: directive?.id, t: Date.now() };
    this.interventions.push(entry);
    this.emit('intervention', entry);

    switch (type) {
      case 'stimulus_injection':
        this.emit('requestStimulusInjection', { risk, directive });
        return { action: 'stimulus_injection', cascadeRisk: risk };

      case 'task_decomposition':
        this.emit('requestTaskDecomposition', { risk, directive });
        return { action: 'task_decomposition', cascadeRisk: risk };

      case 'owner_escalation':
        this._paused = true;
        this.emit('ownerEscalation', {
          risk,
          directive,
          message: `PREDATOR cascade risk critical (ρ=${risk.toFixed(3)}). Awaiting Owner input.`,
        });
        return { action: 'owner_escalation', cascadeRisk: risk, paused: true };

      default:
        return { action: 'none', cascadeRisk: risk };
    }
  }
}
