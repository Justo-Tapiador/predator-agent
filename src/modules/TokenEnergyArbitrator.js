/**
 * TokenEnergyArbitrator.js  (TEA)
 * ─────────────────────────────────────────────────────────────────────────────
 * Controls the emission rate of the Tensorial Praxic Stream based on
 * energy budget consumption and the output AJN's craving level.
 *
 * r_emit(t) = r0 · exp(-κ_E · E_used/E_budget) · (1 + κ_M · M^(12)(t))
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';

export class TokenEnergyArbitrator extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number} opts.tokenBudget   – Total token budget
   * @param {number} opts.energyBudget  – Total energy budget (normalized)
   * @param {number} [opts.r0=1.0]      – Baseline emission rate
   * @param {number} [opts.kappaE=2.0]  – Energy suppression coefficient
   * @param {number} [opts.kappaM=0.5]  – Craving boost coefficient
   */
  constructor(opts = {}) {
    super();
    this.tokenBudget  = opts.tokenBudget  ?? 50_000;
    this.energyBudget = opts.energyBudget ?? 1.0;
    this.r0           = opts.r0    ?? 1.0;
    this.kappaE       = opts.kappaE ?? 2.0;
    this.kappaM       = opts.kappaM ?? 0.5;

    this.tokensUsed   = 0;
    this.energyUsed   = 0;
    this.stepCount    = 0;
    this.emitCount    = 0;

    // Per-step cost constants
    this.E_backbone   = 0.001;   // Fixed forward-pass energy per step
    this.E_ajn        = 0.0005;  // Variable AJN state update energy
    this.E_pse        = 0.002;   // Transducer dispatch energy per praxis unit
    this.T_out_kappa  = 10;      // Token calibration constant
  }

  /**
   * Decide whether to emit a praxis at this step.
   * @param {number} craving  – M^(12)(t) ∈ [0,1]
   * @param {number} praxisNorm – ‖P_t‖_F
   * @returns {{ shouldEmit: boolean, rate: number, tokensOut: number, energyStep: number }}
   */
  arbitrate(craving, praxisNorm) {
    this.stepCount++;

    // Energy per step (Eq. energy)
    const energyStep = this.E_backbone
      + this.E_ajn * craving
      + this.E_pse * praxisNorm;
    this.energyUsed += energyStep;

    // Emission rate (Eq. TEA)
    const eFraction = Math.min(1, this.energyUsed / this.energyBudget);
    const rate = this.r0
      * Math.exp(-this.kappaE * eFraction)
      * (1 + this.kappaM * craving);

    // Token output for this praxis (Eq. token_out)
    const tokensOut = Math.max(0, Math.floor(praxisNorm * this.T_out_kappa));
    const shouldEmit = (Math.random() < rate) && (this.tokensUsed + tokensOut <= this.tokenBudget);

    if (shouldEmit) {
      this.tokensUsed += tokensOut;
      this.emitCount++;
    }

    const status = this.getStatus();
    this.emit('arbitration', { shouldEmit, rate, tokensOut, energyStep, ...status });
    return { shouldEmit, rate, tokensOut, energyStep, ...status };
  }

  getStatus() {
    return {
      tokensUsed:    this.tokensUsed,
      tokenBudget:   this.tokenBudget,
      energyUsed:    this.energyUsed,
      energyBudget:  this.energyBudget,
      tokenFraction: this.tokensUsed / this.tokenBudget,
      energyFraction: this.energyUsed / this.energyBudget,
      budgetExhausted: this.tokensUsed >= this.tokenBudget
                    || this.energyUsed >= this.energyBudget,
      stepCount:  this.stepCount,
      emitCount:  this.emitCount,
    };
  }

  reset() {
    this.tokensUsed = 0;
    this.energyUsed = 0;
    this.stepCount  = 0;
    this.emitCount  = 0;
  }
}


/**
 * PraxicStreamExecutor.js  (PSE)
 * ─────────────────────────────────────────────────────────────────────────────
 * Receives praxis tensors from Layer 12, validates them against constraints,
 * routes them to the registered tool handlers (transducers), and captures
 * environmental feedback for the next ANN-Ψ forward pass.
 * ─────────────────────────────────────────────────────────────────────────────
 */
export class PraxicStreamExecutor extends EventEmitter {
  /**
   * @param {object} opts
   * @param {HierarchicalCommandInterpreter} opts.hci
   * @param {Map<string, Function>} [opts.tools]  – Registered tool handlers
   */
  constructor(opts = {}) {
    super();
    this.hci      = opts.hci;
    this.tools    = opts.tools ?? new Map();
    this.auditLog = [];

    // Register built-in tool stubs
    this._registerDefaults();
  }

  /** Register a new tool transducer */
  registerTool(id, fn, { description = '', reversible = true } = {}) {
    this.tools.set(id, { fn, description, reversible });
  }

  /**
   * Execute a praxis tensor as a structured tool call.
   * @param {Float64Array} praxisTensor  – Raw praxis from Layer 12
   * @param {object}       directive     – Current Owner directive
   * @param {number}       sigmaLevel    – Current frustration σ level
   * @returns {Promise<object>} Environmental feedback object
   */
  async execute(praxisTensor, directive, sigmaLevel = 0) {
    // Decode praxis tensor into structured praxis object
    const praxis = this._decode(praxisTensor, sigmaLevel);

    // Validate against Owner constraints
    const validation = this.hci
      ? this.hci.validatePraxis(praxis, directive)
      : { valid: true, violations: [] };

    if (!validation.valid) {
      const entry = {
        timestamp: Date.now(),
        praxis,
        outcome: 'BLOCKED',
        violations: validation.violations,
        feedback: this._negFeedback('constraint_violation'),
      };
      this.auditLog.push(entry);
      this.emit('praxisBlocked', entry);
      return entry.feedback;
    }

    // Route to tool
    const toolEntry = this.tools.get(praxis.toolId);
    let feedback;

    if (!toolEntry) {
      feedback = this._negFeedback('tool_not_found');
    } else {
      try {
        const result = await toolEntry.fn(praxis.args, directive);
        feedback = this._posFeedback(result, praxis.toolId);
        this.emit('praxisSuccess', { praxis, feedback });
      } catch (err) {
        feedback = this._negFeedback('tool_error', err.message);
        this.emit('praxisError', { praxis, error: err.message });
      }
    }

    const entry = { timestamp: Date.now(), praxis, outcome: feedback.success ? 'OK' : 'FAIL', feedback };
    this.auditLog.push(entry);
    return feedback;
  }

  getAuditLog() { return [...this.auditLog]; }

  clearAuditLog() { this.auditLog = []; }

  // ── Private ────────────────────────────────────────────────────────────────

  /** Decode a raw praxis Float64Array into a structured tool call */
  _decode(praxisTensor, sigmaLevel) {
    const toolIds = [...this.tools.keys()];
    if (toolIds.length === 0) {
      return { toolId: 'noop', args: {}, priority: 0.5, rollbackPlan: null };
    }

    // Map praxis tensor components to tool selection and args
    const norm = this._norm(praxisTensor);
    const toolIdx = Math.floor(Math.abs(praxisTensor[0] ?? 0) * toolIds.length) % toolIds.length;
    const toolId  = toolIds[toolIdx];
    const priority = Math.min(1, norm / (praxisTensor.length * 2));

    // High frustration (σ > threshold): attempt chaotic tool variations
    const chaotic = sigmaLevel > 1.5;

    return {
      toolId,
      args:        { value: praxisTensor[1] ?? 0, chaotic, sigmaLevel },
      priority,
      rollbackPlan: chaotic ? { action: 'revert', snapshotId: Date.now() } : null,
    };
  }

  _posFeedback(result, toolId) {
    return {
      success: true,
      toolId,
      result,
      intensity:          0.8 + Math.random() * 0.2,
      task_progress:      0.6 + Math.random() * 0.4,
      completion_signal:  0.5 + Math.random() * 0.5,
      tool_success:       1.0,
      goal_proximity:     0.6 + Math.random() * 0.3,
      information_gain:   0.4 + Math.random() * 0.4,
    };
  }

  _negFeedback(reason, detail = '') {
    return {
      success:           false,
      reason,
      detail,
      intensity:         0.1 + Math.random() * 0.1,
      task_progress:     0.1,
      completion_signal: 0.0,
      tool_success:      0.0,
      goal_proximity:    0.2,
      information_gain:  0.1,
    };
  }

  _norm(arr) {
    let s = 0;
    for (const v of arr) s += v * v;
    return Math.sqrt(s);
  }

  _registerDefaults() {
    // Stub tools that simulate real capabilities
    this.registerTool('read_file',    async (a) => ({ content: `[file:${a?.path ?? 'unknown'}]` }));
    this.registerTool('write_file',   async (a) => ({ written: a?.path ?? 'unknown' }), { reversible: true });
    this.registerTool('web_search',   async (a) => ({ results: [`[search:${a?.query ?? ''}]`] }));
    this.registerTool('run_code',     async (a) => ({ stdout: '[executed]', exitCode: 0 }));
    this.registerTool('api_call',     async (a) => ({ status: 200, body: '[response]' }));
    this.registerTool('list_dir',     async (a) => ({ entries: [] }));
    this.registerTool('memory_store', async (a) => ({ stored: true }));
    this.registerTool('memory_read',  async (a) => ({ value: null }));
    this.registerTool('noop',         async ()  => ({ done: true }));
  }
}
