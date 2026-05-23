/**
 * Predator.js  –  Main Agent Orchestrator
 * ─────────────────────────────────────────────────────────────────────────────
 * PREDATOR: Praxic Reinforcement and Extinction-Driven Agentic Task
 * Orchestrator and Realizer.
 *
 * Integrates the full pipeline:
 *   ANNPsi backbone → HCI → TEA → PSE → CascadeMonitor
 *
 * Exposes the high-level Owner interface:
 *   predator.execute(directive)  – run a full task
 *   predator.train(config)       – run training pipeline
 *   predator.status()            – get current system state
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 }  from 'uuid';

import { ANNPsi }           from './ANNPsi.js';
import { HierarchicalCommandInterpreter }
                            from '../modules/HierarchicalCommandInterpreter.js';
import { TokenEnergyArbitrator, PraxicStreamExecutor }
                            from '../modules/TokenEnergyArbitrator.js';
import { CascadeMonitor }   from '../modules/CascadeMonitor.js';
import { TrainingPipeline } from '../training/TrainingPipeline.js';

// ── Task completion quality score (simulated) ─────────────────────────────────
function computeQuality(feedbacks) {
  if (!feedbacks.length) return 0;
  const successes = feedbacks.filter(f => f.success).length;
  const avgProgress = feedbacks.reduce((s, f) => s + (f.task_progress ?? 0), 0) / feedbacks.length;
  return (successes / feedbacks.length) * 0.6 + avgProgress * 0.4;
}

// ─────────────────────────────────────────────────────────────────────────────
export class Predator extends EventEmitter {
  /**
   * @param {object} opts
   * @param {object} [opts.ajnParams]          – AJN hyperparameter overrides
   * @param {object} [opts.defaultBudget]      – Default resource budget
   * @param {Map}    [opts.tools]              – Custom tool transducers
   */
  constructor(opts = {}) {
    super();
    this.id      = `predator-${uuidv4().slice(0, 8)}`;
    this.version = '0.1.0';
    this.opts    = opts;

    // ── Build modules ──────────────────────────────────────────────────────
    this.backbone = new ANNPsi({ ajnParams: opts.ajnParams ?? {} });

    this.hci = new HierarchicalCommandInterpreter({
      defaultTokens:  opts.defaultBudget?.tokens      ?? 100_000,
      defaultEnergy:  opts.defaultBudget?.energy      ?? 2.0,
      defaultWallMs:  opts.defaultBudget?.wallClockMs ?? 600_000,
    });

    this.pse = new PraxicStreamExecutor({
      hci:   this.hci,
      tools: opts.tools,
    });

    this.cascadeMonitor = new CascadeMonitor({
      rhoWarn:     opts.rhoWarn     ?? 0.35,
      rhoCritical: opts.rhoCritical ?? 0.65,
    });
    this.cascadeMonitor.attach(this.backbone);

    this.trainingPipeline = new TrainingPipeline({
      backbone: this.backbone,
      hci:      this.hci,
    });

    // ── Wire internal events ───────────────────────────────────────────────
    this.backbone.on('extinction', e => this.emit('extinction', e));
    this.cascadeMonitor.on('ownerEscalation', e => {
      this.emit('ownerEscalation', e);
      this._ownerEscalationPending = e;
    });
    this.cascadeMonitor.on('intervention', e => this.emit('cascadeIntervention', e));

    // ── Runtime state ──────────────────────────────────────────────────────
    this._running                = false;
    this._ownerEscalationPending = null;
    this._taskHistory            = [];
    this._trained                = false;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // Owner Interface
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * Execute a task from a raw Owner directive string.
   *
   * @param {string} rawDirective  – Natural language task instruction
   * @param {object} [budgetOverrides]
   * @returns {Promise<TaskResult>}
   */
  async execute(rawDirective, budgetOverrides = {}) {
    // 1. Parse directive through HCI
    const directive = this.hci.parse(rawDirective, budgetOverrides);
    this.emit('directiveReceived', directive);

    // 2. Build per-layer addiction targets and inject
    const targets = this.hci.buildLayerTargets(directive);
    this.backbone.injectHCITargets(targets);

    // 3. Create a TEA instance for this task
    const tea = new TokenEnergyArbitrator({
      tokenBudget:  directive.budget.tokens,
      energyBudget: directive.budget.energy,
    });

    // 4. Start cascade monitor
    this.cascadeMonitor.start();

    // 5. Run the Tensorial Praxic Stream
    const result = await this._runTPS(directive, tea);

    // 6. Stop monitor
    this.cascadeMonitor.stop();

    // 7. Record task
    this._taskHistory.push(result);
    this.emit('taskComplete', result);
    return result;
  }

  /**
   * Train PREDATOR through all 4 phases.
   * @param {object} [config]  – Training config overrides
   */
  async train(config = {}) {
    this.emit('trainingStart');
    const history = await this.trainingPipeline.run({
      ...config,
      onProgress: (p) => this.emit('trainingProgress', p),
    });
    this._trained = true;
    this.emit('trainingComplete', history);
    return history;
  }

  /**
   * Register a custom tool transducer.
   * @param {string}   id
   * @param {Function} fn         – async (args, directive) => result
   * @param {object}   [meta]
   */
  registerTool(id, fn, meta = {}) {
    this.pse.registerTool(id, fn, meta);
  }

  /** Current system status snapshot */
  status() {
    return {
      id:          this.id,
      version:     this.version,
      trained:     this._trained,
      running:     this._running,
      escalation:  this._ownerEscalationPending,
      tasksDone:   this._taskHistory.length,
      extinctions: this.cascadeMonitor.getExtinctionLog().length,
      backbone:    this.backbone.snapshot(),
      auditLog:    this.pse.getAuditLog().slice(-20),
    };
  }

  /** Resume after an Owner escalation */
  resume(ownerFeedback = '') {
    if (!this._ownerEscalationPending) return;
    this._ownerEscalationPending = null;
    this.emit('ownerResumed', { feedback: ownerFeedback });
  }

  /** Get task execution history */
  history() { return [...this._taskHistory]; }

  // ══════════════════════════════════════════════════════════════════════════
  // Internal TPS execution loop
  // ══════════════════════════════════════════════════════════════════════════

  async _runTPS(directive, tea) {
    this._running = true;
    const taskId  = uuidv4();
    const started = Date.now();

    let stimulus = this._buildInitialStimulus(directive);
    const feedbacks   = [];
    const stepRecords = [];
    let step = 0;
    const maxSteps = 200;

    this.emit('tpsStart', { taskId, directive });

    while (step < maxSteps) {
      step++;

      // ── Check budget ───────────────────────────────────────────────────
      if (tea.getStatus().budgetExhausted) {
        this.emit('budgetExhausted', { step, ...tea.getStatus() });
        break;
      }

      // ── Check Owner escalation ─────────────────────────────────────────
      if (this._ownerEscalationPending) {
        this.emit('tpsPaused', { reason: 'owner_escalation', step });
        // Wait for resume (polling with timeout)
        await this._waitForResume(directive.budget.wallClockMs);
        if (this._ownerEscalationPending) break; // still pending = abort
      }

      // ── ANN-Ψ forward pass ─────────────────────────────────────────────
      const fwd = this.backbone.forward(stimulus);

      // ── Cascade risk evaluation ────────────────────────────────────────
      const cascadeAction = this.cascadeMonitor.evaluate(fwd, directive);
      if (cascadeAction.action === 'owner_escalation') {
        // Will be caught in next iteration
      } else if (cascadeAction.action === 'stimulus_injection') {
        stimulus = { ...stimulus, intensity: Math.min(1, (stimulus.intensity ?? 0) + 0.3) };
      }

      // ── TEA arbitration ────────────────────────────────────────────────
      const arbitration = tea.arbitrate(fwd.outputCraving, fwd.praxisNorm);

      let feedback = null;
      if (arbitration.shouldEmit) {
        // ── PSE tool dispatch ──────────────────────────────────────────
        feedback = await this.pse.execute(
          fwd.outputPraxis,
          directive,
          fwd.cascadeRisk
        );
        feedbacks.push(feedback);

        // ── Update stimulus from feedback ──────────────────────────────
        stimulus = { ...stimulus, ...feedback };
      } else {
        // Saturation: suppress emission, keep existing stimulus
        feedback = { suppressed: true, task_progress: stimulus.task_progress ?? 0 };
      }

      const record = {
        step,
        phase:          fwd.saturated ? 'SATURATED' : 'ACTIVE',
        cascadeRisk:    fwd.cascadeRisk,
        craving:        fwd.outputCraving,
        praxisNorm:     fwd.praxisNorm,
        emitted:        arbitration.shouldEmit,
        tokensOut:      arbitration.tokensOut,
        energyStep:     arbitration.energyStep,
      };
      stepRecords.push(record);
      this.emit('tpsStep', record);

      // ── Early termination: sustained saturation ────────────────────────
      const last5 = stepRecords.slice(-5);
      if (last5.length === 5 && last5.every(r => r.phase === 'SATURATED')) {
        this.emit('taskSaturated', { step, reason: 'sustained_saturation' });
        break;
      }

      await _tick();
    }

    this._running = false;

    const quality   = computeQuality(feedbacks);
    const wallClock = Date.now() - started;

    const taskResult = {
      taskId,
      directiveId:    directive.id,
      goal:           directive.goal,
      priority:       directive.priority,
      steps:          step,
      feedbackCount:  feedbacks.length,
      quality,
      tokenUsage:     tea.getStatus(),
      wallClockMs:    wallClock,
      stepRecords,
      cascadeEvents:  this.cascadeMonitor.getInterventions().length,
      extinctions:    this.cascadeMonitor.getExtinctionLog().length,
      success:        quality >= 0.6,
    };

    this.emit('tpsEnd', taskResult);
    return taskResult;
  }

  _buildInitialStimulus(directive) {
    const targets = directive.stimulusTargets ?? {};
    return {
      intensity:         0.4,
      task_progress:     0.0,
      completion_signal: 0.0,
      goal_proximity:    0.3,
      owner_alignment:   0.8 + directive.urgency * 0.2,
      ...Object.fromEntries(
        Object.entries(targets).map(([k, v]) => [k, v * 0.3])
      ),
    };
  }

  async _waitForResume(timeoutMs = 30_000) {
    const deadline = Date.now() + Math.min(timeoutMs, 30_000);
    while (this._ownerEscalationPending && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 200));
    }
  }
}

const _tick = () => new Promise(resolve => setImmediate(resolve));
