/**
 * TrainingPipeline.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Four-phase PREDATOR training pipeline:
 *
 *  Phase I   – Large-scale pre-training (classical layers + AJN initialisation)
 *  Phase II  – Addiction shaping (stimulus class assignment + gradient curriculum)
 *  Phase III – Hierarchical fine-tuning with Owner directives (HIFT)
 *  Phase IV  – Adversarial frustration hardening (cascade stress-testing)
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { AJNPhase } from '../core/ArtificialJunkyNeuron.js';

// ── Tiny synthetic dataset generators ────────────────────────────────────────
function* pretrainBatch(batchSize = 32) {
  for (let i = 0; i < batchSize; i++) {
    yield {
      intensity:         Math.random(),
      syntax:            Math.random(),
      semantics:         Math.random(),
      structure:         Math.random(),
      task_progress:     Math.random() * 0.3,
      completion_signal: Math.random() * 0.1,
      goal_proximity:    Math.random() * 0.3,
      correctness:       Math.random(),
      novelty:           Math.random(),
    };
  }
}

function* addictionCurriculumBatch(stage, batchSize = 32) {
  for (let i = 0; i < batchSize; i++) {
    const base = stage === 'seeding'   ? 0.8 + Math.random() * 0.2
               : stage === 'tolerance' ? 0.6 + Math.sin(i * 0.3) * 0.2
               : /* frustration */       Math.random() < 0.3 ? 0.9 : 0.0;
    yield {
      intensity:         base,
      syntax:            base * (0.8 + Math.random() * 0.2),
      semantics:         base * (0.7 + Math.random() * 0.3),
      task_progress:     base * 0.8,
      completion_signal: base * 0.9,
      goal_proximity:    base * 0.7,
      correctness:       base,
      novelty:           Math.random() * 0.3,
      tool_success:      base,
    };
  }
}

function* hiftBatch(directives, batchSize = 16) {
  for (let i = 0; i < batchSize; i++) {
    const d = directives[i % directives.length];
    yield {
      directive: d,
      stimulus: {
        intensity:         0.5 + Math.random() * 0.4,
        task_progress:     Math.random(),
        completion_signal: Math.random() * 0.7,
        owner_alignment:   0.7 + Math.random() * 0.3,
        goal_proximity:    Math.random(),
        plan_adherence:    0.6 + Math.random() * 0.4,
      },
      completionLabel: Math.random() > 0.4 ? 1 : 0,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
export class TrainingPipeline extends EventEmitter {
  /**
   * @param {object} opts
   * @param {ANNPsi}  opts.backbone
   * @param {HierarchicalCommandInterpreter} opts.hci
   */
  constructor(opts = {}) {
    super();
    this.backbone = opts.backbone;
    this.hci      = opts.hci;
    this.history  = { phaseI: [], phaseII: [], phaseIII: [], phaseIV: [] };
    this.phase    = null;
  }

  /**
   * Run the full 4-phase training pipeline.
   * @param {object} config  – Training configuration
   */
  async run(config = {}) {
    const cfg = {
      epochsI:      config.epochsI      ?? 10,
      epochsII_T1:  config.epochsII_T1  ?? 5,
      epochsII_T2:  config.epochsII_T2  ?? 5,
      epochsII_T3:  config.epochsII_T3  ?? 5,
      epochsIII:    config.epochsIII    ?? 8,
      epochsIV:     config.epochsIV     ?? 6,
      batchSize:    config.batchSize    ?? 32,
      directives:   config.directives   ?? this._defaultDirectives(),
      onProgress:   config.onProgress   ?? (() => {}),
    };

    this.emit('pipelineStart', { config: cfg });

    await this._phaseI(cfg);
    await this._phaseII(cfg);
    await this._phaseIII(cfg);
    await this._phaseIV(cfg);

    this.emit('pipelineComplete', { history: this.history });
    return this.history;
  }

  // ── Phase I: Large-Scale Pre-Training ─────────────────────────────────────
  async _phaseI(cfg) {
    this.phase = 'I';
    this.emit('phaseStart', { phase: 'I', name: 'Large-Scale Pre-Training' });

    for (let epoch = 0; epoch < cfg.epochsI; epoch++) {
      let lossSum = 0, n = 0;

      for (const sample of pretrainBatch(cfg.batchSize)) {
        const result = this.backbone.forward(sample);
        // Simulated reconstruction loss (lower praxis norm in saturation = good)
        const loss = result.saturated ? 0.05 : (1 - result.outputCraving) * 0.5;
        lossSum += loss;
        n++;
      }

      const avgLoss = lossSum / n;
      const record  = { epoch, loss: avgLoss, phase: 'I' };
      this.history.phaseI.push(record);
      this.emit('epochEnd', record);
      cfg.onProgress({ phase: 'I', epoch, epochs: cfg.epochsI, loss: avgLoss });
      await _tick();
    }
    this.emit('phaseEnd', { phase: 'I' });
  }

  // ── Phase II: Addiction Shaping ───────────────────────────────────────────
  async _phaseII(cfg) {
    this.phase = 'II';
    this.emit('phaseStart', { phase: 'II', name: 'Addiction Shaping' });

    const stages = [
      { name: 'seeding',    epochs: cfg.epochsII_T1 },
      { name: 'tolerance',  epochs: cfg.epochsII_T2 },
      { name: 'frustration',epochs: cfg.epochsII_T3 },
    ];

    for (const stage of stages) {
      for (let epoch = 0; epoch < stage.epochs; epoch++) {
        let satCount = 0, extCount = 0, n = 0;

        for (const sample of addictionCurriculumBatch(stage.name, cfg.batchSize)) {
          const result = this.backbone.forward(sample);
          if (result.saturated)   satCount++;
          if (result.cascadeRisk > 0.5) extCount++;
          n++;
        }

        const record = {
          epoch, stage: stage.name, phase: 'II',
          saturationRate: satCount / n,
          cascadeRate:    extCount / n,
        };
        this.history.phaseII.push(record);
        this.emit('epochEnd', record);
        cfg.onProgress({ phase: 'II', stage: stage.name, epoch, epochs: stage.epochs,
                         satRate: satCount/n, extRate: extCount/n });
        await _tick();
      }
    }
    this.emit('phaseEnd', { phase: 'II' });
  }

  // ── Phase III: Hierarchical Fine-Tuning (HIFT) ────────────────────────────
  async _phaseIII(cfg) {
    this.phase = 'III';
    this.emit('phaseStart', { phase: 'III', name: 'Hierarchical Fine-Tuning (HIFT)' });

    for (let epoch = 0; epoch < cfg.epochsIII; epoch++) {
      let alignSum = 0, completionSum = 0, n = 0;

      for (const sample of hiftBatch(cfg.directives, cfg.batchSize)) {
        // Inject HCI targets
        const targets = this.hci.buildLayerTargets(sample.directive);
        this.backbone.injectHCITargets(targets);

        const result = this.backbone.forward(sample.stimulus);

        // Simulated alignment loss (Eq. L_HCI)
        const alignLoss      = 1 - result.outputCraving * sample.stimulus.owner_alignment;
        const completionLoss = sample.completionLabel === 1
          ? (result.saturated ? 0 : 0.5)
          : (result.saturated ? 0.5 : 0);

        alignSum      += alignLoss;
        completionSum += completionLoss;
        n++;
      }

      const record = {
        epoch, phase: 'III',
        alignLoss:      alignSum / n,
        completionLoss: completionSum / n,
        totalLoss:      (alignSum + completionSum) / n,
      };
      this.history.phaseIII.push(record);
      this.emit('epochEnd', record);
      cfg.onProgress({ phase: 'III', epoch, epochs: cfg.epochsIII,
                       loss: record.totalLoss });
      await _tick();
    }
    this.emit('phaseEnd', { phase: 'III' });
  }

  // ── Phase IV: Adversarial Frustration Hardening ───────────────────────────
  async _phaseIV(cfg) {
    this.phase = 'IV';
    this.emit('phaseStart', { phase: 'IV', name: 'Adversarial Frustration Hardening' });

    for (let epoch = 0; epoch < cfg.epochsIV; epoch++) {
      let cascadesCaught = 0, recoveries = 0, n = 0;

      for (let i = 0; i < cfg.batchSize; i++) {
        // Adversarial stimulus: deliberately starve most stimulus signals
        const adversarial = {
          intensity:         Math.random() < 0.3 ? 0.9 : 0.05,
          task_progress:     0.0,
          completion_signal: 0.0,
          goal_proximity:    Math.random() * 0.15,
          correctness:       0.0,
          tool_success:      0.0,
          // Partial rescue signal (prevents full system collapse)
          novelty:           Math.random() * 0.4,
        };

        const result = this.backbone.forward(adversarial);
        if (result.cascadeRisk > 0.4) cascadesCaught++;

        // Recovery step: provide rich stimulus immediately after adversarial
        const recovery = { intensity: 0.85, task_progress: 0.7,
                           completion_signal: 0.6, goal_proximity: 0.8,
                           correctness: 0.9, tool_success: 0.8 };
        const rResult = this.backbone.forward(recovery);
        if (rResult.outputCraving > 0.3) recoveries++;
        n++;
      }

      const record = {
        epoch, phase: 'IV',
        cascadeRate:    cascadesCaught / n,
        recoveryRate:   recoveries / n,
        resilience:     recoveries / Math.max(1, cascadesCaught),
      };
      this.history.phaseIV.push(record);
      this.emit('epochEnd', record);
      cfg.onProgress({ phase: 'IV', epoch, epochs: cfg.epochsIV,
                       cascadeRate: record.cascadeRate, recoveryRate: record.recoveryRate });
      await _tick();
    }
    this.emit('phaseEnd', { phase: 'IV' });
  }

  _defaultDirectives() {
    const directives = [
      'Implement and test the payment processing module',
      'Search and summarize recent papers on neural architecture search',
      'Debug and fix the authentication service errors urgently',
      'Organize and refactor the database access layer',
      'Generate unit tests for all API endpoints',
      'Analyze and improve the performance of the search algorithm',
    ];
    return directives.map(d => this.hci.parse(d));
  }
}

// ── Async tick helper (yields event loop) ────────────────────────────────────
const _tick = () => new Promise(resolve => setImmediate(resolve));
