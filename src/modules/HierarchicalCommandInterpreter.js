/**
 * HierarchicalCommandInterpreter.js  (HCI)
 * ─────────────────────────────────────────────────────────────────────────────
 * Translates Owner natural-language directives into AJN addiction targets,
 * constraint masks, resource budgets, and urgency scalars.
 *
 * A directive D = (G, C, B, π) where:
 *   G  – goal specification (natural language string)
 *   C  – constraint specification (array of rule strings)
 *   B  – resource budget { tokens, energy, wallClockMs }
 *   π  – priority class: 'routine' | 'expedited' | 'critical'
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';

// ── Urgency map ──────────────────────────────────────────────────────────────
const URGENCY = { routine: 0.2, expedited: 0.6, critical: 0.95 };

// ── Keyword → stimulus class mappings ────────────────────────────────────────
const GOAL_KEYWORDS = {
  syntax:            ['syntax', 'parse', 'format', 'lint', 'style'],
  semantics:         ['mean', 'interpret', 'understand', 'explain', 'analyze'],
  structure:         ['structure', 'organiz', 'architect', 'design', 'layout'],
  novelty:           ['novel', 'creat', 'invent', 'new idea', 'generat'],
  relevance:         ['relev', 'filter', 'select', 'rank', 'match'],
  coherence:         ['coherent', 'consistent', 'logic', 'flow', 'connect'],
  completion:        ['complet', 'finish', 'done', 'final', 'deliver'],
  correctness:       ['correct', 'fix', 'debug', 'test', 'valid', 'error'],
  task_progress:     ['progress', 'step', 'advance', 'proceed', 'continu'],
  tool_success:      ['tool', 'api', 'call', 'fetch', 'execut', 'run'],
  information_gain:  ['search', 'research', 'find', 'discover', 'learn'],
  output_quality:    ['qualit', 'polish', 'refine', 'improve', 'enhance'],
  owner_alignment:   ['align', 'follow', 'instruc', 'guidelin', 'requir'],
  goal_proximity:    ['goal', 'target', 'objectiv', 'aim', 'reach'],
  completion_signal: ['done', 'success', 'achiev', 'complet', 'satisf'],
};

// ── Constraint patterns ───────────────────────────────────────────────────────
const CONSTRAINT_PATTERNS = [
  { pattern: /do not (modify|change|edit|touch)\s+(?:the\s+)?([^\s,\.]+(?:\/[^\s,\.]*)?)/i,
    type: 'file_protection', extract: (m) => m[2].replace(/\/+$/, '').trim() },
  { pattern: /prefer reversible/i,
    type: 'reversible_only', extract: () => true },
  { pattern: /max (\d+) (tokens?|steps?|calls?)/i,
    type: 'resource_limit', extract: (m) => ({ unit: m[2], limit: parseInt(m[1]) }) },
  { pattern: /only use (approved|allowed|safe)/i,
    type: 'safe_tools_only', extract: () => true },
  { pattern: /notify (owner|me) (if|when|before)/i,
    type: 'owner_notification', extract: (m) => m[2] },
];

// ─────────────────────────────────────────────────────────────────────────────
export class HierarchicalCommandInterpreter extends EventEmitter {
  constructor(opts = {}) {
    super();
    this.id = opts.id ?? `hci-${uuidv4().slice(0,8)}`;
    this.defaultBudget = {
      tokens:      opts.defaultTokens ?? 50_000,
      energy:      opts.defaultEnergy ?? 1.0,   // normalized units
      wallClockMs: opts.defaultWallMs  ?? 300_000, // 5 min
    };
  }

  /**
   * Parse an Owner directive string into a structured directive object.
   * @param {string} rawDirective
   * @param {object} [overrides]  – Direct budget/priority overrides
   * @returns {Directive}
   */
  parse(rawDirective, overrides = {}) {
    const directiveId = uuidv4();
    const goal        = rawDirective.trim();

    const priority    = overrides.priority ?? this._inferPriority(goal);
    const urgency     = URGENCY[priority] ?? 0.2;
    const constraints = this._extractConstraints(goal, overrides.constraints ?? []);
    const budget      = this._buildBudget(priority, overrides.budget);
    const stimulusTargets = this._buildStimulusTargets(goal, urgency);

    const directive = {
      id:          directiveId,
      raw:         rawDirective,
      goal,
      priority,
      urgency,
      constraints,
      budget,
      stimulusTargets,
      tauScale:    1 / (1 + urgency * 2),  // higher urgency → shorter extinction horizon
      timestamp:   Date.now(),
    };

    this.emit('directiveParsed', directive);
    return directive;
  }

  /**
   * Project directive targets onto layer-specific addiction prototypes.
   * Returns an object keyed by layer ID → Float64Array prototype.
   */
  buildLayerTargets(directive, praxisDim = 64) {
    const targets = {};
    const activeClasses = Object.keys(directive.stimulusTargets)
      .filter(k => directive.stimulusTargets[k] > 0.5);

    // Build prototype vector biased toward active stimulus classes
    const prototype = new Float64Array(praxisDim);
    for (let i = 0; i < praxisDim; i++) {
      const classIdx = i % activeClasses.length;
      const weight   = directive.stimulusTargets[activeClasses[classIdx]] ?? 0.5;
      prototype[i]   = weight * (Math.random() * 0.4 + 0.8); // add small noise
    }

    // Inject into all AJN layers
    for (const layerId of ['l1','l2','l3','l6','l7','l10','l11','l12']) {
      targets[layerId] = prototype;
    }
    return targets;
  }

  /**
   * Validate a proposed praxis against directive constraints.
   * Returns { valid: boolean, violations: string[] }
   */
  validatePraxis(praxis, directive) {
    const violations = [];

    for (const c of directive.constraints) {
      if (c.type === 'file_protection') {
        const target = String(praxis.args?.path ?? praxis.args?.file ?? '');
        const protected_ = String(c.value ?? '').replace(/\/$/, ''); // strip trailing slash
        if (target.includes(protected_) || target.startsWith(protected_)) {
          violations.push(`Constraint violation: "${target}" is protected by Owner.`);
        }
      }
      if (c.type === 'reversible_only' && !praxis.rollbackPlan) {
        violations.push(`Constraint violation: praxis "${praxis.toolId}" has no rollback plan.`);
      }
      if (c.type === 'safe_tools_only') {
        const unsafePrefixes = ['rm', 'delete', 'drop', 'destroy', 'wipe'];
        if (unsafePrefixes.some(p => String(praxis.toolId ?? '').startsWith(p))) {
          violations.push(`Constraint violation: tool "${praxis.toolId}" is not in safe list.`);
        }
      }
    }

    return { valid: violations.length === 0, violations };
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  _inferPriority(goal) {
    const lower = goal.toLowerCase();
    if (/urgently?|immediately|asap|critical|emergency/i.test(lower)) return 'critical';
    if (/soon|quickly|expedit|high.priority/i.test(lower))           return 'expedited';
    return 'routine';
  }

  _extractConstraints(goal, explicit) {
    const constraints = [...explicit.map(c =>
      typeof c === 'string' ? { type: 'raw', value: c } : c
    )];

    for (const cp of CONSTRAINT_PATTERNS) {
      const m = goal.match(cp.pattern);
      if (m) constraints.push({ type: cp.type, value: cp.extract(m) });
    }

    return constraints;
  }

  _buildBudget(priority, overrides = {}) {
    const scale = priority === 'critical' ? 3.0
                : priority === 'expedited' ? 1.5
                : 1.0;

    return {
      tokens:      overrides.tokens      ?? this.defaultBudget.tokens * scale,
      energy:      overrides.energy      ?? this.defaultBudget.energy * scale,
      wallClockMs: overrides.wallClockMs ?? this.defaultBudget.wallClockMs / scale,
    };
  }

  _buildStimulusTargets(goal, urgency) {
    const lower  = goal.toLowerCase();
    const scores = {};

    for (const [cls, keywords] of Object.entries(GOAL_KEYWORDS)) {
      const hits = keywords.filter(kw => lower.includes(kw)).length;
      scores[cls] = Math.min(1, hits * 0.4 + urgency * 0.1 + (hits > 0 ? 0.3 : 0));
    }

    // Always seed completion_signal and goal_proximity
    scores.completion_signal = Math.max(scores.completion_signal ?? 0, urgency * 0.5 + 0.3);
    scores.goal_proximity    = Math.max(scores.goal_proximity ?? 0, 0.4);

    return scores;
  }
}
