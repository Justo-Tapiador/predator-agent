/**
 * PREDATOR Test Suite
 * Tests for: AJN unit, layers, HCI, TEA, PSE, ANNPsi, and full Predator agent
 */

import { strict as assert } from 'assert';
import { ArtificialJunkyNeuron, AJNPhase } from '../src/core/ArtificialJunkyNeuron.js';
import { HomogeneousAJNLayer, HeterogeneousAJNLayer } from '../src/layers/AJNLayer.js';
import { HierarchicalCommandInterpreter } from '../src/modules/HierarchicalCommandInterpreter.js';
import { TokenEnergyArbitrator, PraxicStreamExecutor } from '../src/modules/TokenEnergyArbitrator.js';
import { ANNPsi } from '../src/core/ANNPsi.js';
import { Predator } from '../src/core/Predator.js';

const pass = (name) => console.log(`  ✓ ${name}`);
const fail = (name, err) => { console.error(`  ✗ ${name}: ${err.message}`); process.exitCode = 1; };

async function runSuite(suiteName, tests) {
  console.log(`\n[${suiteName}]`);
  for (const [name, fn] of Object.entries(tests)) {
    try { await fn(); pass(name); }
    catch (e) { fail(name, e); }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('ArtificialJunkyNeuron', {
  'initial phase is RANDOM': () => {
    const ajn = new ArtificialJunkyNeuron({ intensityFn: () => 0 });
    assert.equal(ajn.phase, AJNPhase.RANDOM);
  },

  'craving increases with rich stimulus': () => {
    const ajn = new ArtificialJunkyNeuron({ intensityFn: () => 0.9 });
    ajn.process({ intensity: 0.9 });
    ajn.process({ intensity: 0.9 });
    assert.ok(ajn.M > 0.1, `craving should grow, got ${ajn.M}`);
  },

  'enters SATURATION phase on strong stimulus': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { betaM: 0.5, thetaSat: 0.5 } });
    for (let i = 0; i < 20; i++) ajn.process({ intensity: 0.95 });
    assert.equal(ajn.phase, AJNPhase.SATURATION);
  },

  'enters EXTINCTION after tau consecutive failures': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { tau: 5, betaM: 0.1 } });
    let extinct = false;
    ajn.on('extinction', () => { extinct = true; });
    for (let i = 0; i < 10; i++) ajn.process({ intensity: 0.0 });
    assert.ok(extinct || ajn.extinctions > 0, 'extinction should fire');
  },

  'praxis tensor has correct dimensionality': () => {
    const dim = 32;
    const ajn = new ArtificialJunkyNeuron({ params: { praximDim: dim } });
    const r = ajn.process({ intensity: 0.5 });
    assert.equal(r.praxis.length, dim);
  },

  'snapshot returns all required fields': () => {
    const ajn = new ArtificialJunkyNeuron();
    const s = ajn.snapshot();
    for (const k of ['id','phase','phaseName','M','theta','nFail','step','extinctions']) {
      assert.ok(k in s, `snapshot missing field: ${k}`);
    }
  },

  'injectAddictionTarget seeds craving': () => {
    const ajn = new ArtificialJunkyNeuron();
    ajn.injectAddictionTarget(new Float64Array(64).fill(0.8));
    assert.ok(ajn.M >= 0.3, 'craving should be seeded after injection');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('HomogeneousAJNLayer', {
  'returns layerPraxis of correct shape': () => {
    const layer = new HomogeneousAJNLayer({
      N: 8, stimulusClass: 'test', intensityFn: () => 0.5,
    });
    const { layerPraxis } = layer.process({ intensity: 0.5 });
    assert.equal(layerPraxis.length, 64);
  },

  'cascadeRisk is between 0 and 1': () => {
    const layer = new HomogeneousAJNLayer({
      N: 4, stimulusClass: 'test', intensityFn: () => 0,
    });
    const { cascadeRisk } = layer.process({ intensity: 0 });
    assert.ok(cascadeRisk >= 0 && cascadeRisk <= 1, `cascadeRisk OOB: ${cascadeRisk}`);
  },

  'collective saturation triggers on strong stimulus': () => {
    const layer = new HomogeneousAJNLayer({
      N: 8, rhoSat: 0.5,
      stimulusClass: 'test', intensityFn: () => 0.95,
      params: { thetaSat: 0.5, betaM: 0.5 },
    });
    let sat = false;
    for (let i = 0; i < 30; i++) {
      const r = layer.process({ intensity: 0.95 });
      if (r.collectiveSaturated) { sat = true; break; }
    }
    assert.ok(sat, 'collective saturation should trigger');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('HeterogeneousAJNLayer', {
  'winner class is one of the defined classes': () => {
    const classes = ['a','b','c'].map(name => ({ name, intensityFn: () => Math.random() }));
    const layer = new HeterogeneousAJNLayer({ K: 3, unitsPerClass: 4, classes });
    const { winnerClass } = layer.process({ intensity: 0.5 });
    assert.ok(classes.map(c => c.name).includes(winnerClass),
      `unexpected winner: ${winnerClass}`);
  },

  'classResults has K entries': () => {
    const K = 4;
    const classes = Array.from({ length: K }, (_, i) => ({
      name: `cls_${i}`, intensityFn: () => Math.random(),
    }));
    const layer = new HeterogeneousAJNLayer({ K, unitsPerClass: 2, classes });
    const { classResults } = layer.process({});
    assert.equal(classResults.length, K);
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('HierarchicalCommandInterpreter', {
  'parses goal correctly': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parse('Debug and fix all failing unit tests urgently');
    assert.ok(d.goal.length > 0);
    assert.equal(d.priority, 'critical');
    assert.ok(d.urgency > 0.5);
  },

  'routine priority for calm language': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parse('Please organize the project documentation at your convenience');
    assert.equal(d.priority, 'routine');
  },

  'stimulus targets are normalized [0,1]': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parse('Implement and test the new search algorithm');
    for (const [k, v] of Object.entries(d.stimulusTargets)) {
      assert.ok(v >= 0 && v <= 1, `${k} OOB: ${v}`);
    }
  },

  'buildLayerTargets returns all 8 layers': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parse('write code for a REST API');
    const targets = hci.buildLayerTargets(d);
    const expected = ['l1','l2','l3','l6','l7','l10','l11','l12'];
    for (const l of expected) {
      assert.ok(l in targets, `missing layer target: ${l}`);
    }
  },

  'constraint extraction: file protection': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parse('Refactor the codebase. Do not modify the auth/ directory');
    const fileConstraint = d.constraints.find(c => c.type === 'file_protection');
    assert.ok(fileConstraint, 'file_protection constraint should be extracted');
  },

  'validatePraxis blocks protected path': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parse('do not modify the secrets/ folder');
    const praxis = { toolId: 'write_file', args: { path: 'secrets/config.json' } };
    const result = hci.validatePraxis(praxis, d);
    assert.equal(result.valid, false);
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('TokenEnergyArbitrator', {
  'arbitrate returns shouldEmit as boolean': () => {
    const tea = new TokenEnergyArbitrator({ tokenBudget: 10000, energyBudget: 1 });
    const r = tea.arbitrate(0.5, 1.0);
    assert.equal(typeof r.shouldEmit, 'boolean');
  },

  'does not exceed token budget': () => {
    const tea = new TokenEnergyArbitrator({ tokenBudget: 100, energyBudget: 999 });
    for (let i = 0; i < 500; i++) tea.arbitrate(0.9, 5.0);
    assert.ok(tea.tokensUsed <= 100, `tokens exceeded budget: ${tea.tokensUsed}`);
  },

  'energy accumulates monotonically': () => {
    const tea = new TokenEnergyArbitrator({ tokenBudget: 9999999, energyBudget: 999 });
    let prev = 0;
    for (let i = 0; i < 10; i++) {
      tea.arbitrate(0.5, 1.0);
      assert.ok(tea.energyUsed >= prev, 'energy should not decrease');
      prev = tea.energyUsed;
    }
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('PraxicStreamExecutor', {
  'executes registered tool and returns success feedback': async () => {
    const hci = new HierarchicalCommandInterpreter();
    const pse = new PraxicStreamExecutor({ hci });
    pse.registerTool('test_tool', async () => ({ result: 'ok' }));
    const directive = hci.parse('test the tool');
    const praxis = new Float64Array(64).fill(0.1);
    // Force to our test tool
    pse.tools.delete('read_file'); pse.tools.delete('write_file');
    pse.tools.delete('web_search'); pse.tools.delete('run_code');
    pse.tools.delete('api_call'); pse.tools.delete('list_dir');
    pse.tools.delete('memory_store'); pse.tools.delete('memory_read');
    pse.tools.delete('noop');
    const fb = await pse.execute(praxis, directive);
    assert.ok('intensity' in fb, 'feedback should have intensity');
  },

  'audit log grows with each execution': async () => {
    const hci = new HierarchicalCommandInterpreter();
    const pse = new PraxicStreamExecutor({ hci });
    const dir = hci.parse('run something');
    const p   = new Float64Array(64).fill(0.2);
    await pse.execute(p, dir);
    await pse.execute(p, dir);
    assert.ok(pse.getAuditLog().length >= 2);
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('ANNPsi', {
  'forward pass returns required fields': () => {
    const ann = new ANNPsi();
    const r = ann.forward({ intensity: 0.5, task_progress: 0.3, completion_signal: 0.2,
                            goal_proximity: 0.4, correctness: 0.6 });
    for (const k of ['outputPraxis','praxisNorm','saturated','cascadeRisk','layerTrace']) {
      assert.ok(k in r, `forward missing: ${k}`);
    }
  },

  'layerTrace has 8 entries': () => {
    const ann = new ANNPsi();
    const r = ann.forward({ intensity: 0.5 });
    assert.ok(r.layerTrace.length >= 8, `expected ≥8 trace entries, got ${r.layerTrace.length}`);
  },

  'cascadeRisk is in [0,1]': () => {
    const ann = new ANNPsi();
    const r = ann.forward({ intensity: 0 });
    assert.ok(r.cascadeRisk >= 0 && r.cascadeRisk <= 1);
  },

  'injectHCITargets does not throw': () => {
    const ann = new ANNPsi();
    assert.doesNotThrow(() => {
      ann.injectHCITargets({ l1: new Float64Array(64).fill(0.5) });
    });
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('Predator (integration)', {
  'status() returns well-formed object': () => {
    const p = new Predator();
    const s = p.status();
    for (const k of ['id','version','trained','running','tasksDone']) {
      assert.ok(k in s, `status missing: ${k}`);
    }
  },

  'execute() completes and returns task result': async function() {
    this?.timeout?.(30000);
    const p = new Predator();
    const result = await p.execute('Write a quick hello world script', {
      priority: 'routine', budget: { tokens: 5000, energy: 0.3 },
    });
    assert.ok('taskId'     in result);
    assert.ok('steps'      in result);
    assert.ok('quality'    in result);
    assert.ok('tokenUsage' in result);
    assert.ok(result.steps > 0, 'should have taken at least 1 step');
  },

  'task history grows after execute': async () => {
    const p = new Predator();
    await p.execute('list directory contents', { budget: { tokens: 3000, energy: 0.2 } });
    assert.equal(p.history().length, 1);
  },

  'registerTool makes tool available to PSE': () => {
    const p = new Predator();
    p.registerTool('my_tool', async () => ({ custom: true }));
    assert.ok(p.pse.tools.has('my_tool'));
  },
});

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
if ((process.exitCode ?? 0) === 0) {
  console.log('  All tests passed ✓');
} else {
  console.log('  Some tests FAILED ✗');
}
