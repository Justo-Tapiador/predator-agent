/**
 * PREDATOR built-in demonstration
 * Runs 3 tasks showcasing different priority classes and domains
 */
import chalk  from 'chalk';
import ora    from 'ora';
import { Predator } from '../../src/index.js';
import { formatTaskResult } from './format.js';

const DEMO_TASKS = [
  {
    directive: 'Debug and fix all failing unit tests in the authentication module',
    opts:      { priority: 'critical', budget: { tokens: 30_000, energy: 0.8 } },
    label:     'Autonomous Debugging (critical)',
  },
  {
    directive: 'Search and summarize the latest research on transformer efficiency improvements',
    opts:      { priority: 'expedited', budget: { tokens: 20_000, energy: 0.6 } },
    label:     'Scientific Research Assistance (expedited)',
  },
  {
    directive: 'Organize and refactor the database access layer with proper connection pooling',
    opts:      { priority: 'routine', budget: { tokens: 15_000, energy: 0.5 } },
    label:     'Code Refactoring (routine)',
  },
];

export async function runDemo() {
  console.log(chalk.red.bold(`
╔══════════════════════════════════════════════════════════════════╗
║            PREDATOR  –  Built-in Demonstration                  ║
║   Agentic Theory · Justo Tapiador García (UA) · 2024-2026       ║
╚══════════════════════════════════════════════════════════════════╝
`));

  // Build and train once
  const agent = new Predator();

  const spinner = ora({ text: chalk.yellow('Initialising PREDATOR (4-phase training)…'), color: 'yellow' }).start();

  let lastPhase = null;
  agent.on('phaseStart', e => {
    if (lastPhase !== e.phase) {
      lastPhase = e.phase;
      spinner.text = chalk.yellow(`Phase ${e.phase}: ${e.name}…`);
    }
  });

  await agent.train({
    epochsI: 4, epochsII_T1: 3, epochsII_T2: 3, epochsII_T3: 3,
    epochsIII: 4, epochsIV: 3,
  });
  spinner.succeed(chalk.green('Training complete'));

  // Run demo tasks
  for (let i = 0; i < DEMO_TASKS.length; i++) {
    const t = DEMO_TASKS[i];
    console.log(chalk.cyan.bold(`\n  [${i+1}/${DEMO_TASKS.length}] ${t.label}`));
    console.log(chalk.gray(`  "${t.directive}"`));

    const stepLog = [];
    agent.on('tpsStep', r => stepLog.push(r));

    const spin = ora({ text: chalk.magenta('Running…'), color: 'magenta' }).start();
    const result = await agent.execute(t.directive, t.opts);
    spin.succeed(result.success ? chalk.green('Success') : chalk.yellow('Partial'));

    // Efficiency mini-chart
    const steps   = result.stepRecords;
    const satSteps = steps.filter(s => s.phase === 'SATURATED').length;
    const tokenEff = ((1 - result.tokenUsage.tokensUsed / result.tokenUsage.tokenBudget) * 100).toFixed(1);

    console.log(chalk.gray(
      `  Steps: ${result.steps} | Saturated: ${satSteps}/${result.steps} | ` +
      `Token budget remaining: ${tokenEff}% | Quality: ${(result.quality*100).toFixed(1)}%`
    ));
    console.log(formatTaskResult(result));
  }

  console.log(chalk.green.bold('\n  Demo complete. PREDATOR is operational.\n'));
}
