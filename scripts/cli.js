#!/usr/bin/env node
/**
 * PREDATOR CLI
 * Usage:
 *   predator task "Implement a REST API for user management"
 *   predator train [--epochs 10]
 *   predator status
 *   predator demo
 */

import { Command }  from 'commander';
import chalk        from 'chalk';
import ora          from 'ora';
import { Predator } from '../src/index.js';
import { formatTaskResult, formatStatus, formatTrainingHistory } from './lib/format.js';

const program = new Command();

program
  .name('predator')
  .description(chalk.red.bold('PREDATOR') + ' – Deep Agentic AI System (AJN Framework)\n' +
               chalk.gray('  Based on Agentic Theory by Justo Tapiador García (UA)'))
  .version('0.1.0');

// ── task command ──────────────────────────────────────────────────────────────
program
  .command('task <directive>')
  .description('Execute a task from a natural language Owner directive')
  .option('-p, --priority <level>', 'Priority: routine | expedited | critical', 'routine')
  .option('-t, --tokens <n>',       'Token budget', '50000')
  .option('-e, --energy <n>',       'Energy budget (normalized)', '1.0')
  .option('--no-train',             'Skip quick pre-training')
  .action(async (directive, opts) => {
    console.log(banner());

    const agent = new Predator();
    attachProgressListeners(agent);

    if (opts.train !== false) {
      const spinner = ora({ text: chalk.yellow('Running quick pre-training…'), color: 'yellow' }).start();
      await agent.train({ epochsI: 3, epochsII_T1: 2, epochsII_T2: 2, epochsII_T3: 2,
                          epochsIII: 3, epochsIV: 2 });
      spinner.succeed(chalk.green('Pre-training complete'));
    }

    console.log(chalk.cyan('\n┌─ Owner Directive ────────────────────────────────────────────────'));
    console.log(chalk.white(`│  ${directive}`));
    console.log(chalk.cyan(`│  Priority: ${opts.priority}  |  Budget: ${opts.tokens} tokens`));
    console.log(chalk.cyan('└──────────────────────────────────────────────────────────────────\n'));

    const spinner = ora({ text: chalk.magenta('PREDATOR executing task…'), color: 'magenta' }).start();
    const result  = await agent.execute(directive, {
      priority: opts.priority,
      budget: { tokens: parseInt(opts.tokens), energy: parseFloat(opts.energy) },
    });
    spinner.succeed(chalk.green('Task complete'));

    console.log(formatTaskResult(result));
  });

// ── train command ─────────────────────────────────────────────────────────────
program
  .command('train')
  .description('Run the full 4-phase PREDATOR training pipeline')
  .option('--epochs-i  <n>',   'Phase I epochs',  '10')
  .option('--epochs-ii <n>',   'Phase II epochs per stage', '5')
  .option('--epochs-iii <n>',  'Phase III epochs', '8')
  .option('--epochs-iv <n>',   'Phase IV epochs',  '6')
  .action(async (opts) => {
    console.log(banner());
    const agent = new Predator();
    attachTrainingListeners(agent);

    const history = await agent.train({
      epochsI:     parseInt(opts.epochsI  ?? 10),
      epochsII_T1: parseInt(opts.epochsIi ??  5),
      epochsII_T2: parseInt(opts.epochsIi ??  5),
      epochsII_T3: parseInt(opts.epochsIi ??  5),
      epochsIII:   parseInt(opts.epochsIii ?? 8),
      epochsIV:    parseInt(opts.epochsIv  ?? 6),
    });

    console.log(formatTrainingHistory(history));
    console.log(chalk.green.bold('\n✓ Training pipeline complete.'));
  });

// ── status command ────────────────────────────────────────────────────────────
program
  .command('status')
  .description('Print current PREDATOR system status')
  .action(() => {
    console.log(banner());
    const agent = new Predator();
    console.log(formatStatus(agent.status()));
  });

// ── demo command ──────────────────────────────────────────────────────────────
program
  .command('demo')
  .description('Run the built-in demonstration with 3 representative tasks')
  .action(async () => {
    const { runDemo } = await import('./lib/demo.js');
    await runDemo();
  });

program.parse();

// ── Helpers ───────────────────────────────────────────────────────────────────

function banner() {
  return chalk.red(`
  ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗
  ██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
  ██████╔╝██████╔╝█████╗  ██║  ██║███████║   ██║   ██║   ██║██████╔╝
  ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██╔══██║   ██║   ██║   ██║██╔══██╗
  ██║     ██║  ██║███████╗██████╔╝██║  ██║   ██║   ╚██████╔╝██║  ██║
  ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
`) + chalk.gray('  Agentic Theory · Justo Tapiador García (UA) · 2024-2026\n');
}

function attachProgressListeners(agent) {
  agent.on('tpsStep', r => {
    if (r.step % 10 === 0) {
      process.stdout.write(
        chalk.gray(`  [step ${String(r.step).padStart(3)}] `) +
        chalk.yellow(`craving=${r.craving.toFixed(3)} `) +
        chalk.blue(`cascadeRisk=${r.cascadeRisk.toFixed(3)} `) +
        chalk.green(`tokens+=${r.tokensOut}`) + '\n'
      );
    }
  });
  agent.on('ownerEscalation', e => {
    console.log(chalk.red.bold(`\n⚠ OWNER ESCALATION: ${e.message}`));
  });
  agent.on('extinction', e => {
    console.log(chalk.red(`  ⚡ Extinction event on unit ${e.id} (#${e.extinctions})`));
  });
}

function attachTrainingListeners(agent) {
  let lastPhase = null;
  agent.on('phaseStart', e => {
    if (lastPhase !== e.phase) {
      lastPhase = e.phase;
      console.log(chalk.cyan.bold(`\n  Phase ${e.phase}: ${e.name}`));
    }
  });
  agent.on('trainingProgress', p => {
    const bar = '█'.repeat(Math.floor((p.epoch / p.epochs) * 20)).padEnd(20, '░');
    process.stdout.write(
      `\r  [${bar}] ${p.epoch}/${p.epochs}` +
      (p.loss     !== undefined ? chalk.gray(` loss=${p.loss.toFixed(4)}`) : '') +
      (p.satRate  !== undefined ? chalk.green(` sat=${p.satRate.toFixed(2)}`) : '') +
      (p.recoveryRate !== undefined ? chalk.yellow(` rec=${p.recoveryRate.toFixed(2)}`) : '')
    );
  });
}
