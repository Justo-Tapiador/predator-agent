/**
 * PREDATOR – Main entry point
 * ─────────────────────────────────────────────────────────────────────────────
 * Praxic Reinforcement and Extinction-Driven Agentic Task Orchestrator
 * and Realizer (PREDATOR)
 *
 * Based on the Agentic Theory by Justo Tapiador García (UA)
 * Preprints: WALLERMAX-AI 2604.00012, 2604.00013, 2604.00014
 * ─────────────────────────────────────────────────────────────────────────────
 */

export { Predator }           from './core/Predator.js';
export { ANNPsi }             from './core/ANNPsi.js';
export { ArtificialJunkyNeuron, AJNPhase } from './core/ArtificialJunkyNeuron.js';
export { HomogeneousAJNLayer, HeterogeneousAJNLayer, HybridAJNLayer }
                              from './layers/AJNLayer.js';
export { HierarchicalCommandInterpreter }
                              from './modules/HierarchicalCommandInterpreter.js';
export { TokenEnergyArbitrator, PraxicStreamExecutor }
                              from './modules/TokenEnergyArbitrator.js';
export { CascadeMonitor }     from './modules/CascadeMonitor.js';
export { TrainingPipeline }   from './training/TrainingPipeline.js';
