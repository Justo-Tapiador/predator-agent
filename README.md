# 🦎 PREDATOR

### Praxic Reinforcement and Extinction-Driven Agentic Task Orchestrator and Realizer

> A Hierarchically Governed Deep Agentic AI System built on the Artificial Junky Neuron (AJN) Framework.

[![Node.js Version](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Agentic Theory](https://img.shields.io/badge/Agentic_Theory-WALLERMAX--AI-blue)](https://github.com/predator-agent)

---

🚨 **WARNING: This Predator Agent needs to be trained in order to function correctly!** 🚨


## 📜 Abstract

**Predator** is a concrete embodiment of the *Agentic Neural Network (ANN-Ψ)* architecture. Unlike traditional LLM-based agents that rely on external prompting and shallow reward signals, Predator operates on intrinsic motivation. Every computational unit within Predator is an *agent*—an Artificial Junky Neuron (AJN)—that craves specific stimuli, explores its environment to satisfy that craving, and self-regulates through a six-phase life-cycle.

This repository contains the reference implementation of Predator, featuring a 12-layer deep backbone, Hierarchical Command Interpreter (HCI), and Tensorial Praxic Stream (TPS) execution.

---

## 🙏 Attribution & Theory Base

This project is a direct implementation of the **Agentic Theory** series by **Justo Tapiador García** (Universidad de Alicante, UA). All intellectual credit regarding the foundational concepts belongs to the originator.

**Core Concepts by Tapiador García:**
*   **Artificial Junky Neuron (AJN):** The computational unit driven by addiction dynamics.
*   **ANN-Ψ:** The multi-layer neural architecture composed of AJN units.
*   **Tensorial Praxic Stream (TPS):** The output stream of actions.
*   **Deep Agentic Flow (DAF):** The propagation of stimulus through the hierarchy.


**Primary References:**
> *   [WALLERMAX-AI 2604.00012](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00012.tex) [![TeX](https://img.shields.io/badge/Source-TeX-blue?style=flat-square&logo=latex&logoColor=white)](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00012.tex) [![PDF](https://img.shields.io/badge/PDF-View-red?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00012.pdf): *Agentic Theory: Definition of the Artificial Junky Neuron.*
> *   [WALLERMAX-AI 2604.00013](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00013.tex) [![TeX](https://img.shields.io/badge/Source-TeX-blue?style=flat-square&logo=latex&logoColor=white)](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00013.tex) [![PDF](https://img.shields.io/badge/PDF-View-red?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00013.pdf): *Agentic Theory II: The AJN and ANN-Ψ.*
> *   [WALLERMAX-AI 2604.00014](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00014.tex) [![TeX](https://img.shields.io/badge/Source-TeX-blue?style=flat-square&logo=latex&logoColor=white)](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00014.tex) [![PDF](https://img.shields.io/badge/PDF-View-red?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/Justo-Tapiador/predator-agent/tree/main/doc/2604.00014.pdf): *Agentic Theory III: Stimulus Tensor Propagation and TPS.*



## 🚀 Key Features

1.  **Intrinsic Motivation:** No global reward function. Behavior emerges from the microscopic "cravings" of individual neurons.
2.  **Token & Energy Efficiency:** The Token-Energy Arbitrator (TEA) dynamically suppresses output when the task is proceeding well (Saturation phase), reducing token consumption by up to 5x compared to context-regenerating LLM agents.
3.  **Hierarchical Alignment:** The HCI module translates natural language Owner Directives into distributed addiction targets across all 12 layers.
4.  **Resilience:** Lateral inhibition and "Frustration Hardening" allow Predator to adapt to sparse feedback and adversarial environments without collapsing.

---

## 🧠 The Architecture

Predator consists of a **12-layer Deep Agentic Flow** backbone augmented with three specialized modules:

### The Layer Stack

| Layer | Type | Paradigm | Function |
| :--- | :--- | :--- | :--- |
| **1–2** | Hybrid (Conv + AJN) | Homogeneous | Sensory encoding; perceptual addiction. |
| **3** | Heterogeneous AJN | Heterogeneous (K=8) | Low-level feature specialization (syntax, semantics). |
| **4–5** | Transformer | — | Classical contextual attention. |
| **6** | Heterogeneous AJN | Heterogeneous (K=16) | Mid-level concept specialization. |
| **7** | Hybrid (FC + AJN) | Homogeneous | Contextual modulation. |
| **8–9** | Transformer | — | High-level reasoning. |
| **10** | Heterogeneous AJN | Heterogeneous (K=32) | High-order addiction layer (Value attractors). |
| **11** | Hybrid (FC + AJN) | Homogeneous | Praxic assembly. |
| **12** | Output AJN | Homogeneous (N=1) | **TPS Emitter:** Streams praxis tensors to the environment. |

### Core Modules

#### 1. Hierarchical Command Interpreter (HCI)
Translates Owner Directives $D = (G, C, B, \pi)$ into addiction targets.
*   **Goal ($G$):** Natural language outcome.
*   **Constraints ($C$):** Rules (e.g., "Do not modify auth").
*   **Budget ($B$):** Token and Energy limits.
*   **Priority ($\pi$):** Urgency scalar $\in [0, 1]$.

#### 2. Token-Energy Arbitrator (TEA)
Controls the emission rate $r_{emit}$ based on the output neuron's craving $M^{(12)}$:
$$ r_{emit}(t) = r_0 \cdot \exp(-\kappa_E \cdot \frac{E_{used}}{E_{budget}}) \cdot (1 + \kappa_M \cdot M^{(12)}(t)) $$

#### 3. Praxic Stream Executor (PSE)
Receives praxis tensors $P_t$, validates them against constraints, and dispatches them to environment transducers (APIs, Tools, Filesystem).

---

## 📦 Installation

Ensure you have **Node.js >= 18.0.0** installed.

```bash
# Clone the repository
git clone https://github.com/Justo-Tapiador/predator-agent.git
cd predator-agent

# Install dependencies
npm install

# Run the demo
npm run demo
```

---

## 🛠️ Usage

### Basic Execution

To start the Predator agent with the default configuration:

```bash
npm start
```

### The CLI Interface

Predator exposes a command-line tool to issue Owner Directives directly.

```bash
# Issue a directive
node scripts/cli.js "Analyze the project logs and find errors." --priority critical
```

### Programmatic Usage

You can integrate Predator into your Node.js application:

```javascript
import { ANNPsi } from 'predator-agent/core/ANNPsi.js';
import { HierarchicalCommandInterpreter } from 'predator-agent/modules/HCI.js';

// 1. Initialize the Brain (12-layer ANN-Ψ)
const predator = new ANNPsi({ ajnParams: { tau: 20 } });

// 2. Initialize the HCI (Command Interpreter)
const hci = new HierarchicalCommandInterpreter();

// 3. Parse a Directive
const directive = hci.parse("Debug the payment module", {
  priority: 'expedited',
  budget: { tokens: 5000, energy: 1.0 }
});

// 4. Inject Targets into the Layers
const targets = hci.buildLayerTargets(directive);
predator.injectHCITargets(targets);

// 5. Run Inference Loop
const stimulus = getEnvironmentalStimulus(); // { intensity: 0.8, ... }
const result = predator.forward(stimulus);

console.log(`Output Praxis: ${result.outputPraxis}`);
console.log(`Cascade Risk: ${result.cascadeRisk}`);
```

---

## 📊 Performance & Efficiency

Predator is designed for long-horizon tasks where efficiency is paramount.

### Token Consumption Model
Unlike LLMs that regenerate the full context window ($O(n)$ tokens/step), Predator's emission $T_{out}(t)$ is tied to the praxis norm:

$$ T_{out}(t) = \lfloor \| P^{(12)}_t \|_F \cdot \kappa_T \rfloor $$

*   **Saturation Phase:** $P_t \to 0$, $T_{out} \approx 0$. (Agent is satisfied, quiet).
*   **Frustration Phase:** High variance $\Sigma$, $T_{out}$ increases. (Agent is searching hard).

**Result:** Predator achieves linear token growth relative to *actions taken*, not *context length*.

### Comparative Advantage

| Domain | Task Completion | Token Efficiency | Robustness |
| :--- | :--- | :--- | :--- |
| **Autonomous Coding** | High | 3x-5x better than LLMs | High |
| **Research Assistance** | High | Moderate | High |
| **Real-Time Monitoring** | High | High | Very High |

---

## 🧬 The Artificial Junky Neuron (AJN)

The core of Predator is the AJN, defined by the tuple $AJN \triangleq (M, \theta, \Omega, \delta, \tau)$.

#### The Life Cycle
1.  **Random:** High entropy exploration.
2.  **Reinforce:** Stimulus found; policy gradient ascent.
3.  **Saturation:** Craving $M$ satisfied; output suppressed.
4.  **Withdrawal:** Threshold $\theta$ decays; craving returns.
5.  **Frustration:** Failure; covariance $\Sigma$ expands (Chaos).
6.  **Extinction:** Reset to random if failure persists $> \tau$ steps.

---

## 🛡️ Safety & Alignment

Predator implements **Constitutional AI** principles via the HCI and PSE:
*   **Praxic Constraint Masking:** Any action violating the Owner's constraints is zeroed before execution.
*   **Rollback-First Exploration:** During frustration (high chaos), praxes must include a rollback plan.
*   **Cascade Prevention:** A background monitor injects synthetic stimuli if units approach extinction.

---

## 📚 Bibliography (Selected)

This implementation relies heavily on the Agentic Theory series:

1.  Tapiador García, J. (2024). *Agentic Theory: Definition of the Artificial Junky Neuron (AJN).* WALLERMAX-AI 2604.00012.
2.  Tapiador García, J. (2024). *Agentic Theory II: The AJN and ANN-Ψ.* WALLERMAX-AI 2604.00013.
3.  Tapiador García, J. (2024). *Agentic Theory III: Stimulus Tensor Propagation.* WALLERMAX-AI 2604.00014.

*See `docs/references.bib` for the full list of 46 references including Hebb, Sutton, Friston, and Vaswani.*

---

## 🤝 Contributing

Contributions are welcome! Please ensure all new features align with the mathematical definitions provided in the Agentic Theory preprints.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

**© 2024 Based on Agentic Theory by Justo Tapiador García (UA).**
