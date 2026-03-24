# SpecRepro: Structured Specification-Driven Paper Reproduction

> **Research Proposal** — Automatic Paper Reproduction via Structured Intermediate Specifications

---

## Motivation

Reproducing results from AI research papers is a critical but laborious process.
Recent work such as [AutoReproduce (Zhao et al., 2025)](https://arxiv.org/abs/2505.20662)
shows that multi-agent LLM frameworks can automate this process.
However, AutoReproduce uses **free-form text summaries** as the intermediate
representation between paper understanding and code generation, which leads to:

| Problem | Impact |
|---------|--------|
| No completeness check | Components may be silently omitted |
| Monolithic generation | One huge prompt → harder for the LLM to focus |
| Post-hoc evaluation only | No feedback during generation |
| Opaque progress | Hard to know what has been implemented |

**SpecRepro** addresses all four problems with a single idea:
**replace free-form summaries with a machine-readable structured specification.**

---

## Key Contribution: The Structured Paper Specification (SPS)

A **PaperSpec** is a JSON object extracted from the paper that explicitly lists:

```json
{
  "title": "Decoupled Knowledge Distillation",
  "task": "Reproduce DKD on CIFAR-100 with resnet32x4→resnet8x4",
  "model_components": [
    {
      "name": "DKDLoss",
      "description": "Decoupled KD loss: α·TCKD + β·NCKD",
      "hyperparams": {"alpha": 1.0, "beta": 8.0, "temperature": 4.0},
      "status": "pending"
    }
  ],
  "training_config": {
    "optimizer": "SGD", "learning_rate": 0.05, "epochs": 240,
    "lr_schedule": "cosine", "status": "pending"
  },
  "eval_metrics": [
    {"name": "Top-1 Accuracy", "dataset": "CIFAR-100 test",
     "expected_value": 76.32, "tolerance": 1.0, "status": "pending"}
  ]
}
```

Each item carries a **status** field (`pending` → `implemented` → `verified`)
that is updated live as the pipeline runs, giving a **coverage score** at every step.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SpecRepro Pipeline                          │
│                                                                 │
│  Paper Text                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────┐                                               │
│  │ SpecExtractor│  LLM reads paper → structured PaperSpec JSON │
│  └──────┬───────┘                                               │
│         │  PaperSpec (all items: pending)                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  CodeAgent   │  Data Acquisition → validate dataloader       │
│  │  (modular)   │  Component Gen    → one call per component    │
│  │              │  Algorithm Gen    → one call per algorithm     │
│  │              │  Training Loop    → full train+eval code       │
│  └──────┬───────┘                                               │
│         │  Generated Code  +  Updated PaperSpec statuses        │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ CoverageCheck│  For each spec item: is it in the code?       │
│  └──────┬───────┘                                               │
│         │  Coverage Report (% implemented)                      │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ ResultVerify │  Run code → parse "METRIC name: value" lines  │
│  └──────┬───────┘  Compare against spec expected_value          │
│         │                                                       │
│         ▼                                                       │
│  SpecRepro Score = 0.4·Coverage + 0.4·Alignment + 0.2·Quality  │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison with AutoReproduce

| Feature | AutoReproduce | **SpecRepro** |
|---------|--------------|---------------|
| Intermediate representation | Free-form text summary | **Structured JSON spec** |
| Code generation strategy | Monolithic (full paper → full code) | **Modular** (one spec item → one code module) |
| Completeness verification | None | **Spec coverage check** |
| Numeric result verification | Post-hoc eval only | **Integrated with pipeline** |
| Progress visibility | Phase names | **Per-item status + coverage %** |
| Evaluation metric | Align-score | **SpecRepro Score (3 components)** |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
# OR for OpenAI:
export OPENAI_API_KEY="your-key-here"
```

### 3. Run on the DKD example

```bash
python main.py
```

This runs the full pipeline on the Decoupled Knowledge Distillation paper
and reports the coverage score and result alignment.

### 4. Run on a custom paper

```bash
python main.py \
    --paper-path path/to/paper.txt \
    --task "Reproduce experiment X on dataset Y, report metric Z" \
    --dataloader-path path/to/dataloader.py \
    --arxiv-id 2203.08679
```

### 5. Extract spec only (inspect before generating code)

```bash
python main.py --paper-path paper.txt --task "..." --spec-only
```

### 6. Evaluate a completed run

```bash
python evaluation/eval.py \
    --spec output/specs/paper_spec.json \
    --code output/final_code.py \
    --run-code
```

---

## Project Structure

```
SpecRepro/
├── main.py                          # Entry point
├── requirements.txt
├── specrepro/
│   ├── pipeline.py                  # Main orchestrator
│   ├── spec/
│   │   ├── schema.py                # PaperSpec dataclass (the core IR)
│   │   └── extractor.py             # LLM-based spec extraction
│   ├── agents/
│   │   ├── base.py                  # BaseAgent
│   │   └── code_agent.py            # Modular code generation agent
│   ├── verification/
│   │   ├── coverage.py              # Spec coverage checker
│   │   └── results.py               # Numeric result verifier
│   └── utils/
│       ├── llm.py                   # LLM query wrapper (Claude + OpenAI)
│       ├── code.py                  # Code execution utilities
│       └── paper.py                 # Paper text reading
├── evaluation/
│   └── eval.py                      # SpecRepro Score computation
└── examples/
    └── dkd/                         # Decoupled Knowledge Distillation
        ├── paper.txt
        └── dataloader.py
```

---

## The SpecRepro Score

SpecRepro evaluates reproduction quality with a three-component score:

```
SpecRepro Score = 0.4 × Coverage + 0.4 × Result Alignment + 0.2 × Code Quality
```

| Component | Measures | Range |
|-----------|---------|-------|
| **Coverage Score** | Fraction of spec items implemented | 0–1 |
| **Result Alignment Score** | Fraction of metrics within paper-reported tolerance | 0–1 |
| **Code Quality Score** | Syntax validity, training loop, eval function, main() | 0–1 |

---

## Research Proposal

### Problem Statement

Automatic paper reproduction is an unsolved problem in reproducibility research.
Existing automated approaches use LLMs to translate paper text directly to code,
but lack structured mechanisms to verify that all paper components are implemented
or that numeric results match reported values.

### Research Questions

1. **RQ1**: Does a structured intermediate specification (PaperSpec) improve
   completeness of paper reproduction compared to free-form summaries?

2. **RQ2**: Does modular code generation (one spec item per LLM call) reduce
   errors compared to monolithic generation?

3. **RQ3**: Can automatic spec coverage checking identify missing implementations
   without human inspection?

### Methodology

1. Evaluate on a subset of the ReproduceBench dataset (10 papers across different domains).
2. Compare SpecRepro vs. AutoReproduce on:
   - Coverage Score (fraction of components implemented)
   - Result Alignment Score (gap from reported metrics)
   - Debug iterations needed (fewer = better structured generation)
3. Ablation study: spec vs. no spec, modular vs. monolithic generation.

### Expected Contributions

1. **SpecRepro framework**: open-source, multi-LLM paper reproduction system.
2. **PaperSpec schema**: reusable JSON schema for structured paper specifications.
3. **SpecRepro Score**: three-component evaluation metric for reproduction quality.
4. **Empirical findings**: effect of structured intermediate representation on
   automated code reproduction.

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{specrepro2025,
  title   = {SpecRepro: Structured Specification-Driven Paper Reproduction},
  author  = {[Your Name]},
  year    = {2025},
  note    = {Research proposal / prototype implementation}
}
```

Also cite the AutoReproduce paper that inspired this work:

```bibtex
@misc{zhao2025autoreproduceautomaticaiexperiment,
  title   = {AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage},
  author  = {Xuanle Zhao and Zilin Sang and Yuxuan Li and Qi Shi and
             Shuo Wang and Duzhen Zhang and Xu Han and Zhiyuan Liu and Maosong Sun},
  year    = {2025},
  eprint  = {2505.20662},
  archivePrefix = {arXiv},
}
```

---

## Contact

For questions or collaboration opportunities, please reach out.
