# SpecRepro: Structured Specification Driven Paper Reproduction

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
<img width="3888" height="880" alt="image" src="https://github.com/user-attachments/assets/d2b5c770-908b-4b08-87f3-6cdc0f404110" />

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

## Contact

For questions or collaboration opportunities, please reach out.
