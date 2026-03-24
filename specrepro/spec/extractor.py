"""
SpecAgent: Extracts a structured PaperSpec from raw paper text using an LLM.

This is the first phase of SpecRepro and its core novelty compared to
AutoReproduce: instead of free-form summaries, we produce a machine-readable
JSON specification that drives all subsequent code generation and verification.
"""

import json
import re
from typing import Optional

from specrepro.utils.llm import query_llm
from specrepro.utils.paper import read_paper, truncate_paper, clean_paper_text
from specrepro.spec.schema import (
    PaperSpec, ModelComponent, TrainingConfig,
    EvalMetric, AlgorithmStep, Status
)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research engineer specialized in reading AI/ML papers and extracting
structured, machine-readable specifications for code reproduction.

Your output must be a single valid JSON object — no extra commentary, no markdown
fences. If you are unsure about a value, use null or an empty list/dict.
"""


# ── Extraction prompt ─────────────────────────────────────────────────────────

def _build_extraction_prompt(paper_text: str, task_instruction: str) -> str:
    return f"""\
Read the following research paper and extract a structured specification (PaperSpec)
for reproducing the experiment described below.

TASK TO REPRODUCE:
{task_instruction}

PAPER TEXT:
{paper_text}

Extract and return ONLY a JSON object with this exact schema:
{{
  "title": "<paper title>",
  "arxiv_id": "<arxiv id if mentioned, else empty string>",
  "task": "<one-sentence description of what experiment to reproduce>",
  "dataset_name": "<name of dataset used>",
  "data_splits": {{"train": <int or null>, "val": <int or null>, "test": <int or null>}},
  "preprocessing": ["<step 1>", "<step 2>", "..."],
  "model_components": [
    {{
      "name": "<component name, e.g. TransformerEncoder>",
      "description": "<what this component does>",
      "hyperparams": {{"layers": <int>, "d_model": <int>, "...other params": "..."}},
      "input_shape": "<e.g. (B, T, C) or null>",
      "output_shape": "<e.g. (B, T, C) or null>",
      "status": "pending"
    }}
  ],
  "algorithms": [
    {{
      "name": "<algorithm or loss name>",
      "description": "<description including LaTeX formula if available>",
      "inputs": ["<tensor/variable names>"],
      "outputs": ["<tensor/variable names>"],
      "status": "pending"
    }}
  ],
  "training_config": {{
    "optimizer": "<Adam / SGD / AdamW / ...>",
    "learning_rate": <float>,
    "lr_schedule": "<cosine / step / none / description>",
    "epochs": <int>,
    "batch_size": <int>,
    "loss_functions": ["<loss1>", "<loss2>"],
    "regularization": ["<weight_decay=...>", "..."],
    "extra": {{"momentum": 0.9, "...other": "..."}},
    "status": "pending"
  }},
  "eval_metrics": [
    {{
      "name": "<Top-1 Accuracy / MSE / ADE / ...>",
      "dataset": "<dataset split, e.g. CIFAR-100 test>",
      "expected_value": <float or null>,
      "tolerance": 1.0,
      "higher_is_better": <true/false>,
      "status": "pending"
    }}
  ],
  "implementation_notes": [
    "<any important implementation detail not captured above>"
  ]
}}

Be thorough: capture ALL model components, ALL novel algorithms/losses, ALL
reported metric values from results tables. Each component must have all its
hyperparameters extracted from the paper (layer count, hidden dims, heads, etc.).
"""


# ── Main extractor ────────────────────────────────────────────────────────────

class SpecExtractor:
    """
    Uses an LLM to extract a PaperSpec from a paper.

    This is SpecRepro's first agent. It reads the paper text and produces
    a structured JSON spec — the intermediate representation that drives
    all downstream agents.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.max_retries = max_retries
        self.verbose = verbose

    def extract(
        self,
        paper_path: str,
        task_instruction: str,
        arxiv_id: str = "",
    ) -> PaperSpec:
        """
        Read the paper at paper_path and return a PaperSpec.
        """
        if self.verbose:
            print(f"[SpecExtractor] Reading paper: {paper_path}")
        raw_text = read_paper(paper_path)
        raw_text = clean_paper_text(raw_text)
        paper_text = truncate_paper(raw_text, max_chars=60_000)

        if self.verbose:
            print(f"[SpecExtractor] Paper length: {len(paper_text)} chars. Extracting spec...")

        prompt = _build_extraction_prompt(paper_text, task_instruction)

        for attempt in range(1, self.max_retries + 1):
            if self.verbose:
                print(f"[SpecExtractor] LLM call attempt {attempt}/{self.max_retries}")
            response = query_llm(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                model=self.model,
                temperature=0.1,
                max_tokens=4096,
            )
            spec = self._parse_response(response, arxiv_id)
            if spec is not None:
                if self.verbose:
                    print(f"[SpecExtractor] Extracted spec with "
                          f"{len(spec.model_components)} components, "
                          f"{len(spec.algorithms)} algorithms, "
                          f"{len(spec.eval_metrics)} metrics.")
                return spec
            if self.verbose:
                print(f"[SpecExtractor] Parse failed, retrying...")

        raise RuntimeError(
            f"SpecExtractor: failed to parse a valid PaperSpec after "
            f"{self.max_retries} attempts."
        )

    def _parse_response(self, response: str, arxiv_id: str) -> Optional[PaperSpec]:
        """Parse LLM response into a PaperSpec. Returns None on failure."""
        # Strip any accidental markdown fences
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract a JSON block from the middle of the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print(f"  [SpecExtractor] JSON parse error: {e}")
                    return None
            else:
                print(f"  [SpecExtractor] No JSON found in response: {e}")
                return None

        # Override arxiv_id if provided by caller
        if arxiv_id:
            data["arxiv_id"] = arxiv_id

        try:
            return PaperSpec.from_dict(data)
        except Exception as e:
            print(f"  [SpecExtractor] Schema construction error: {e}")
            return None
