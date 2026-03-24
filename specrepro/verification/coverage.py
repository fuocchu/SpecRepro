"""
Coverage Checker: Verifies that every spec item is reflected in generated code.

This is the first half of SpecRepro's automated verification — it answers:
  "Did we implement everything the paper requires?"

For each spec item (ModelComponent, AlgorithmStep, TrainingConfig) we:
  1. Search for the item's name in the generated code (fast check).
  2. If not found by name, ask the LLM to verify presence using semantic search.
  3. Update the item's status accordingly.
"""

import re
from specrepro.spec.schema import PaperSpec, Status, ModelComponent, AlgorithmStep
from specrepro.utils.llm import query_llm


_COVERAGE_SYSTEM = """\
You are a code review expert. Your job is to check whether a specific paper
component has been implemented in the provided code.
Answer ONLY with "YES" or "NO" and a one-sentence reason.
"""


def _semantic_check_prompt(item_name: str, item_desc: str, code: str) -> str:
    return f"""\
Check if the following paper component is implemented in the code below.

COMPONENT: {item_name}
DESCRIPTION: {item_desc}

CODE:
```python
{code[:6000]}
```

Is this component implemented? Answer YES or NO and one sentence reason.
"""


class CoverageChecker:
    """
    Checks spec coverage against generated code.

    For each spec item:
      - A fast name-based search is done first.
      - If not found, an LLM semantic check is performed.
      - Items found are marked Status.IMPLEMENTED (if not already VERIFIED).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",   # Use fast/cheap model for coverage checks
        use_llm_fallback: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.use_llm_fallback = use_llm_fallback
        self.verbose = verbose

    def check(self, spec: PaperSpec, code: str) -> dict:
        """
        Check coverage of all spec items against code.
        Updates item statuses in-place.

        Returns a coverage_report dict:
          {
            "coverage": float (0.0–1.0),
            "implemented": int,
            "total": int,
            "missing": [{"name": ..., "type": ..., "reason": ...}]
          }
        """
        missing = []

        # ── Check model components ────────────────────────────────────────────
        for comp in spec.model_components:
            if comp.status == Status.VERIFIED:
                continue
            found = self._check_item(comp.name, comp.description, code)
            if found:
                comp.status = Status.IMPLEMENTED
            else:
                missing.append({"name": comp.name, "type": "ModelComponent",
                                 "reason": "Not found in generated code"})
                if self.verbose:
                    print(f"  [Coverage] MISSING component: {comp.name}")

        # ── Check algorithms ──────────────────────────────────────────────────
        for algo in spec.algorithms:
            if algo.status == Status.VERIFIED:
                continue
            found = self._check_item(algo.name, algo.description, code)
            if found:
                algo.status = Status.IMPLEMENTED
            else:
                missing.append({"name": algo.name, "type": "AlgorithmStep",
                                 "reason": "Not found in generated code"})
                if self.verbose:
                    print(f"  [Coverage] MISSING algorithm: {algo.name}")

        # ── Check training config ─────────────────────────────────────────────
        if spec.training_config and spec.training_config.status != Status.VERIFIED:
            # Training config is "found" if a training loop exists
            has_train_loop = (
                "def train" in code
                or "optimizer.step()" in code
                or "loss.backward()" in code
            )
            if has_train_loop:
                spec.training_config.status = Status.IMPLEMENTED
            else:
                missing.append({"name": "TrainingLoop", "type": "TrainingConfig",
                                 "reason": "No training loop found"})
                if self.verbose:
                    print("  [Coverage] MISSING: training loop")

        implemented = spec.implemented_items
        total = spec.total_items
        coverage = implemented / total if total > 0 else 0.0

        if self.verbose:
            print(f"  [Coverage] {implemented}/{total} items  ({coverage*100:.1f}%)")

        return {
            "coverage": coverage,
            "implemented": implemented,
            "total": total,
            "missing": missing,
        }

    def _check_item(self, name: str, description: str, code: str) -> bool:
        """Return True if the item is present in code (by name or semantic check)."""
        # Fast check: case-insensitive name search
        # e.g. "TransformerEncoder" → look for class/def TransformerEncoder
        normalized_name = name.replace(" ", "").replace("-", "")
        if re.search(rf"\b{re.escape(name)}\b", code, re.IGNORECASE):
            return True
        if re.search(rf"\b{re.escape(normalized_name)}\b", code, re.IGNORECASE):
            return True

        # Fallback: LLM semantic check (slower but more robust)
        if self.use_llm_fallback:
            prompt = _semantic_check_prompt(name, description, code)
            response = query_llm(
                prompt=prompt,
                system_prompt=_COVERAGE_SYSTEM,
                model=self.model,
                temperature=0.0,
                max_tokens=128,
                print_cost=False,
            )
            return response.strip().upper().startswith("YES")

        return False
