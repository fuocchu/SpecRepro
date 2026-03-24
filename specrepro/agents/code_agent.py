"""
CodeAgent: Spec-Guided Modular Code Generation

The key innovation over AutoReproduce's CodeAgent:
  - Generates code MODULE BY MODULE according to the PaperSpec, not as one
    monolithic block.
  - Each ModelComponent, AlgorithmStep, and TrainingConfig becomes its own
    code generation call.
  - After each module is generated and tested, the spec item is marked
    Status.IMPLEMENTED, giving live coverage tracking.
  - Assembled code is always runnable and can be executed at each step.
"""

import json
from typing import Optional

from specrepro.agents.base import BaseAgent
from specrepro.spec.schema import PaperSpec, ModelComponent, AlgorithmStep, TrainingConfig, Status
from specrepro.utils.code import (
    execute_code, extract_python_code, number_code_lines,
    edit_code_lines, save_code
)

_SYSTEM_PROMPT = """\
You are an expert ML engineer specializing in reproducing AI research papers.
You generate clean, runnable PyTorch code based on a structured paper specification.
Always generate complete, self-contained code snippets. Use standard libraries.
When writing code for a component, make sure it integrates with previously generated modules.
"""


def _dataloader_prompt(spec: PaperSpec, dataloader_code: str) -> str:
    return f"""\
PAPER: {spec.title}
TASK: {spec.task}
DATASET: {spec.dataset_name}
PREPROCESSING STEPS: {json.dumps(spec.preprocessing)}
DATA SPLITS: {json.dumps(spec.data_splits)}

EXISTING DATALOADER CODE (use and extend this):
```python
{dataloader_code}
```

Task: Generate or validate the data loading code.
  1. Verify the dataloader works (check shapes, dtypes, split sizes).
  2. Print the shape and dtype of the first batch for each data split.
  3. Add brief comments explaining each data dimension (batch, channels, height, width, etc.).

Return ONLY the complete Python code in a single ```python ... ``` block.
"""


def _component_prompt(component: ModelComponent, spec: PaperSpec, codebase: str) -> str:
    return f"""\
PAPER: {spec.title}
TASK: {spec.task}

Implement the following model component from the paper specification:

{component.to_prompt()}

EXISTING CODEBASE (so far):
```python
{codebase}
```

Rules:
  1. Generate PyTorch code for this specific component only.
  2. The class/function name must match the component name: {component.name}
  3. All hyperparameters from the spec must be constructor arguments with the paper's exact values as defaults.
  4. After the class definition, write a short smoke test that:
       - Instantiates the component with the spec's hyperparameters.
       - Passes a dummy tensor of the correct input shape ({component.input_shape or "infer from context"}).
       - Prints: "SHAPE_CHECK {component.name}: input=<shape> output=<shape>"
  5. Do NOT duplicate anything already in the codebase.

Return ONLY the code for this component + its smoke test in a ```python ... ``` block.
"""


def _algorithm_prompt(algo: AlgorithmStep, spec: PaperSpec, codebase: str) -> str:
    return f"""\
PAPER: {spec.title}
TASK: {spec.task}

Implement the following algorithm / loss function from the paper specification:

{algo.to_prompt()}

EXISTING CODEBASE (so far):
```python
{codebase}
```

Rules:
  1. Implement as a Python function or nn.Module as appropriate.
  2. Follow the formula exactly as described.
  3. After the implementation, write a smoke test that calls the function with dummy tensors
     and prints: "ALGO_CHECK {algo.name}: OK  output_shape=<shape>"
  4. Do NOT duplicate anything already in the codebase.

Return ONLY the implementation + smoke test in a ```python ... ``` block.
"""


def _training_prompt(tc: TrainingConfig, spec: PaperSpec, codebase: str) -> str:
    return f"""\
PAPER: {spec.title}
TASK: {spec.task}

Implement the complete training + evaluation loop using the training config below.

TRAINING CONFIG:
{tc.to_prompt()}

EVAL METRICS: {json.dumps([m.to_prompt() for m in spec.eval_metrics])}

EXISTING CODEBASE (so far — contains dataloader, model components, and loss functions):
```python
{codebase}
```

Rules:
  1. Define a train_one_epoch(model, loader, optimizer, criterion, device) function.
  2. Define an evaluate(model, loader, device) function that returns a dict of metric values.
  3. Define a main() function that:
       - Sets up device, model, optimizer, scheduler, loss functions using the spec values.
       - Runs 1 epoch of training (with a break after 2 batches so we can debug quickly).
       - Runs evaluation and prints: "METRIC <name>: <value>"
       - Prints: "TRAINING_LOOP: OK"
  4. End the file with: if __name__ == "__main__": main()
  5. Do NOT duplicate class/function definitions already in the codebase.

Return the complete updated code (codebase + training loop) in a ```python ... ``` block.
"""


def _debug_prompt(code: str, error: str, spec: PaperSpec) -> str:
    return f"""\
The following code produced an error during execution.

TASK: {spec.task}

ERROR:
{error}

CODE:
{number_code_lines(code)}

Diagnose the error and provide a corrected version.
Return ONLY the complete corrected code in a ```python ... ``` block.
Do not truncate or omit any part of the code.
"""

class CodeAgent(BaseAgent):
    """
    Generates code for each item in a PaperSpec, module by module.

    Generation phases:
      1. data_acquisition   — validate/generate dataloader
      2. component_<name>   — one call per ModelComponent in spec
      3. algorithm_<name>   — one call per AlgorithmStep in spec
      4. training_loop      — full train + eval loop
    """

    MAX_DEBUG_TRIES = 5

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        output_dir: str = "output",
        verbose: bool = True,
        print_cost: bool = True,
    ):
        super().__init__(model=model, verbose=verbose, print_cost=print_cost)
        self.output_dir = output_dir
        self.codebase: str = ""     
        self.checkpoints: list[str] = []

    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def data_acquisition(self, spec: PaperSpec, dataloader_code: str) -> str:
        """Generate and validate data loading code."""
        if self.verbose:
            print("\n[CodeAgent] ── Phase: data_acquisition ──")

        prompt = _dataloader_prompt(spec, dataloader_code)
        code = self._generate_and_debug(prompt, "data_acquisition", spec)
        self.codebase = code
        save_code(f"{self.output_dir}/ckpts", "data.py", code)
        return code

    def implement_component(self, component: ModelComponent, spec: PaperSpec) -> str:
        """Generate code for a single model component."""
        phase = f"component_{component.name}"
        if self.verbose:
            print(f"\n[CodeAgent] ── Phase: {phase} ──")

        prompt = _component_prompt(component, spec, self.codebase)
        new_code = self._generate_and_debug(prompt, phase, spec)

        self.codebase = self.codebase + "\n\n" + new_code
        component.status = Status.IMPLEMENTED
        save_code(f"{self.output_dir}/ckpts", f"{phase}.py", self.codebase)
        return self.codebase

    def implement_algorithm(self, algo: AlgorithmStep, spec: PaperSpec) -> str:
        """Generate code for a single algorithm/loss function."""
        phase = f"algorithm_{algo.name}"
        if self.verbose:
            print(f"\n[CodeAgent] ── Phase: {phase} ──")

        prompt = _algorithm_prompt(algo, spec, self.codebase)
        new_code = self._generate_and_debug(prompt, phase, spec)

        self.codebase = self.codebase + "\n\n" + new_code
        algo.status = Status.IMPLEMENTED
        save_code(f"{self.output_dir}/ckpts", f"{phase}.py", self.codebase)
        return self.codebase

    def implement_training_loop(self, spec: PaperSpec) -> str:
        """Generate the complete training + evaluation loop."""
        if self.verbose:
            print("\n[CodeAgent] ── Phase: training_loop ──")

        if spec.training_config is None:
            raise ValueError("PaperSpec has no training_config. Cannot generate training loop.")

        clean_cb = self._clean_codebase(self.codebase)
        prompt = _training_prompt(spec.training_config, spec, clean_cb)
        code = self._generate_and_debug(prompt, "training_loop", spec, max_tokens=8192)
        self.codebase = code
        spec.training_config.status = Status.IMPLEMENTED
        save_code(f"{self.output_dir}/ckpts", "run.py", code)
        return code

    @staticmethod
    def _clean_codebase(code: str) -> str:
        """
        Remove smoke-test blocks (the if __name__ == '__main__' / standalone calls
        that were added for shape checks) before passing codebase to training prompt.
        Keeps class/function definitions; removes ad-hoc print/assert statements
        at module level.
        """
        import ast, textwrap
        lines = code.split("\n")
        cleaned = []
        skip = False
        for line in lines:
            if any(marker in line for marker in [
                "SHAPE_CHECK", "ALGO_CHECK", "print(\"SHAPE_CHECK",
                "print(\"ALGO_CHECK", "print('SHAPE_CHECK", "print('ALGO_CHECK",
            ]):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _generate_and_debug(self, prompt: str, phase: str, spec: PaperSpec, max_tokens: int = 8192) -> str:
        """
        Generate code, try to execute it, and debug if needed.
        Returns the final working code (or best attempt after MAX_DEBUG_TRIES).
        """
        response = self.inference(prompt, phase, temperature=0.2, max_tokens=max_tokens)
        code = extract_python_code(response)

        for attempt in range(self.MAX_DEBUG_TRIES):
            stdout, error = execute_code(code, timeout=120)

            if error is None:
                if self.verbose:
                    print(f"  [CodeAgent] {phase}: code ran successfully.")
                    if stdout.strip():
                        print(f"  [stdout]\n{stdout[:500]}")
                return code

            if self.verbose:
                print(f"  [CodeAgent] {phase}: attempt {attempt+1} — error:\n{error[:400]}")

            if attempt < self.MAX_DEBUG_TRIES - 1:
                debug_prompt = _debug_prompt(code, error, spec)
                debug_response = self.inference(debug_prompt, f"{phase}_debug", temperature=0.1, max_tokens=max_tokens)
                code = extract_python_code(debug_response)
                save_code(f"{self.output_dir}/ckpts", f"{phase}_attempt{attempt+1}_fail.py", code)
            else:
                if self.verbose:
                    print(f"  [CodeAgent] {phase}: max debug tries reached. Returning best code.")
                save_code(f"{self.output_dir}/ckpts", f"{phase}_final_fail.py", code)

        return code
