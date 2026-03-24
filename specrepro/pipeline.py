"""
SpecRepro Pipeline

The main orchestrator that ties together all phases:

  Phase 1 — SPEC EXTRACTION
    SpecExtractor reads the paper text and produces a PaperSpec (structured JSON).

  Phase 2 — DATA ACQUISITION
    CodeAgent generates and validates the data loading code.

  Phase 3 — MODULAR CODE GENERATION
    For each ModelComponent in the spec → CodeAgent.implement_component()
    For each AlgorithmStep in the spec  → CodeAgent.implement_algorithm()

  Phase 4 — TRAINING LOOP
    CodeAgent generates the full training + evaluation loop.

  Phase 5 — COVERAGE VERIFICATION
    CoverageChecker checks that all spec items appear in the generated code.

  Phase 6 — RESULT VERIFICATION
    Runs the generated code and compares outputs against paper-reported values.

At each phase the PaperSpec is updated with implementation statuses,
providing a live progress report (coverage score).

Key difference from AutoReproduce:
  AutoReproduce uses a single ResearchAgent + CodeAgent with free-form summaries.
  SpecRepro uses a structured spec as the intermediate representation, enabling:
    1. Completeness verification (did we miss any component?)
    2. Modular generation (one LLM call per component, not one giant call)
    3. Numeric result verification (compare actual vs. expected metrics)
"""

import os
import json
import time
from typing import Optional

from specrepro.spec.schema import PaperSpec, Status
from specrepro.spec.extractor import SpecExtractor
from specrepro.agents.code_agent import CodeAgent
from specrepro.verification.coverage import CoverageChecker
from specrepro.verification.results import ResultVerifier
from specrepro.utils.code import execute_code, save_code
from specrepro.utils.paper import read_paper


class SpecReproPipeline:
    """
    End-to-end pipeline for reproducing a research paper using
    structured specification as an intermediate representation.
    """

    def __init__(
        self,
        paper_path: str,
        task_instruction: str,
        dataloader_code: str = "",
        arxiv_id: str = "",

        spec_model: str = "claude-sonnet-4-6",
        code_model: str = "claude-sonnet-4-6",
        verify_model: str = "claude-haiku-4-5",

        output_dir: str = "output",
        verbose: bool = True,
        print_cost: bool = True,
    ):
        self.paper_path = paper_path
        self.task_instruction = task_instruction
        self.dataloader_code = dataloader_code
        self.arxiv_id = arxiv_id
        self.output_dir = output_dir
        self.verbose = verbose
        self.print_cost = print_cost

        self.spec_extractor = SpecExtractor(model=spec_model, verbose=verbose)
        self.code_agent = CodeAgent(
            model=code_model, output_dir=output_dir,
            verbose=verbose, print_cost=print_cost
        )
        self.coverage_checker = CoverageChecker(
            model=verify_model, verbose=verbose
        )
        self.result_verifier = ResultVerifier(verbose=verbose)

        self.spec: Optional[PaperSpec] = None
        self.phase_times: dict[str, float] = {}
        self.final_code: str = ""
        self.coverage_report: dict = {}
        self.verify_report: dict = {}

        os.makedirs(os.path.join(output_dir, "ckpts"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "specs"), exist_ok=True)


    def run(self) -> PaperSpec:
        """
        Run the full SpecRepro pipeline.
        Returns the final PaperSpec with all status fields updated.
        """
        self._banner("SpecRepro: Starting Pipeline")

        self.spec = self._run_phase(
            "spec_extraction", self._phase_spec_extraction
        )
        self.spec.save(os.path.join(self.output_dir, "specs", "paper_spec.json"))
        if self.verbose:
            print(self.spec.coverage_report())

        self._run_phase("data_acquisition", self._phase_data_acquisition)
        self._save_spec()

        self._run_phase("component_generation", self._phase_component_generation)
        self._run_phase("algorithm_generation", self._phase_algorithm_generation)
        self._save_spec()

        self._run_phase("training_loop", self._phase_training_loop)
        self._save_spec()

        self._run_phase("coverage_verification", self._phase_coverage_verification)
        self._save_spec()

        self._run_phase("result_verification", self._phase_result_verification)
        self._save_spec()

        self._print_final_report()
        return self.spec

    def _phase_spec_extraction(self) -> PaperSpec:
        return self.spec_extractor.extract(
            paper_path=self.paper_path,
            task_instruction=self.task_instruction,
            arxiv_id=self.arxiv_id,
        )

    def _phase_data_acquisition(self):
        self.code_agent.data_acquisition(self.spec, self.dataloader_code)

    def _phase_component_generation(self):
        for comp in self.spec.model_components:
            if comp.status == Status.VERIFIED:
                continue
            self.code_agent.implement_component(comp, self.spec)
            if self.verbose:
                print(f"  → Coverage: {self.spec.coverage*100:.1f}%")

    def _phase_algorithm_generation(self):
        for algo in self.spec.algorithms:
            if algo.status == Status.VERIFIED:
                continue
            self.code_agent.implement_algorithm(algo, self.spec)
            if self.verbose:
                print(f"  → Coverage: {self.spec.coverage*100:.1f}%")

    def _phase_training_loop(self):
        if self.spec.training_config is not None:
            self.final_code = self.code_agent.implement_training_loop(self.spec)
        else:
            if self.verbose:
                print("  [Pipeline] No training_config in spec — skipping training loop.")
            self.final_code = self.code_agent.codebase

        save_code(self.output_dir, "final_code.py", self.final_code)

    def _phase_coverage_verification(self):
        self.coverage_report = self.coverage_checker.check(
            spec=self.spec,
            code=self.final_code,
        )
        if self.verbose:
            print(self.spec.coverage_report())

    def _phase_result_verification(self):
        if self.verbose:
            print("  [Pipeline] Running final code for result verification...")

        stdout, error = execute_code(self.final_code, timeout=300)

        if error:
            if self.verbose:
                print(f"  [Pipeline] Code execution error:\n{error[:500]}")
            self.verify_report = {
                "verified_metrics": [],
                "unverified_metrics": [m.name for m in self.spec.eval_metrics],
                "overall_pass": False,
                "execution_error": error,
            }
            return

        self.verify_report = self.result_verifier.verify(self.spec, stdout)
        if self.verbose:
            print(self.result_verifier.gap_report(self.verify_report, self.spec))

    def _run_phase(self, phase_name: str, fn):
        self._banner(f"Phase: {phase_name}")
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        self.phase_times[phase_name] = elapsed
        if self.verbose:
            print(f"  [Pipeline] {phase_name} completed in {elapsed:.1f}s")
        return result

    def _save_spec(self):
        if self.spec:
            self.spec.save(os.path.join(self.output_dir, "specs", "paper_spec.json"))

    def _banner(self, msg: str):
        if self.verbose:
            print(f"\n{'='*60}\n  {msg}\n{'='*60}")

    def _print_final_report(self):
        self._banner("SpecRepro: Final Report")
        print(self.spec.coverage_report())
        print()
        if self.verify_report:
            print(self.result_verifier.gap_report(self.verify_report, self.spec))
        print()
        print("Phase timing:")
        for phase, t in self.phase_times.items():
            print(f"  {phase:30s}  {t:.1f}s")
        total = sum(self.phase_times.values())
        print(f"  {'TOTAL':30s}  {total:.1f}s")
        print()
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
