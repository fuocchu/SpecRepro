"""
SpecRepro Evaluation

Evaluates the quality of paper reproduction by computing:

  1. Spec Coverage Score — % of spec items implemented
  2. Result Alignment Score — how close generated metrics are to paper-reported values
  3. Code Quality Score — static analysis checks (syntax, imports, structure)

These three scores compose the overall SpecRepro Score.

Usage:
    python evaluation/eval.py \
        --spec output/specs/paper_spec.json \
        --code output/final_code.py \
        --run-code   # actually execute the code to get metric outputs
"""

import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specrepro.spec.schema import PaperSpec, Status
from specrepro.verification.coverage import CoverageChecker
from specrepro.verification.results import ResultVerifier
from specrepro.utils.code import execute_code


def compute_coverage_score(spec: PaperSpec) -> float:
    """
    Spec coverage score: fraction of spec items with status IMPLEMENTED or VERIFIED.
    Range: 0.0 – 1.0
    """
    return spec.coverage


def compute_result_alignment_score(spec: PaperSpec) -> float:
    """
    Result alignment score: fraction of eval metrics within tolerance.
    Metrics without expected_value are counted as aligned.
    Range: 0.0 – 1.0
    """
    metrics = spec.eval_metrics
    if not metrics:
        return 1.0  

    aligned = sum(
        1 for m in metrics
        if m.status == Status.VERIFIED or m.expected_value is None
    )
    return aligned / len(metrics)


def compute_code_quality_score(code: str) -> float:
    """
    Code quality score based on static checks:
      - Syntax validity (0.4 weight)
      - Has training loop (0.3 weight)
      - Has evaluation function (0.2 weight)
      - Has main() (0.1 weight)
    Range: 0.0 – 1.0
    """
    score = 0.0

    try:
        ast.parse(code)
        score += 0.4
    except SyntaxError:
        return 0.0 

    if "loss.backward()" in code or "optimizer.step()" in code:
        score += 0.3

    if "def eval" in code or "def test" in code or "def validate" in code:
        score += 0.2

    if "def main(" in code or 'if __name__' in code:
        score += 0.1

    return score


def compute_specrepro_score(
    spec: PaperSpec,
    code: str,
    weights: dict = None,
) -> dict:
    """
    Compute the overall SpecRepro Score and all subscores.

    Default weights:
      coverage     0.40  — Did we implement everything the spec requires?
      alignment    0.40  — Are numeric results close to paper-reported values?
      quality      0.20  — Is the code syntactically valid and well-structured?
    """
    if weights is None:
        weights = {"coverage": 0.40, "alignment": 0.40, "quality": 0.20}

    cov  = compute_coverage_score(spec)
    aln  = compute_result_alignment_score(spec)
    qual = compute_code_quality_score(code)

    overall = (
        weights["coverage"]  * cov +
        weights["alignment"] * aln +
        weights["quality"]   * qual
    )

    return {
        "specrepro_score": round(overall, 4),
        "coverage_score":  round(cov,  4),
        "alignment_score": round(aln,  4),
        "quality_score":   round(qual, 4),
        "weights": weights,
    }

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a SpecRepro reproduction run")
    p.add_argument("--spec", required=True, help="Path to paper_spec.json")
    p.add_argument("--code", required=True, help="Path to final_code.py")
    p.add_argument("--run-code", action="store_true",
                   help="Execute the code and parse metric outputs for result alignment")
    p.add_argument("--timeout", type=int, default=300,
                   help="Code execution timeout (seconds)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading spec: {args.spec}")
    spec = PaperSpec.load(args.spec)

    print(f"Loading code: {args.code}")
    with open(args.code, "r", encoding="utf-8") as f:
        code = f.read()

    if args.run_code:
        print("Running code for metric verification...")
        stdout, error = execute_code(code, timeout=args.timeout)
        if error:
            print(f"Code execution error:\n{error[:500]}")
        else:
            verifier = ResultVerifier(verbose=True)
            result = verifier.verify(spec, stdout)
            print(verifier.gap_report(result, spec))

    scores = compute_specrepro_score(spec, code)

    print("\n" + "="*50)
    print("  SpecRepro Evaluation Results")
    print("="*50)
    print(f"  Overall SpecRepro Score : {scores['specrepro_score']:.4f}")
    print(f"  Coverage Score          : {scores['coverage_score']:.4f}  (weight {scores['weights']['coverage']})")
    print(f"  Result Alignment Score  : {scores['alignment_score']:.4f}  (weight {scores['weights']['alignment']})")
    print(f"  Code Quality Score      : {scores['quality_score']:.4f}  (weight {scores['weights']['quality']})")
    print("="*50)
    print()
    print(spec.coverage_report())

    output_path = os.path.join(os.path.dirname(args.spec), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
