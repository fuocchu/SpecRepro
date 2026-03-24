"""
Result Verifier: Compares generated code's outputs against paper-reported values.

This is the second half of SpecRepro's verification.

For each EvalMetric in the spec that has an expected_value:
  1. Run the generated code (or read from a previous run's output).
  2. Parse metric values from stdout using "METRIC <name>: <value>" patterns.
  3. Compare against expected_value ± tolerance.
  4. Mark metric status as Status.VERIFIED or keep as IMPLEMENTED.

Also provides a human-readable gap report.
"""

import re
from typing import Optional
from specrepro.spec.schema import PaperSpec, EvalMetric, Status

_METRIC_RE = re.compile(
    r"METRIC\s+(?P<name>[^:]+):\s*(?P<value>[\d.]+)",
    re.IGNORECASE
)


class ResultVerifier:
    """
    Parses metric values from code execution output and compares against
    paper-reported expected values in the PaperSpec.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def verify(self, spec: PaperSpec, stdout: str) -> dict:
        """
        Parse stdout for metric values and compare against spec.

        Returns:
          {
            "verified_metrics": [{"name": ..., "actual": ..., "expected": ..., "within_tolerance": bool}],
            "unverified_metrics": [<metric names not found in stdout>],
            "overall_pass": bool,
          }
        """
        parsed = {}
        for m in _METRIC_RE.finditer(stdout):
            parsed[m.group("name").strip().lower()] = float(m.group("value"))

        verified = []
        unverified = []

        for metric in spec.eval_metrics:
            actual = self._find_metric(metric.name, parsed)

            if actual is None:
                unverified.append(metric.name)
                if self.verbose:
                    print(f"  [ResultVerifier] NOT FOUND in output: {metric.name}")
                continue

            if metric.expected_value is not None:
                gap = abs(actual - metric.expected_value)
                gap_pct = gap / abs(metric.expected_value) * 100 if metric.expected_value != 0 else gap
                within = gap_pct <= metric.tolerance
                entry = {
                    "name": metric.name,
                    "actual": actual,
                    "expected": metric.expected_value,
                    "gap": round(gap, 4),
                    "gap_pct": round(gap_pct, 2),
                    "within_tolerance": within,
                    "tolerance": metric.tolerance,
                }
                if within:
                    metric.status = Status.VERIFIED
                    if self.verbose:
                        print(f"  [ResultVerifier] VERIFIED  {metric.name}: "
                              f"{actual:.4f} (expected {metric.expected_value}, gap {gap_pct:.2f}%)")
                else:
                    if self.verbose:
                        print(f"  [ResultVerifier] GAP TOO LARGE  {metric.name}: "
                              f"{actual:.4f} vs expected {metric.expected_value} "
                              f"(gap {gap_pct:.2f}% > tolerance {metric.tolerance}%)")
            else:
                metric.status = Status.IMPLEMENTED
                entry = {
                    "name": metric.name,
                    "actual": actual,
                    "expected": None,
                    "gap": None,
                    "gap_pct": None,
                    "within_tolerance": True,
                    "tolerance": metric.tolerance,
                }
                if self.verbose:
                    print(f"  [ResultVerifier] FOUND (no expected value)  {metric.name}: {actual:.4f}")

            verified.append(entry)

        overall_pass = (
            len(unverified) == 0
            and all(e["within_tolerance"] for e in verified)
        )

        return {
            "verified_metrics": verified,
            "unverified_metrics": unverified,
            "overall_pass": overall_pass,
        }

    def _find_metric(self, name: str, parsed: dict) -> Optional[float]:
        """Case-insensitive partial match for metric names."""
        lower_name = name.lower()
        if lower_name in parsed:
            return parsed[lower_name]
        for key, val in parsed.items():
            if lower_name in key or key in lower_name:
                return val
        return None

    def gap_report(self, verify_result: dict, spec: PaperSpec) -> str:
        """Format a human-readable gap report."""
        lines = ["=== SpecRepro Result Verification Report ===", ""]
        vms = verify_result["verified_metrics"]
        if vms:
            lines.append("Metrics found in output:")
            for e in vms:
                status = "PASS" if e["within_tolerance"] else "FAIL"
                if e["expected"] is not None:
                    lines.append(
                        f"  [{status}]  {e['name']:40s}  "
                        f"actual={e['actual']:.4f}  expected={e['expected']}  "
                        f"gap={e['gap_pct']:.2f}%  tol=±{e['tolerance']}%"
                    )
                else:
                    lines.append(
                        f"  [----]  {e['name']:40s}  actual={e['actual']:.4f}  (no expected)"
                    )

        if verify_result["unverified_metrics"]:
            lines.append("\nMetrics NOT found in output:")
            for name in verify_result["unverified_metrics"]:
                lines.append(f"  [MISSING]  {name}")

        lines.append("")
        lines.append(f"Overall: {'PASS' if verify_result['overall_pass'] else 'FAIL'}")
        return "\n".join(lines)
