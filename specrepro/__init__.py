"""
SpecRepro: Structured Specification-Driven Paper Reproduction

Key innovation over AutoReproduce:
  Instead of free-form text summaries as the intermediate representation,
  SpecRepro extracts a machine-readable PaperSpec (JSON) from the paper.
  This spec drives modular code generation and enables automated
  completeness verification and numeric result checking.

Usage:
    from specrepro import SpecReproPipeline

    pipeline = SpecReproPipeline(
        paper_path="examples/dkd/paper.txt",
        task_instruction="Reproduce DKD on CIFAR-100 with resnet32x4→resnet8x4",
        dataloader_code=open("examples/dkd/dataloader.py").read(),
        arxiv_id="2203.08679",
    )
    spec = pipeline.run()
    print(spec.coverage_report())
"""

from specrepro.pipeline import SpecReproPipeline
from specrepro.spec.schema import PaperSpec
