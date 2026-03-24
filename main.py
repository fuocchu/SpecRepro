"""
SpecRepro

Usage:
    # Reproduce a paper using the default DKD example:
    python main.py

    # Reproduce a custom paper:
    python main.py \
        --paper-path path/to/paper.txt \
        --task "Reproduce experiment X on dataset Y" \
        --dataloader-path path/to/dataloader.py \
        --arxiv-id 2203.08679 \
        --model claude-sonnet-4-6 \
        --output-dir output

Environment variables (set before running):
    ANTHROPIC_API_KEY   — for Claude models (default)
    OPENAI_API_KEY      — for GPT models
"""

import argparse
import os
import sys

from specrepro import SpecReproPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpecRepro: Structured Specification-Driven Paper Reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                            
  python main.py --paper-path paper.txt \\
                 --task "Reproduce DKD..."   
        """,
    )

    parser.add_argument(
        "--paper-path", type=str,
        default="examples/dkd/paper.txt",
        help="Path to paper text file (.txt or .pdf)",
    )
    parser.add_argument(
        "--task", type=str,
        default=(
            "Reproduce the Decoupled Knowledge Distillation (DKD) experiment "
            "on CIFAR-100 using teacher=resnet32x4 and student=resnet8x4. "
            "Report Top-1 accuracy."
        ),
        help="Task description: what experiment to reproduce",
    )
    parser.add_argument(
        "--dataloader-path", type=str,
        default="examples/dkd/dataloader.py",
        help="Path to existing dataloader code (optional)",
    )
    parser.add_argument(
        "--arxiv-id", type=str, default="2203.08679",
        help="arXiv ID of the paper",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-6",
        help="LLM model for spec extraction and code generation",
    )
    parser.add_argument(
        "--verify-model", type=str, default="claude-haiku-4-5",
        help="LLM model for coverage verification (cheaper)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for all output files",
    )
    parser.add_argument(
        "--spec-only", action="store_true",
        help="Only extract the spec and print it, do not generate code",
    )
    parser.add_argument(
        "--from-spec", type=str, default=None,
        help="Skip spec extraction: load spec from this JSON file",
    )
    parser.add_argument(
        "--no-cost", action="store_true",
        help="Suppress LLM cost tracking output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)
    if args.model.startswith("gpt") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("  export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    dataloader_code = ""
    if args.dataloader_path and os.path.exists(args.dataloader_path):
        with open(args.dataloader_path, "r", encoding="utf-8") as f:
            dataloader_code = f.read()
    elif args.dataloader_path:
        print(f"Warning: dataloader path not found: {args.dataloader_path}")

    if args.spec_only:
        from specrepro.spec.extractor import SpecExtractor
        extractor = SpecExtractor(model=args.model)
        spec = extractor.extract(
            paper_path=args.paper_path,
            task_instruction=args.task,
            arxiv_id=args.arxiv_id,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        spec_path = os.path.join(args.output_dir, "paper_spec.json")
        spec.save(spec_path)
        print(f"\nSpec saved to: {spec_path}\n")
        print(spec.coverage_report())
        print("\nSpec JSON:")
        print(spec.to_json())
        return

    preloaded_spec = None
    if args.from_spec:
        from specrepro.spec.schema import PaperSpec
        print(f"Loading spec from: {args.from_spec}")
        preloaded_spec = PaperSpec.load(args.from_spec)

    pipeline = SpecReproPipeline(
        paper_path=args.paper_path,
        task_instruction=args.task,
        dataloader_code=dataloader_code,
        arxiv_id=args.arxiv_id,
        spec_model=args.model,
        code_model=args.model,
        verify_model=args.verify_model,
        output_dir=args.output_dir,
        verbose=True,
        print_cost=not args.no_cost,
    )

    if preloaded_spec is not None:
        pipeline.spec = preloaded_spec

    spec = pipeline.run()

    print("\nDone! Output files:")
    for root, _, files in os.walk(args.output_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {path}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
