"""
PaperSpec: Structured Paper Specification Schema

The core innovation of SpecRepro. Instead of free-form text summaries,
we extract a machine-readable JSON specification from the paper.
This spec acts as a "contract" between paper understanding and code generation.

Each spec item tracks its own implementation status, enabling:
  - Completeness verification (did we implement everything?)
  - Modular code generation (generate component by component)
  - Automated result verification (compare output against expected values)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

class Status:
    PENDING     = "pending"     
    IMPLEMENTED = "implemented" 
    VERIFIED    = "verified"     

@dataclass
class ModelComponent:
    """A single component of the model architecture."""
    name: str                       
    description: str                 
    hyperparams: dict = field(default_factory=dict)  
    input_shape: Optional[str] = None  
    output_shape: Optional[str] = None 
    status: str = Status.PENDING

    def to_prompt(self) -> str:
        hp = json.dumps(self.hyperparams, indent=2) if self.hyperparams else "not specified"
        lines = [
            f"Component: {self.name}",
            f"Description: {self.description}",
            f"Hyperparameters: {hp}",
        ]
        if self.input_shape:
            lines.append(f"Input shape: {self.input_shape}")
        if self.output_shape:
            lines.append(f"Output shape: {self.output_shape}")
        return "\n".join(lines)


@dataclass
class TrainingConfig:
    """Training hyperparameters and strategy."""
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    lr_schedule: Optional[str] = None   
    epochs: int = 100
    batch_size: int = 32
    loss_functions: list = field(default_factory=list)  
    regularization: list = field(default_factory=list)  
    extra: dict = field(default_factory=dict)           
    status: str = Status.PENDING

    def to_prompt(self) -> str:
        lines = [
            f"Optimizer: {self.optimizer}  (lr={self.learning_rate})",
            f"LR Schedule: {self.lr_schedule or 'none specified'}",
            f"Epochs: {self.epochs},  Batch size: {self.batch_size}",
            f"Loss functions: {self.loss_functions}",
            f"Regularization: {self.regularization}",
        ]
        if self.extra:
            lines.append(f"Extra: {json.dumps(self.extra)}")
        return "\n".join(lines)


@dataclass
class EvalMetric:
    """An evaluation metric with an optional expected value from the paper."""
    name: str                        
    dataset: str                     
    expected_value: Optional[float] = None   
    tolerance: float = 1.0             
    higher_is_better: bool = True
    status: str = Status.PENDING

    def to_prompt(self) -> str:
        exp = f"{self.expected_value}" if self.expected_value is not None else "not specified"
        return (f"Metric: {self.name} on {self.dataset}  "
                f"(expected ≈ {exp},  tolerance ±{self.tolerance}%)")


@dataclass
class AlgorithmStep:
    """A key algorithm or formula extracted from the paper."""
    name: str                
    description: str          
    inputs: list = field(default_factory=list)   
    outputs: list = field(default_factory=list)
    status: str = Status.PENDING

    def to_prompt(self) -> str:
        return (f"Algorithm: {self.name}\n"
                f"  {self.description}\n"
                f"  Inputs: {self.inputs}  →  Outputs: {self.outputs}")


@dataclass
class PaperSpec:
    """
    The Structured Paper Specification — the core intermediate representation
    of SpecRepro.

    Extracted from paper text by SpecAgent, then consumed by CodeAgent
    to drive modular code generation and by VerifyAgent to check completeness.
    """
    title: str = ""
    arxiv_id: str = ""
    task: str = ""                

    model_components: list = field(default_factory=list)  
    algorithms: list = field(default_factory=list)         
    training_config: Optional[TrainingConfig] = None

    eval_metrics: list = field(default_factory=list)       

    dataset_name: str = ""
    preprocessing: list = field(default_factory=list)     
    data_splits: dict = field(default_factory=dict)       
    implementation_notes: list = field(default_factory=list)
    @property
    def total_items(self) -> int:
        n = len(self.model_components) + len(self.algorithms) + len(self.eval_metrics)
        if self.training_config:
            n += 1
        return n

    @property
    def implemented_items(self) -> int:
        items = (self.model_components + self.algorithms + self.eval_metrics
                 + ([self.training_config] if self.training_config else []))
        return sum(1 for i in items if i.status in (Status.IMPLEMENTED, Status.VERIFIED))

    @property
    def coverage(self) -> float:
        """Implementation coverage: 0.0 – 1.0"""
        if self.total_items == 0:
            return 0.0
        return self.implemented_items / self.total_items

    def coverage_report(self) -> str:
        lines = [
            f"=== SpecRepro Coverage Report ===",
            f"Paper : {self.title}",
            f"Task  : {self.task}",
            f"",
            f"Coverage: {self.implemented_items}/{self.total_items} "
            f"items  ({self.coverage*100:.1f}%)",
            "",
            "── Model Components ──",
        ]
        for c in self.model_components:
            lines.append(f"  [{c.status:12s}]  {c.name}")
        lines.append("── Algorithms ──")
        for a in self.algorithms:
            lines.append(f"  [{a.status:12s}]  {a.name}")
        if self.training_config:
            lines.append("── Training Config ──")
            lines.append(f"  [{self.training_config.status:12s}]  TrainingConfig")
        lines.append("── Eval Metrics ──")
        for m in self.eval_metrics:
            exp = f"(expected {m.expected_value})" if m.expected_value is not None else ""
            lines.append(f"  [{m.status:12s}]  {m.name} on {m.dataset} {exp}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict) -> "PaperSpec":
        spec = cls(
            title=d.get("title", ""),
            arxiv_id=d.get("arxiv_id", ""),
            task=d.get("task", ""),
            dataset_name=d.get("dataset_name", ""),
            preprocessing=d.get("preprocessing", []),
            data_splits=d.get("data_splits", {}),
            implementation_notes=d.get("implementation_notes", []),
        )
        for c in d.get("model_components", []):
            spec.model_components.append(ModelComponent(**c))
        for a in d.get("algorithms", []):
            spec.algorithms.append(AlgorithmStep(**a))
        tc = d.get("training_config")
        if tc:
            spec.training_config = TrainingConfig(**tc)
        for m in d.get("eval_metrics", []):
            spec.eval_metrics.append(EvalMetric(**m))
        return spec

    @classmethod
    def load(cls, path: str) -> "PaperSpec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
