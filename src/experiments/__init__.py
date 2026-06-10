"""Experiment definitions for Snowdrop 4q ver2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qiskit import QuantumCircuit


@dataclass
class ParameterSpec:
    """Declares a tweakable parameter for an experiment (rendered in GUI)."""
    name: str
    label: str
    kind: str   # 'choice' | 'int' | 'text' | 'bool'
    default: Any
    choices: list = field(default_factory=list)
    min_value: int | None = None
    max_value: int | None = None
    help: str = ""

    def coerce(self, raw_value):
        if self.kind == "int":
            try:
                v = int(raw_value)
            except (TypeError, ValueError):
                return self.default
            if self.min_value is not None and v < self.min_value:
                return self.min_value
            if self.max_value is not None and v > self.max_value:
                return self.max_value
            return v
        if self.kind == "bool":
            return bool(raw_value)
        return str(raw_value)


@dataclass
class ExperimentDef:
    """Full description of an experiment (parameterizable, GUI-driven)."""
    key: str
    title: str
    description: str
    qubits_used: int
    parameters: list[ParameterSpec] = field(default_factory=list)

    build: Callable[[dict], QuantumCircuit] = None
    expected: Callable[[dict], dict[str, float]] = None
    metric_fn: Callable[[dict, dict], float] = None
    metric_name: str = ""
    interpretation_hint: str = ""

    def default_params(self) -> dict:
        return {p.name: p.default for p in self.parameters}

    def title_with_params(self, params: dict) -> str:
        if not params:
            return self.title
        bits = [f"{p.label}={params.get(p.name, p.default)}" for p in self.parameters]
        return f"{self.title}  [{', '.join(bits)}]"


def list_experiments() -> list[ExperimentDef]:
    """4 canonical experiments designed to fit on Snowdrop 4q ver2."""
    from src.experiments.bell_state import EXPERIMENT as bell
    from src.experiments.ghz_state import EXPERIMENT as ghz
    from src.experiments.bernstein_vazirani import EXPERIMENT as bv
    from src.experiments.shor_n15 import EXPERIMENT as shor
    return [bell, ghz, bv, shor]

