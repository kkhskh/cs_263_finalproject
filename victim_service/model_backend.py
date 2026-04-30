# victim_service/model_backend.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math


@dataclass
class FakeModel:
    name: str
    default_steps: int
    kernel: Callable[[int], float]

    def generate(self, prompt: str, steps: int | None = None) -> str:
        n = steps or self.default_steps
        acc = self.kernel(n)
        # acc just prevents compiler from optimizing away the work
        return f"[fake-{self.name} acc={acc:.3f}] {prompt[:128]}"


def _kernel_a(steps: int) -> float:
    """Fake model A: sin/cos pattern."""
    acc = 0.0
    for i in range(1, steps):
        acc += math.sin(i) * math.cos(i / 2.0)
    return acc


def _kernel_b(steps: int) -> float:
    """Fake model B: sqrt/log pattern with different cache behavior."""
    acc = 1.0
    for i in range(1, steps):
        acc += math.log(i + 1.0) * math.sqrt(i * 0.5)
    return acc


class ModelBackend:
    """
    Wrapper that can later be extended to real HuggingFace models.
    For now we expose two distinct compute patterns: fake_a, fake_b.
    """

    def __init__(self, name: str = "fake_a"):
        self.name = name

        if name == "fake_a":
            self.impl = FakeModel("a", default_steps=300_000, kernel=_kernel_a)
        elif name == "fake_b":
            self.impl = FakeModel("b", default_steps=300_000, kernel=_kernel_b)
        else:
            raise ValueError(f"Unknown model backend: {name!r}")

    def generate(self, prompt: str, steps: int | None = None) -> str:
        return self.impl.generate(prompt, steps=steps)
