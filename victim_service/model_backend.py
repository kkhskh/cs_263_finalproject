<<<<<<< HEAD
"""
Model Backend for LLM Side-Channel Research
Supports synthetic and real HuggingFace models with optional timing obfuscation.
"""

import os
import time
import math
import random
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Timing Obfuscation (Mitigation)
# =============================================================================

class TimingObfuscator:
    """
    Timing obfuscation to defeat fingerprinting attacks.
    
    Strategies:
      - random_delay: Add random delay to each request
      - bucket: Round response time to fixed buckets
      - constant: Pad all responses to constant time
    """
    
    def __init__(self, strategy: str = "none", param: float = 0.0):
        """
        Args:
            strategy: "none", "random", "bucket", or "constant"
            param: Strategy-specific parameter
                   - random: max delay in ms
                   - bucket: bucket size in ms
                   - constant: target time in ms
        """
        self.strategy = strategy
        self.param = param
        
    def obfuscate(self, start_time: float) -> None:
        """
        Apply timing obfuscation. Call after processing completes.
        
        Args:
            start_time: time.perf_counter() value when request started
        """
        if self.strategy == "none":
            return
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if self.strategy == "random":
            # Add random delay up to param ms
            delay_ms = random.uniform(0, self.param)
            time.sleep(delay_ms / 1000.0)
            
        elif self.strategy == "bucket":
            # Round up to next bucket
            bucket_ms = self.param
            target_ms = math.ceil(elapsed_ms / bucket_ms) * bucket_ms
            remaining_ms = target_ms - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000.0)
                
        elif self.strategy == "constant":
            # Pad to constant time
            target_ms = self.param
            remaining_ms = target_ms - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000.0)


# =============================================================================
# Synthetic Model Kernels
# =============================================================================

def _kernel_a(steps: int) -> float:
    """Kernel A: Trigonometric-heavy computation."""
    acc = 0.0
    for i in range(steps):
        acc += math.sin(i * 0.001) * math.cos(i * 0.002)
        acc += math.tan(i * 0.0001 + 0.01) * 0.001
=======
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
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b
    return acc


def _kernel_b(steps: int) -> float:
<<<<<<< HEAD
    """Kernel B: Memory-heavy computation with array operations."""
    size = min(steps, 10000)
    arr = [float(i) for i in range(size)]
    acc = 0.0
    for i in range(steps):
        idx = i % size
        arr[idx] = arr[idx] * 1.0001 + 0.1
        acc += arr[idx]
        if i % 100 == 0:
            acc += sum(arr[::10])
    return acc


def _kernel_c(steps: int) -> float:
    """Kernel C: Mixed integer and floating point operations."""
    acc = 0.0
    x = 1
    for i in range(steps):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        acc += float(x) / 2147483647.0
        acc *= 0.999999
    return acc


# =============================================================================
# Fake Model Implementation
# =============================================================================

@dataclass
class FakeModel:
    """Synthetic model with configurable computation kernel."""
    name: str
    default_steps: int
    kernel: Callable[[int], float]
    description: str = ""

    def generate(self, prompt: str, steps: Optional[int] = None) -> str:
        actual_steps = steps if steps is not None else self.default_steps
        result = self.kernel(actual_steps)
        return f"[fake-{self.name} acc={result:.3f}] Echo: {prompt}"


FAKE_MODELS: Dict[str, FakeModel] = {
    "fake_a": FakeModel("fake_a", 50000, _kernel_a, "Trigonometric kernel (~50ms)"),
    "fake_b": FakeModel("fake_b", 50000, _kernel_b, "Memory-heavy kernel (~60ms)"),
    "fake_c": FakeModel("fake_c", 50000, _kernel_c, "Mixed int/float kernel (~45ms)"),
}


# =============================================================================
# Real HuggingFace Model Implementation
# =============================================================================

class RealModelBackend:
    """Wrapper for real HuggingFace transformer models."""
    
    MODEL_REGISTRY = {
        "distilgpt2": ("distilgpt2", "causal"),
        "gpt2": ("gpt2", "causal"),
        "gpt2-medium": ("gpt2-medium", "causal"),
        "opt-125m": ("facebook/opt-125m", "causal"),
        "bert-tiny": ("prajjwal1/bert-tiny", "masked"),
        "distilbert": ("distilbert-base-uncased", "masked"),
    }
    
    def __init__(self, model_name: str):
        if model_name not in self.MODEL_REGISTRY:
            available = ", ".join(self.MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        self.model_name = model_name
        self.hf_model_id, self.model_type = self.MODEL_REGISTRY[model_name]
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def _ensure_loaded(self):
        if self._loaded:
            return
            
        logger.info(f"Loading model: {self.hf_model_id}")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
        
        torch.set_num_threads(1)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.hf_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        
        self.model.eval()
        self._loaded = True
        logger.info(f"Model {self.model_name} loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        self._ensure_loaded()
        
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            if self.model_type == "causal":
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = self.model(**inputs)
                return f"[{self.model_name}] Processed {outputs.logits.shape[1]} tokens"


# =============================================================================
# Unified Model Backend
# =============================================================================

class ModelBackend:
    """Unified backend for fake or real models with optional timing obfuscation."""
    
    def __init__(self, model_name: str, use_real: bool = False, 
                 obfuscation_strategy: str = "none", obfuscation_param: float = 0.0):
        self.model_name = model_name
        self.use_real = use_real
        self.obfuscator = TimingObfuscator(obfuscation_strategy, obfuscation_param)
        
        if use_real:
            self._backend = RealModelBackend(model_name)
        else:
            if model_name not in FAKE_MODELS:
                available = ", ".join(FAKE_MODELS.keys())
                raise ValueError(f"Unknown fake model '{model_name}'. Available: {available}")
            self._backend = FAKE_MODELS[model_name]
    
    def generate(self, prompt: str, **kwargs) -> tuple:
        """
        Generate output with timing obfuscation.
        Returns (output, actual_elapsed_ms, obfuscated_elapsed_ms)
        """
        start = time.perf_counter()
        
        if self.use_real:
            output = self._backend.generate(prompt, **kwargs)
        else:
            output = self._backend.generate(prompt, steps=kwargs.get("steps"))
        
        actual_elapsed = (time.perf_counter() - start) * 1000
        
        # Apply obfuscation
        self.obfuscator.obfuscate(start)
        
        obfuscated_elapsed = (time.perf_counter() - start) * 1000
        
        return output, actual_elapsed, obfuscated_elapsed
    
    @property
    def is_real(self) -> bool:
        return self.use_real
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, model in FAKE_MODELS.items():
            result[name] = {"type": "synthetic", "description": model.description}
        for name, (hf_id, mtype) in RealModelBackend.MODEL_REGISTRY.items():
            result[name] = {"type": "real", "hf_model_id": hf_id, "model_type": mtype}
        return result
=======
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
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b
