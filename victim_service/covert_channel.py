# victim_service/covert_channel.py

import logging
import time
from typing import Iterable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SENSITIVE_TOPICS = [
    "crypto",
    "password",
    "secret",
    "attack",
    "keys",
    "pii",
]


def bits_from_int(x: int, width: int = 6) -> str:
    return format(x, f"0{width}b")


def topic_to_id(prompt: str) -> int | None:
    """
    Map prompts mentioning SENSITIVE_TOPICS to a small integer ID.
    Up to 64 topics -> 6 bits.

    Return None if no sensitive topic detected.
    """
    prompt_lower = prompt.lower()
    for idx, word in enumerate(SENSITIVE_TOPICS):
        if word in prompt_lower:
            return idx
    return None


def _busy_loop(duration_ms: float) -> None:
    """Consume CPU and memory bandwidth for ~duration_ms."""
    end_time = time.perf_counter() + duration_ms / 1000.0
    x = 0.0
    arr = [0.0] * 128
    i = 0
    while time.perf_counter() < end_time:
        # arbitrary work that touches a small array
        x += (i % 7) * 1.0001
        arr[i % 128] = x
        i += 1


def _idle(duration_ms: float) -> None:
    """Sleep (mostly idle) for ~duration_ms."""
    time.sleep(duration_ms / 1000.0)


def send_bits(bitstring: str, slot_ms: float = 2.0) -> None:
    """
    Simple time-division covert channel:
    - Each bit gets a time slot of slot_ms milliseconds.
    - '1' => busy CPU (busy loop).
    - '0' => sleep.
    An external receiver that probes its own memory latency can decode this.
    """
    logger.info("CO-CHANNEL: sending bits=%s", bitstring)
    for bit in bitstring:
        t_start = time.perf_counter()
        if bit == "1":
            _busy_loop(slot_ms)
        else:
            _idle(slot_ms)

        # Align to slot boundary in case work finished early
        elapsed = (time.perf_counter() - t_start) * 1000.0
        if elapsed < slot_ms:
            _idle(slot_ms - elapsed)


def maybe_send_covert(prompt: str, enabled: bool) -> bool:
    """
    Decide whether to leak, and if so, transmit bits.
    Returns True if a covert transmission was attempted.
    """
    if not enabled:
        return False

    topic_id = topic_to_id(prompt)
    if topic_id is None:
        return False

    bits = bits_from_int(topic_id, width=6)
    send_bits(bits)
    return True
