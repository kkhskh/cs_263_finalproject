# experiments/covert_receiver.py
"""
Simple receiver that measures latency of a local memory load as a proxy
for CPU contention caused by the model's covert channel.

This is not a proper FLUSH+RELOAD, but it gives us a sense of signal strength
even on non-x86 hardware.
"""

import time
import csv
from pathlib import Path


def measure(iterations: int = 5000) -> list[tuple[int, float]]:
    buf = bytearray(64)
    view = memoryview(buf)
    out = []

    for i in range(iterations):
        t0 = time.perf_counter()
        # "probe" load
        x = view[0]
        t1 = time.perf_counter()
        dt_ns = (t1 - t0) * 1e9
        out.append((i, dt_ns))

        # tiny sleep to avoid hogging the CPU completely
        time.sleep(0.0005)
        _ = x  # prevent optimization

    return out


def main():
    traces = measure(iterations=8000)
    out_path = Path("covert_trace.csv")
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "latency_ns"])
        w.writerows(traces)
    print(f"Wrote {len(traces)} samples to {out_path}")


if __name__ == "__main__":
    main()
