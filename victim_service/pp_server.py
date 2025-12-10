# victim_service/server.py

import csv
import os
import time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from model_backend import ModelBackend
from hf_model_backend import HFModelBackend
from covert_channel import maybe_send_covert

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
QUANT_MODE = os.getenv("QUANT_MODE", "fp16")

COVERT_ENABLED = os.getenv("COVERT_ENABLED", "0") == "1"
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQ_LOG_PATH = LOG_DIR / "requests.csv"

backend = HFModelBackend(MODEL_NAME, QUANT_MODE)


class GenerateRequest(BaseModel):
    prompt: str
    steps: int | None = None  # for fake models controls CPU work
    tag: str | None = None    # optional experiment label


class GenerateResponse(BaseModel):
    model_name: str
    output: str
    elapsed_ms: float
    covert_triggered: bool
    tag: str | None


def _append_request_log(
    prompt: str,
    tag: str | None,
    elapsed_ms: float,
    covert_triggered: bool,
) -> None:
    """Append one line to a CSV log (request-level metadata)."""
    with REQ_LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                time.time(),  # unix timestamp
                MODEL_NAME,
                tag or "",
                len(prompt),
                f"{elapsed_ms:.3f}",
                int(covert_triggered),
            ]
        )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    prompt = req.prompt
    steps = req.steps
    tag = req.tag

    covert_triggered = maybe_send_covert(prompt, enabled=COVERT_ENABLED)

    start = time.time()
    output = backend.generate(prompt, steps=steps)
    elapsed_ms = (time.time() - start) * 1000.0

    _append_request_log(prompt, tag, elapsed_ms, covert_triggered)

    return GenerateResponse(
        model_name=MODEL_NAME,
        output=output,
        elapsed_ms=elapsed_ms,
        covert_triggered=covert_triggered,
        tag=tag,
    )
